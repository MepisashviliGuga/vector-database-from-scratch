//! Axis 2: when and what to merge.
//!
//! Orthogonal to the growth scheme. A growth scheme says how large level `i` may
//! be; a compaction policy decides what to do about it when that budget is
//! exceeded. Phase 3 varies one while holding the other fixed, so the two must
//! not know about each other's internals — they meet only through
//! [`GrowthScheme::level_capacity_bytes`].
//!
//! # Runs, and why they are not files
//!
//! A **run** is a sorted sequence of entries with no duplicate keys. A run may
//! be split across several SSTable files with disjoint key ranges — searching a
//! run means finding the one file whose range covers the key, so a run costs one
//! lookup no matter how many files it contains.
//!
//! The number of *runs* is what drives read cost; the number of *files* only
//! bounds how much has to be rewritten per compaction. Conflating the two makes
//! leveling look far more expensive than it is, which would poison the Phase 3
//! comparison. Hence [`RunShape`] rather than a flat file list.
//!
//! # The two baselines
//!
//! - **[`Leveling`]**: one run per level. Cheap reads (one run to probe per
//!   level), expensive writes (a record is rewritten every time its level takes
//!   in new data).
//! - **[`Tiering`]**: up to `T` runs per level, merged all at once when full.
//!   Cheap writes (a record moves down a level once per `T` arrivals),
//!   expensive reads (`T` runs to probe per level).
//!
//! These sit at opposite ends of the read-write trade-off. EcoTune (Phase 2)
//! frames the choice between them as an investment decision rather than a fixed
//! configuration, which is only meaningful once both endpoints are measurable.

use std::fmt::Debug;

use super::growth::GrowthScheme;
use super::Key;

pub mod leveling;
pub mod tiering;

pub use leveling::Leveling;
pub use tiering::Tiering;

/// One run's summary: enough for a policy to reason about size and overlap,
/// without touching the files themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunShape {
    /// Position within its level, newest first.
    pub index: usize,
    pub bytes: u64,
    /// `None` for an empty run, which cannot overlap anything.
    pub min_key: Option<Key>,
    pub max_key: Option<Key>,
}

impl RunShape {
    /// Whether two runs' key ranges intersect. Non-overlapping runs can be moved
    /// between levels without merging anything.
    pub fn overlaps(&self, other: &RunShape) -> bool {
        match (&self.min_key, &self.max_key, &other.min_key, &other.max_key) {
            (Some(a_min), Some(a_max), Some(b_min), Some(b_max)) => a_min <= b_max && b_min <= a_max,
            _ => false,
        }
    }

    /// Whether this run's range intersects the closed interval `[min, max]`.
    pub fn overlaps_range(&self, min: &Key, max: &Key) -> bool {
        match (&self.min_key, &self.max_key) {
            (Some(self_min), Some(self_max)) => self_min <= max && min <= self_max,
            _ => false,
        }
    }
}

/// One level's summary.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LevelShape {
    /// Runs, newest first. That ordering is what the read path relies on.
    pub runs: Vec<RunShape>,
}

impl LevelShape {
    pub fn from_sizes(sizes: &[u64]) -> Self {
        Self {
            runs: sizes
                .iter()
                .enumerate()
                .map(|(index, &bytes)| RunShape {
                    index,
                    bytes,
                    min_key: None,
                    max_key: None,
                })
                .collect(),
        }
    }

    pub fn bytes(&self) -> u64 {
        self.runs.iter().map(|run| run.bytes).sum()
    }

    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }
}

/// The whole tree's shape, as a policy sees it.
#[derive(Debug, Clone, Default)]
pub struct TreeShape {
    /// Level 0 first.
    pub levels: Vec<LevelShape>,
}

impl TreeShape {
    pub fn total_bytes(&self) -> u64 {
        self.levels.iter().map(LevelShape::bytes).sum()
    }

    pub fn level(&self, level: usize) -> Option<&LevelShape> {
        self.levels.get(level)
    }

    /// Bytes in `level`, or 0 if it does not exist.
    pub fn level_bytes(&self, level: usize) -> u64 {
        self.levels.get(level).map_or(0, LevelShape::bytes)
    }

    /// Deepest level holding anything. `None` for an empty tree.
    pub fn bottom_level(&self) -> Option<usize> {
        self.levels
            .iter()
            .rposition(|level| !level.is_empty())
    }
}

/// A unit of compaction work: merge these runs, write the result to
/// `target_level`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionJob {
    pub source_level: usize,
    /// Indices into `levels[source_level].runs`, newest first.
    pub source_runs: Vec<usize>,
    pub target_level: usize,
    /// Runs in the target level that overlap and must be rewritten. Empty for
    /// tiering, which appends a run rather than merging into one.
    pub target_runs: Vec<usize>,
    /// Whether tombstones may be discarded.
    ///
    /// True only when the output is the deepest level holding data, so nothing
    /// older can exist for a tombstone to shadow. Setting this anywhere else
    /// resurrects deleted keys.
    pub drop_tombstones: bool,
}

impl CompactionJob {
    /// Total input runs, which is what the merge iterator will be built over.
    pub fn input_run_count(&self) -> usize {
        self.source_runs.len() + self.target_runs.len()
    }
}

/// When and what to merge.
///
/// Implementations are pure decision functions over [`TreeShape`]: no I/O, no
/// interior state. That is deliberate — it makes both baselines and, later,
/// EcoTune's dynamic-programming scheduler unit-testable against handcrafted
/// tree shapes rather than only through a running database.
pub trait CompactionPolicy: Debug + Send + Sync {
    fn name(&self) -> &'static str;

    /// Pick the next compaction, or `None` if the tree is within budget.
    ///
    /// Called after every flush and after every completed compaction, so a
    /// policy that always returns work would loop forever; each job must
    /// strictly reduce the condition that triggered it.
    fn pick(&self, tree: &TreeShape, growth: &dyn GrowthScheme) -> Option<CompactionJob>;

    /// Runs a level may hold before it is considered full. 1 for leveling, `T`
    /// for tiering. Used to size read-path expectations in benchmarks.
    fn runs_per_level(&self) -> usize;
}

/// Whether every level strictly below `target_level` is empty.
///
/// A necessary condition for discarding tombstones, but **not a sufficient
/// one** — the target level itself may still hold older runs. Each policy adds
/// its own reasoning about those:
///
/// - Leveling consumes every target run overlapping the merged key range, and
///   the runs it leaves behind are disjoint from that range by construction, so
///   they cannot hold a key the output has a tombstone for. This condition alone
///   is enough.
/// - Tiering never consumes target runs at all, and the runs it leaves behind
///   *do* overlap. It must additionally require the target level to be empty.
///
/// Getting this wrong is the same catastrophic bug either way: a tombstone
/// dropped while a live entry survives below it silently resurrects a deleted
/// key, with no error and no way to notice until the data is read.
pub(crate) fn is_deepest_level(tree: &TreeShape, target_level: usize) -> bool {
    tree.levels
        .iter()
        .enumerate()
        .all(|(level, shape)| level <= target_level || shape.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(s: &str) -> Key {
        s.as_bytes().to_vec()
    }

    fn run(index: usize, bytes: u64, min: &str, max: &str) -> RunShape {
        RunShape {
            index,
            bytes,
            min_key: Some(k(min)),
            max_key: Some(k(max)),
        }
    }

    #[test]
    fn overlap_detection_handles_touching_and_disjoint_ranges() {
        let left = run(0, 100, "a", "m");
        let right = run(1, 100, "n", "z");
        let straddling = run(2, 100, "f", "t");
        let touching = run(3, 100, "m", "q");

        assert!(!left.overlaps(&right));
        assert!(left.overlaps(&straddling));
        assert!(
            left.overlaps(&touching),
            "ranges sharing an endpoint overlap: the key 'm' is in both"
        );
        assert!(left.overlaps(&left));
    }

    #[test]
    fn an_empty_run_overlaps_nothing() {
        let empty = RunShape {
            index: 0,
            bytes: 0,
            min_key: None,
            max_key: None,
        };
        assert!(!empty.overlaps(&run(1, 100, "a", "z")));
        assert!(!run(1, 100, "a", "z").overlaps(&empty));
        assert!(!empty.overlaps_range(&k("a"), &k("z")));
    }

    #[test]
    fn range_overlap_matches_run_overlap() {
        let subject = run(0, 100, "d", "h");
        assert!(subject.overlaps_range(&k("a"), &k("e")));
        assert!(subject.overlaps_range(&k("e"), &k("f")));
        assert!(subject.overlaps_range(&k("h"), &k("z")));
        assert!(!subject.overlaps_range(&k("a"), &k("c")));
        assert!(!subject.overlaps_range(&k("i"), &k("z")));
    }

    #[test]
    fn tree_shape_reports_sizes_and_the_bottom_level() {
        let tree = TreeShape {
            levels: vec![
                LevelShape::from_sizes(&[10, 20]),
                LevelShape::from_sizes(&[300]),
                LevelShape::default(),
            ],
        };

        assert_eq!(tree.level_bytes(0), 30);
        assert_eq!(tree.level_bytes(1), 300);
        assert_eq!(tree.level_bytes(2), 0);
        assert_eq!(tree.level_bytes(99), 0);
        assert_eq!(tree.total_bytes(), 330);
        assert_eq!(
            tree.bottom_level(),
            Some(1),
            "an empty trailing level is not the bottom"
        );
    }

    #[test]
    fn an_empty_tree_has_no_bottom_level() {
        assert_eq!(TreeShape::default().bottom_level(), None);
        assert_eq!(
            TreeShape {
                levels: vec![LevelShape::default()]
            }
            .bottom_level(),
            None
        );
    }

    #[test]
    fn the_deepest_level_is_the_one_with_nothing_under_it() {
        let tree = TreeShape {
            levels: vec![
                LevelShape::from_sizes(&[10]),
                LevelShape::from_sizes(&[100]),
                LevelShape::from_sizes(&[1000]),
            ],
        };

        assert!(!is_deepest_level(&tree, 0));
        assert!(!is_deepest_level(&tree, 1));
        assert!(is_deepest_level(&tree, 2));
        assert!(
            is_deepest_level(&tree, 3),
            "writing below the current bottom creates the new bottom"
        );

        assert!(
            is_deepest_level(&TreeShape::default(), 0),
            "in an empty tree there is nothing for a tombstone to shadow"
        );
    }

    /// An empty trailing level must not make a shallower level look deepest.
    #[test]
    fn trailing_empty_levels_do_not_confuse_the_check() {
        let tree = TreeShape {
            levels: vec![
                LevelShape::from_sizes(&[10]),
                LevelShape::default(),
                LevelShape::from_sizes(&[1000]),
            ],
        };
        assert!(!is_deepest_level(&tree, 0));
        assert!(
            !is_deepest_level(&tree, 1),
            "level 2 still holds data below level 1"
        );
        assert!(is_deepest_level(&tree, 2));
    }
}
