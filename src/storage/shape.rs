//! A read-only view of the tree's structure, shared by both design axes.
//!
//! The growth scheme (when to compact) and the merge policy (how) both need to
//! see the tree, and neither should need the other. Putting the view here keeps
//! them independent, which is what lets Phase 3 vary one while holding the other
//! fixed.
//!
//! # Runs, not files
//!
//! A **run** is a sorted sequence with no duplicate keys, possibly split across
//! several files with disjoint key ranges. Searching a run costs one probe
//! whatever its file count, so read cost tracks *runs*; file count only bounds
//! how much a compaction rewrites. Conflating them makes leveling look far more
//! expensive than it is.

use super::Key;

/// One run's summary: enough to reason about size and overlap without touching
/// the files.
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
    /// Whether two runs' key ranges intersect.
    pub fn overlaps(&self, other: &RunShape) -> bool {
        match (&self.min_key, &self.max_key, &other.min_key, &other.max_key) {
            (Some(a_min), Some(a_max), Some(b_min), Some(b_max)) => {
                a_min <= b_max && b_min <= a_max
            }
            _ => false,
        }
    }

    /// Whether this run intersects the closed interval `[min, max]`.
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
    /// Runs, newest first — the order the read path consults them in.
    pub runs: Vec<RunShape>,
}

impl LevelShape {
    /// Build a level from run sizes alone, for tests that do not care about key
    /// ranges.
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

/// The whole tree's shape.
#[derive(Debug, Clone, Default)]
pub struct TreeShape {
    /// Level 0 first. Level 0 receives memtable flushes.
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

    /// Runs in `level`, or 0 if it does not exist.
    pub fn run_count(&self, level: usize) -> usize {
        self.levels.get(level).map_or(0, LevelShape::run_count)
    }

    /// Deepest level holding anything. `None` for an empty tree.
    pub fn bottom_level(&self) -> Option<usize> {
        self.levels.iter().rposition(|level| !level.is_empty())
    }
}

/// Whether every level strictly below `target_level` is empty.
///
/// A necessary condition for discarding tombstones, but **not sufficient** — the
/// target level itself may hold older runs. Each merge policy adds its own
/// reasoning about those; see `compaction::leveling` and `compaction::tiering`.
///
/// Getting this wrong resurrects deleted keys, silently and with no error.
pub fn is_deepest_level(tree: &TreeShape, target_level: usize) -> bool {
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
        assert_eq!(tree.run_count(0), 2);
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
        assert!(is_deepest_level(&TreeShape::default(), 0));
    }

    #[test]
    fn trailing_empty_levels_do_not_confuse_the_check() {
        let tree = TreeShape {
            levels: vec![
                LevelShape::from_sizes(&[10]),
                LevelShape::default(),
                LevelShape::from_sizes(&[1000]),
            ],
        };
        assert!(
            !is_deepest_level(&tree, 1),
            "level 2 still holds data below level 1"
        );
        assert!(is_deepest_level(&tree, 2));
    }
}
