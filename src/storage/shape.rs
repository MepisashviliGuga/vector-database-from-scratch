//! A read-only view of the tree's structure, shared by both design axes.
//!
//! The growth scheme (when to compact, and at what granularity) and the merge
//! policy (how to merge) both need to see the tree, and neither should need the
//! other. Putting the view here keeps them independent, which is what lets
//! Phase 3 vary one while holding the other fixed.
//!
//! # Files, runs, levels
//!
//! A **run** is a sorted sequence with no duplicate keys, split across one or
//! more **files** with disjoint key ranges. Searching a run costs one probe
//! whatever its file count, so read cost tracks *runs*; file count bounds how
//! finely a compaction can be sliced.
//!
//! That second point is why files appear here at all. Full compaction merges
//! whole runs; partial compaction merges a few files and the ones they overlap
//! below. Paper 01 shows the choice is not cosmetic — full compaction is the
//! mechanism behind the horizontal scheme's space cost, since inputs cannot be
//! freed until a merge of the entire level completes.

use super::Key;

/// One file's summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileShape {
    /// Position within its run, ascending by key.
    pub index: usize,
    pub bytes: u64,
    pub min_key: Option<Key>,
    pub max_key: Option<Key>,
}

impl FileShape {
    /// Whether this file's range intersects the closed interval `[min, max]`.
    pub fn overlaps_range(&self, min: &Key, max: &Key) -> bool {
        match (&self.min_key, &self.max_key) {
            (Some(self_min), Some(self_max)) => self_min <= max && min <= self_max,
            _ => false,
        }
    }
}

/// One run's summary: its files, in ascending key order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunShape {
    /// Position within its level, newest first.
    pub index: usize,
    pub files: Vec<FileShape>,
    /// How many *unit runs* this run represents.
    ///
    /// A unit run is one memtable flush's worth of data; a run produced by
    /// merging others carries the sum of theirs. Only EcoTune uses this: its
    /// schedule specifies merge widths in unit runs rather than in physical
    /// runs, because a width-4 merge may absorb one previously-merged width-2
    /// run plus two new ones. Every other scheme ignores it.
    pub units: usize,
}

impl RunShape {
    pub fn bytes(&self) -> u64 {
        self.files.iter().map(|file| file.bytes).sum()
    }

    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Smallest key in the run, taken from its first file.
    pub fn min_key(&self) -> Option<&Key> {
        self.files.first().and_then(|file| file.min_key.as_ref())
    }

    /// Largest key in the run, taken from its last file.
    pub fn max_key(&self) -> Option<&Key> {
        self.files.last().and_then(|file| file.max_key.as_ref())
    }

    /// Whether two runs' key ranges intersect.
    pub fn overlaps(&self, other: &RunShape) -> bool {
        match (other.min_key(), other.max_key()) {
            (Some(min), Some(max)) => self.overlaps_range(min, max),
            _ => false,
        }
    }

    /// Whether this run intersects the closed interval `[min, max]`.
    pub fn overlaps_range(&self, min: &Key, max: &Key) -> bool {
        match (self.min_key(), self.max_key()) {
            (Some(self_min), Some(self_max)) => self_min <= max && min <= self_max,
            _ => false,
        }
    }

    /// Indices of the files intersecting `[min, max]`.
    pub fn files_overlapping(&self, min: &Key, max: &Key) -> Vec<usize> {
        self.files
            .iter()
            .filter(|file| file.overlaps_range(min, max))
            .map(|file| file.index)
            .collect()
    }
}

/// One level's summary.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LevelShape {
    /// Runs, newest first — the order the read path consults them in.
    pub runs: Vec<RunShape>,
}

impl LevelShape {
    /// Build a level of single-file runs from sizes alone, for tests that do not
    /// care about key ranges.
    pub fn from_sizes(sizes: &[u64]) -> Self {
        Self {
            runs: sizes
                .iter()
                .enumerate()
                .map(|(index, &bytes)| RunShape {
                    index,
                    files: vec![FileShape {
                        index: 0,
                        bytes,
                        min_key: None,
                        max_key: None,
                    }],
                    units: 1,
                })
                .collect(),
        }
    }

    pub fn bytes(&self) -> u64 {
        self.runs.iter().map(RunShape::bytes).sum()
    }

    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    pub fn file_count(&self) -> usize {
        self.runs.iter().map(RunShape::file_count).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.runs.iter().all(RunShape::is_empty)
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

    fn file(index: usize, bytes: u64, min: &str, max: &str) -> FileShape {
        FileShape {
            index,
            bytes,
            min_key: Some(k(min)),
            max_key: Some(k(max)),
        }
    }

    /// A run of one file spanning `min..max`.
    fn run(index: usize, bytes: u64, min: &str, max: &str) -> RunShape {
        RunShape {
            index,
            files: vec![file(0, bytes, min, max)],
            units: 1,
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
            files: Vec::new(),
            units: 1,
        };
        assert!(!empty.overlaps(&run(1, 100, "a", "z")));
        assert!(!run(1, 100, "a", "z").overlaps(&empty));
        assert!(!empty.overlaps_range(&k("a"), &k("z")));
    }

    /// A run's range spans its files: first file's min to last file's max.
    #[test]
    fn a_multi_file_run_reports_its_full_span() {
        let subject = RunShape {
            index: 0,
            files: vec![
                file(0, 10, "a", "c"),
                file(1, 10, "d", "m"),
                file(2, 10, "n", "z"),
            ],
            units: 1,
        };

        assert_eq!(subject.min_key(), Some(&k("a")));
        assert_eq!(subject.max_key(), Some(&k("z")));
        assert_eq!(subject.bytes(), 30);
        assert_eq!(subject.file_count(), 3);
    }

    /// The point of tracking files: a partial compaction consumes only the
    /// files a key range actually touches.
    #[test]
    fn only_the_files_intersecting_a_range_are_selected() {
        let subject = RunShape {
            index: 0,
            files: vec![
                file(0, 10, "a", "c"),
                file(1, 10, "d", "m"),
                file(2, 10, "n", "z"),
            ],
            units: 1,
        };

        assert_eq!(subject.files_overlapping(&k("e"), &k("f")), vec![1]);
        assert_eq!(subject.files_overlapping(&k("c"), &k("e")), vec![0, 1]);
        assert_eq!(subject.files_overlapping(&k("a"), &k("z")), vec![0, 1, 2]);
        assert!(subject.files_overlapping(&k("A"), &k("B")).is_empty());
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
