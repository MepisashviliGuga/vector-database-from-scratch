//! Leveling: merge into the run already at the target level.
//!
//! The read-optimised policy, and what LevelDB and RocksDB do below level 0.
//!
//! Every level below level 0 holds exactly **one run**, so a lookup probes at
//! most one run per level. Keeping that invariant is what costs: data arriving
//! at level `i` must be merged into the run already there, rewriting the
//! overlapping part of it. A record is rewritten roughly `T` times per level on
//! its way to the bottom.
//!
//! ```text
//!   L0 │ run  run  run          overlapping, straight from flushes
//!   L1 │ ───────one run───────  merged eagerly
//!   L2 │ ───────one run───────
//! ```
//!
//! Level 0 is the standard exception: its runs come directly from memtable
//! flushes, so their key ranges overlap arbitrarily and cannot form one sorted
//! run. All of them are therefore compacted together — merging a subset would
//! leave the rest overlapping the output and break level 1's invariant.

use super::{CompactionJob, MergePolicy};
use crate::storage::shape::{is_deepest_level, TreeShape};
use crate::storage::Key;

/// One run per level below level 0.
#[derive(Debug, Clone, Copy, Default)]
pub struct Leveling;

impl MergePolicy for Leveling {
    fn name(&self) -> &'static str {
        "leveling"
    }

    fn plan(&self, tree: &TreeShape, source_level: usize) -> Option<CompactionJob> {
        let source = tree.level(source_level)?;
        if source.is_empty() {
            return None;
        }

        // All of the source level. Below level 0 there is normally only one run;
        // at level 0 the runs overlap each other, so they must go together.
        let source_runs: Vec<usize> = source.runs.iter().map(|run| run.index).collect();
        let target_level = source_level + 1;
        let target_runs = overlapping_target_runs(tree, source_level, &source_runs, target_level);

        Some(CompactionJob {
            source_level,
            source_runs,
            target_level,
            target_runs,
            // Safe on the deepest-level check alone: every target run
            // overlapping the merged range is consumed above, so whatever
            // remains at the target cannot hold these keys.
            drop_tombstones: is_deepest_level(tree, target_level),
        })
    }

    fn runs_per_level(&self) -> usize {
        1
    }
}

/// Runs at `target_level` whose key range intersects the chosen sources.
///
/// These must be rewritten as part of the merge: leaving an overlapping run
/// behind would put two runs holding the same key in one level, with no way for
/// the read path to know which is newer.
fn overlapping_target_runs(
    tree: &TreeShape,
    source_level: usize,
    source_runs: &[usize],
    target_level: usize,
) -> Vec<usize> {
    let (Some(source), Some(target)) = (tree.level(source_level), tree.level(target_level)) else {
        return Vec::new();
    };

    let selected = source
        .runs
        .iter()
        .filter(|run| source_runs.contains(&run.index));

    // Combined span of everything being merged down.
    let mut span: Option<(Key, Key)> = None;
    for run in selected {
        if let (Some(min), Some(max)) = (&run.min_key, &run.max_key) {
            span = Some(match span {
                None => (min.clone(), max.clone()),
                Some((lo, hi)) => (lo.min(min.clone()), hi.max(max.clone())),
            });
        }
    }
    let Some((min, max)) = span else {
        return Vec::new();
    };

    target
        .runs
        .iter()
        .filter(|run| run.overlaps_range(&min, &max))
        .map(|run| run.index)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::shape::{LevelShape, RunShape};

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
    fn all_source_runs_are_merged_together() {
        let tree = TreeShape {
            levels: vec![LevelShape::from_sizes(&[10, 10, 10, 10])],
        };
        let job = Leveling.plan(&tree, 0).expect("job");
        assert_eq!(
            job.source_runs,
            vec![0, 1, 2, 3],
            "level 0 runs overlap arbitrarily, so they must all go together"
        );
        assert_eq!(job.target_level, 1);
    }

    /// Only the overlapping slice of the level below is rewritten.
    #[test]
    fn only_overlapping_target_runs_are_included() {
        let tree = TreeShape {
            levels: vec![
                LevelShape {
                    runs: vec![run(0, 1000, "d", "f")],
                },
                LevelShape {
                    runs: vec![
                        run(0, 100, "a", "c"), // before the source range
                        run(1, 100, "e", "g"), // overlaps
                        run(2, 100, "x", "z"), // after
                    ],
                },
            ],
        };

        let job = Leveling.plan(&tree, 0).expect("job");
        assert_eq!(
            job.target_runs,
            vec![1],
            "only the run intersecting d..f should be rewritten"
        );
    }

    #[test]
    fn a_disjoint_source_moves_down_without_rewriting_anything() {
        let tree = TreeShape {
            levels: vec![
                LevelShape {
                    runs: vec![run(0, 1000, "m", "n")],
                },
                LevelShape {
                    runs: vec![run(0, 100, "a", "c"), run(1, 100, "x", "z")],
                },
            ],
        };
        assert!(Leveling.plan(&tree, 0).expect("job").target_runs.is_empty());
    }

    /// The span is the union across *all* source runs, not just the first.
    #[test]
    fn the_merged_span_covers_every_source_run() {
        let tree = TreeShape {
            levels: vec![
                LevelShape {
                    runs: vec![run(0, 10, "a", "c"), run(1, 10, "w", "z")],
                },
                LevelShape {
                    runs: vec![run(0, 100, "m", "n")],
                },
            ],
        };
        let job = Leveling.plan(&tree, 0).expect("job");
        assert_eq!(
            job.target_runs,
            vec![0],
            "m..n lies inside the a..z span of the two source runs together"
        );
    }

    #[test]
    fn tombstones_are_dropped_only_when_nothing_lives_below() {
        let shallow = TreeShape {
            levels: vec![LevelShape::from_sizes(&[10, 10])],
        };
        assert!(Leveling.plan(&shallow, 0).expect("job").drop_tombstones);

        let deep = TreeShape {
            levels: vec![
                LevelShape::from_sizes(&[10, 10]),
                LevelShape::from_sizes(&[100]),
                LevelShape::from_sizes(&[1000]),
            ],
        };
        assert!(
            !Leveling.plan(&deep, 0).expect("job").drop_tombstones,
            "a tombstone dropped above level 2 would resurrect deleted keys"
        );
    }

    #[test]
    fn leveling_reports_one_run_per_level() {
        assert_eq!(Leveling.runs_per_level(), 1);
        assert_eq!(Leveling.name(), "leveling");
    }
}
