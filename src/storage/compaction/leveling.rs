//! Leveling: one run per level, merged eagerly.
//!
//! The read-optimised baseline, and what LevelDB and RocksDB do by default below
//! level 0.
//!
//! Every level from 1 down holds exactly **one run**, so a point lookup probes
//! at most one run per level. Keeping that invariant is expensive: whenever data
//! arrives at level `i`, it must be merged into the run already there, which
//! rewrites the overlapping part of that run. A record is therefore rewritten
//! roughly `T` times per level on its way to the bottom.
//!
//! ```text
//!   L0 │ run  run  run          overlapping, straight from flushes
//!   L1 │ ───────one run───────  merged eagerly
//!   L2 │ ───────one run───────
//! ```
//!
//! # Level 0 is special
//!
//! Runs at level 0 come directly from memtable flushes, so their key ranges
//! overlap arbitrarily and they cannot form a single sorted run. Level 0 is
//! therefore allowed several runs, and is compacted on **run count** rather than
//! bytes. Every real leveling implementation makes this exception; without it,
//! each flush would trigger a full rewrite of level 1.
//!
//! # Partial compaction
//!
//! Below level 0 only *part* of a level is compacted at a time: one run's worth
//! of key range, merged with the overlapping portion of the level below. Since
//! runs here are physically split into disjoint files, that means rewriting a
//! bounded slice rather than the whole level. Compacting whole levels would
//! multiply write amplification by the level's size ratio and make the Phase 3
//! comparison against tiering meaningless.

use super::{is_deepest_level, CompactionJob, CompactionPolicy, GrowthScheme, TreeShape};

/// One run per level below level 0.
#[derive(Debug, Clone, Copy)]
pub struct Leveling {
    /// Runs level 0 may accumulate before it is compacted into level 1.
    ///
    /// This is the classic read-write knob: a larger trigger batches more flush
    /// output into one merge (less write amplification) at the cost of more runs
    /// to probe on every lookup.
    level0_run_trigger: usize,
}

impl Leveling {
    pub fn new(level0_run_trigger: usize) -> Self {
        assert!(
            level0_run_trigger >= 1,
            "level 0 must be allowed at least one run"
        );
        Self { level0_run_trigger }
    }

    pub fn level0_run_trigger(&self) -> usize {
        self.level0_run_trigger
    }
}

impl Default for Leveling {
    /// Four runs at level 0, matching LevelDB's default trigger.
    fn default() -> Self {
        Self::new(4)
    }
}

impl CompactionPolicy for Leveling {
    fn name(&self) -> &'static str {
        "leveling"
    }

    fn pick(&self, tree: &TreeShape, growth: &dyn GrowthScheme) -> Option<CompactionJob> {
        let total_bytes = tree.total_bytes();

        // Level 0 first: it is the only level that blocks incoming flushes, so
        // relieving it takes priority over tidying deeper levels.
        if let Some(level0) = tree.level(0) {
            if level0.run_count() >= self.level0_run_trigger {
                // All of level 0 at once. The runs overlap each other, so
                // merging a subset would leave the remainder still overlapping
                // the output and violate level 1's single-run invariant.
                let source_runs: Vec<usize> = level0.runs.iter().map(|run| run.index).collect();
                let target_runs = overlapping_target_runs(tree, &source_runs, 0, 1);

                return Some(CompactionJob {
                    source_level: 0,
                    source_runs,
                    target_level: 1,
                    target_runs,
                    // Safe on the deepest-level check alone: every target run
                    // overlapping the merged range is consumed above, so what
                    // remains at level 1 cannot hold these keys.
                    drop_tombstones: is_deepest_level(tree, 1),
                });
            }
        }

        // Then the deeper levels, shallowest first: data should move down one
        // step at a time, and compacting a deep level while a shallow one is
        // over budget just delays the inevitable.
        let deepest = tree.levels.len();
        for level in 1..deepest {
            let capacity = growth.level_capacity_bytes(level, total_bytes);
            if capacity == 0 || tree.level_bytes(level) <= capacity {
                continue;
            }

            // Take the largest run: it relieves the most pressure per
            // compaction, and at levels below 0 there is normally only one.
            let source = tree
                .level(level)?
                .runs
                .iter()
                .max_by_key(|run| run.bytes)?;

            let source_runs = vec![source.index];
            let target_runs = overlapping_target_runs(tree, &source_runs, level, level + 1);

            return Some(CompactionJob {
                source_level: level,
                source_runs,
                target_level: level + 1,
                target_runs,
                drop_tombstones: is_deepest_level(tree, level + 1),
            });
        }

        None
    }

    fn runs_per_level(&self) -> usize {
        1
    }
}

/// Runs in `target_level` whose key range intersects the chosen sources.
///
/// These must be rewritten as part of the merge: leaving an overlapping run
/// behind would put two runs holding the same key in one level, and the read
/// path would have no way to know which is newer.
fn overlapping_target_runs(
    tree: &TreeShape,
    source_runs: &[usize],
    source_level: usize,
    target_level: usize,
) -> Vec<usize> {
    let (Some(source), Some(target)) = (tree.level(source_level), tree.level(target_level)) else {
        return Vec::new();
    };

    let selected: Vec<_> = source
        .runs
        .iter()
        .filter(|run| source_runs.contains(&run.index))
        .collect();

    // Combined span of everything being merged down.
    let Some(min) = selected.iter().filter_map(|run| run.min_key.clone()).min() else {
        return Vec::new();
    };
    let Some(max) = selected.iter().filter_map(|run| run.max_key.clone()).max() else {
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
    use crate::storage::compaction::{LevelShape, RunShape};
    use crate::storage::growth::Vertical;
    use crate::storage::Key;

    const KIB: u64 = 1024;

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

    fn growth() -> Vertical {
        Vertical::new(KIB, 10)
    }

    #[test]
    fn an_empty_tree_needs_no_compaction() {
        assert_eq!(Leveling::default().pick(&TreeShape::default(), &growth()), None);
    }

    #[test]
    fn level_zero_below_its_trigger_is_left_alone() {
        let tree = TreeShape {
            levels: vec![LevelShape::from_sizes(&[10, 10, 10])],
        };
        assert_eq!(
            Leveling::new(4).pick(&tree, &growth()),
            None,
            "three runs is under a trigger of four"
        );
    }

    #[test]
    fn level_zero_compacts_all_its_runs_at_once() {
        let tree = TreeShape {
            levels: vec![LevelShape::from_sizes(&[10, 10, 10, 10])],
        };

        let job = Leveling::new(4).pick(&tree, &growth()).expect("a job");
        assert_eq!(job.source_level, 0);
        assert_eq!(job.target_level, 1);
        assert_eq!(
            job.source_runs,
            vec![0, 1, 2, 3],
            "level 0 runs overlap arbitrarily, so they must all go together"
        );
    }

    /// Level 0 blocks incoming flushes, so it wins even when a deeper level is
    /// also over budget.
    #[test]
    fn level_zero_takes_priority_over_deeper_levels() {
        let tree = TreeShape {
            levels: vec![
                LevelShape::from_sizes(&[10, 10, 10, 10]),
                LevelShape::from_sizes(&[100 * KIB]),
            ],
        };

        let job = Leveling::new(4).pick(&tree, &growth()).expect("a job");
        assert_eq!(job.source_level, 0);
    }

    #[test]
    fn a_level_within_its_budget_is_left_alone() {
        // Level 1 capacity at ratio 10 from a 1 KiB base is 10 KiB.
        let tree = TreeShape {
            levels: vec![
                LevelShape::default(),
                LevelShape::from_sizes(&[9 * KIB]),
            ],
        };
        assert_eq!(Leveling::new(4).pick(&tree, &growth()), None);
    }

    #[test]
    fn a_level_over_its_budget_is_compacted_downward() {
        let tree = TreeShape {
            levels: vec![
                LevelShape::default(),
                LevelShape::from_sizes(&[11 * KIB]),
            ],
        };

        let job = Leveling::new(4).pick(&tree, &growth()).expect("a job");
        assert_eq!(job.source_level, 1);
        assert_eq!(job.target_level, 2);
    }

    #[test]
    fn shallower_levels_are_compacted_first() {
        let tree = TreeShape {
            levels: vec![
                LevelShape::default(),
                LevelShape::from_sizes(&[11 * KIB]),   // over its 10 KiB budget
                LevelShape::from_sizes(&[200 * KIB]),  // over its 100 KiB budget
            ],
        };

        let job = Leveling::new(4).pick(&tree, &growth()).expect("a job");
        assert_eq!(
            job.source_level, 1,
            "data should move down one step at a time"
        );
    }

    /// Only the overlapping slice of the level below is rewritten. Rewriting the
    /// whole level would inflate write amplification by the size ratio.
    #[test]
    fn only_overlapping_target_runs_are_included() {
        let tree = TreeShape {
            levels: vec![
                LevelShape::default(),
                LevelShape {
                    runs: vec![run(0, 11 * KIB, "d", "f")],
                },
                LevelShape {
                    runs: vec![
                        run(0, KIB, "a", "c"), // before the source range
                        run(1, KIB, "e", "g"), // overlaps
                        run(2, KIB, "x", "z"), // after
                    ],
                },
            ],
        };

        let job = Leveling::new(4).pick(&tree, &growth()).expect("a job");
        assert_eq!(job.source_level, 1);
        assert_eq!(
            job.target_runs,
            vec![1],
            "only the run whose range intersects d..f should be rewritten"
        );
    }

    #[test]
    fn a_source_overlapping_nothing_merges_into_an_empty_target() {
        let tree = TreeShape {
            levels: vec![
                LevelShape::default(),
                LevelShape {
                    runs: vec![run(0, 11 * KIB, "m", "n")],
                },
                LevelShape {
                    runs: vec![run(0, KIB, "a", "c"), run(1, KIB, "x", "z")],
                },
            ],
        };

        let job = Leveling::new(4).pick(&tree, &growth()).expect("a job");
        assert!(
            job.target_runs.is_empty(),
            "a disjoint run moves down without rewriting anything"
        );
    }

    #[test]
    fn tombstones_are_dropped_only_when_writing_to_the_bottom() {
        // Writing into level 1, which is the deepest level holding data.
        let shallow = TreeShape {
            levels: vec![LevelShape::from_sizes(&[10, 10, 10, 10])],
        };
        let job = Leveling::new(4).pick(&shallow, &growth()).expect("a job");
        assert!(
            job.drop_tombstones,
            "nothing lives below level 1 here, so tombstones can go"
        );

        // Same compaction, but level 2 holds data underneath.
        let deep = TreeShape {
            levels: vec![
                LevelShape::from_sizes(&[10, 10, 10, 10]),
                LevelShape::from_sizes(&[KIB]),
                LevelShape::from_sizes(&[100 * KIB]),
            ],
        };
        let job = Leveling::new(4).pick(&deep, &growth()).expect("a job");
        assert!(
            !job.drop_tombstones,
            "a tombstone dropped above level 2 would resurrect deleted keys"
        );
    }

    /// A policy that keeps returning the same job would spin forever. Each job
    /// must strictly reduce what triggered it.
    #[test]
    fn repeated_compaction_terminates() {
        let policy = Leveling::new(4);
        let growth = growth();

        let mut tree = TreeShape {
            levels: vec![LevelShape::from_sizes(&[KIB; 8])],
        };

        // Apply jobs by moving the merged bytes down a level, as the executor
        // will, and check the loop ends.
        for step in 0..100 {
            let Some(job) = policy.pick(&tree, &growth) else {
                assert!(step > 0, "the first tree was already over budget");
                return;
            };

            let moved: u64 = job
                .source_runs
                .iter()
                .map(|&index| tree.levels[job.source_level].runs[index].bytes)
                .sum();

            tree.levels[job.source_level].runs.clear();
            while tree.levels.len() <= job.target_level {
                tree.levels.push(LevelShape::default());
            }
            let target = &mut tree.levels[job.target_level];
            let existing: u64 = target.runs.iter().map(|run| run.bytes).sum();
            *target = LevelShape::from_sizes(&[existing + moved]);
        }
        panic!("compaction did not converge after 100 steps");
    }

    #[test]
    fn leveling_reports_one_run_per_level() {
        assert_eq!(Leveling::default().runs_per_level(), 1);
        assert_eq!(Leveling::default().name(), "leveling");
    }
}
