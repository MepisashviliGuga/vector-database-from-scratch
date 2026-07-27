//! Tiering: several runs per level, merged only when the level fills.
//!
//! The write-optimised baseline, and leveling's opposite.
//!
//! Each level accumulates up to `T` runs. Nothing is merged until the level is
//! full; then all `T` runs are merged in one pass and the single result is
//! **appended** to the level below, without touching what is already there.
//!
//! ```text
//!   L0 │ run  run  run  run     ← at T=4 these merge into one run at L1
//!   L1 │ run  run               ← and this level fills the same way
//!   L2 │ run
//! ```
//!
//! # The trade-off against leveling
//!
//! A record is rewritten roughly **once per level** on its way down, against
//! leveling's `T` times — tiering does far less write work. The bill arrives at
//! read time: a point lookup may have to probe `T` runs at every level instead
//! of one, so read cost is multiplied by the size ratio.
//!
//! Bloom filters blunt that considerably, since most of those probes are
//! answered from memory, which is why tiering is viable at all in practice. The
//! Phase 3 benchmarks should report read cost both in runs probed and in blocks
//! actually read, because the filter makes those two numbers diverge sharply.
//!
//! # Why the target level is not merged into
//!
//! Leveling merges its output with the overlapping run below to preserve one run
//! per level. Tiering has no such invariant, so it simply appends — which is
//! exactly where its write-amplification saving comes from. The appended run
//! overlaps existing runs at that level, and the read path resolves that by
//! consulting runs newest-first.

use super::{is_bottom_level, CompactionJob, CompactionPolicy, GrowthScheme, TreeShape};

/// Up to `T` runs per level, merged all at once.
#[derive(Debug, Clone, Copy)]
pub struct Tiering {
    /// Runs a level may hold before merging. Conventionally the size ratio `T`,
    /// so a level's total size stays within its budget.
    runs_per_level: usize,
}

impl Tiering {
    pub fn new(runs_per_level: usize) -> Self {
        assert!(
            runs_per_level >= 2,
            "tiering needs at least two runs per level; with one it is leveling"
        );
        Self { runs_per_level }
    }

    /// Match the run limit to a growth scheme's size ratio, which is the
    /// standard pairing: `T` runs each roughly `1/T` of the level's budget.
    pub fn matching(growth: &dyn GrowthScheme) -> Self {
        Self::new(growth.size_ratio().max(2) as usize)
    }
}

impl Default for Tiering {
    fn default() -> Self {
        Self::new(4)
    }
}

impl CompactionPolicy for Tiering {
    fn name(&self) -> &'static str {
        "tiering"
    }

    fn pick(&self, tree: &TreeShape, _growth: &dyn GrowthScheme) -> Option<CompactionJob> {
        // Shallowest first: level 0 blocks incoming flushes, and in general a
        // full shallow level feeds the ones below it.
        for (level, shape) in tree.levels.iter().enumerate() {
            if shape.run_count() < self.runs_per_level {
                continue;
            }

            // All of them, in one pass. Merging a subset would leave runs behind
            // that overlap the output, gaining nothing on read cost while paying
            // the full write cost.
            let source_runs: Vec<usize> = shape.runs.iter().map(|run| run.index).collect();

            return Some(CompactionJob {
                source_level: level,
                source_runs,
                target_level: level + 1,
                // Appended, not merged: this is where the write saving lives.
                target_runs: Vec::new(),
                drop_tombstones: is_bottom_level(tree, level + 1),
            });
        }
        None
    }

    fn runs_per_level(&self) -> usize {
        self.runs_per_level
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::compaction::LevelShape;
    use crate::storage::growth::Vertical;

    const KIB: u64 = 1024;

    fn growth() -> Vertical {
        Vertical::new(KIB, 4)
    }

    #[test]
    fn an_empty_tree_needs_no_compaction() {
        assert_eq!(Tiering::default().pick(&TreeShape::default(), &growth()), None);
    }

    #[test]
    fn a_level_below_its_run_limit_is_left_alone() {
        let tree = TreeShape {
            levels: vec![LevelShape::from_sizes(&[KIB; 3])],
        };
        assert_eq!(
            Tiering::new(4).pick(&tree, &growth()),
            None,
            "three runs is under a limit of four"
        );
    }

    #[test]
    fn a_full_level_merges_all_of_its_runs() {
        let tree = TreeShape {
            levels: vec![LevelShape::from_sizes(&[KIB; 4])],
        };

        let job = Tiering::new(4).pick(&tree, &growth()).expect("a job");
        assert_eq!(job.source_level, 0);
        assert_eq!(job.target_level, 1);
        assert_eq!(job.source_runs, vec![0, 1, 2, 3]);
    }

    /// The defining difference from leveling: nothing at the target level is
    /// rewritten. This is where tiering's write saving comes from.
    #[test]
    fn the_target_level_is_appended_to_not_rewritten() {
        let tree = TreeShape {
            levels: vec![
                LevelShape::from_sizes(&[KIB; 4]),
                LevelShape::from_sizes(&[10 * KIB, 10 * KIB]),
            ],
        };

        let job = Tiering::new(4).pick(&tree, &growth()).expect("a job");
        assert!(
            job.target_runs.is_empty(),
            "tiering never rewrites the level it writes into"
        );
        assert_eq!(job.input_run_count(), 4, "only the source runs are merged");
    }

    #[test]
    fn shallower_levels_are_compacted_first() {
        let tree = TreeShape {
            levels: vec![
                LevelShape::from_sizes(&[KIB; 2]), // not full
                LevelShape::from_sizes(&[KIB; 4]), // full
                LevelShape::from_sizes(&[KIB; 5]), // also full
            ],
        };

        let job = Tiering::new(4).pick(&tree, &growth()).expect("a job");
        assert_eq!(
            job.source_level, 1,
            "the shallowest full level should be picked, not the fullest"
        );
    }

    #[test]
    fn tombstones_are_dropped_only_when_writing_to_the_bottom() {
        let shallow = TreeShape {
            levels: vec![LevelShape::from_sizes(&[KIB; 4])],
        };
        assert!(
            Tiering::new(4)
                .pick(&shallow, &growth())
                .expect("a job")
                .drop_tombstones
        );

        let deep = TreeShape {
            levels: vec![
                LevelShape::from_sizes(&[KIB; 4]),
                LevelShape::from_sizes(&[KIB]),
                LevelShape::from_sizes(&[KIB]),
            ],
        };
        assert!(
            !Tiering::new(4)
                .pick(&deep, &growth())
                .expect("a job")
                .drop_tombstones
        );
    }

    #[test]
    fn the_run_limit_can_track_the_growth_scheme() {
        let scheme = Vertical::new(KIB, 8);
        assert_eq!(Tiering::matching(&scheme).runs_per_level(), 8);
    }

    /// Tiering must converge too: each merge removes `T` runs from a level and
    /// adds one below.
    #[test]
    fn repeated_compaction_terminates() {
        let policy = Tiering::new(4);
        let growth = growth();

        let mut tree = TreeShape {
            levels: vec![LevelShape::from_sizes(&[KIB; 16])],
        };

        for step in 0..100 {
            let Some(job) = policy.pick(&tree, &growth) else {
                assert!(step > 0, "the first tree was already full");
                return;
            };

            let merged: u64 = job
                .source_runs
                .iter()
                .map(|&index| tree.levels[job.source_level].runs[index].bytes)
                .sum();

            tree.levels[job.source_level].runs.clear();
            while tree.levels.len() <= job.target_level {
                tree.levels.push(LevelShape::default());
            }
            // Appended as one new run, rather than merged into what is there.
            let mut sizes: Vec<u64> = tree.levels[job.target_level]
                .runs
                .iter()
                .map(|run| run.bytes)
                .collect();
            sizes.insert(0, merged);
            tree.levels[job.target_level] = LevelShape::from_sizes(&sizes);
        }
        panic!("compaction did not converge after 100 steps");
    }

    /// The headline contrast, asserted directly so the Phase 3 comparison has a
    /// documented starting point.
    #[test]
    fn tiering_merges_fewer_runs_than_leveling_for_the_same_tree() {
        use crate::storage::compaction::Leveling;

        let tree = TreeShape {
            levels: vec![
                LevelShape::from_sizes(&[KIB; 4]),
                LevelShape {
                    runs: vec![crate::storage::compaction::RunShape {
                        index: 0,
                        bytes: 40 * KIB,
                        min_key: Some(b"a".to_vec()),
                        max_key: Some(b"z".to_vec()),
                    }],
                },
            ],
        };

        // Level 0 runs have no key ranges in this fixture, so leveling finds no
        // overlap to rewrite; give them one to make the comparison real.
        let mut with_ranges = tree.clone();
        for run in &mut with_ranges.levels[0].runs {
            run.min_key = Some(b"a".to_vec());
            run.max_key = Some(b"z".to_vec());
        }

        let tiering_job = Tiering::new(4).pick(&with_ranges, &growth()).expect("job");
        let leveling_job = Leveling::new(4).pick(&with_ranges, &growth()).expect("job");

        assert_eq!(tiering_job.input_run_count(), 4);
        assert_eq!(
            leveling_job.input_run_count(),
            5,
            "leveling additionally rewrites the overlapping run below"
        );
        assert!(
            tiering_job.input_run_count() < leveling_job.input_run_count(),
            "tiering should move the same data while touching less"
        );
    }
}
