//! Tiering: append the merged result as a new run.
//!
//! The write-optimised policy, and leveling's opposite.
//!
//! A compaction merges the source level's runs and **appends** the single result
//! to the level below, without touching what is already there. A record is
//! therefore rewritten roughly once per level instead of `T` times. The bill
//! arrives at read time: a lookup may probe several runs at every level.
//!
//! ```text
//!   L0 │ run  run  run  run     ← merge into one run, appended to L1
//!   L1 │ run  run               ← untouched by that compaction
//!   L2 │ run
//! ```
//!
//! Bloom filters blunt the read cost considerably, since most of those probes
//! are answered from memory — which is why tiering is viable at all. Phase 3
//! should report read cost both in runs probed and in blocks actually read,
//! because the filter makes those two diverge sharply.

use super::{all_files, CompactionJob, Granularity, MergePolicy};
use crate::storage::shape::{is_deepest_level, TreeShape};

/// Several runs per level, merged all at once and appended below.
#[derive(Debug, Clone, Copy)]
pub struct Tiering {
    runs_per_level: usize,
}

impl Tiering {
    /// # Panics
    ///
    /// If `runs_per_level < 2`. With one run per level this is leveling.
    pub fn new(runs_per_level: usize) -> Self {
        assert!(
            runs_per_level >= 2,
            "tiering needs at least two runs per level; with one it is leveling"
        );
        Self { runs_per_level }
    }
}

impl Default for Tiering {
    fn default() -> Self {
        Self::new(4)
    }
}

impl MergePolicy for Tiering {
    fn name(&self) -> &'static str {
        "tiering"
    }

    /// `granularity` is ignored: a tiering compaction always merges the whole
    /// level.
    ///
    /// Slicing would gain nothing. Under tiering the source level's runs overlap
    /// each other, so a key-range slice of one still overlaps the rest and the
    /// level's run count — the thing read cost actually tracks — would not fall.
    /// The caller would pay to rewrite data without buying a single probe back.
    fn plan(
        &self,
        tree: &TreeShape,
        source_level: usize,
        _granularity: Granularity,
    ) -> Option<CompactionJob> {
        let source = tree.level(source_level)?;
        if source.is_empty() {
            return None;
        }

        let sources = all_files(tree, source_level);
        let target_level = source_level + 1;

        // Runs already at the target survive this compaction *and* overlap the
        // output. A tombstone dropped here would leave the older value in one of
        // them reachable, so the target must be empty as well as deepest.
        let target_is_empty = tree
            .level(target_level)
            .is_none_or(|shape| shape.is_empty());

        Some(CompactionJob {
            source_level,
            sources,
            target_level,
            // Appended, not merged: this is where the write saving lives.
            targets: Vec::new(),
            drop_tombstones: is_deepest_level(tree, target_level) && target_is_empty,
        })
    }

    fn runs_per_level(&self) -> usize {
        self.runs_per_level
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::shape::LevelShape;

    #[test]
    fn all_source_runs_merge_into_one_appended_run() {
        let tree = TreeShape {
            levels: vec![
                LevelShape::from_sizes(&[100; 4]),
                LevelShape::from_sizes(&[900, 900]),
            ],
        };

        let job = Tiering::new(4).plan(&tree, 0, Granularity::Full).expect("job");
        assert_eq!(job.sources.len(), 4);
        assert!(
            job.targets.is_empty(),
            "tiering never rewrites the level it writes into"
        );
        assert_eq!(job.input_run_count(), 4);
    }

    /// Granularity does not apply: a key-range slice of one run would still
    /// overlap the others, so the level's run count would not fall.
    #[test]
    fn a_partial_request_still_merges_the_whole_level() {
        let tree = TreeShape {
            levels: vec![
                LevelShape::from_sizes(&[100; 4]),
                LevelShape::from_sizes(&[900]),
            ],
        };

        let full = Tiering::new(4).plan(&tree, 0, Granularity::Full).expect("job");
        let partial = Tiering::new(4)
            .plan(&tree, 0, Granularity::Partial)
            .expect("job");
        assert_eq!(full, partial);
    }

    #[test]
    fn tombstones_are_dropped_only_into_an_empty_bottom_level() {
        let empty_target = TreeShape {
            levels: vec![LevelShape::from_sizes(&[100; 4])],
        };
        assert!(
            Tiering::new(4)
                .plan(&empty_target, 0, Granularity::Full)
                .expect("job")
                .drop_tombstones,
            "nothing exists below, so nothing can be resurrected"
        );
    }

    /// Regression test. Tiering appends without consuming the target level, so
    /// runs already there survive *and* overlap the output. If the tombstone
    /// check only asked whether this was the deepest level, a delete written
    /// into level 1 would be discarded while the value it deletes still sat in
    /// an older level-1 run — and the key would come back from the dead.
    #[test]
    fn tombstones_are_kept_when_the_target_level_still_holds_older_runs() {
        let tree = TreeShape {
            levels: vec![
                LevelShape::from_sizes(&[100; 4]),
                // Deepest level, but not empty.
                LevelShape::from_sizes(&[900]),
            ],
        };

        let job = Tiering::new(4).plan(&tree, 0, Granularity::Full).expect("job");
        assert!(
            job.targets.is_empty(),
            "the surviving target run is exactly the danger"
        );
        assert!(
            !job.drop_tombstones,
            "that run can hold values these tombstones delete"
        );
    }

    #[test]
    fn the_run_limit_is_reported() {
        assert_eq!(Tiering::new(8).runs_per_level(), 8);
        assert_eq!(Tiering::default().runs_per_level(), 4);
        assert_eq!(Tiering::default().name(), "tiering");
    }

    #[test]
    #[should_panic(expected = "at least two runs per level")]
    fn a_single_run_per_level_is_rejected() {
        Tiering::new(1);
    }
}
