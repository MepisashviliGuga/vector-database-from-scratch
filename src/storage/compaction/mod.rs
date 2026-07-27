//! Axis 2: **how** a compaction merges.
//!
//! Orthogonal to the growth scheme, which decides *when*. The growth scheme
//! hands over a source level; the merge policy turns that into a concrete job.
//! Neither knows anything about the other, which is what lets Phase 3 vary one
//! while holding the other fixed.
//!
//! # The two policies
//!
//! - **[`Leveling`]** merges the source into the overlapping run already at the
//!   target level, keeping one run per level. Cheap reads (one run to probe per
//!   level), expensive writes (a record is rewritten every time its level takes
//!   in data).
//! - **[`Tiering`]** appends the source as a *new* run at the target level,
//!   touching nothing already there. Cheap writes (a record moves down a level
//!   once), expensive reads (several runs to probe per level).
//!
//! These are the endpoints of the read-write trade-off. EcoTune (Phase 2) treats
//! the choice between them as an investment decision rather than a fixed
//! configuration, which is only meaningful once both endpoints are measurable.

use std::fmt::Debug;

use super::growth::Granularity;
use super::shape::TreeShape;

pub mod leveling;
pub mod tiering;

pub use leveling::Leveling;
pub use tiering::Tiering;

/// Files selected from one run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunFiles {
    /// Index into the level's `runs`.
    pub run: usize,
    /// Indices into that run's `files`, ascending by key.
    pub files: Vec<usize>,
}

/// A unit of compaction work.
///
/// Inputs are named down to the file, so a partial compaction can take a slice
/// of a run rather than all of it. A full compaction simply names every file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionJob {
    pub source_level: usize,
    /// Runs and files to merge, **newest run first**. The merge depends on that
    /// ordering to resolve key collisions by recency.
    pub sources: Vec<RunFiles>,
    pub target_level: usize,
    /// Files at the target level to rewrite. Always empty for tiering, which
    /// appends rather than merging.
    pub targets: Vec<RunFiles>,
    /// Whether tombstones may be discarded.
    ///
    /// True only when nothing that could hold an older version of these keys
    /// survives the merge. Setting it anywhere else resurrects deleted keys,
    /// silently.
    pub drop_tombstones: bool,
}

impl CompactionJob {
    /// Input runs, which is what the merge iterator is built over.
    pub fn input_run_count(&self) -> usize {
        self.sources.len() + self.targets.len()
    }

    /// Input files, which bounds how much this compaction rewrites.
    pub fn input_file_count(&self) -> usize {
        self.sources
            .iter()
            .chain(self.targets.iter())
            .map(|selection| selection.files.len())
            .sum()
    }
}

/// Turns "compact level `i`" into a concrete merge.
pub trait MergePolicy: Debug + Send + Sync {
    fn name(&self) -> &'static str;

    /// Plan the compaction of `source_level` into `source_level + 1`.
    ///
    /// `granularity` comes from the growth scheme, which owns that decision —
    /// see [`Granularity`]. Returns `None` when the level holds nothing to move.
    fn plan(
        &self,
        tree: &TreeShape,
        source_level: usize,
        granularity: Granularity,
    ) -> Option<CompactionJob>;

    /// Runs a level may hold: 1 for leveling, `T` for tiering. The executor uses
    /// this to decide whether disjoint runs at a level should be folded into one.
    fn runs_per_level(&self) -> usize;
}

/// Select every file of every run in a level, newest run first.
pub(crate) fn all_files(tree: &TreeShape, level: usize) -> Vec<RunFiles> {
    tree.level(level).map_or_else(Vec::new, |shape| {
        shape
            .runs
            .iter()
            .filter(|run| !run.is_empty())
            .map(|run| RunFiles {
                run: run.index,
                files: run.files.iter().map(|file| file.index).collect(),
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::shape::LevelShape;

    use crate::storage::shape::{FileShape, RunShape};

    fn run(index: usize, bytes: u64, min: &str, max: &str) -> RunShape {
        RunShape {
            index,
            files: vec![FileShape {
                index: 0,
                bytes,
                min_key: Some(min.as_bytes().to_vec()),
                max_key: Some(max.as_bytes().to_vec()),
            }],
        }
    }

    /// Both policies must agree on the shape of the interface, so the harness
    /// can hold the growth scheme fixed and swap the merge policy.
    #[test]
    fn both_policies_satisfy_the_trait() {
        let tree = TreeShape {
            levels: vec![
                LevelShape::from_sizes(&[100, 100]),
                LevelShape::from_sizes(&[900]),
            ],
        };
        let policies: Vec<Box<dyn MergePolicy>> =
            vec![Box::new(Leveling), Box::new(Tiering::new(4))];

        for policy in policies {
            assert!(!policy.name().is_empty());
            let job = policy
                .plan(&tree, 0, Granularity::Full)
                .expect("level 0 holds data");
            assert_eq!(job.source_level, 0);
            assert_eq!(job.target_level, 1);
            assert!(!job.sources.is_empty());
        }
    }

    #[test]
    fn an_empty_level_yields_no_job() {
        let tree = TreeShape {
            levels: vec![LevelShape::default(), LevelShape::from_sizes(&[900])],
        };
        assert_eq!(Leveling.plan(&tree, 0, Granularity::Full), None);
        assert_eq!(Tiering::new(4).plan(&tree, 0, Granularity::Full), None);
        assert_eq!(
            Leveling.plan(&tree, 99, Granularity::Full),
            None,
            "a level that does not exist"
        );
    }

    /// The headline contrast: tiering leaves the target level alone.
    #[test]
    fn leveling_rewrites_the_target_and_tiering_does_not() {
        let tree = TreeShape {
            levels: vec![
                LevelShape {
                    runs: vec![run(0, 100, "a", "z")],
                },
                LevelShape {
                    runs: vec![run(0, 900, "a", "z")],
                },
            ],
        };

        let leveling = Leveling.plan(&tree, 0, Granularity::Full).expect("job");
        let tiering = Tiering::new(4)
            .plan(&tree, 0, Granularity::Full)
            .expect("job");

        assert_eq!(leveling.targets.len(), 1);
        assert!(tiering.targets.is_empty());
        assert!(
            tiering.input_file_count() < leveling.input_file_count(),
            "tiering moves the same data while touching less"
        );
    }
}
