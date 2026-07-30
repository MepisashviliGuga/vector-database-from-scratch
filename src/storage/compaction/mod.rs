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

use super::growth::CompactionRequest;
use super::shape::TreeShape;

pub mod ecotune;
pub mod leveling;
pub mod tiering;

pub use ecotune::{EcoTuneConfig, EcoTunePolicy};
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

/// Files selected from one level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelFiles {
    pub level: usize,
    /// Runs within that level, **newest first**.
    pub runs: Vec<RunFiles>,
}

impl LevelFiles {
    pub fn file_count(&self) -> usize {
        self.runs.iter().map(|run| run.files.len()).sum()
    }
}

/// A unit of compaction work.
///
/// Inputs are named down to the file, so a partial compaction can take a slice
/// of a run rather than all of it. A full compaction simply names every file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionJob {
    /// Levels to merge, **shallowest first** — shallower levels hold newer
    /// data, and the merge resolves key collisions by that order. Usually one
    /// level; Vertiorizon drains several at once.
    pub sources: Vec<LevelFiles>,
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
        self.sources
            .iter()
            .map(|level| level.runs.len())
            .sum::<usize>()
            + self.targets.len()
    }

    /// Input files, which bounds how much this compaction rewrites.
    pub fn input_file_count(&self) -> usize {
        self.sources
            .iter()
            .map(LevelFiles::file_count)
            .sum::<usize>()
            + self
                .targets
                .iter()
                .map(|selection| selection.files.len())
                .sum::<usize>()
    }
}

/// Turns a scheduled compaction into a concrete merge.
pub trait MergePolicy: Debug + Send + Sync {
    fn name(&self) -> &'static str;

    /// Plan the compaction the growth scheme has requested.
    ///
    /// Granularity and the level span both come from the request, since the
    /// growth scheme owns those decisions. Returns `None` when the named levels
    /// hold nothing to move.
    fn plan(&self, tree: &TreeShape, request: CompactionRequest) -> Option<CompactionJob>;

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

/// Every file across a request's source levels, shallowest level first.
///
/// Honours [`CompactionRequest::merge_units`] when set, taking only the newest
/// runs whose unit counts reach it.
pub(crate) fn all_source_levels(tree: &TreeShape, request: CompactionRequest) -> Vec<LevelFiles> {
    request
        .source_levels()
        .filter_map(|level| {
            let runs = match request.merge_units {
                Some(units) => newest_runs_by_units(tree, level, units),
                None => all_files(tree, level),
            };
            (!runs.is_empty()).then_some(LevelFiles { level, runs })
        })
        .collect()
}

/// The newest runs in a level whose combined unit count reaches `units`.
///
/// Runs are stored newest first, and EcoTune's merges always consume a suffix of
/// the most recent ones — a width-`w` merge at position `p` covers unit runs
/// `(p−w, p]`, which are exactly the newest. Accumulating from the front until
/// the width is met therefore selects the right set.
///
/// Stops short rather than overshooting: taking a run that carries more units
/// than remain would merge data the schedule assigned to a later, separate final
/// run.
pub(crate) fn newest_runs_by_units(tree: &TreeShape, level: usize, units: usize) -> Vec<RunFiles> {
    let Some(shape) = tree.level(level) else {
        return Vec::new();
    };

    let mut selected = Vec::new();
    let mut accumulated = 0usize;
    for run in &shape.runs {
        if run.is_empty() {
            continue;
        }
        if accumulated + run.units > units {
            break;
        }
        accumulated += run.units;
        selected.push(RunFiles {
            run: run.index,
            files: run.files.iter().map(|file| file.index).collect(),
        });
        if accumulated >= units {
            break;
        }
    }

    // A single run wider than the whole request is not a merge worth doing.
    if selected.len() < 2 {
        return Vec::new();
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::shape::LevelShape;

    use crate::storage::growth::Granularity;
    use crate::storage::shape::{FileShape, RunShape};

    fn full(level: usize) -> CompactionRequest {
        CompactionRequest::single(level, Granularity::Full)
    }

    fn run(index: usize, bytes: u64, min: &str, max: &str) -> RunShape {
        RunShape {
            index,
            files: vec![FileShape {
                index: 0,
                bytes,
                min_key: Some(min.as_bytes().to_vec()),
                max_key: Some(max.as_bytes().to_vec()),
            }],
            units: 1,
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
            let job = policy.plan(&tree, full(0)).expect("level 0 holds data");
            assert_eq!(job.sources[0].level, 0);
            assert_eq!(job.target_level, 1);
            assert!(!job.sources.is_empty());
        }
    }

    #[test]
    fn an_empty_level_yields_no_job() {
        let tree = TreeShape {
            levels: vec![LevelShape::default(), LevelShape::from_sizes(&[900])],
        };
        assert_eq!(Leveling.plan(&tree, full(0)), None);
        assert_eq!(Tiering::new(4).plan(&tree, full(0)), None);
        assert_eq!(
            Leveling.plan(&tree, full(99)),
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

        let leveling = Leveling.plan(&tree, full(0)).expect("job");
        let tiering = Tiering::new(4).plan(&tree, full(0)).expect("job");

        assert_eq!(leveling.targets.len(), 1);
        assert!(tiering.targets.is_empty());
        assert!(
            tiering.input_file_count() < leveling.input_file_count(),
            "tiering moves the same data while touching less"
        );
    }
}
