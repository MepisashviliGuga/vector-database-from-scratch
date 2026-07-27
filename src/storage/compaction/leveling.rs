//! Leveling: merge into the run already at the target level.
//!
//! The read-optimised policy, and what LevelDB and RocksDB do below level 0.
//!
//! Every level below level 0 holds exactly **one run**, so a lookup probes at
//! most one run per level. Keeping that invariant is what costs: data arriving
//! at level `i` must be merged into the run already there, rewriting the
//! overlapping part of it.
//!
//! ```text
//!   L0 │ run  run  run          overlapping, straight from flushes
//!   L1 │ ──file──file──file──   one run, several disjoint files
//!   L2 │ ─file─file─file─file─
//! ```
//!
//! # Full versus partial
//!
//! Under [`Granularity::Full`] the whole source level is merged with everything
//! it overlaps below. Under [`Granularity::Partial`] one file's worth of key
//! range moves at a time, together with just the files it overlaps.
//!
//! Write *amplification* is the same either way — both rewrite about `T+1` bytes
//! per byte moved down a level. What differs is everything else: a partial
//! compaction holds far less data hostage while it runs, so it does not spike
//! space usage or stall the levels above. Paper 01 identifies exactly this as
//! why industry adopted the vertical scheme despite its worse asymptotic
//! read-write trade-off, and why the horizontal scheme — which is *defined* on
//! full compaction — pays a space cost.
//!
//! Level 0 is always merged whole: its runs come straight from memtable flushes,
//! so their key ranges overlap arbitrarily and cannot be sliced by key.

use super::{all_source_levels, CompactionJob, LevelFiles, MergePolicy, RunFiles};
use crate::storage::growth::{CompactionRequest, Granularity};
use crate::storage::shape::{is_deepest_level, TreeShape};
use crate::storage::Key;

/// One run per level below level 0.
#[derive(Debug, Clone, Copy, Default)]
pub struct Leveling;

impl MergePolicy for Leveling {
    fn name(&self) -> &'static str {
        "leveling"
    }

    fn plan(&self, tree: &TreeShape, request: CompactionRequest) -> Option<CompactionJob> {
        let target_level = request.target_level;
        let source_level = request.first_level;

        // A span of levels is always merged whole: slicing would leave part of
        // each level behind, which is the opposite of what a drain is for.
        let sliceable = !request.spans_multiple_levels()
            && request.granularity == Granularity::Partial
            // Level 0's runs overlap each other, so a key-range slice of one
            // would still overlap the others. Same argument for any level that
            // transiently holds several runs.
            && source_level > 0
            && tree.level(source_level).is_some_and(|l| l.run_count() == 1);

        let sources = if sliceable {
            let slice = pick_slice(tree, source_level)?;
            vec![LevelFiles {
                level: source_level,
                runs: vec![slice],
            }]
        } else {
            all_source_levels(tree, request)
        };

        if sources.is_empty() {
            return None;
        }

        // No known key range means nothing can be shown to overlap, so the merge
        // proceeds with no target files rather than being abandoned.
        let targets = match merged_span(tree, &sources) {
            Some(span) => overlapping_target_files(tree, target_level, &span),
            None => Vec::new(),
        };

        Some(CompactionJob {
            sources,
            target_level,
            targets,
            // Safe on the deepest-level check alone: every target file
            // overlapping the merged range is consumed above, so whatever
            // remains at the target is disjoint from these keys.
            drop_tombstones: is_deepest_level(tree, target_level),
        })
    }

    fn runs_per_level(&self) -> usize {
        1
    }
}

/// Choose one file to move down.
///
/// Takes the first file of the level's single run. Since that file is consumed
/// by the compaction, successive calls sweep the level from low keys to high.
///
/// **Engineering choice, not from the paper.** LevelDB rotates a per-level
/// pointer so successive compactions spread across the key space instead of
/// marching through it, which distributes work more evenly under skew. Sweeping
/// is simpler and gives the same amortised cost; if the Phase 3 skew workloads
/// show it mattering, the rotating pointer goes in and gets labelled.
fn pick_slice(tree: &TreeShape, source_level: usize) -> Option<RunFiles> {
    let run = tree.level(source_level)?.runs.first()?;
    let file = run.files.first()?;
    Some(RunFiles {
        run: run.index,
        files: vec![file.index],
    })
}

/// Combined key range of every selected source file, across all source levels.
fn merged_span(tree: &TreeShape, sources: &[LevelFiles]) -> Option<(Key, Key)> {
    let mut span: Option<(Key, Key)> = None;

    for level_files in sources {
        let Some(shape) = tree.level(level_files.level) else {
            continue;
        };
        for selection in &level_files.runs {
            let Some(run) = shape.runs.iter().find(|run| run.index == selection.run) else {
                continue;
            };
            for file_index in &selection.files {
                let Some(file) = run.files.iter().find(|file| file.index == *file_index) else {
                    continue;
                };
                if let (Some(min), Some(max)) = (&file.min_key, &file.max_key) {
                    span = Some(match span {
                        None => (min.clone(), max.clone()),
                        Some((lo, hi)) => (lo.min(min.clone()), hi.max(max.clone())),
                    });
                }
            }
        }
    }
    span
}

/// Files at `target_level` intersecting `span`.
///
/// These must be rewritten as part of the merge: leaving an overlapping file
/// behind would put two versions of a key in one level with no way for the read
/// path to know which is newer.
fn overlapping_target_files(
    tree: &TreeShape,
    target_level: usize,
    span: &(Key, Key),
) -> Vec<RunFiles> {
    let Some(target) = tree.level(target_level) else {
        return Vec::new();
    };
    let (min, max) = span;

    target
        .runs
        .iter()
        .filter_map(|run| {
            let files = run.files_overlapping(min, max);
            (!files.is_empty()).then_some(RunFiles {
                run: run.index,
                files,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::shape::{FileShape, LevelShape, RunShape};

    fn k(s: &str) -> Key {
        s.as_bytes().to_vec()
    }

    fn full(level: usize) -> CompactionRequest {
        CompactionRequest::single(level, Granularity::Full)
    }

    fn partial(level: usize) -> CompactionRequest {
        CompactionRequest::single(level, Granularity::Partial)
    }

    /// Runs selected from the job's single source level.
    fn source_runs(job: &CompactionJob) -> &[RunFiles] {
        assert_eq!(job.sources.len(), 1, "expected one source level");
        &job.sources[0].runs
    }

    fn file(index: usize, bytes: u64, min: &str, max: &str) -> FileShape {
        FileShape {
            index,
            bytes,
            min_key: Some(k(min)),
            max_key: Some(k(max)),
        }
    }

    /// A run of one file.
    fn run(index: usize, bytes: u64, min: &str, max: &str) -> RunShape {
        RunShape {
            index,
            files: vec![file(0, bytes, min, max)],
        }
    }

    /// A run split across the given `(min, max)` file ranges.
    fn multi_file_run(index: usize, ranges: &[(&str, &str)]) -> RunShape {
        RunShape {
            index,
            files: ranges
                .iter()
                .enumerate()
                .map(|(i, (min, max))| file(i, 100, min, max))
                .collect(),
        }
    }

    #[test]
    fn a_full_compaction_takes_every_source_run() {
        let tree = TreeShape {
            levels: vec![LevelShape::from_sizes(&[10, 10, 10, 10])],
        };
        let job = Leveling.plan(&tree, full(0)).expect("job");
        assert_eq!(
            source_runs(&job).len(),
            4,
            "level 0 runs overlap arbitrarily, so they must all go together"
        );
        assert_eq!(job.target_level, 1);
    }

    /// Vertiorizon drains its whole horizontal part in one merge, so a job must
    /// be able to span several source levels — shallowest (newest) first.
    #[test]
    fn a_request_spanning_levels_takes_all_of_them() {
        let tree = TreeShape {
            levels: vec![
                LevelShape::from_sizes(&[10]),
                LevelShape::from_sizes(&[20]),
                LevelShape::from_sizes(&[30]),
                LevelShape::default(),
            ],
        };

        let job = Leveling
            .plan(
                &tree,
                CompactionRequest {
                    first_level: 0,
                    last_level: 2,
                    target_level: 3,
                    granularity: Granularity::Full,
                },
            )
            .expect("job");

        assert_eq!(job.sources.len(), 3);
        assert_eq!(
            job.sources.iter().map(|s| s.level).collect::<Vec<_>>(),
            vec![0, 1, 2],
            "source levels must be shallowest first, since they hold newer data"
        );
        assert_eq!(job.target_level, 3);
    }

    /// A span is never sliced: leaving part of each level behind is the opposite
    /// of what a drain is for.
    #[test]
    fn a_spanning_request_is_never_sliced() {
        let tree = TreeShape {
            levels: vec![
                LevelShape {
                    runs: vec![multi_file_run(0, &[("a", "c"), ("d", "f")])],
                },
                LevelShape {
                    runs: vec![multi_file_run(0, &[("a", "b"), ("e", "g")])],
                },
                LevelShape::default(),
            ],
        };

        let job = Leveling
            .plan(
                &tree,
                CompactionRequest {
                    first_level: 0,
                    last_level: 1,
                    target_level: 2,
                    granularity: Granularity::Partial,
                },
            )
            .expect("job");

        assert_eq!(job.input_file_count(), 4, "every file from both levels");
    }

    /// The point of partial compaction: one file's worth of key range moves,
    /// with only the files it overlaps below.
    #[test]
    fn a_partial_compaction_takes_one_file_and_its_overlap() {
        let tree = TreeShape {
            levels: vec![
                LevelShape::default(),
                LevelShape {
                    runs: vec![multi_file_run(0, &[("a", "c"), ("d", "f"), ("g", "i")])],
                },
                LevelShape {
                    runs: vec![multi_file_run(
                        0,
                        &[("a", "b"), ("c", "e"), ("f", "h"), ("x", "z")],
                    )],
                },
            ],
        };

        let job = Leveling.plan(&tree, partial(1)).expect("job");

        assert_eq!(source_runs(&job).len(), 1);
        assert_eq!(
            source_runs(&job)[0].files,
            vec![0],
            "only the first file of the source run should move"
        );
        // Source file spans a..c; target files a..b and c..e intersect it.
        assert_eq!(job.targets.len(), 1);
        assert_eq!(job.targets[0].files, vec![0, 1]);
        assert_eq!(
            job.input_file_count(),
            3,
            "one source file plus two overlapping target files"
        );
    }

    /// The whole reason partial compaction exists: it touches dramatically less.
    #[test]
    fn partial_compaction_rewrites_far_less_than_full() {
        let wide_source = multi_file_run(
            0,
            &[("a", "c"), ("d", "f"), ("g", "i"), ("j", "l"), ("m", "o")],
        );
        let wide_target = multi_file_run(
            0,
            &[("a", "b"), ("c", "e"), ("f", "h"), ("i", "k"), ("l", "o")],
        );
        let tree = TreeShape {
            levels: vec![
                LevelShape::default(),
                LevelShape {
                    runs: vec![wide_source],
                },
                LevelShape {
                    runs: vec![wide_target],
                },
            ],
        };

        let whole = Leveling.plan(&tree, full(1)).expect("job");
        let sliced = Leveling.plan(&tree, partial(1)).expect("job");

        assert_eq!(whole.input_file_count(), 10, "everything, both levels");
        assert!(
            sliced.input_file_count() < whole.input_file_count() / 2,
            "partial touched {} files against full's {}",
            sliced.input_file_count(),
            whole.input_file_count()
        );
    }

    /// Level 0's runs overlap each other, so slicing one by key range would
    /// leave the others overlapping the output.
    #[test]
    fn level_zero_is_never_sliced() {
        let tree = TreeShape {
            levels: vec![
                LevelShape {
                    runs: vec![run(0, 10, "a", "z"), run(1, 10, "a", "z")],
                },
                LevelShape::default(),
            ],
        };

        let job = Leveling.plan(&tree, partial(0)).expect("job");
        assert_eq!(
            source_runs(&job).len(),
            2,
            "a partial request at level 0 must still take both overlapping runs"
        );
    }

    /// A level holding several runs cannot be sliced either — the same argument
    /// as level 0. It can happen transiently after a policy switch.
    #[test]
    fn a_multi_run_level_is_never_sliced() {
        let tree = TreeShape {
            levels: vec![
                LevelShape::default(),
                LevelShape {
                    runs: vec![run(0, 10, "a", "m"), run(1, 10, "d", "z")],
                },
            ],
        };
        let job = Leveling.plan(&tree, partial(1)).expect("job");
        assert_eq!(source_runs(&job).len(), 2);
    }

    #[test]
    fn a_disjoint_source_moves_down_without_rewriting_anything() {
        let tree = TreeShape {
            levels: vec![
                LevelShape {
                    runs: vec![run(0, 1000, "m", "n")],
                },
                LevelShape {
                    runs: vec![multi_file_run(0, &[("a", "c"), ("x", "z")])],
                },
            ],
        };
        assert!(Leveling.plan(&tree, full(0)).expect("job").targets.is_empty());
    }

    /// The span is the union across *all* source files, not just the first.
    #[test]
    fn the_merged_span_covers_every_source_file() {
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
        let job = Leveling.plan(&tree, full(0)).expect("job");
        assert_eq!(
            job.targets.len(),
            1,
            "m..n lies inside the a..z span of the two source runs together"
        );
    }

    #[test]
    fn tombstones_are_dropped_only_when_nothing_lives_below() {
        let shallow = TreeShape {
            levels: vec![LevelShape::from_sizes(&[10, 10])],
        };
        assert!(Leveling.plan(&shallow, full(0)).expect("job").drop_tombstones);

        let deep = TreeShape {
            levels: vec![
                LevelShape::from_sizes(&[10, 10]),
                LevelShape::from_sizes(&[100]),
                LevelShape::from_sizes(&[1000]),
            ],
        };
        assert!(
            !Leveling.plan(&deep, full(0)).expect("job").drop_tombstones,
            "a tombstone dropped above level 2 would resurrect deleted keys"
        );
    }

    #[test]
    fn leveling_reports_one_run_per_level() {
        assert_eq!(Leveling.runs_per_level(), 1);
        assert_eq!(Leveling.name(), "leveling");
    }
}
