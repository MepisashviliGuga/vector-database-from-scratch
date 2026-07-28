//! Driving the engine from EcoTune's solved schedule.
//!
//! The dynamic program itself lives in
//! [`crate::storage::compaction::ecotune`], which is a faithful reproduction of
//! paper 02's Algorithm 1. This module is the **glue** that runs it against a
//! levelled tree, and the mapping it makes is ours, not the paper's.
//!
//! # The mapping, and why it is glue
//!
//! Paper 02's whole argument is that physical levels should not exist: an
//! optimal policy varies its aggressiveness through a round, so runs at the same
//! "level" end up different sizes and the concept dissolves. It replaces levels
//! with three *logical* ones. This engine is levelled, so:
//!
//! ```text
//!   level 0  →  top level    never compacted internally; drains when full
//!   level 1  →  main level   the DP schedules merges here
//!   level 2  →  last level   one run, capping space amplification
//! ```
//!
//! Two things the paper specifies that this does **not** reproduce: the top
//! level's full key index (~0.8 bits/key, which absorbs the CPU cost of probing
//! many uncompacted runs) and SNARF range filters. Our bloom filters stand in,
//! and `f` in the cost model is taken from their configured false-positive rate.
//!
//! # Parameters that must be measured, not guessed
//!
//! `T_c` (time to rewrite one unit run) and `β` (query speed while merge threads
//! are busy) are *measured on the hardware* in the paper. They are configuration
//! here with documented defaults, which means **any EcoTune result from this
//! engine is only as good as those two numbers**. Phase 3 should measure them
//! before any EcoTune figure is reported.
//!
//! # Widths are in unit runs
//!
//! The schedule says things like "merge width 4 at position 4". That counts
//! *unit runs* absorbed, and merges nest — a width-4 merge may take one earlier
//! width-2 run plus two fresh ones. Runs therefore carry a unit count, and the
//! request asks for a total width rather than a number of runs.

use super::{CompactionRequest, Granularity, GrowthScheme, TreeShape};
use crate::storage::compaction::{EcoTuneConfig, EcoTunePolicy};

/// Logical level indices in the mapped tree.
const TOP_LEVEL: usize = 0;
const MAIN_LEVEL: usize = 1;
const LAST_LEVEL: usize = 2;

/// Runs a levelled tree from EcoTune's solved schedule.
#[derive(Debug)]
pub struct EcoTune {
    policy: EcoTunePolicy,
    /// Top level capacity `S`, in bytes.
    top_capacity_bytes: u64,
    /// Unit runs created so far in this compaction round.
    position: usize,
    /// Merges the schedule places, indexed by the position they happen at.
    schedule: Vec<(usize, usize)>,
    /// Whether this flush's scan has already been served.
    scanned: bool,
}

impl EcoTune {
    /// # Panics
    ///
    /// If `top_capacity_bytes` is 0.
    pub fn new(config: EcoTuneConfig, top_capacity_bytes: u64) -> Self {
        assert!(top_capacity_bytes > 0, "top level capacity must be positive");
        let policy = EcoTunePolicy::solve(config);
        let schedule = policy
            .chosen_merges()
            .into_iter()
            .map(|merge| (merge.after_unit_runs, merge.width))
            .collect();

        Self {
            policy,
            top_capacity_bytes,
            position: 0,
            schedule,
            scanned: true,
        }
    }

    pub fn config(&self) -> &EcoTuneConfig {
        self.policy.config()
    }

    /// Unit runs created so far in the current round.
    pub fn position(&self) -> usize {
        self.position
    }

    /// The merge the schedule places at the current position, if any.
    fn merge_at_position(&self) -> Option<usize> {
        self.schedule
            .iter()
            .find(|(at, _)| *at == self.position)
            .map(|(_, width)| *width)
    }

    fn round_length(&self) -> usize {
        self.policy.config().runs_per_round.max(1)
    }
}

impl GrowthScheme for EcoTune {
    fn name(&self) -> &'static str {
        "ecotune"
    }

    fn note_flush(&mut self) {
        self.scanned = false;
    }

    fn next_compaction(&mut self, tree: &TreeShape) -> Option<CompactionRequest> {
        if self.scanned {
            return None;
        }

        // 1. The round-ending global compaction: everything above merges into
        //    the last level, and the schedule restarts.
        if self.position >= self.round_length() {
            self.position = 0;
            self.scanned = true;
            if tree.level_bytes(TOP_LEVEL) + tree.level_bytes(MAIN_LEVEL) == 0 {
                return None;
            }
            // The last level is named as a source as well as the target, so it
            // is consumed rather than appended to. That is what keeps it at one
            // run, which is the invariant capping space amplification.
            return Some(CompactionRequest {
                first_level: TOP_LEVEL,
                last_level: LAST_LEVEL,
                target_level: LAST_LEVEL,
                granularity: Granularity::Full,
                merge_units: None,
            });
        }

        // 2. An ML compaction the schedule placed here. Merges runs *within* the
        //    main level, which is why the target equals the source — the paper's
        //    main level is one logical place, not a step in a cascade.
        if let Some(width) = self.merge_at_position() {
            if tree.run_count(MAIN_LEVEL) > 1 {
                self.scanned = true;
                return Some(CompactionRequest {
                    first_level: MAIN_LEVEL,
                    last_level: MAIN_LEVEL,
                    target_level: MAIN_LEVEL,
                    granularity: Granularity::Full,
                    merge_units: Some(width),
                });
            }
        }

        // 3. Otherwise drain the top level once it is full, creating a unit run
        //    in the main level. The top level is never compacted internally:
        //    §4.1 measured that merging small runs barely helps point, seek or
        //    short-scan I/O, so the work would be wasted.
        if tree.level_bytes(TOP_LEVEL) >= self.top_capacity_bytes {
            self.position += 1;
            self.scanned = true;
            return Some(CompactionRequest::single(TOP_LEVEL, Granularity::Full));
        }

        self.scanned = true;
        None
    }

    fn max_levels(&self) -> Option<usize> {
        Some(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::shape::LevelShape;

    fn tree(level_bytes: &[u64]) -> TreeShape {
        TreeShape {
            levels: level_bytes
                .iter()
                .map(|&bytes| {
                    if bytes == 0 {
                        LevelShape::default()
                    } else {
                        LevelShape::from_sizes(&[bytes])
                    }
                })
                .collect(),
        }
    }

    /// A main level holding `runs` separate unit runs.
    fn tree_with_main_runs(top: u64, runs: usize) -> TreeShape {
        TreeShape {
            levels: vec![
                if top == 0 {
                    LevelShape::default()
                } else {
                    LevelShape::from_sizes(&[top])
                },
                LevelShape::from_sizes(&vec![100u64; runs]),
                LevelShape::default(),
            ],
        }
    }

    fn scheme() -> EcoTune {
        EcoTune::new(
            EcoTuneConfig {
                runs_per_round: 8,
                long_range_ratio: 0.5,
                ..EcoTuneConfig::default()
            },
            1024,
        )
    }

    #[test]
    fn the_mapped_tree_has_three_levels() {
        assert_eq!(scheme().max_levels(), Some(3));
        assert_eq!(scheme().name(), "ecotune");
    }

    /// The top level drains only when full, and the drain creates a unit run.
    #[test]
    fn a_full_top_level_drains_into_main() {
        let mut scheme = scheme();
        scheme.note_flush();
        assert_eq!(
            scheme.next_compaction(&tree(&[512, 0, 0])),
            None,
            "an under-full top level should be left alone"
        );

        scheme.note_flush();
        let request = scheme
            .next_compaction(&tree(&[2048, 0, 0]))
            .expect("a drain");
        assert_eq!(request.first_level, TOP_LEVEL);
        assert_eq!(request.target_level, MAIN_LEVEL);
        assert_eq!(scheme.position(), 1, "the drain creates one unit run");
    }

    /// §4.1: the top level is never compacted internally, because merging small
    /// runs barely helps the queries that matter.
    #[test]
    fn the_top_level_is_never_compacted_into_itself() {
        let mut scheme = scheme();
        for _ in 0..50 {
            scheme.note_flush();
            while let Some(request) = scheme.next_compaction(&tree_with_main_runs(2048, 3)) {
                assert!(
                    !(request.first_level == TOP_LEVEL && request.target_level == TOP_LEVEL),
                    "the top level must never merge into itself: {request:?}"
                );
            }
        }
    }

    /// The defining shape of an ML compaction: it merges runs *within* the main
    /// level, so source and target are the same place.
    #[test]
    fn a_scheduled_merge_stays_inside_the_main_level() {
        let mut scheme = scheme();
        // Advance to a position the schedule places a merge at.
        let Some(&(position, width)) = scheme.schedule.first() else {
            panic!("the schedule placed no merges");
        };
        scheme.position = position;

        scheme.note_flush();
        let request = scheme
            .next_compaction(&tree_with_main_runs(0, 4))
            .expect("a merge");

        assert_eq!(request.first_level, MAIN_LEVEL);
        assert_eq!(request.target_level, MAIN_LEVEL);
        assert_eq!(
            request.merge_units,
            Some(width),
            "the width is in unit runs, taken straight from the schedule"
        );
    }

    /// A merge needs something to merge.
    #[test]
    fn a_single_main_run_is_not_merged_with_itself() {
        let mut scheme = scheme();
        let Some(&(position, _)) = scheme.schedule.first() else {
            panic!("the schedule placed no merges");
        };
        scheme.position = position;

        scheme.note_flush();
        let request = scheme.next_compaction(&tree_with_main_runs(0, 1));
        assert!(
            request.is_none_or(|request| request.target_level != MAIN_LEVEL),
            "one run in the main level is nothing to merge"
        );
    }

    /// At the end of a round everything above the last level is merged into it,
    /// and the schedule starts again.
    #[test]
    fn the_round_ends_with_a_global_compaction() {
        let mut scheme = scheme();
        scheme.position = scheme.round_length();

        scheme.note_flush();
        let request = scheme
            .next_compaction(&tree(&[1024, 4096, 0]))
            .expect("a global compaction");

        assert_eq!(request.first_level, TOP_LEVEL);
        assert_eq!(
            request.last_level, LAST_LEVEL,
            "the last level is a source too, so it collapses to one run"
        );
        assert_eq!(request.target_level, LAST_LEVEL);
        assert!(request.spans_multiple_levels());
        assert_eq!(scheme.position(), 0, "the round restarts");
    }

    /// An empty tree at the end of a round has nothing to compact.
    #[test]
    fn an_empty_round_end_yields_no_job() {
        let mut scheme = scheme();
        scheme.position = scheme.round_length();
        scheme.note_flush();
        assert_eq!(scheme.next_compaction(&tree(&[0, 0, 0])), None);
    }

    /// One flush yields at most one scheduled action, so a caller looping until
    /// `None` cannot spin.
    #[test]
    fn each_flush_yields_at_most_one_action() {
        let mut scheme = scheme();
        for _ in 0..100 {
            scheme.note_flush();
            let mut actions = 0;
            while scheme.next_compaction(&tree_with_main_runs(4096, 3)).is_some() {
                actions += 1;
                assert!(actions <= 1, "a flush produced {actions} actions");
            }
        }
    }

    /// Position must cycle rather than run away, so the schedule stays valid
    /// over a long-lived database.
    #[test]
    fn the_position_cycles_within_the_round() {
        let mut scheme = scheme();
        let round = scheme.round_length();
        for _ in 0..500 {
            scheme.note_flush();
            while scheme.next_compaction(&tree_with_main_runs(4096, 3)).is_some() {}
            assert!(
                scheme.position() <= round,
                "position {} escaped the round length {round}",
                scheme.position()
            );
        }
    }

    #[test]
    #[should_panic(expected = "top level capacity must be positive")]
    fn a_zero_capacity_top_level_is_rejected() {
        EcoTune::new(EcoTuneConfig::default(), 0);
    }
}
