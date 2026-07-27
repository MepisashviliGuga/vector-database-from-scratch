//! Horizontal growth under leveling: fixed level count, compaction counters.
//!
//! **Faithful reproduction of Algorithm 1, paper 01 §3.** Used by BigTable,
//! HBase and AsterixDB.
//!
//! The tree is created with `ℓ` levels and keeps exactly `ℓ` forever; as data
//! grows, every level's capacity grows with it. But capacity is the *effect*,
//! not the mechanism. The scheme is defined by a compaction counter per level:
//!
//! ```text
//!   C_i ← 0 for all i
//!   on each buffer flush:
//!     C_1 ← C_1 + 1
//!     for i in 1..ℓ-1:
//!       if C_i > C_(i+1):
//!         full compaction from level i to level i+1
//!         C_(i+1) ← C_(i+1) + 1
//!         C_i ← 0
//! ```
//!
//! `C_i` counts compactions into level `i` since level `i` last compacted
//! onward, and `C_1` counts buffer flushes. The condition `C_i > C_(i+1)` fires
//! when level `i` has taken in more merges than it has passed on.
//!
//! # Why this beats a capacity check
//!
//! The counters produce a *decreasing compaction frequency*. With `ℓ = 2` the
//! compactions land at flushes 1, 3, 6, 10, … — gaps of 1, 2, 3, 4. Level 1
//! merges down constantly while level 2 is nearly empty, and progressively less
//! often as level 2 grows.
//!
//! That is the whole point. Under leveling a compaction rewrites the target
//! level, so it costs what that level currently holds; doing the merges while
//! they are cheap and deferring them once they are expensive is what minimises
//! total write cost. Bentley and Saxe (1980) proved this schedule optimal for a
//! fixed level count, and since leveling's read cost is proportional to level
//! count, the horizontal scheme sits on the optimal read-write frontier.
//!
//! A capacity formula reproduces roughly the right level *sizes* with the wrong
//! *timing*, and the timing is the result.
//!
//! # Its weakness
//!
//! The scheme is defined on **full compaction**: one merge moves an entire level
//! at once. The inputs cannot be freed until it completes, so a large level
//! transiently needs room for both inputs and outputs, and levels above it
//! cannot drain meanwhile. That operational cost — not any steady-state
//! occupancy — is why industry chose the vertical scheme, and what Vertiorizon
//! addresses by giving the bottom two levels to a vertical part that can compact
//! partially.

use super::{CompactionRequest, Granularity, GrowthScheme, TreeShape};

/// Fixed level count, compaction driven by per-level counters.
#[derive(Debug, Clone)]
pub struct HorizontalLeveling {
    /// `C_i`, 0-indexed: `counters[0]` is the paper's `C_1`.
    counters: Vec<u64>,
    /// Where the current flush's downward scan has reached.
    ///
    /// Algorithm 1's inner loop is a single ascending pass per flush, not a
    /// repeat-until-settled loop. Resuming from a cursor reproduces that;
    /// rescanning from level 0 each call would fire extra compactions the paper
    /// does not.
    scan_cursor: usize,
}

impl HorizontalLeveling {
    /// # Panics
    ///
    /// If `levels < 2`. With one level there is nowhere to compact to.
    pub fn new(levels: usize) -> Self {
        assert!(levels >= 2, "a horizontal tree needs at least 2 levels");
        Self {
            counters: vec![0; levels],
            // No flush has happened yet, so no scan is in progress.
            scan_cursor: levels,
        }
    }

    pub fn levels(&self) -> usize {
        self.counters.len()
    }

    /// Current counter values, for tests and for tracing a schedule.
    pub fn counters(&self) -> &[u64] {
        &self.counters
    }
}

impl GrowthScheme for HorizontalLeveling {
    fn name(&self) -> &'static str {
        "horizontal-leveling"
    }

    fn note_flush(&mut self) {
        self.counters[0] += 1;
        // Begin this flush's single ascending pass.
        self.scan_cursor = 0;
    }

    fn next_compaction(&mut self, tree: &TreeShape) -> Option<CompactionRequest> {
        // The deepest level never compacts onward; there is nothing below it.
        let last_source = self.counters.len().saturating_sub(1);

        while self.scan_cursor < last_source {
            let level = self.scan_cursor;
            self.scan_cursor += 1;

            if self.counters[level] > self.counters[level + 1] {
                // An empty level satisfies the counter condition but has nothing
                // to move. Leaving the counters untouched keeps the schedule
                // aligned with the data rather than running ahead of it.
                if tree.level_bytes(level) == 0 {
                    continue;
                }
                self.counters[level + 1] += 1;
                self.counters[level] = 0;
                return Some(CompactionRequest {
                    level,
                    // The scheme is *defined* on full compaction: the counters
                    // measure whole-level merges, so slicing one would break the
                    // correspondence between a counter tick and a level's worth
                    // of data. This is also the source of its space cost.
                    granularity: Granularity::Full,
                });
            }
        }
        None
    }

    fn max_levels(&self) -> Option<usize> {
        Some(self.counters.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::shape::LevelShape;

    /// A tree with plenty of data everywhere, so only the counters decide.
    fn full_tree(levels: usize) -> TreeShape {
        TreeShape {
            levels: (0..levels)
                .map(|_| LevelShape::from_sizes(&[1024]))
                .collect(),
        }
    }

    /// Flush numbers at which a compaction from level 0 fires, over `flushes`
    /// buffer flushes.
    fn compaction_schedule(scheme: &mut HorizontalLeveling, flushes: usize) -> Vec<usize> {
        let tree = full_tree(scheme.levels());
        let mut fired = Vec::new();
        for flush in 1..=flushes {
            scheme.note_flush();
            while let Some(request) = scheme.next_compaction(&tree) {
                assert_eq!(
                    request.granularity,
                    Granularity::Full,
                    "the horizontal scheme is defined on full compaction"
                );
                if request.level == 0 {
                    fired.push(flush);
                }
            }
        }
        fired
    }

    /// **Paper 01, Figure 2, horizontal scheme with ℓ = 2.**
    ///
    /// The paper's trace, quoted: after the first flush `C_1 = 1` and `C_2 = 0`,
    /// so `C_1 > C_2` fires a compaction, after which `C_1` resets to 0 and `C_2`
    /// becomes 1. At `n = 3`, `C_1 = 2` surpasses `C_2 = 1` and fires again. "A
    /// similar process recurs at n = 6."
    ///
    /// Compactions therefore land at flushes 1, 3 and 6.
    #[test]
    fn reproduces_the_papers_horizontal_trace() {
        let mut scheme = HorizontalLeveling::new(2);
        let fired = compaction_schedule(&mut scheme, 6);
        assert_eq!(
            fired,
            vec![1, 3, 6],
            "paper 01 Figure 2 places compactions at flushes 1, 3 and 6"
        );
    }

    /// The counter values the paper states explicitly along the way.
    #[test]
    fn counters_follow_the_papers_values() {
        let mut scheme = HorizontalLeveling::new(2);
        let tree = full_tree(2);

        scheme.note_flush();
        assert_eq!(scheme.counters(), &[1, 0], "after the first flush");
        assert_eq!(scheme.next_compaction(&tree).map(|r| r.level), Some(0));
        assert_eq!(scheme.counters(), &[0, 1], "C_1 resets, C_2 increments");

        scheme.note_flush();
        assert_eq!(scheme.next_compaction(&tree), None, "1 > 1 is false");

        scheme.note_flush();
        assert_eq!(scheme.counters(), &[2, 1]);
        assert_eq!(scheme.next_compaction(&tree).map(|r| r.level), Some(0));
        assert_eq!(scheme.counters(), &[0, 2]);
    }

    /// The property that makes the scheme optimal: gaps between compactions grow.
    #[test]
    fn compaction_frequency_decreases_as_the_tree_grows() {
        let mut scheme = HorizontalLeveling::new(2);
        let fired = compaction_schedule(&mut scheme, 60);

        // 1, 3, 6, 10, 15, ... — the triangular numbers.
        assert_eq!(&fired[..6], &[1, 3, 6, 10, 15, 21]);

        let gaps: Vec<usize> = fired.windows(2).map(|pair| pair[1] - pair[0]).collect();
        assert!(
            gaps.windows(2).all(|pair| pair[1] >= pair[0]),
            "gaps between compactions must never shrink: {gaps:?}"
        );
        assert!(
            gaps.last().unwrap() > gaps.first().unwrap(),
            "the schedule must actually slow down over time"
        );
    }

    /// Contrast with vertical, whose gaps are constant. This is the difference
    /// the whole paper turns on.
    #[test]
    fn the_schedule_differs_from_a_fixed_frequency() {
        let mut scheme = HorizontalLeveling::new(2);
        let fired = compaction_schedule(&mut scheme, 30);
        let gaps: Vec<usize> = fired.windows(2).map(|pair| pair[1] - pair[0]).collect();
        assert!(
            gaps.iter().collect::<std::collections::HashSet<_>>().len() > 1,
            "a fixed compaction frequency would give identical gaps: {gaps:?}"
        );
    }

    #[test]
    fn deeper_levels_compact_too() {
        let mut scheme = HorizontalLeveling::new(3);
        let tree = full_tree(3);
        let mut deep_fired = 0;

        for _ in 0..40 {
            scheme.note_flush();
            while let Some(request) = scheme.next_compaction(&tree) {
                if request.level == 1 {
                    deep_fired += 1;
                }
            }
        }
        assert!(deep_fired > 0, "level 1 should have compacted into level 2");
    }

    /// The deepest level has nowhere to go, so it must never be returned.
    #[test]
    fn the_deepest_level_never_compacts_onward() {
        let mut scheme = HorizontalLeveling::new(3);
        let tree = full_tree(3);

        for _ in 0..200 {
            scheme.note_flush();
            while let Some(request) = scheme.next_compaction(&tree) {
                assert!(
                    request.level < 2,
                    "level {} is the deepest and cannot compact",
                    request.level
                );
            }
        }
    }

    /// An empty level satisfies the counter condition but has nothing to move.
    /// Firing anyway would advance the schedule past data that never existed.
    #[test]
    fn an_empty_level_does_not_fire_or_disturb_the_counters() {
        let mut scheme = HorizontalLeveling::new(2);
        let empty = TreeShape {
            levels: vec![LevelShape::default(), LevelShape::default()],
        };

        scheme.note_flush();
        assert_eq!(scheme.next_compaction(&empty), None);
        assert_eq!(
            scheme.counters(),
            &[1, 0],
            "the counters must not advance for a compaction that did not happen"
        );
    }

    /// Each flush drives a single ascending pass, not a loop to quiescence.
    #[test]
    fn one_flush_yields_at_most_one_compaction_per_level() {
        let mut scheme = HorizontalLeveling::new(4);
        let tree = full_tree(4);

        for _ in 0..50 {
            scheme.note_flush();
            let mut seen = Vec::new();
            while let Some(request) = scheme.next_compaction(&tree) {
                assert!(
                    !seen.contains(&request.level),
                    "level {} fired twice in one flush: {seen:?}",
                    request.level
                );
                seen.push(request.level);
            }
            assert!(
                seen.windows(2).all(|pair| pair[0] < pair[1]),
                "the pass must ascend: {seen:?}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "at least 2 levels")]
    fn a_single_level_tree_is_rejected() {
        HorizontalLeveling::new(1);
    }
}
