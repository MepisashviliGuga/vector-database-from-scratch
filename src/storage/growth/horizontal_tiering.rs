//! Horizontal growth under tiering.
//!
//! **This is a contribution of paper 01 (Algorithm 2, §4), not a baseline.** The
//! paper's Table 1 shows the horizontal-scheme/tiering-policy combination had
//! never been built: horizontal had only ever been paired with leveling. Any
//! reporting of results must label this as a reproduction of their algorithm,
//! never as prior art.
//!
//! ```text
//!   k ← smallest integer with N/B ≤ C(k+ℓ-1, ℓ)      // C = binomial coefficient
//!   C_i ← k for all i
//!   on each buffer flush:
//!     C_1 ← C_1 - 1
//!     for i in 1..ℓ-1:
//!       if C_i = 0:
//!         full compaction from level i to level i+1
//!         C_(i+1) ← C_(i+1) - 1
//!         C_j ← C_(i+1) for j in 1..i
//! ```
//!
//! # Why the schedule runs the other way
//!
//! Under tiering a compaction *appends* a new run to the level below instead of
//! merging into what is there, so **each entry is rewritten exactly once per
//! level regardless of when the compaction fires**. Write amplification is
//! therefore fixed, and timing affects only read cost — how many runs are alive,
//! and for how long.
//!
//! That inverts the leveling calculus. Paper 01's Figure 4 gives the intuition:
//! a run created while level `i` is nearly empty will sit there a long time and
//! serve many lookups, whereas one created when the level is nearly full will
//! shortly be merged away. So compact **less** often when a level is empty and
//! **more** often as it fills — an *increasing* frequency, the opposite
//! direction to [`super::HorizontalLeveling`], and for an entirely different
//! reason.
//!
//! Counters therefore count *down* and are reset to the value of the level below,
//! which shrinks over time, so the gaps between compactions shrink with it.
//!
//! # Optimality
//!
//! Theorem 4.2 proves this sequence minimises total read cost when the data size
//! is exactly `C(k+ℓ-1, ℓ)·B`. The proof is the paper's "non-trivial
//! generalization of Bentley and Saxe's theory": a dynamic program splitting
//! `ψ(n, ℓ)` at the first compaction into the largest level, into `ψ(i, ℓ-1)`
//! and `ψ(n-i, ℓ)`.
//!
//! # One place this departs from the paper
//!
//! Lemma 4.1 states that after `C(k+ℓ-1, ℓ)` flushes every counter reaches zero.
//! The algorithm is written for a workload of exactly that size and says nothing
//! about what follows — but a real database keeps accepting writes. When the
//! counters are exhausted this implementation re-initialises them at the next
//! `k`, which restarts the schedule for a larger data size. That is **engineering
//! glue, not the paper's algorithm**, and it is the same need Vertiorizon's
//! "dynamic resizing of the horizontal part" (§5.1) addresses.

use super::{binomial, CompactionRequest, Granularity, GrowthScheme, TreeShape};

/// Fixed level count with down-counting compaction triggers.
#[derive(Debug, Clone)]
pub struct HorizontalTiering {
    /// `C_i`, 0-indexed: `counters[0]` is the paper's `C_1`.
    counters: Vec<u64>,
    /// Current initial counter value.
    k: u64,
    /// Buffer flush size `B`, used to re-derive `k` when the schedule restarts.
    buffer_bytes: u64,
    /// Data size the current `k` was chosen for.
    horizon_bytes: u64,
    scan_cursor: usize,
}

impl HorizontalTiering {
    /// Build a scheme for `levels` levels, a buffer of `buffer_bytes`, and an
    /// expected total data size of `expected_data_bytes`.
    ///
    /// The data size sets the initial counter value: a larger workload gets a
    /// larger `k` and so a longer, more gradual schedule. The paper notes `N`
    /// may be the storage device's capacity or a user estimate, and that
    /// reasonable deviations do not significantly affect behaviour.
    ///
    /// # Panics
    ///
    /// If `levels < 2` or `buffer_bytes` is 0.
    pub fn new(levels: usize, buffer_bytes: u64, expected_data_bytes: u64) -> Self {
        assert!(levels >= 2, "a horizontal tree needs at least 2 levels");
        assert!(buffer_bytes > 0, "buffer size must be positive");

        let k = Self::initial_counter(levels, buffer_bytes, expected_data_bytes);
        Self {
            counters: vec![k; levels],
            k,
            buffer_bytes,
            horizon_bytes: expected_data_bytes,
            scan_cursor: levels,
        }
    }

    /// The smallest `k` satisfying `N/B ≤ C(k+ℓ-1, ℓ)` (Algorithm 2, lines 1-3).
    fn initial_counter(levels: usize, buffer_bytes: u64, data_bytes: u64) -> u64 {
        let flushes = data_bytes.div_ceil(buffer_bytes).max(1);
        let mut k = 0u64;
        // C(k+ℓ-1, ℓ) is 0 until k reaches 1, then rises quickly; the loop is
        // short even for very large data sizes.
        while binomial(k + levels as u64 - 1, levels as u64) < flushes {
            k += 1;
            if k > 10_000 {
                // Defensive: the binomial saturates long before this, so failing
                // to terminate would mean an arithmetic bug rather than a large
                // workload.
                break;
            }
        }
        k.max(1)
    }

    pub fn levels(&self) -> usize {
        self.counters.len()
    }

    pub fn counters(&self) -> &[u64] {
        &self.counters
    }

    /// Initial counter value currently in force.
    pub fn k(&self) -> u64 {
        self.k
    }

    /// Restart the schedule for a larger data size.
    ///
    /// Engineering glue — see the module docs. Called when every counter has
    /// reached zero, which Lemma 4.1 says happens exactly at the end of the
    /// workload the current `k` was sized for.
    fn restart_for_more_data(&mut self) {
        self.horizon_bytes = self.horizon_bytes.saturating_mul(2).max(self.buffer_bytes);
        let levels = self.counters.len();
        self.k = Self::initial_counter(levels, self.buffer_bytes, self.horizon_bytes)
            .max(self.k + 1);
        self.counters = vec![self.k; levels];
    }
}

impl GrowthScheme for HorizontalTiering {
    fn name(&self) -> &'static str {
        "horizontal-tiering"
    }

    fn note_flush(&mut self) {
        // Every counter exhausted: the workload this schedule was sized for is
        // complete, so start a longer one rather than counting below zero.
        if self.counters.iter().all(|&counter| counter == 0) {
            self.restart_for_more_data();
        }
        self.counters[0] = self.counters[0].saturating_sub(1);
        self.scan_cursor = 0;
    }

    fn next_compaction(&mut self, tree: &TreeShape) -> Option<CompactionRequest> {
        let last_source = self.counters.len().saturating_sub(1);

        while self.scan_cursor < last_source {
            let level = self.scan_cursor;
            self.scan_cursor += 1;

            if self.counters[level] == 0 {
                // Nothing to move: leave the counters alone so the schedule
                // stays aligned with the data.
                if tree.level_bytes(level) == 0 {
                    continue;
                }
                self.counters[level + 1] = self.counters[level + 1].saturating_sub(1);
                let reset_to = self.counters[level + 1];
                for counter in self.counters[..=level].iter_mut() {
                    *counter = reset_to;
                }
                return Some(CompactionRequest {
                    level,
                    // As with horizontal-leveling, the counters measure
                    // whole-level merges.
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

    fn full_tree(levels: usize) -> TreeShape {
        TreeShape {
            levels: (0..levels)
                .map(|_| LevelShape::from_sizes(&[1024]))
                .collect(),
        }
    }

    fn compaction_schedule(scheme: &mut HorizontalTiering, flushes: usize) -> Vec<usize> {
        let tree = full_tree(scheme.levels());
        let mut fired = Vec::new();
        for flush in 1..=flushes {
            scheme.note_flush();
            while let Some(request) = scheme.next_compaction(&tree) {
                assert_eq!(request.granularity, Granularity::Full);
                if request.level == 0 {
                    fired.push(flush);
                }
            }
        }
        fired
    }

    /// **Paper 01, Figure 5: ℓ = 2, k = 3.**
    ///
    /// The paper's trace, quoted: all counters start at 3. At `n = 3`, `C_1`
    /// reaches zero and fires a compaction; `C_2` drops to 2 and `C_1` resets to
    /// 2. "After two additional buffer flushes" — `n = 5` — it fires again, `C_2`
    /// drops to 1 and `C_1` resets to 1. "The next compaction then occurs after
    /// only one buffer flush", so `n = 6`.
    ///
    /// Compactions land at flushes 3, 5 and 6 — gaps of 3, 2, 1, the increasing
    /// frequency the scheme is built for.
    #[test]
    fn reproduces_the_papers_horizontal_tiering_trace() {
        let mut scheme = HorizontalTiering::new(2, 1, 1);
        // Force the paper's k rather than deriving it from a data size.
        scheme.counters = vec![3, 3];
        scheme.k = 3;

        let fired = compaction_schedule(&mut scheme, 6);
        assert_eq!(
            fired,
            vec![3, 5, 6],
            "paper 01 Figure 5 places compactions at flushes 3, 5 and 6"
        );
    }

    /// The counter values the paper states explicitly along that trace.
    #[test]
    fn counters_follow_the_papers_values() {
        let mut scheme = HorizontalTiering::new(2, 1, 1);
        scheme.counters = vec![3, 3];
        let tree = full_tree(2);

        for _ in 0..2 {
            scheme.note_flush();
            assert_eq!(scheme.next_compaction(&tree), None);
        }
        assert_eq!(scheme.counters(), &[1, 3], "two flushes consumed two counts");

        scheme.note_flush();
        assert_eq!(scheme.next_compaction(&tree).map(|r| r.level), Some(0));
        assert_eq!(
            scheme.counters(),
            &[2, 2],
            "C_2 decrements to 2 and C_1 resets to that value"
        );

        scheme.note_flush();
        assert_eq!(scheme.next_compaction(&tree), None);
        scheme.note_flush();
        assert_eq!(scheme.next_compaction(&tree).map(|r| r.level), Some(0));
        assert_eq!(scheme.counters(), &[1, 1]);
    }

    /// The defining property, and the exact opposite of horizontal-leveling.
    #[test]
    fn compaction_frequency_increases_as_the_level_fills() {
        let mut scheme = HorizontalTiering::new(2, 1, 1);
        scheme.counters = vec![6, 6];
        scheme.k = 6;

        let fired = compaction_schedule(&mut scheme, 21);
        let gaps: Vec<usize> = fired.windows(2).map(|pair| pair[1] - pair[0]).collect();

        assert!(
            gaps.windows(2).all(|pair| pair[1] <= pair[0]),
            "gaps must never grow under tiering: {gaps:?}"
        );
        assert!(
            gaps.first().unwrap() > gaps.last().unwrap(),
            "the schedule must actually speed up: {gaps:?}"
        );
    }

    /// Leveling slows down, tiering speeds up. Asserting the contrast directly
    /// guards against one being implemented in terms of the other.
    #[test]
    fn the_two_horizontal_schedules_run_in_opposite_directions() {
        use super::super::HorizontalLeveling;

        let mut tiering = HorizontalTiering::new(2, 1, 1);
        tiering.counters = vec![6, 6];
        let tiering_gaps: Vec<usize> = compaction_schedule(&mut tiering, 21)
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect();

        let mut leveling = HorizontalLeveling::new(2);
        let tree = full_tree(2);
        let mut leveling_fired = Vec::new();
        for flush in 1..=21 {
            leveling.note_flush();
            while let Some(request) = leveling.next_compaction(&tree) {
                if request.level == 0 {
                    leveling_fired.push(flush);
                }
            }
        }
        let leveling_gaps: Vec<usize> = leveling_fired
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect();

        assert!(leveling_gaps.last() > leveling_gaps.first(), "leveling slows down");
        assert!(tiering_gaps.last() < tiering_gaps.first(), "tiering speeds up");
    }

    /// Algorithm 2 lines 1-3: `k` is the smallest integer with `N/B ≤ C(k+ℓ-1, ℓ)`.
    #[test]
    fn the_initial_counter_follows_the_papers_formula() {
        // ℓ = 2: C(k+1, 2) = k(k+1)/2, the triangular numbers 1, 3, 6, 10, 15.
        // 10 flushes needs k = 4 (C(5,2) = 10); 11 needs k = 5.
        assert_eq!(HorizontalTiering::initial_counter(2, 1, 10), 4);
        assert_eq!(HorizontalTiering::initial_counter(2, 1, 11), 5);
        assert_eq!(HorizontalTiering::initial_counter(2, 1, 6), 3);

        // A bigger workload gets a longer schedule.
        let small = HorizontalTiering::initial_counter(3, 1024, 1024 * 100);
        let large = HorizontalTiering::initial_counter(3, 1024, 1024 * 100_000);
        assert!(large > small);
    }

    /// Lemma 4.1: after `C(k+ℓ-1, ℓ)` flushes every counter reaches zero. Our
    /// engineering glue must then restart rather than count below zero.
    #[test]
    fn the_schedule_restarts_instead_of_underflowing() {
        let mut scheme = HorizontalTiering::new(2, 1, 6);
        let starting_k = scheme.k();
        let tree = full_tree(2);

        // Run well past the horizon the initial k was sized for.
        for _ in 0..200 {
            scheme.note_flush();
            while let Some(request) = scheme.next_compaction(&tree) {
                assert!(request.level < 2);
            }
        }

        assert!(
            scheme.k() > starting_k,
            "the schedule should have restarted at a larger k"
        );
        assert!(
            scheme.counters().iter().any(|&counter| counter > 0),
            "counters must be replenished, not stuck at zero"
        );
    }

    #[test]
    fn the_deepest_level_never_compacts_onward() {
        let mut scheme = HorizontalTiering::new(3, 1, 100);
        let tree = full_tree(3);

        for _ in 0..300 {
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

    #[test]
    fn an_empty_level_does_not_fire_or_disturb_the_counters() {
        let mut scheme = HorizontalTiering::new(2, 1, 1);
        scheme.counters = vec![1, 3];
        let empty = TreeShape {
            levels: vec![LevelShape::default(), LevelShape::default()],
        };

        scheme.note_flush();
        assert_eq!(scheme.next_compaction(&empty), None);
        assert_eq!(
            scheme.counters(),
            &[0, 3],
            "the flush decrement stands, but no compaction bookkeeping happened"
        );
    }

    #[test]
    #[should_panic(expected = "at least 2 levels")]
    fn a_single_level_tree_is_rejected() {
        HorizontalTiering::new(1, 1024, 1024);
    }
}
