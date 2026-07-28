//! EcoTune: compaction as an investment, scheduled by dynamic programming.
//!
//! **Faithful reproduction of Algorithm 1, paper 02 §4.3.2.**
//!
//! # The premise, which is not obvious
//!
//! Every other policy in this crate optimises the write-amplification /
//! read-amplification trade-off. Paper 02 argues that trade-off is an HDD-era
//! artefact and measures the claim: modern NVMe exceeds 2 GB/s, while Meta
//! reports their highest real write speed as ~45 MB/s. Reserve enough bandwidth
//! and CPU that flush never stalls and there is still ~50x of write-amplification
//! headroom left. Their Table 2 measures write latency across three compaction
//! policies on two SSDs and finds it unchanged — 2.8-2.9 µs on Optane, 3.3-3.5 µs
//! on NVMe.
//!
//! So **write speed is a constant**, and the only question left is how to spend
//! the leftover CPU and I/O to serve the most queries.
//!
//! # Average, not instantaneous
//!
//! Read amplification is usually measured immediately after a compaction — the
//! best moment, and a fleeting one. Paper 02 measures the *average* over a
//! **compaction round** (between two global compactions) and finds that leveling,
//! despite lower read amplification than lazy leveling, delivers **64%** of its
//! throughput: leveling's compactions consumed **62% of the CPU** that queries
//! needed. Compaction and queries draw from one pool.
//!
//! # Why timing dominates
//!
//! A compaction's cost is the resources it burns; its return is the improved
//! query speed *integrated over how long that improvement lasts*. Hence the
//! central insight:
//!
//! > **The earlier in a round a compaction happens, the greater its cumulative
//! > return** — its output survives longer and serves more queries before the
//! > next global compaction erases it.
//!
//! So the optimal policy is aggressive early in a round and lazy near the end.
//! That breaks every policy built on physical levels, including this crate's
//! [`super::Leveling`] and [`super::Tiering`], because grouping runs into `L`
//! allowed sizes forces *fixed aggressiveness at all times*.
//!
//! # The model
//!
//! Physical levels are replaced by three logical ones — top (never compacted),
//! main (where this policy lives), last (one run, capping space amplification).
//! With `e` runs in the main level, `r` the proportion of long-range scans and
//! `f` the filter false-positive rate:
//!
//! ```text
//!   q(e) = 1 / ( (e+2)·r + (1−r)·(1+f) )
//! ```
//!
//! `e+2` because a long scan costs one I/O in the top level, `e` in the main
//! level and one in the last. **Score** is query speed × time, i.e. queries
//! served.
//!
//! # The recurrence
//!
//! A problem `(e, c, m)`: `e` runs already in the main level, `c` incoming unit
//! runs to schedule, and `m` further unit runs' worth of data that a later merge
//! will have to carry. The root is `(0, R, C·R)` — the global compaction at the
//! end of the round rewrites `C·R` units in the last level.
//!
//! ```text
//!   f(e, 1, m) = (m + 1)·T_c·q′(e)
//!   f(e, c, m) = max of
//!       x = 1:  T_w·q(e+1) + f(e+1, c−1, m+1)
//!       x ≥ 2:  f(e, x, 0) + (T_w − x·T_c)·q(e+1) + f(e+1, c−x, m+x)
//! ```
//!
//! `x` is how many unit runs are merged into the first *final* run — one that
//! survives to the end of the round. `x = 1` means "leave this run alone", which
//! is how a lazy policy is expressed. `O(R⁴)`.
//!
//! # What is not reproduced
//!
//! §4.3.3 relaxes the last assumption — that an ML compaction fits inside `T_w`
//! — by tracking *pending* sorted runs with a fourth parameter, at `O(R⁵)`. The
//! paper states outright: "Due to space limitations, we do not delve deeply into
//! this final version of EcoTune." Its Algorithm 2 is given without the
//! accompanying derivation, so this module implements Algorithm 1 and says so,
//! rather than guessing at the remainder.
//!
//! One consequence: `T_w − x·T_c` can go negative for a wide merge, which
//! Algorithm 1 does not guard against. That is the very case §4.3.3 exists to
//! handle. Left as written; a negative term simply makes wide merges
//! unattractive, which is directionally right.

use std::collections::HashMap;

/// Workload and hardware parameters the schedule is optimised against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EcoTuneConfig {
    /// `R`: unit sorted runs created per compaction round.
    pub runs_per_round: usize,
    /// `C`: capped size ratio between the main and last levels. Sets how much
    /// data the round-ending global compaction has to rewrite.
    pub last_level_ratio: usize,
    /// `T_w`: time between two consecutive top-to-main compactions.
    pub tm_interval: f64,
    /// `T_c`: time to rewrite one unit run's worth of data with the merge
    /// threads. Measured on the hardware, per the paper.
    pub rewrite_time: f64,
    /// `r`: proportion of long-range scans in the workload, in `[0, 1]`.
    pub long_range_ratio: f64,
    /// `f`: filter false-positive rate.
    pub false_positive_rate: f64,
    /// `β`: query speed multiplier while merge threads are busy, in `[0, 1]`.
    /// Measured, per §4.3.2.
    pub mlc_query_factor: f64,
}

impl Default for EcoTuneConfig {
    fn default() -> Self {
        Self {
            runs_per_round: 8,
            last_level_ratio: 3,
            tm_interval: 1.0,
            rewrite_time: 0.1,
            long_range_ratio: 0.3,
            false_positive_rate: 0.01,
            mlc_query_factor: 0.5,
        }
    }
}

impl EcoTuneConfig {
    /// `q(e) = 1 / ((e+2)·r + (1−r)·(1+f))`.
    ///
    /// Queries served per unit time with `e` runs in the main level. Strictly
    /// decreasing in `e` whenever the workload contains any range scans.
    pub fn query_speed(&self, main_runs: usize) -> f64 {
        let r = self.long_range_ratio;
        let cost = (main_runs as f64 + 2.0) * r + (1.0 - r) * (1.0 + self.false_positive_rate);
        if cost <= 0.0 {
            return 0.0;
        }
        1.0 / cost
    }

    /// `q′(e) = β·q(e)`: query speed while a merge is running.
    pub fn mlc_query_speed(&self, main_runs: usize) -> f64 {
        self.mlc_query_factor * self.query_speed(main_runs)
    }
}

/// The optimiser's choice at one state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    /// Best achievable score from this state onward.
    pub score: f64,
    /// Unit runs merged into the first final sorted run. `1` means "do not
    /// merge this one".
    pub merge_width: usize,
}

/// One merge in the materialised timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledMerge {
    /// Unit runs created before this merge happens — its position in the round.
    pub after_unit_runs: usize,
    /// Unit runs it combines.
    pub width: usize,
    /// Whether this is the round-ending global compaction rather than a merge
    /// the policy chose. It falls out of the model as the right-most base case
    /// of the root problem, and carries the last level's `C·R` units too.
    pub is_global: bool,
}

/// A solved compaction policy for one round.
#[derive(Debug, Clone)]
pub struct EcoTunePolicy {
    config: EcoTuneConfig,
    decisions: HashMap<(usize, usize, usize), Decision>,
    score: f64,
}

impl EcoTunePolicy {
    /// Solve the dynamic program for `config`.
    pub fn solve(config: EcoTuneConfig) -> Self {
        let mut decisions = HashMap::new();
        let runs = config.runs_per_round.max(1);
        let root_m = config.last_level_ratio * runs;

        let tail = solve_state(&config, &mut decisions, 0, runs, root_m);
        // Algorithm 1, line 14: the first T_w of the round is served with an
        // empty main level and is the same under every policy, so it sits
        // outside the recursion.
        let score = config.tm_interval * config.query_speed(0) + tail;

        Self {
            config,
            decisions,
            score,
        }
    }

    /// Total score for the round: queries served under the optimal policy.
    pub fn score(&self) -> f64 {
        self.score
    }

    pub fn config(&self) -> &EcoTuneConfig {
        &self.config
    }

    /// The optimiser's choice at a state, if that state was reachable.
    pub fn decision(&self, existing_runs: usize, incoming: usize, pending: usize) -> Option<Decision> {
        self.decisions
            .get(&(existing_runs, incoming, pending))
            .copied()
    }

    /// How many unit runs to merge at a state. `1` means leave it alone.
    pub fn merge_width(&self, existing_runs: usize, incoming: usize, pending: usize) -> Option<usize> {
        self.decision(existing_runs, incoming, pending)
            .map(|decision| decision.merge_width)
    }

    /// Walk the solved table into a timeline of merges for the round.
    ///
    /// Every merge sits at a base case of the recursion: §4.3.2 places the merge
    /// of `(m + c)·S` data at the end of a sub-problem, with its score charged in
    /// that sub-problem's right-most descendant. So a `c == 1` state *is* a
    /// merge, and there are no others — an earlier version of this walk also
    /// emitted one per `x ≥ 2` branch and double-counted every merge.
    pub fn schedule(&self) -> Vec<ScheduledMerge> {
        let mut merges = Vec::new();
        let runs = self.config.runs_per_round.max(1);
        self.walk(
            0,
            runs,
            self.config.last_level_ratio * runs,
            0,
            true,
            &mut merges,
        );
        merges.sort_by_key(|merge| merge.after_unit_runs);
        merges
    }

    /// Merges the policy actually chose, excluding the round-ending global
    /// compaction that happens regardless.
    pub fn chosen_merges(&self) -> Vec<ScheduledMerge> {
        self.schedule()
            .into_iter()
            .filter(|merge| !merge.is_global)
            .collect()
    }

    /// `rightmost` tracks whether only right branches have been taken from the
    /// root, which identifies the global compaction.
    fn walk(
        &self,
        existing: usize,
        incoming: usize,
        pending: usize,
        base: usize,
        rightmost: bool,
        merges: &mut Vec<ScheduledMerge>,
    ) {
        if incoming == 0 {
            return;
        }
        if incoming == 1 {
            merges.push(ScheduledMerge {
                after_unit_runs: base + 1,
                width: pending + 1,
                is_global: rightmost,
            });
            return;
        }

        let Some(decision) = self.decision(existing, incoming, pending) else {
            return;
        };
        let x = decision.merge_width;

        if x == 1 {
            // No merge here; this run becomes final on its own and joins the
            // pending set for a later merge.
            self.walk(
                existing + 1,
                incoming - 1,
                pending + 1,
                base + 1,
                rightmost,
                merges,
            );
            return;
        }

        // Left sub-problem organises the first x runs; its own right-most base
        // case is the merge that makes them one final run.
        self.walk(existing, x, 0, base, false, merges);
        self.walk(
            existing + 1,
            incoming - x,
            pending + x,
            base + x,
            rightmost,
            merges,
        );
    }
}

/// `f(e, c, m)` from Algorithm 1, memoised.
fn solve_state(
    config: &EcoTuneConfig,
    memo: &mut HashMap<(usize, usize, usize), Decision>,
    existing: usize,
    incoming: usize,
    pending: usize,
) -> f64 {
    if incoming == 0 {
        return 0.0;
    }
    if let Some(decision) = memo.get(&(existing, incoming, pending)) {
        return decision.score;
    }

    // Line 4: the last unit run of the problem. The merge that follows carries
    // this run plus everything pending, and runs at the reduced query speed.
    if incoming == 1 {
        let score =
            (pending + 1) as f64 * config.rewrite_time * config.mlc_query_speed(existing);
        memo.insert(
            (existing, incoming, pending),
            Decision {
                score,
                merge_width: 1,
            },
        );
        return score;
    }

    // Line 6: x = 1 — leave this run alone and move on.
    let mut best = Decision {
        score: config.tm_interval * config.query_speed(existing + 1)
            + solve_state(config, memo, existing + 1, incoming - 1, pending + 1),
        merge_width: 1,
    };

    // Lines 7-11: merge the first x runs into one final run.
    for x in 2..incoming {
        let left = solve_state(config, memo, existing, x, 0);
        let during = (config.tm_interval - x as f64 * config.rewrite_time)
            * config.query_speed(existing + 1);
        let right = solve_state(config, memo, existing + 1, incoming - x, pending + x);

        let score = left + during + right;
        if score > best.score {
            best = Decision {
                score,
                merge_width: x,
            };
        }
    }

    memo.insert((existing, incoming, pending), best);
    best.score
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> EcoTuneConfig {
        EcoTuneConfig::default()
    }

    // -----------------------------------------------------------------
    // The cost model
    // -----------------------------------------------------------------

    #[test]
    fn query_speed_matches_the_formula() {
        let config = EcoTuneConfig {
            long_range_ratio: 0.5,
            false_positive_rate: 0.02,
            ..config()
        };
        // q(3) = 1 / ((3+2)·0.5 + 0.5·1.02) = 1 / (2.5 + 0.51)
        let expected = 1.0 / (5.0 * 0.5 + 0.5 * 1.02);
        assert!((config.query_speed(3) - expected).abs() < 1e-12);
    }

    /// More runs in the main level means slower queries — the entire reason a
    /// compaction has any return at all.
    #[test]
    fn query_speed_falls_as_runs_accumulate() {
        let config = config();
        let speeds: Vec<f64> = (0..10).map(|e| config.query_speed(e)).collect();
        assert!(
            speeds.windows(2).all(|pair| pair[1] < pair[0]),
            "query speed must strictly decrease in the run count: {speeds:?}"
        );
    }

    /// With no range scans at all, the run count stops mattering: point lookups
    /// are answered from filters regardless of how many runs exist. That is the
    /// §4.1 finding the three-level model rests on.
    #[test]
    fn without_range_scans_the_run_count_is_irrelevant() {
        let config = EcoTuneConfig {
            long_range_ratio: 0.0,
            ..config()
        };
        assert!((config.query_speed(1) - config.query_speed(50)).abs() < 1e-12);
    }

    #[test]
    fn the_merge_query_speed_is_scaled_by_beta() {
        let config = EcoTuneConfig {
            mlc_query_factor: 0.25,
            ..config()
        };
        assert!((config.mlc_query_speed(4) - 0.25 * config.query_speed(4)).abs() < 1e-12);
    }

    // -----------------------------------------------------------------
    // The dynamic program
    // -----------------------------------------------------------------

    /// Algorithm 1's base case, line 4, computed by hand.
    #[test]
    fn the_base_case_matches_the_paper() {
        let config = config();
        let mut memo = HashMap::new();
        // f(2, 1, 5) = (5 + 1)·T_c·q′(2)
        let expected = 6.0 * config.rewrite_time * config.mlc_query_speed(2);
        let score = solve_state(&config, &mut memo, 2, 1, 5);
        assert!((score - expected).abs() < 1e-12);
    }

    /// Root state is `(0, R, C·R)`: the global compaction ending the round
    /// rewrites `C·R` units in the last level (Algorithm 1, line 14).
    #[test]
    fn the_root_problem_accounts_for_the_global_compaction() {
        let config = EcoTuneConfig {
            runs_per_round: 7,
            last_level_ratio: 3,
            ..config()
        };
        let policy = EcoTunePolicy::solve(config);
        assert!(
            policy.decision(0, 7, 21).is_some(),
            "the root should be (0, R, C·R) = (0, 7, 21), as in the paper's Figure 6"
        );
    }

    /// Paper 02's Figure 6 partitions `(0, 7, 21)` with `C = 3`. Whatever `x` the
    /// workload happens to select, the `m` parameter must thread through exactly
    /// as the figure shows: the right sub-problem inherits `m + x`.
    #[test]
    fn the_pending_parameter_threads_through_as_in_figure_six() {
        let config = EcoTuneConfig {
            runs_per_round: 7,
            last_level_ratio: 3,
            ..config()
        };
        let policy = EcoTunePolicy::solve(config);

        let root = policy.decision(0, 7, 21).expect("root");
        let x = root.merge_width;
        // Figure 6's root has x = 3, giving the right child (1, 4, 24).
        let expected_right = if x == 1 {
            (1, 6, 22)
        } else {
            (1, 7 - x, 21 + x)
        };
        assert!(
            policy.decision(expected_right.0, expected_right.1, expected_right.2).is_some(),
            "expected the right sub-problem {expected_right:?} to have been solved"
        );
    }

    #[test]
    fn the_score_is_positive_and_finite() {
        let policy = EcoTunePolicy::solve(config());
        assert!(policy.score().is_finite());
        assert!(policy.score() > 0.0);
    }

    /// The optimiser must never do worse than never merging at all, which is
    /// always available to it as `x = 1` at every step.
    #[test]
    fn the_optimum_beats_never_merging() {
        let config = config();
        let policy = EcoTunePolicy::solve(config);

        // Score of the all-x=1 policy, computed directly.
        let runs = config.runs_per_round;
        let mut lazy = config.tm_interval * config.query_speed(0);
        for e in 1..runs {
            lazy += config.tm_interval * config.query_speed(e);
        }
        lazy += (config.last_level_ratio * runs + runs) as f64
            * config.rewrite_time
            * config.mlc_query_speed(runs - 1);

        assert!(
            policy.score() >= lazy - 1e-9,
            "optimal {} should be at least the lazy policy's {lazy}",
            policy.score()
        );
    }

    /// **The paper's central claim**: aggressive early in a round, lazy near the
    /// end, because a run created early survives to serve more queries before
    /// the global compaction erases it.
    ///
    /// Aggressiveness is measured as merge *frequency* — the gaps between
    /// merges — not as a count per half. Merges nest: a merge of width 4 at
    /// position 4 absorbs the earlier width-2 merge plus two new runs, so widths
    /// are cumulative and say nothing on their own about when work happens.
    ///
    /// The workload here is deliberately point-heavy. With cheap merges and a
    /// scan-heavy workload the optimiser saturates — it merges at *every* step,
    /// keeping exactly one run, which is simply leveling — and a saturated
    /// policy has no taper to observe.
    #[test]
    fn merge_frequency_decreases_through_the_round() {
        let config = EcoTuneConfig {
            runs_per_round: 12,
            long_range_ratio: 0.02,
            rewrite_time: 0.02,
            ..config()
        };
        let policy = EcoTunePolicy::solve(config);
        // The round-ending global compaction is not a choice, so it is excluded.
        let chosen = policy.chosen_merges();
        assert!(chosen.len() >= 3, "expected several merges: {chosen:?}");

        // Gaps between merges, counting from the start of the round.
        let mut gaps = Vec::new();
        let mut previous = 0usize;
        for merge in &chosen {
            gaps.push(merge.after_unit_runs - previous);
            previous = merge.after_unit_runs;
        }

        assert!(
            gaps.windows(2).all(|pair| pair[1] >= pair[0]),
            "the policy should get lazier, so gaps must never shrink: {gaps:?} \
             from {chosen:?}"
        );
        assert!(
            gaps.last() > gaps.first(),
            "the policy must actually slow down over the round: {gaps:?}"
        );
    }

    /// The opposite corner, and a good sanity check on the model: when merges
    /// are nearly free and the workload is scan-heavy, the optimiser should
    /// merge at every opportunity — which is exactly the leveling policy.
    #[test]
    fn cheap_merges_and_heavy_scans_rediscover_leveling() {
        let config = EcoTuneConfig {
            runs_per_round: 10,
            long_range_ratio: 0.6,
            rewrite_time: 0.001,
            ..config()
        };
        let chosen = EcoTunePolicy::solve(config).chosen_merges();

        // Merges land at positions 2..R-1: the first needs two runs to combine,
        // and position R is the global compaction.
        assert_eq!(
            chosen.len(),
            config.runs_per_round - 2,
            "a merge at every opportunity: {chosen:?}"
        );
        assert!(
            chosen
                .iter()
                .all(|merge| merge.width == merge.after_unit_runs),
            "each merge should absorb everything so far, keeping one run: {chosen:?}"
        );
    }

    /// A workload with more range scans values a low run count more, so it
    /// should be willing to merge at least as much.
    #[test]
    fn range_heavy_workloads_merge_at_least_as_much() {
        let scan_heavy = EcoTunePolicy::solve(EcoTuneConfig {
            runs_per_round: 10,
            long_range_ratio: 0.9,
            ..config()
        });
        let point_heavy = EcoTunePolicy::solve(EcoTuneConfig {
            runs_per_round: 10,
            long_range_ratio: 0.02,
            ..config()
        });

        let merged = |policy: &EcoTunePolicy| -> usize {
            policy.schedule().iter().map(|merge| merge.width).sum()
        };
        assert!(
            merged(&scan_heavy) >= merged(&point_heavy),
            "scan-heavy merged {} units, point-heavy {}",
            merged(&scan_heavy),
            merged(&point_heavy)
        );
    }

    /// When merging is very expensive, the optimiser should stop doing it. This
    /// is the cost side of the investment view.
    #[test]
    fn expensive_merges_are_declined() {
        let cheap = EcoTunePolicy::solve(EcoTuneConfig {
            runs_per_round: 10,
            rewrite_time: 0.001,
            long_range_ratio: 0.8,
            ..config()
        });
        let ruinous = EcoTunePolicy::solve(EcoTuneConfig {
            runs_per_round: 10,
            // Each merged unit costs more than the whole interval between runs.
            rewrite_time: 5.0,
            long_range_ratio: 0.8,
            ..config()
        });

        let widest = |policy: &EcoTunePolicy| -> usize {
            policy
                .schedule()
                .iter()
                .map(|merge| merge.width)
                .max()
                .unwrap_or(0)
        };
        assert!(
            widest(&ruinous) <= widest(&cheap),
            "a ruinous rewrite cost should not merge more widely than a cheap one"
        );
    }

    #[test]
    fn a_single_run_round_is_handled() {
        let policy = EcoTunePolicy::solve(EcoTuneConfig {
            runs_per_round: 1,
            ..config()
        });
        assert!(policy.score().is_finite());
        let schedule = policy.schedule();
        assert_eq!(schedule.len(), 1, "just the round-ending merge: {schedule:?}");
    }

    /// The schedule must account for every unit run exactly once, or the policy
    /// describes a tree that cannot exist.
    #[test]
    fn the_schedule_is_internally_consistent() {
        for runs in [2usize, 5, 9, 14] {
            let policy = EcoTunePolicy::solve(EcoTuneConfig {
                runs_per_round: runs,
                ..config()
            });
            let schedule = policy.schedule();

            assert!(
                schedule
                    .windows(2)
                    .all(|pair| pair[0].after_unit_runs <= pair[1].after_unit_runs),
                "the timeline must be ordered: {schedule:?}"
            );
            for merge in &schedule {
                assert!(merge.width >= 1);
                assert!(
                    merge.after_unit_runs <= runs,
                    "a merge cannot happen after more runs than the round creates: \
                     {merge:?} with R = {runs}"
                );
            }
        }
    }

    /// `O(R⁴)` with memoisation. A large round must still solve promptly; without
    /// the memo this is exponential.
    #[test]
    fn a_large_round_solves_quickly() {
        let policy = EcoTunePolicy::solve(EcoTuneConfig {
            runs_per_round: 40,
            last_level_ratio: 4,
            ..config()
        });
        assert!(policy.score().is_finite());
        assert!(!policy.schedule().is_empty());
    }
}
