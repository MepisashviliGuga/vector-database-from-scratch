//! Lerp: learning the compaction policy from measured latency — paper 06 §5.
//!
//! # What is being learned
//!
//! One number: `K`, the runs a level may hold. `K = 1` is leveling and `K = T` is
//! tiering, so the learner is choosing where to sit on that dial as the workload
//! moves. Under uniform bloom bits — what this engine uses — §5.2.1 says every
//! level shares the same optimal policy, so a single `K` is propagated down the
//! tree by [`super::propagate_uniform`] rather than learned per level.
//!
//! # The loop
//!
//! Work is divided into **missions**, batches of operations. After each mission
//! the tuner sees what that mission cost and picks the policy for the next one:
//!
//! ```text
//! mission runs under K  →  observe (read ratio, latency)  →  update  →  choose next K
//! ```
//!
//! Two design choices from §5.1.2 make this tractable, and both are reproduced:
//!
//! - **The action space is three.** A policy vector over `L` levels would give
//!   `O(T^L)` choices. Lerp instead moves `K` by at most one per mission — down,
//!   stay, or up. Real workloads drift rather than jump, and a drastic policy
//!   change is usually the wrong response even when the workload does jump.
//! - **State is the workload, not the tree.** What decides the right `K` is the
//!   read/write mix; the tree's own statistics follow from it.
//!
//! # Simplification: tabular Q-learning, not DDPG
//!
//! **Labelled simplification.** The paper implements Lerp as DDPG — an
//! actor-critic pair of 3-layer, 128-neuron networks in PyTorch (§5.1.4, §7).
//! This uses tabular Q-learning over a discretised state instead.
//!
//! The justification is that the problem this reduces to is small: three actions,
//! and a state of (read-ratio bucket, current `K`). A table covers it exactly,
//! converges from far fewer samples, and — the reason that matters here — can be
//! *checked*, since the optimal policy is analytically known for a given cost
//! model and the learner can be asserted to find it.
//!
//! What it gives up: DDPG handles a continuous, high-dimensional state, so the
//! paper can feed in level capacities, I/O counts and per-level statistics
//! wholesale. Discretising that many dimensions would blow up the table. If the
//! state ever needs to grow beyond a couple of dimensions, this must be replaced
//! rather than extended.
//!
//! # Simplification: the reward is end-to-end latency only
//!
//! §5.1.3 blends level-local and end-to-end latency, `α·t_i + (1−α)·t'`, with
//! `α = ½`. This engine does not instrument per-level latency, so only `t'` is
//! available — equivalent to `α = 0`. Under uniform bloom bits, where a single
//! policy governs every level anyway, the level-local term has much less to say
//! than it does under Monkey.

use std::collections::HashMap;

use super::propagation::{propagate_uniform, Policy};
use crate::workload::Rng;

/// How the learner explores and updates.
#[derive(Debug, Clone, Copy)]
pub struct LerpConfig {
    /// `T`: the ceiling on `K`, and the tiering endpoint.
    pub size_ratio: usize,
    /// Policy before anything is learned.
    pub initial_policy: Policy,
    /// Q-learning step size.
    pub learning_rate: f64,
    /// Discount on the value of the next state.
    ///
    /// A mission's cost depends mostly on the policy it ran under rather than on
    /// policies chosen after it, so this is deliberately low — the problem is
    /// closer to a contextual bandit than to a long-horizon control task.
    pub discount: f64,
    /// Probability of taking a random action, before decay.
    pub exploration: f64,
    /// Multiplier applied to `exploration` after each mission.
    pub exploration_decay: f64,
    /// Floor on exploration, so a shifted workload can still be discovered.
    ///
    /// This is what makes the learner *dynamic* rather than merely convergent:
    /// with exploration decayed to zero it would sit on a policy that was optimal
    /// for a workload that has since changed.
    pub min_exploration: f64,
    /// Read-ratio buckets the state is discretised into.
    pub ratio_buckets: usize,
    pub seed: u64,
}

impl Default for LerpConfig {
    fn default() -> Self {
        Self {
            size_ratio: 10,
            initial_policy: 1,
            learning_rate: 0.3,
            discount: 0.4,
            exploration: 0.5,
            exploration_decay: 0.97,
            min_exploration: 0.05,
            ratio_buckets: 5,
            seed: 0x1EA5_1EA5,
        }
    }
}

/// What one mission cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mission {
    /// Share of the mission's operations that were lookups, in `[0, 1]`.
    pub read_ratio: f64,
    /// Mean latency per operation. Any consistent unit; only ratios matter.
    pub latency: f64,
}

/// The learner's discretised view of the world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct State {
    ratio_bucket: usize,
    policy: Policy,
}

/// The three moves of §5.1.2.
const ACTIONS: [i32; 3] = [-1, 0, 1];

/// A level-based Q-learning tuner for the compaction policy.
#[derive(Debug)]
pub struct Lerp {
    config: LerpConfig,
    /// Action values, indexed by state. Missing entries read as zero.
    q: HashMap<State, [f64; ACTIONS.len()]>,
    policy: Policy,
    exploration: f64,
    /// The state and action awaiting a reward, set when a policy is chosen.
    pending: Option<(State, usize)>,
    missions: u64,
    rng: Rng,
}

impl Lerp {
    pub fn new(config: LerpConfig) -> Self {
        let ceiling = config.size_ratio.max(1);
        Self {
            policy: config.initial_policy.clamp(1, ceiling),
            exploration: config.exploration,
            config,
            q: HashMap::new(),
            pending: None,
            missions: 0,
            rng: Rng::new(config.seed),
        }
    }

    /// The policy to run the next mission under.
    pub fn policy(&self) -> Policy {
        self.policy
    }

    /// The policy expanded to every level — §5.2.1's uniform propagation.
    pub fn policies(&self, levels: usize) -> Vec<Policy> {
        propagate_uniform(self.policy, levels)
    }

    pub fn missions(&self) -> u64 {
        self.missions
    }

    /// Current exploration rate, after decay.
    pub fn exploration(&self) -> f64 {
        self.exploration
    }

    /// Report a finished mission and receive the policy for the next one.
    ///
    /// Reward is **negative latency**: the learner maximises reward, and lower
    /// latency is better. Using latency directly rather than a normalised score
    /// keeps the scale meaningful across missions, which matters because a
    /// workload shift changes what "good" costs.
    pub fn observe(&mut self, mission: Mission) -> Policy {
        let state = self.state_for(mission.read_ratio);
        let reward = -mission.latency;

        // Credit the action that produced this mission, using the state we are
        // now in as the successor.
        if let Some((previous, action)) = self.pending.take() {
            let best_next = self.best_value(state);
            let entry = self.q.entry(previous).or_insert([0.0; ACTIONS.len()]);
            let current = entry[action];
            entry[action] = current
                + self.config.learning_rate * (reward + self.config.discount * best_next - current);
        }

        let action = self.choose(state);
        self.policy = self.apply(state.policy, action);
        self.pending = Some((state, action));

        self.missions += 1;
        self.exploration =
            (self.exploration * self.config.exploration_decay).max(self.config.min_exploration);

        self.policy
    }

    /// Highest action value available in a state; zero if unvisited.
    fn best_value(&self, state: State) -> f64 {
        self.q.get(&state).map_or(0.0, |values| {
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        })
    }

    fn state_for(&self, read_ratio: f64) -> State {
        let buckets = self.config.ratio_buckets.max(1);
        let clamped = read_ratio.clamp(0.0, 1.0);
        // `* buckets` lands exactly on `buckets` for a ratio of 1.0, hence the min.
        let bucket = ((clamped * buckets as f64) as usize).min(buckets - 1);
        State {
            ratio_bucket: bucket,
            policy: self.policy,
        }
    }

    /// ε-greedy over the three actions.
    fn choose(&mut self, state: State) -> usize {
        // `below(100)` gives whole percents, which is resolution enough and keeps
        // the choice reproducible from the seed.
        let roll = self.rng.below(100) as f64 / 100.0;
        if roll < self.exploration {
            return self.rng.below(ACTIONS.len() as u64) as usize;
        }

        let values = self.q.get(&state).copied().unwrap_or([0.0; ACTIONS.len()]);
        let mut best = 0usize;
        for index in 1..values.len() {
            if values[index] > values[best] {
                best = index;
            }
        }
        best
    }

    /// Move the policy by an action, staying within `[1, T]`.
    fn apply(&self, policy: Policy, action: usize) -> Policy {
        let ceiling = self.config.size_ratio.max(1);
        let delta = ACTIONS[action];
        let next = policy as i64 + i64::from(delta);
        next.clamp(1, ceiling as i64) as Policy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The standard LSM cost shape, and the reason it is the right test oracle.
    ///
    /// A lookup probes up to `K` runs per level, so read cost rises with `K`. An
    /// entry is rewritten `T/K` times, so write cost falls with `K`. This is Eq 5
    /// of the paper stripped to its two dominant terms:
    ///
    /// ```text
    /// latency(K) = γ·K + (1−γ)·T/K
    /// ```
    ///
    /// Differentiating gives an optimum at `K* = √((1−γ)·T/γ)`, so a learner can
    /// be checked against a known answer rather than against itself.
    fn latency(policy: Policy, read_ratio: f64, size_ratio: usize) -> f64 {
        let k = policy as f64;
        read_ratio * k + (1.0 - read_ratio) * size_ratio as f64 / k
    }

    /// The analytic optimum, rounded to a valid policy.
    fn optimal(read_ratio: f64, size_ratio: usize) -> Policy {
        if read_ratio <= 0.0 {
            return size_ratio;
        }
        let exact = ((1.0 - read_ratio) * size_ratio as f64 / read_ratio).sqrt();
        (exact.round() as usize).clamp(1, size_ratio)
    }

    /// Run the learner against the cost model for a fixed workload.
    fn converge(read_ratio: f64, missions: usize, config: LerpConfig) -> (Lerp, Policy) {
        let mut lerp = Lerp::new(config);
        let mut policy = lerp.policy();
        for _ in 0..missions {
            let observed = latency(policy, read_ratio, config.size_ratio);
            policy = lerp.observe(Mission {
                read_ratio,
                latency: observed,
            });
        }
        (lerp, policy)
    }

    #[test]
    fn the_cost_model_has_the_optimum_it_claims() {
        // Guards the oracle itself: if this is wrong, every test below is wrong.
        for &ratio in &[0.1f64, 0.3, 0.5, 0.9] {
            let best = optimal(ratio, 10);
            let cost = latency(best, ratio, 10);
            for k in 1..=10 {
                assert!(
                    latency(k, ratio, 10) >= cost - 1e-9,
                    "K={k} beats the analytic optimum {best} at γ={ratio}"
                );
            }
        }
    }

    #[test]
    fn a_write_heavy_workload_learns_to_compact_lazily() {
        // 10% reads: the cost model wants K near tiering.
        let config = LerpConfig::default();
        let (_, policy) = converge(0.1, 400, config);
        let want = optimal(0.1, config.size_ratio);
        assert!(
            policy >= want - 1,
            "expected a lazy policy near {want}, settled at {policy}"
        );
    }

    #[test]
    fn a_read_heavy_workload_learns_to_compact_aggressively() {
        // 90% reads: the cost model wants leveling.
        let config = LerpConfig::default();
        let (_, policy) = converge(0.9, 400, config);
        let want = optimal(0.9, config.size_ratio);
        assert!(
            policy <= want + 1,
            "expected an aggressive policy near {want}, settled at {policy}"
        );
    }

    #[test]
    fn the_two_extremes_settle_in_different_places() {
        // Guards the two tests above against both converging to the same policy
        // and passing for the wrong reason.
        let config = LerpConfig::default();
        let (_, write_heavy) = converge(0.1, 400, config);
        let (_, read_heavy) = converge(0.9, 400, config);
        assert!(
            write_heavy > read_heavy,
            "write-heavy settled at {write_heavy}, read-heavy at {read_heavy} — \
             the learner is not responding to the workload"
        );
    }

    /// Figure 2's scenario: the workload shifts underneath a converged learner.
    #[test]
    fn the_learner_follows_a_shifting_workload() {
        let config = LerpConfig::default();
        let mut lerp = Lerp::new(config);
        let mut policy = lerp.policy();

        let run = |lerp: &mut Lerp, policy: &mut Policy, ratio: f64, missions: usize| {
            for _ in 0..missions {
                let observed = latency(*policy, ratio, config.size_ratio);
                *policy = lerp.observe(Mission {
                    read_ratio: ratio,
                    latency: observed,
                });
            }
            *policy
        };

        // Write-heavy first, as in Figure 2.
        let after_writes = run(&mut lerp, &mut policy, 0.1, 300);
        // Then read-heavy: the previously learned policy is now wrong.
        let after_reads = run(&mut lerp, &mut policy, 0.9, 300);
        // And back again.
        let after_writes_again = run(&mut lerp, &mut policy, 0.1, 300);

        assert!(
            after_reads < after_writes,
            "shifting to reads should tighten the policy: {after_writes} → {after_reads}"
        );
        assert!(
            after_writes_again > after_reads,
            "shifting back to writes should loosen it again: {after_reads} → {after_writes_again}"
        );
    }

    #[test]
    fn a_state_is_remembered_across_a_workload_shift() {
        // The reason state includes the read ratio: returning to a workload seen
        // before should reuse what was learned rather than start over. The
        // learner is asked for a policy it already knows, with exploration off.
        let config = LerpConfig {
            exploration: 0.0,
            min_exploration: 0.0,
            ..Default::default()
        };
        let (mut lerp, settled) = converge(0.9, 300, config);

        // One mission of a different workload, then back.
        lerp.observe(Mission {
            read_ratio: 0.1,
            latency: latency(lerp.policy(), 0.1, config.size_ratio),
        });
        let returned = lerp.observe(Mission {
            read_ratio: 0.9,
            latency: latency(lerp.policy(), 0.9, config.size_ratio),
        });
        assert!(
            returned.abs_diff(settled) <= 1,
            "returning to a known workload jumped from {settled} to {returned}"
        );
    }

    #[test]
    fn the_policy_moves_by_at_most_one_per_mission() {
        // §5.1.2's action space. Checked under full exploration, so every action
        // is exercised rather than only the greedy one.
        let config = LerpConfig {
            exploration: 1.0,
            min_exploration: 1.0,
            initial_policy: 5,
            ..Default::default()
        };
        let mut lerp = Lerp::new(config);
        let mut policy = lerp.policy();
        for _ in 0..500 {
            let next = lerp.observe(Mission {
                read_ratio: 0.5,
                latency: latency(policy, 0.5, config.size_ratio),
            });
            assert!(
                next.abs_diff(policy) <= 1,
                "policy jumped {policy} → {next}"
            );
            policy = next;
        }
    }

    #[test]
    fn the_policy_stays_within_one_and_the_size_ratio() {
        let config = LerpConfig {
            exploration: 1.0,
            min_exploration: 1.0,
            size_ratio: 6,
            ..Default::default()
        };
        let mut lerp = Lerp::new(config);
        for _ in 0..1000 {
            let policy = lerp.observe(Mission {
                read_ratio: 0.5,
                latency: 1.0,
            });
            assert!((1..=6).contains(&policy), "policy {policy} out of range");
        }
    }

    #[test]
    fn exploration_decays_to_its_floor_and_stops() {
        let config = LerpConfig::default();
        let (lerp, _) = converge(0.5, 1000, config);
        assert!(
            (lerp.exploration() - config.min_exploration).abs() < 1e-9,
            "exploration should rest at its floor, got {}",
            lerp.exploration()
        );
    }

    #[test]
    fn exploration_never_reaches_zero() {
        // Without a floor a converged learner cannot notice a workload change,
        // which is the whole point of the system.
        let config = LerpConfig::default();
        let (lerp, _) = converge(0.5, 5000, config);
        assert!(lerp.exploration() > 0.0);
    }

    #[test]
    fn learning_is_reproducible_from_a_seed() {
        let config = LerpConfig::default();
        let (_, first) = converge(0.3, 200, config);
        let (_, second) = converge(0.3, 200, config);
        assert_eq!(first, second);

        let (_, other_seed) = converge(0.3, 200, LerpConfig { seed: 99, ..config });
        // Not asserting inequality — two seeds may agree on the optimum, which
        // is the desired outcome. Only that a seed change is accepted.
        assert!((1..=config.size_ratio).contains(&other_seed));
    }

    #[test]
    fn the_learned_policy_propagates_to_every_level() {
        let config = LerpConfig::default();
        let (lerp, policy) = converge(0.5, 200, config);
        assert_eq!(lerp.policies(4), vec![policy; 4]);
    }

    #[test]
    fn missions_are_counted() {
        let config = LerpConfig::default();
        let (lerp, _) = converge(0.5, 37, config);
        assert_eq!(lerp.missions(), 37);
    }

    #[test]
    fn degenerate_configurations_are_handled() {
        // A size ratio of 1 leaves leveling as the only policy.
        let mut lerp = Lerp::new(LerpConfig {
            size_ratio: 1,
            ..Default::default()
        });
        assert_eq!(
            lerp.observe(Mission {
                read_ratio: 0.5,
                latency: 1.0
            }),
            1
        );
        // An out-of-range initial policy is clamped rather than trusted.
        let lerp = Lerp::new(LerpConfig {
            size_ratio: 4,
            initial_policy: 99,
            ..Default::default()
        });
        assert_eq!(lerp.policy(), 4);
    }
}
