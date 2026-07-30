//! Deriving deep-level compaction policies from shallow ones — paper 06 §5.2.
//!
//! # The problem this solves
//!
//! RusKey tunes `K_i`, the number of sorted runs allowed in level `i`, by
//! reinforcement learning. But a level compacts exponentially less often the
//! deeper it sits, so the deepest levels produce training data at a trickle
//! while holding most of the data. Learning them directly would converge far too
//! slowly to track a moving workload.
//!
//! §5.2's answer: **learn the top of the tree, derive the rest.** RL tunes level
//! 1 (and level 2 under Monkey), and a closed-form recurrence extends those
//! policies downward. The recurrence is white-box — it comes from differentiating
//! a cost model, not from data — so it needs no samples at all.
//!
//! # Two cases, because the bloom filters differ
//!
//! **Uniform bits per key** (§5.2.1), which is what RocksDB does by default and
//! what this project does: every level gets the same false-positive rate, so
//! every level sees the same read/write cost ratio. There is nothing to derive —
//! the optimal policy is the *same* at every level, and level 1's learned value
//! is simply copied down.
//!
//! **Monkey allocation** (§5.2.2), where level `i` gets exponentially lower bits
//! per key than level `i+1`: now the ratio differs sharply by level and each
//! level wants its own policy. That is what [`propagate_monkey`] computes.
//!
//! # The recurrence
//!
//! Lemma 5.1, for three consecutive levels under Monkey:
//!
//! ```text
//! 1/K*_{i+1} = √( 1/K*_i² + T·(1/K*_i² − 1/K*_{i−1}²) )
//! ```
//!
//! It falls out of setting the derivative of Eq 5 — query I/O plus query CPU plus
//! update I/O plus update CPU — to zero via Lagrange multipliers, then taking the
//! ratio between adjacent levels so that the unknown system constants `X`, `Y`,
//! `Z` cancel. That cancellation is the point: the formula needs the *shape* of
//! the cost model but none of its hardware-specific coefficients.
//!
//! Read it as: the gap between consecutive levels' policies widens by a factor of
//! `T` each level down, so policies fall towards `K = 1` — the deepest levels
//! hold the most data and can least afford lazy compaction.

/// The compaction policy of a level: at most this many sorted runs.
///
/// `1` is leveling, `size_ratio` is tiering, and everything between is a Fluid
/// LSM-tree hybrid.
pub type Policy = usize;

/// Extend a learned level-1 policy to `levels` levels, for uniform bloom bits.
///
/// §5.2.1: with the same false-positive rate everywhere, every level has the same
/// read and write amplification, so the optimal policy does not vary by level.
/// The paper says so plainly, and this function is that sentence.
pub fn propagate_uniform(level_one: Policy, levels: usize) -> Vec<Policy> {
    vec![level_one.max(1); levels]
}

/// Extend learned policies for levels 1 and 2 down the tree — Lemma 5.1.
///
/// `size_ratio` is `T`. Returns one policy per level, the first two being the
/// inputs, clamped to `[1, size_ratio]` and rounded to the nearest integer as
/// §5.2.2 prescribes.
///
/// # Why it can stop early
///
/// The recurrence drives `K` downward and it bottoms out at 1, which is leveling
/// — the most aggressive policy available. Once a level reaches 1 every deeper
/// level stays there, so the remainder is filled without further arithmetic.
/// That is not a shortcut: `1/K` cannot exceed 1 for a valid policy, and the
/// paper's own example reaches `K*_4 ≈ 1` by the fourth level.
pub fn propagate_monkey(
    level_one: Policy,
    level_two: Policy,
    size_ratio: usize,
    levels: usize,
) -> Vec<Policy> {
    let ceiling = size_ratio.max(1);
    let clamp = |k: Policy| k.clamp(1, ceiling);

    let mut policies = Vec::with_capacity(levels);
    if levels == 0 {
        return policies;
    }
    policies.push(clamp(level_one));
    if levels == 1 {
        return policies;
    }
    policies.push(clamp(level_two));

    for index in 2..levels {
        // `previous` is K*_{i−1}, `current` is K*_i, and we are solving for
        // K*_{i+1}. Work in reciprocals throughout: the lemma is stated in 1/K²
        // and converting back and forth would only lose precision.
        let previous = policies[index - 2] as f64;
        let current = policies[index - 1] as f64;

        let inverse_sq_current = 1.0 / (current * current);
        let inverse_sq_previous = 1.0 / (previous * previous);
        let inner =
            inverse_sq_current + size_ratio as f64 * (inverse_sq_current - inverse_sq_previous);

        // Under Monkey the optimal policies satisfy K*_i ≤ K*_{i−1}, so `inner`
        // is non-negative whenever the inputs are ordered that way. A caller who
        // supplies an increasing pair is outside the lemma's premise; saturating
        // at the most aggressive policy is the safe reading, since a negative
        // radicand means the cost model wants a policy tighter than any available.
        if inner <= 0.0 {
            policies.resize(levels, 1);
            break;
        }

        let next = 1.0 / inner.sqrt();
        let rounded = clamp(next.round() as usize);
        policies.push(rounded);

        if rounded <= 1 {
            // Already at the most aggressive policy; nothing deeper can be lower.
            policies.resize(levels, 1);
            break;
        }
    }

    policies
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_repeats_the_learned_policy() {
        assert_eq!(propagate_uniform(5, 4), vec![5, 5, 5, 5]);
        assert_eq!(propagate_uniform(1, 3), vec![1, 1, 1]);
    }

    #[test]
    fn uniform_never_emits_an_invalid_policy() {
        // K = 0 would mean a level allowed no runs at all.
        assert_eq!(propagate_uniform(0, 2), vec![1, 1]);
    }

    /// The worked example of §5.2.2, both steps.
    ///
    /// "We assume that the tuning result of Level 1 and Level 2 are 9 and 7 …
    /// we set the compaction policy of Level 3 as K*_3 ≈ 3. Similarly, Level 4
    /// has K*_4 ≈ 1."
    #[test]
    fn monkey_reproduces_the_papers_worked_example() {
        let policies = propagate_monkey(9, 7, 10, 4);
        assert_eq!(policies, vec![9, 7, 3, 1], "§5.2.2 gives 9, 7, ≈3, ≈1");
    }

    #[test]
    fn the_unrounded_values_match_the_papers_arithmetic() {
        // Level 3: 1/K² = 1/7² + 10·(1/7² − 1/9²) = 0.101032, so K = 3.146.
        let inner: f64 = 1.0 / 49.0 + 10.0 * (1.0 / 49.0 - 1.0 / 81.0);
        let k3 = 1.0 / inner.sqrt();
        assert!(
            (k3 - 3.146).abs() < 0.01,
            "expected ≈3.15 before rounding, got {k3:.3}"
        );

        // Level 4, from the *rounded* level 3 as the paper does: 1/K² = 1/9 +
        // 10·(1/9 − 1/49) = 1.018, so K = 0.991, which clamps to 1.
        let inner: f64 = 1.0 / 9.0 + 10.0 * (1.0 / 9.0 - 1.0 / 49.0);
        let k4 = 1.0 / inner.sqrt();
        assert!(
            (k4 - 0.991).abs() < 0.01,
            "expected ≈0.99 before clamping, got {k4:.3}"
        );
    }

    #[test]
    fn policies_never_increase_with_depth() {
        // The lemma's premise, and the shape that makes it meaningful: deeper
        // levels hold more data and can least afford lazy compaction.
        for (one, two) in [(10usize, 9usize), (8, 5), (6, 6), (4, 2)] {
            let policies = propagate_monkey(one, two, 10, 6);
            for pair in policies.windows(2) {
                assert!(
                    pair[1] <= pair[0],
                    "policies rose with depth: {policies:?} from ({one}, {two})"
                );
            }
        }
    }

    #[test]
    fn policies_stay_within_one_and_the_size_ratio() {
        for ratio in [2usize, 4, 10, 16] {
            for one in 1..=ratio {
                for two in 1..=one {
                    let policies = propagate_monkey(one, two, ratio, 8);
                    for &k in &policies {
                        assert!(
                            (1..=ratio).contains(&k),
                            "policy {k} outside [1, {ratio}] from ({one}, {two})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn equal_inputs_hold_the_policy_flat() {
        // With K*_1 = K*_2 the difference term vanishes and the recurrence
        // reduces to 1/K*_{i+1} = 1/K*_i — the uniform case, reached from the
        // Monkey formula rather than assumed.
        let policies = propagate_monkey(6, 6, 10, 5);
        assert_eq!(policies, vec![6, 6, 6, 6, 6]);
    }

    #[test]
    fn a_steep_drop_bottoms_out_at_leveling() {
        // A large gap between the first two levels sends the recurrence straight
        // to K = 1, and it must stay there rather than going below or oscillating.
        let policies = propagate_monkey(10, 2, 10, 6);
        assert_eq!(policies[0], 10);
        assert_eq!(policies[1], 2);
        assert!(
            policies[2..].iter().all(|&k| k == 1),
            "should saturate at leveling: {policies:?}"
        );
    }

    #[test]
    fn an_increasing_pair_saturates_rather_than_producing_nonsense() {
        // Outside the lemma's premise (K*_i ≤ K*_{i−1}). The radicand goes
        // negative, which means the model wants a policy tighter than exists.
        let policies = propagate_monkey(3, 9, 10, 5);
        assert_eq!(policies[0], 3);
        assert_eq!(policies[1], 9);
        assert!(policies[2..].iter().all(|&k| k == 1), "{policies:?}");
    }

    #[test]
    fn degenerate_inputs_are_handled() {
        assert!(propagate_monkey(5, 3, 10, 0).is_empty());
        assert_eq!(propagate_monkey(5, 3, 10, 1), vec![5]);
        assert_eq!(propagate_monkey(5, 3, 10, 2), vec![5, 3]);
        // A size ratio of 1 leaves leveling as the only valid policy.
        assert_eq!(propagate_monkey(5, 3, 1, 3), vec![1, 1, 1]);
    }
}
