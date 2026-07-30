//! Compaction policy as a dial rather than a switch — the Fluid LSM-tree.
//!
//! Paper 06 §2, from Dostoevsky (Dayan & Idreos). RusKey's entire tuning surface
//! is `K_i`, the number of sorted runs level `i` may hold:
//!
//! ```text
//! K_i = 1  →  at most one run per level  →  leveling  (read-optimised)
//! K_i = T  →  up to T runs per level     →  tiering   (write-optimised)
//! ```
//!
//! Everything between is a hybrid. This project already had leveling and tiering
//! as separate policies; they are the two endpoints of this one knob, and
//! [`Fluid`] is what lets a tuner move between them — including *per level*,
//! which is the whole point of §5.2's policy propagation, where shallow levels
//! stay lazy and deep ones tighten towards leveling.
//!
//! # Which level's `K` decides
//!
//! The **target** level's. A compaction merges the source level's runs and
//! deposits the result one level down, so what matters is whether that
//! destination is allowed to accumulate runs:
//!
//! - `K_target = 1` — the target may hold one run, so the output must be merged
//!   into what is already there. That is leveling.
//! - `K_target > 1` — the target may accumulate, so the output is appended and
//!   the target compacts later, on its own schedule. That is tiering.
//!
//! # Delegation, deliberately
//!
//! [`Fluid`] does not reimplement either behaviour. It holds a [`Leveling`] and a
//! [`Tiering`] and forwards to whichever the target's `K` selects.
//!
//! That is not laziness. The subtle part of both policies is *when a tombstone
//! may be dropped* — leveling can drop at the deepest level because it consumes
//! every overlapping target file, while tiering additionally requires the target
//! level to be empty, since runs that survive alongside the output could still
//! hold an older value. Getting that wrong resurrects deleted keys silently, and
//! this project has already made that exact mistake once. Reusing the tested
//! implementations means a hybrid policy cannot reintroduce it.
//!
//! # A limitation of the trait, not of the idea
//!
//! [`MergePolicy::runs_per_level`] returns one number, but `Fluid` varies `K` by
//! level. It reports the **maximum** across levels, which is the conservative
//! reading: the executor uses it to decide whether disjoint runs at a level may
//! be folded into one, and over-reporting only declines a fold that would have
//! been safe. A fully faithful FLSM-tree would make that query per level.

use std::fmt::Debug;

use super::propagation::Policy;
use super::{CompactionJob, Leveling, MergePolicy, Tiering};
use crate::storage::growth::CompactionRequest;
use crate::storage::shape::TreeShape;

/// A per-level compaction policy: `policies[i]` is `K_i`.
#[derive(Debug, Clone)]
pub struct Fluid {
    policies: Vec<Policy>,
    /// Applied to levels deeper than `policies` describes.
    ///
    /// A tree grows levels at runtime while a policy vector is fixed at
    /// construction, so there must be an answer for a level nobody tuned. The
    /// deepest level holds the most data and can least afford lazy compaction,
    /// so the safe default is the most aggressive policy.
    default_policy: Policy,
    size_ratio: usize,
    leveling: Leveling,
    tiering: Tiering,
}

impl Fluid {
    /// One policy for every level.
    ///
    /// `size_ratio` is `T`, the ceiling on any `K`. Policies are clamped into
    /// `[1, size_ratio]`.
    pub fn new(policies: Vec<Policy>, size_ratio: usize) -> Self {
        let size_ratio = size_ratio.max(1);
        let policies: Vec<Policy> = policies
            .into_iter()
            .map(|k| k.clamp(1, size_ratio))
            .collect();
        Self {
            policies,
            default_policy: 1,
            size_ratio,
            leveling: Leveling,
            tiering: Tiering::new(size_ratio),
        }
    }

    /// The same `K` at every level — the §5.2.1 uniform-bloom case.
    pub fn uniform(policy: Policy, size_ratio: usize) -> Self {
        Self::new(vec![policy.clamp(1, size_ratio.max(1))], size_ratio).with_default(policy)
    }

    /// Set the policy applied to levels beyond the configured vector.
    pub fn with_default(mut self, policy: Policy) -> Self {
        self.default_policy = policy.clamp(1, self.size_ratio);
        self
    }

    /// `K` for one level.
    pub fn policy_for(&self, level: usize) -> Policy {
        self.policies
            .get(level)
            .copied()
            .unwrap_or(self.default_policy)
    }

    pub fn policies(&self) -> &[Policy] {
        &self.policies
    }

    pub fn size_ratio(&self) -> usize {
        self.size_ratio
    }

    /// Whether this level merges into its target rather than appending.
    pub fn is_leveled(&self, level: usize) -> bool {
        self.policy_for(level) <= 1
    }
}

impl MergePolicy for Fluid {
    fn name(&self) -> &'static str {
        "fluid"
    }

    fn plan(&self, tree: &TreeShape, request: CompactionRequest) -> Option<CompactionJob> {
        if self.is_leveled(request.target_level) {
            self.leveling.plan(tree, request)
        } else {
            self.tiering.plan(tree, request)
        }
    }

    fn runs_per_level(&self) -> usize {
        self.policies
            .iter()
            .copied()
            .chain(std::iter::once(self.default_policy))
            .max()
            .unwrap_or(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::growth::Granularity;
    use crate::storage::shape::{FileShape, LevelShape, RunShape};

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

    /// Three levels with **known key ranges**, which matters: leveling only
    /// selects target files it can prove overlap the merged span, so a fixture
    /// without ranges makes leveling and tiering plan identically and every
    /// comparison below vacuous.
    fn tree() -> TreeShape {
        TreeShape {
            levels: vec![
                LevelShape {
                    runs: vec![run(0, 100, "a", "m"), run(1, 100, "b", "n")],
                },
                LevelShape {
                    runs: vec![run(0, 900, "a", "z")],
                },
                LevelShape {
                    runs: vec![run(0, 4000, "a", "z")],
                },
            ],
        }
    }

    fn request() -> CompactionRequest {
        CompactionRequest::single(0, Granularity::Full)
    }

    #[test]
    fn k_of_one_plans_exactly_what_leveling_plans() {
        // The correctness anchor: at its endpoints the dial must reproduce the
        // already-tested policies rather than approximate them.
        let tree = tree();
        let fluid = Fluid::uniform(1, 10);
        assert_eq!(
            fluid.plan(&tree, request()),
            Leveling.plan(&tree, request()),
            "K=1 must be leveling"
        );
    }

    #[test]
    fn k_of_t_plans_exactly_what_tiering_plans() {
        let tree = tree();
        let fluid = Fluid::uniform(10, 10);
        assert_eq!(
            fluid.plan(&tree, request()),
            Tiering::new(10).plan(&tree, request()),
            "K=T must be tiering"
        );
    }

    #[test]
    fn the_endpoints_differ_from_each_other() {
        // Guards the two tests above against both arms collapsing to the same
        // plan, which would make them pass vacuously.
        let tree = tree();
        assert_ne!(
            Fluid::uniform(1, 10).plan(&tree, request()),
            Fluid::uniform(10, 10).plan(&tree, request()),
        );
    }

    #[test]
    fn leveling_rewrites_the_target_and_tiering_appends() {
        // The structural difference the dial interpolates, stated directly.
        let tree = tree();
        let levelled = Fluid::uniform(1, 10).plan(&tree, request()).expect("plan");
        let tiered = Fluid::uniform(10, 10).plan(&tree, request()).expect("plan");
        assert!(
            !levelled.targets.is_empty(),
            "leveling merges into the target"
        );
        assert!(tiered.targets.is_empty(), "tiering appends");
    }

    #[test]
    fn the_target_levels_policy_decides_not_the_sources() {
        // A compaction deposits its output at the target, so the target's
        // tolerance for extra runs is what selects the behaviour.
        let tree = tree();
        // Level 0 lazy, level 1 (the target) aggressive → expect leveling.
        let fluid = Fluid::new(vec![10, 1], 10);
        let plan = fluid.plan(&tree, request()).expect("plan");
        assert!(
            !plan.targets.is_empty(),
            "target level K=1 should merge, whatever the source's K"
        );

        // Now the reverse: aggressive source, lazy target → expect tiering.
        let fluid = Fluid::new(vec![1, 10], 10);
        let plan = fluid.plan(&tree, request()).expect("plan");
        assert!(plan.targets.is_empty(), "target level K>1 should append");
    }

    #[test]
    fn policies_are_clamped_to_the_size_ratio() {
        let fluid = Fluid::new(vec![0, 99, 4], 10);
        assert_eq!(fluid.policies(), &[1, 10, 4]);
    }

    #[test]
    fn levels_beyond_the_vector_take_the_default() {
        let fluid = Fluid::new(vec![8, 6], 10);
        assert_eq!(fluid.policy_for(0), 8);
        assert_eq!(fluid.policy_for(1), 6);
        // Untuned depths default to the most aggressive policy: they hold the
        // most data and can least afford laziness.
        assert_eq!(fluid.policy_for(2), 1);
        assert_eq!(fluid.policy_for(50), 1);
    }

    #[test]
    fn runs_per_level_reports_the_maximum() {
        // Conservative by design; see the module docs on the trait's limitation.
        let fluid = Fluid::new(vec![1, 7, 3], 10);
        assert_eq!(fluid.runs_per_level(), 7);
    }

    #[test]
    fn a_propagated_policy_vector_is_accepted_whole() {
        // The output of §5.2.2 feeds straight in, which is the point of both
        // modules existing.
        let policies = super::super::propagate_monkey(9, 7, 10, 4);
        let fluid = Fluid::new(policies, 10);
        assert_eq!(fluid.policies(), &[9, 7, 3, 1]);
        assert!(!fluid.is_leveled(0), "shallow levels stay lazy");
        assert!(fluid.is_leveled(3), "the deepest tightens to leveling");
    }
}
