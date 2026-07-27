//! Axis 1: how the tree accommodates growth.
//!
//! An LSM-tree stores data in levels of geometrically increasing size. When the
//! data set doubles, something has to give — and there are exactly two things
//! that can:
//!
//! - **[`Vertical`]**: keep each level's capacity fixed and *add another level*.
//!   This is what RocksDB, LevelDB and WiredTiger do.
//! - **[`Horizontal`]**: keep the number of levels fixed and *grow every level's
//!   capacity*.
//!
//! Neither is free, and the trade-off is the subject of paper 01 ("How to Grow
//! an LSM-tree?", Mo/Luo/Idreos, SIGMOD 2025):
//!
//! - Vertical adds a level roughly every time the data grows by a factor of `T`.
//!   Every extra level is another place a point lookup may have to probe and
//!   another merge step a record must pass through on its way to the bottom. The
//!   paper's analysis is that vertical does not reach an optimal read-write
//!   trade-off: it pays more read cost than its write cost should require.
//! - Horizontal never adds a level, so lookup cost stays bounded — but the
//!   levels above the bottom hold a fraction of the data that grows with `N`,
//!   and those are all *duplicate or obsolete* versions of records that also
//!   exist below. That is space amplification, and it is horizontal's weakness.
//!
//! Vertiorizon (Phase 1) combines the two. Both baselines here exist so that
//! claim can be measured rather than repeated.
//!
//! # What a growth scheme actually decides
//!
//! Only two things, both pure functions of the data size:
//!
//! 1. [`GrowthScheme::level_capacity_bytes`] — how much level `i` may hold.
//! 2. [`GrowthScheme::target_level_count`] — how many levels there should be.
//!
//! It does *not* decide when to merge or which runs to merge; that is the
//! compaction policy, a separate axis. Keeping the two apart is what lets the
//! Phase 3 benchmarks vary one while holding the other fixed.

use std::fmt::Debug;

pub mod horizontal;
pub mod vertical;

pub use horizontal::Horizontal;
pub use vertical::Vertical;

/// How the tree is shaped as data grows.
///
/// Implementations must be pure: the same inputs always give the same answer,
/// with no reference to the tree's current contents. Compaction consumes these
/// numbers to decide whether a level is over its budget.
pub trait GrowthScheme: Debug + Send + Sync {
    /// Short name for benchmark output and plot legends.
    fn name(&self) -> &'static str;

    /// Bytes level `level` may hold before it is considered full.
    ///
    /// `total_bytes` is the size of the live data set. Vertical ignores it;
    /// horizontal is defined in terms of it.
    fn level_capacity_bytes(&self, level: usize, total_bytes: u64) -> u64;

    /// How many levels the tree should have for `total_bytes` of data.
    ///
    /// The last level is the bottom one, which in a well-shaped tree holds the
    /// large majority of the data.
    fn target_level_count(&self, total_bytes: u64) -> usize;

    /// The ratio between consecutive level capacities.
    fn size_ratio(&self) -> u64;
}

/// `base * ratio^exponent`, saturating instead of overflowing.
///
/// Deep levels in a vertical tree reach enormous nominal capacities. Saturating
/// is correct here: a capacity larger than any possible data set means "this
/// level is never full", which is exactly what it should mean.
pub(crate) fn saturating_geometric(base: u64, ratio: u64, exponent: usize) -> u64 {
    let mut value = base as u128;
    for _ in 0..exponent {
        value = value.saturating_mul(ratio as u128);
        if value > u64::MAX as u128 {
            return u64::MAX;
        }
    }
    value.min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometric_growth_saturates_rather_than_wrapping() {
        assert_eq!(saturating_geometric(4, 10, 0), 4);
        assert_eq!(saturating_geometric(4, 10, 3), 4000);
        // Deep levels must clamp, not wrap around to a small number, or a level
        // would appear permanently full.
        assert_eq!(saturating_geometric(1024, 10, 64), u64::MAX);
        assert_eq!(saturating_geometric(u64::MAX, 2, 1), u64::MAX);
    }

    /// Both schemes have to agree on the shape of the interface, so the
    /// benchmark harness can hold one and swap the other.
    #[test]
    fn both_baselines_satisfy_the_trait() {
        let schemes: Vec<Box<dyn GrowthScheme>> = vec![
            Box::new(Vertical::new(1024, 10)),
            Box::new(Horizontal::new(4, 10, 1024)),
        ];
        for scheme in schemes {
            assert!(!scheme.name().is_empty());
            assert!(scheme.size_ratio() > 1);
            assert!(scheme.target_level_count(1_000_000) >= 1);
            assert!(scheme.level_capacity_bytes(0, 1_000_000) > 0);
        }
    }
}
