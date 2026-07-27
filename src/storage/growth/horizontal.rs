//! Horizontal growth: fixed number of levels, capacities grow with the data.
//!
//! The inverse of [`super::Vertical`]. The tree is created with `L` levels and
//! keeps exactly `L` levels forever; when the data set grows, every level's
//! capacity grows with it:
//!
//! ```text
//!   T = 10, L = 4
//!
//!   N = 1 GiB                     N = 10 GiB
//!   L0 │▏          ~1 MiB         L0 │▎          ~10 MiB
//!   L1 │▍         ~10 MiB         L1 │█         ~100 MiB
//!   L2 │██       ~100 MiB         L2 │████        ~1 GiB
//!   L3 │████████  ~900 MiB        L3 │████████    ~9 GiB
//!            same shape, scaled up
//! ```
//!
//! # The capacity formula
//!
//! Capacities form a geometric series with ratio `T` that must sum to the data
//! size `N`. With `c` the capacity of level 0:
//!
//! ```text
//!   c + cT + cT² + ... + cT^(L-1) = N
//!   c (T^L - 1) / (T - 1)         = N
//!   c                             = N (T - 1) / (T^L - 1)
//! ```
//!
//! so level `i` gets `capacity_i = N (T - 1) T^i / (T^L - 1)`. The bottom level
//! takes `(T-1)/T` of the total — 90% at `T = 10` — and everything above it
//! shares the remaining tenth.
//!
//! # What it costs
//!
//! Lookup cost is bounded: there are always `L` levels to probe, no matter how
//! large the data set becomes. That is the scheme's appeal, and why it sits at a
//! different point on the read-write trade-off curve than vertical.
//!
//! Its weakness is space. Paper 01 reports that Vertiorizon cuts space cost by
//! roughly **6x** versus the horizontal scheme, measured inside RocksDB. This
//! module does not attempt to re-derive why — that analysis is read carefully in
//! Phase 1, before Vertiorizon is implemented, and the mechanism is written up
//! then. What matters here is that horizontal is implemented faithfully enough
//! for that comparison to be measured on the same engine rather than asserted.

use super::{saturating_geometric, GrowthScheme};

/// Fixed level count; capacities scale with the data set.
#[derive(Debug, Clone, Copy)]
pub struct Horizontal {
    level_count: usize,
    size_ratio: u64,
    /// Floor on any level's capacity.
    ///
    /// Without it, a small database computes fractional-byte capacities that
    /// round to zero, and every level would read as permanently over budget —
    /// compaction would thrash on an almost empty tree.
    min_level_bytes: u64,
}

impl Horizontal {
    /// # Panics
    ///
    /// If `level_count` is 0 or `size_ratio < 2`.
    pub fn new(level_count: usize, size_ratio: u64, min_level_bytes: u64) -> Self {
        assert!(level_count > 0, "a tree needs at least one level");
        assert!(
            size_ratio >= 2,
            "size ratio must be at least 2, got {size_ratio}"
        );
        assert!(min_level_bytes > 0, "minimum level capacity must be positive");
        Self {
            level_count,
            size_ratio,
            min_level_bytes,
        }
    }

    pub fn min_level_bytes(&self) -> u64 {
        self.min_level_bytes
    }
}

impl GrowthScheme for Horizontal {
    fn name(&self) -> &'static str {
        "horizontal"
    }

    /// `N (T-1) T^i / (T^L - 1)`, floored at `min_level_bytes`.
    ///
    /// Computed in `u128` so the `N * T^i` numerator does not overflow before
    /// the division brings it back into range.
    fn level_capacity_bytes(&self, level: usize, total_bytes: u64) -> u64 {
        // A level below the bottom of this tree holds nothing. Compaction uses
        // this to know it must never create one.
        if level >= self.level_count {
            return 0;
        }

        let ratio = self.size_ratio as u128;
        let denominator = saturating_geometric(1, self.size_ratio, self.level_count) as u128 - 1;
        let numerator = (total_bytes as u128)
            .saturating_mul(ratio - 1)
            .saturating_mul(saturating_geometric(1, self.size_ratio, level) as u128);

        let capacity = (numerator / denominator.max(1)).min(u64::MAX as u128) as u64;
        capacity.max(self.min_level_bytes)
    }

    /// Constant, by definition. This is the whole point of the scheme.
    fn target_level_count(&self, _total_bytes: u64) -> usize {
        self.level_count
    }

    fn size_ratio(&self) -> u64 {
        self.size_ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;

    /// The defining property, and the exact inverse of vertical's.
    #[test]
    fn level_count_never_changes() {
        let scheme = Horizontal::new(4, 10, 1024);
        for total in [0, MIB, GIB, 1000 * GIB, u64::MAX] {
            assert_eq!(scheme.target_level_count(total), 4);
        }
    }

    #[test]
    fn capacities_scale_with_the_data_size() {
        let scheme = Horizontal::new(4, 10, 1024);

        let at_one_gib = scheme.level_capacity_bytes(3, GIB);
        let at_ten_gib = scheme.level_capacity_bytes(3, 10 * GIB);

        let growth = at_ten_gib as f64 / at_one_gib as f64;
        assert!(
            (9.9..=10.1).contains(&growth),
            "tenfold data should give a tenfold capacity, got {growth:.3}x"
        );
    }

    #[test]
    fn capacities_sum_to_the_data_size() {
        let scheme = Horizontal::new(5, 10, 1);
        let total = 100 * GIB;

        let sum: u64 = (0..5).map(|i| scheme.level_capacity_bytes(i, total)).sum();
        let error = (sum as f64 - total as f64).abs() / total as f64;
        assert!(
            error < 0.001,
            "level capacities should sum to the data size; sum {sum}, total {total}"
        );
    }

    /// The bottom level takes `(T-1)/T` of everything — 90% at ratio 10.
    #[test]
    fn the_bottom_level_holds_the_large_majority() {
        let scheme = Horizontal::new(4, 10, 1);
        let total = 100 * GIB;

        let bottom = scheme.level_capacity_bytes(3, total) as f64 / total as f64;
        assert!(
            (0.89..=0.91).contains(&bottom),
            "bottom level should hold ~90% of the data, got {:.1}%",
            bottom * 100.0
        );
    }

    #[test]
    fn consecutive_capacities_keep_the_size_ratio() {
        let scheme = Horizontal::new(5, 10, 1);
        let total = 1000 * GIB;

        for level in 0..4 {
            let lower = scheme.level_capacity_bytes(level, total) as f64;
            let higher = scheme.level_capacity_bytes(level + 1, total) as f64;
            let ratio = higher / lower;
            assert!(
                (9.9..=10.1).contains(&ratio),
                "level {level} to {} ratio was {ratio:.3}, expected ~10",
                level + 1
            );
        }
    }

    /// A level past the bottom holds nothing, so compaction never invents one.
    #[test]
    fn levels_beyond_the_bottom_have_no_capacity() {
        let scheme = Horizontal::new(3, 10, 1024);
        assert!(scheme.level_capacity_bytes(2, GIB) > 0);
        assert_eq!(scheme.level_capacity_bytes(3, GIB), 0);
        assert_eq!(scheme.level_capacity_bytes(99, GIB), 0);
    }

    /// On an almost empty tree the formula yields fractional bytes. Without a
    /// floor those round to zero and every level looks permanently over budget,
    /// which would make compaction thrash.
    #[test]
    fn a_tiny_database_still_gets_usable_capacities() {
        let scheme = Horizontal::new(4, 10, 4096);

        for total in [0, 1, 100, 4096] {
            for level in 0..4 {
                assert!(
                    scheme.level_capacity_bytes(level, total) >= 4096,
                    "level {level} at total {total} fell below the floor"
                );
            }
        }
    }

    #[test]
    fn enormous_data_sizes_do_not_overflow() {
        let scheme = Horizontal::new(7, 10, 1024);
        let bottom = scheme.level_capacity_bytes(6, u64::MAX);
        // The bottom level should still be ~90% of the total. Anything small
        // here means the u128 intermediate wrapped or truncated.
        assert!(
            bottom > u64::MAX / 2,
            "bottom level came out as {bottom}, which means the arithmetic overflowed"
        );
    }

    #[test]
    #[should_panic(expected = "at least one level")]
    fn zero_levels_is_rejected() {
        Horizontal::new(0, 10, 1024);
    }
}
