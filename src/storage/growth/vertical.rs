//! Vertical growth: fixed capacity per level, add levels as data grows.
//!
//! This is the classic scheme, used by RocksDB, LevelDB and WiredTiger, and the
//! baseline paper 01 measures Vertiorizon against.
//!
//! Level `i` holds `base * T^i` bytes, forever. As the data set grows the tree
//! grows *downward*, gaining a new level roughly every time the data multiplies
//! by `T`:
//!
//! ```text
//!   T = 10, base = 1 MiB
//!
//!   L0 │█                    1 MiB
//!   L1 │██                  10 MiB
//!   L2 │████               100 MiB
//!   L3 │████████             1 GiB     ← added when the data passes ~1 GiB
//!   L4 │████████████████    10 GiB     ← added when the data passes ~10 GiB
//! ```
//!
//! # What it costs
//!
//! Level count grows as `log_T(N)`. Each level is somewhere a point lookup may
//! have to probe, and a step every record must be merged through on its way
//! down, so **both read cost and write cost grow with the logarithm of the data
//! size**. Space is the scheme's strength: the bottom level dominates, so
//! obsolete copies above it are a small fraction of the total.
//!
//! Paper 01's criticism is that this trade-off is not optimal — vertical pays
//! more read cost than is theoretically necessary for the write cost it incurs.
//! That claim is what Phase 3 measures.

use super::{saturating_geometric, GrowthScheme};

/// Fixed level capacities; the tree gains levels as it grows.
#[derive(Debug, Clone, Copy)]
pub struct Vertical {
    /// Capacity of level 0. Conventionally a small multiple of the memtable
    /// flush threshold, so a level-0 compaction merges a handful of runs.
    base_level_bytes: u64,
    /// Ratio `T` between consecutive level capacities.
    size_ratio: u64,
}

impl Vertical {
    /// # Panics
    ///
    /// If `size_ratio < 2`. A ratio of 1 would make every level the same size,
    /// so the tree could never accommodate growth by adding levels — it would
    /// add them without bound and never converge.
    pub fn new(base_level_bytes: u64, size_ratio: u64) -> Self {
        assert!(
            size_ratio >= 2,
            "size ratio must be at least 2, got {size_ratio}"
        );
        assert!(base_level_bytes > 0, "base level capacity must be positive");
        Self {
            base_level_bytes,
            size_ratio,
        }
    }

    pub fn base_level_bytes(&self) -> u64 {
        self.base_level_bytes
    }
}

impl GrowthScheme for Vertical {
    fn name(&self) -> &'static str {
        "vertical"
    }

    /// `base * T^level`, independent of how much data exists.
    fn level_capacity_bytes(&self, level: usize, _total_bytes: u64) -> u64 {
        saturating_geometric(self.base_level_bytes, self.size_ratio, level)
    }

    /// Enough levels that the bottom one can hold the whole data set.
    ///
    /// Computed by walking capacities rather than with a logarithm, because
    /// `f64::log` on byte counts near `u64::MAX` loses precision exactly where
    /// an off-by-one adds a spurious level.
    fn target_level_count(&self, total_bytes: u64) -> usize {
        if total_bytes == 0 {
            return 1;
        }
        let mut levels = 1usize;
        let mut bottom_capacity = self.base_level_bytes;
        while bottom_capacity < total_bytes {
            if bottom_capacity == u64::MAX {
                break;
            }
            bottom_capacity = bottom_capacity.saturating_mul(self.size_ratio);
            levels += 1;
        }
        levels
    }

    fn size_ratio(&self) -> u64 {
        self.size_ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    #[test]
    fn capacities_grow_geometrically() {
        let scheme = Vertical::new(MIB, 10);
        assert_eq!(scheme.level_capacity_bytes(0, 0), MIB);
        assert_eq!(scheme.level_capacity_bytes(1, 0), 10 * MIB);
        assert_eq!(scheme.level_capacity_bytes(2, 0), 100 * MIB);
        assert_eq!(scheme.level_capacity_bytes(3, 0), 1000 * MIB);
    }

    /// The defining property: capacities do not depend on the data size. This is
    /// exactly what horizontal inverts.
    #[test]
    fn capacities_ignore_the_data_size() {
        let scheme = Vertical::new(MIB, 10);
        for total in [0, MIB, 1000 * MIB, u64::MAX] {
            assert_eq!(scheme.level_capacity_bytes(2, total), 100 * MIB);
        }
    }

    #[test]
    fn levels_are_added_as_data_grows() {
        let scheme = Vertical::new(MIB, 10);

        assert_eq!(scheme.target_level_count(0), 1);
        assert_eq!(scheme.target_level_count(MIB), 1, "exactly full needs no new level");
        assert_eq!(scheme.target_level_count(MIB + 1), 2);
        assert_eq!(scheme.target_level_count(10 * MIB), 2);
        assert_eq!(scheme.target_level_count(10 * MIB + 1), 3);
        assert_eq!(scheme.target_level_count(100 * MIB), 3);
    }

    /// Level count should track `log_T(N)`: a thousandfold more data buys three
    /// more levels at `T = 10`, not thirty.
    #[test]
    fn level_count_is_logarithmic_in_the_data_size() {
        let scheme = Vertical::new(MIB, 10);
        let small = scheme.target_level_count(MIB);
        let thousand_times = scheme.target_level_count(1000 * MIB);
        assert_eq!(thousand_times - small, 3);

        let million_times = scheme.target_level_count(1_000_000 * MIB);
        assert_eq!(million_times - small, 6);
    }

    #[test]
    fn a_larger_ratio_means_fewer_levels() {
        let shallow = Vertical::new(MIB, 100);
        let steep = Vertical::new(MIB, 2);
        let data = 10_000 * MIB;
        assert!(
            shallow.target_level_count(data) < steep.target_level_count(data),
            "a bigger fanout should need fewer levels for the same data"
        );
    }

    #[test]
    fn enormous_data_sizes_do_not_overflow_or_hang() {
        let scheme = Vertical::new(1, 2);
        let levels = scheme.target_level_count(u64::MAX);
        assert!(
            (60..=70).contains(&levels),
            "doubling from 1 byte should reach u64::MAX in ~64 levels, got {levels}"
        );
        assert_eq!(scheme.level_capacity_bytes(200, u64::MAX), u64::MAX);
    }

    #[test]
    #[should_panic(expected = "size ratio must be at least 2")]
    fn a_ratio_of_one_is_rejected() {
        Vertical::new(MIB, 1);
    }
}
