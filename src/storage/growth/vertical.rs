//! Vertical growth: fixed capacity per level, add levels as data grows.
//!
//! The classic scheme — RocksDB, LevelDB, WiredTiger, Cassandra — and the
//! baseline paper 01 measures Vertiorizon against.
//!
//! Level `i` holds `B·T^(i+1)` bytes forever, where `B` is the memtable flush
//! size. (Paper 01 indexes levels from 1 and writes `B·T^i`; this module indexes
//! from 0, so the exponent shifts by one. Level 0 therefore holds `T` flushes
//! before it merges down, which is the familiar level-0 trigger.) As data grows
//! the tree grows *downward*, gaining a level roughly every time the data
//! multiplies by `T`.
//!
//! # What it costs
//!
//! Level count grows as `log_T(N)`. Each level is somewhere a lookup may probe
//! and a step every record is merged through, so read and write cost both grow
//! with the logarithm of the data size.
//!
//! Its trigger is a *fixed* compaction frequency: equal-sized chunks merged down
//! at a constant rate. Paper 01's Figure 3 shows why that is suboptimal — a
//! compaction costs what the target level currently holds, so merging in equal
//! chunks does the expensive merges as often as the cheap ones. Moving 60 MB in
//! three equal steps writes 20 + 40 + 60 = 120 MB, where the horizontal
//! schedule's 10, 20, 30 MB steps write 10 + 30 + 60 = 100 MB.
//!
//! Space is the scheme's strength, and the reason industry adopted it: fixed
//! capacities permit *partial* compaction, merging a slice of a level rather
//! than all of it.

use super::{saturating_geometric, CompactionRequest, Granularity, GrowthScheme, TreeShape};

/// Fixed level capacities; the tree gains levels as it grows.
#[derive(Debug, Clone, Copy)]
pub struct Vertical {
    /// Memtable flush size, `B` in the paper.
    buffer_bytes: u64,
    /// Ratio `T` between consecutive level capacities.
    size_ratio: u64,
}

impl Vertical {
    /// # Panics
    ///
    /// If `size_ratio < 2`. A ratio of 1 makes every level the same size, so the
    /// tree could never accommodate growth by adding levels.
    pub fn new(buffer_bytes: u64, size_ratio: u64) -> Self {
        assert!(
            size_ratio >= 2,
            "size ratio must be at least 2, got {size_ratio}"
        );
        assert!(buffer_bytes > 0, "buffer size must be positive");
        Self {
            buffer_bytes,
            size_ratio,
        }
    }

    /// Bytes level `level` may hold: `B·T^(level+1)`.
    pub fn level_capacity_bytes(&self, level: usize) -> u64 {
        saturating_geometric(self.buffer_bytes, self.size_ratio, level + 1)
    }

    /// Levels needed to hold `total_bytes`.
    ///
    /// Walks capacities rather than taking a logarithm: `f64::log` on byte counts
    /// near [`u64::MAX`] loses precision exactly where an off-by-one adds a
    /// spurious level.
    pub fn levels_needed(&self, total_bytes: u64) -> usize {
        if total_bytes == 0 {
            return 1;
        }
        let mut levels = 1usize;
        let mut bottom = self.level_capacity_bytes(0);
        while bottom < total_bytes && bottom != u64::MAX {
            bottom = bottom.saturating_mul(self.size_ratio);
            levels += 1;
        }
        levels
    }

    pub fn size_ratio(&self) -> u64 {
        self.size_ratio
    }
}

impl GrowthScheme for Vertical {
    fn name(&self) -> &'static str {
        "vertical"
    }

    /// Nothing to record: the vertical trigger reads the tree's byte sizes
    /// directly rather than counting events.
    fn note_flush(&mut self) {}

    /// The shallowest level at or over its capacity.
    ///
    /// Rescans from level 0 each call rather than resuming where it left off,
    /// because a compaction changes the sizes of two levels and may push the
    /// target over its own capacity. Data only ever moves downward, so the
    /// cascade terminates.
    ///
    /// Compaction is **partial** below level 0 — merging one file's worth of key
    /// range rather than the whole level. That is the vertical scheme's
    /// practical advantage and the reason industry adopted it. Level 0 is the
    /// exception: its runs come straight from memtable flushes and overlap each
    /// other arbitrarily, so they cannot be sliced by key range and are merged
    /// whole.
    fn next_compaction(&mut self, tree: &TreeShape) -> Option<CompactionRequest> {
        let level = (0..tree.levels.len()).find(|&level| {
            let bytes = tree.level_bytes(level);
            // Paper 01's running example (Figure 2, T=2) triggers at exactly
            // capacity: level 1 holds 2 buffers, capacity 2 buffers, and merges.
            // So the comparison is `>=`, not `>`.
            bytes > 0 && bytes >= self.level_capacity_bytes(level)
        })?;

        Some(CompactionRequest::single(
            level,
            if level == 0 {
                Granularity::Full
            } else {
                Granularity::Partial
            },
        ))
    }

    /// Unbounded: the scheme adds levels as needed.
    fn max_levels(&self) -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::shape::LevelShape;

    const MIB: u64 = 1024 * 1024;

    /// The level a request names, discarding granularity.
    fn level_of(request: Option<CompactionRequest>) -> Option<usize> {
        request.map(|request| request.first_level)
    }

    /// Build a tree whose levels hold the given byte counts.
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

    #[test]
    fn capacities_grow_geometrically_from_the_buffer_size() {
        let scheme = Vertical::new(MIB, 10);
        assert_eq!(scheme.level_capacity_bytes(0), 10 * MIB);
        assert_eq!(scheme.level_capacity_bytes(1), 100 * MIB);
        assert_eq!(scheme.level_capacity_bytes(2), 1000 * MIB);
    }

    /// The defining property, and exactly what horizontal inverts: capacities do
    /// not depend on how much data exists.
    #[test]
    fn capacities_ignore_the_data_size() {
        let scheme = Vertical::new(MIB, 10);
        let small = tree(&[MIB]);
        let huge = tree(&[MIB, 1000 * MIB, 100_000 * MIB]);
        assert_eq!(scheme.level_capacity_bytes(1), 100 * MIB);
        // Same capacity regardless; only whether it is exceeded differs.
        assert_eq!(Vertical::new(MIB, 10).next_compaction(&small), None);
        assert!(Vertical::new(MIB, 10).next_compaction(&huge).is_some());
    }

    /// The vertical scheme's practical advantage: below level 0 it moves a slice
    /// rather than the whole level. Level 0 cannot be sliced, because its runs
    /// overlap each other.
    #[test]
    fn compaction_is_partial_below_level_zero() {
        let mut scheme = Vertical::new(1, 2);

        let at_level_zero = scheme.next_compaction(&tree(&[9])).expect("a request");
        assert_eq!(at_level_zero.first_level, 0);
        assert_eq!(at_level_zero.target_level, 1);
        assert!(!at_level_zero.spans_multiple_levels());
        assert_eq!(
            at_level_zero.granularity,
            Granularity::Full,
            "level 0 runs overlap arbitrarily and cannot be sliced by key range"
        );

        let deeper = scheme.next_compaction(&tree(&[0, 99])).expect("a request");
        assert_eq!(deeper.first_level, 1);
        assert_eq!(deeper.granularity, Granularity::Partial);
    }

    #[test]
    fn levels_are_added_as_data_grows() {
        let scheme = Vertical::new(MIB, 10);
        assert_eq!(scheme.levels_needed(0), 1);
        assert_eq!(scheme.levels_needed(10 * MIB), 1);
        assert_eq!(scheme.levels_needed(10 * MIB + 1), 2);
        assert_eq!(scheme.levels_needed(100 * MIB), 2);
        assert_eq!(scheme.levels_needed(100 * MIB + 1), 3);
    }

    #[test]
    fn level_count_is_logarithmic_in_the_data_size() {
        let scheme = Vertical::new(MIB, 10);
        let base = scheme.levels_needed(10 * MIB);
        assert_eq!(scheme.levels_needed(10_000 * MIB) - base, 3);
        assert_eq!(scheme.levels_needed(10_000_000 * MIB) - base, 6);
    }

    /// **Paper 01, Figure 2, vertical scheme with T = 2.**
    ///
    /// Level 1 (our level 0) holds 2 buffers, level 2 holds 4. The paper's trace:
    /// at `n = 2` level 1 reaches capacity and merges into level 2; at `n = 4`
    /// level 1 merges again, level 2 then reaches *its* capacity and merges into
    /// a newly created level 3.
    ///
    /// Reproducing the trace pins this implementation against the source rather
    /// than against a reading of it.
    #[test]
    fn reproduces_the_papers_vertical_trace() {
        // One buffer is 1 unit; T = 2, so level 0 holds 2 and level 1 holds 4.
        let mut scheme = Vertical::new(1, 2);
        assert_eq!(scheme.level_capacity_bytes(0), 2);
        assert_eq!(scheme.level_capacity_bytes(1), 4);

        // n = 1: one buffer in level 0, under its capacity of 2.
        scheme.note_flush();
        assert_eq!(level_of(scheme.next_compaction(&tree(&[1]))), None);

        // n = 2: level 0 reaches capacity and merges into level 1.
        scheme.note_flush();
        assert_eq!(level_of(scheme.next_compaction(&tree(&[2]))), Some(0));

        // n = 3: level 0 holds one buffer again, level 1 holds two.
        scheme.note_flush();
        assert_eq!(level_of(scheme.next_compaction(&tree(&[1, 2]))), None);

        // n = 4: level 0 is full again and merges down...
        scheme.note_flush();
        assert_eq!(level_of(scheme.next_compaction(&tree(&[2, 2]))), Some(0));
        // ...which fills level 1 to its capacity of 4, so it cascades into a
        // newly created level 2.
        assert_eq!(level_of(scheme.next_compaction(&tree(&[0, 4]))), Some(1));
        assert_eq!(level_of(scheme.next_compaction(&tree(&[0, 0, 4]))), None);
    }

    #[test]
    fn the_shallowest_over_capacity_level_is_picked_first() {
        let mut scheme = Vertical::new(1, 2);
        // Both level 0 (cap 2) and level 1 (cap 4) are over budget.
        assert_eq!(
            level_of(scheme.next_compaction(&tree(&[3, 9]))),
            Some(0),
            "data should move down one step at a time"
        );
    }

    #[test]
    fn an_empty_level_never_triggers() {
        let mut scheme = Vertical::new(1, 2);
        assert_eq!(scheme.next_compaction(&tree(&[0, 0, 0])), None);
        assert_eq!(scheme.next_compaction(&TreeShape::default()), None);
    }

    #[test]
    #[should_panic(expected = "size ratio must be at least 2")]
    fn a_ratio_of_one_is_rejected() {
        Vertical::new(MIB, 1);
    }
}
