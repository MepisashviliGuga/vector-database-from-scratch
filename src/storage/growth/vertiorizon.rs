//! Vertiorizon: a horizontal part above a two-level vertical part.
//!
//! **The central contribution of paper 01 (§5).** Reproduces the overall layout,
//! the size-ratio optimisation and the dynamic resizing. Does **not** reproduce
//! §5.2's self-tuning cost model, §5.3's skew adaptation, or the dynamic Bloom
//! filter layout — those are listed as not reproduced in the project README.
//!
//! # The problem it solves
//!
//! The two classic schemes each win on one axis and lose on the other:
//!
//! - The horizontal scheme sits on the optimal read-write frontier (Bentley and
//!   Saxe), but is *defined* on full compaction, so it pays heavily in space and
//!   write stalls — the inputs of a merge cannot be freed until the merge of an
//!   entire level completes.
//! - The vertical scheme permits partial compaction, which is cheap in space and
//!   stalls, but its fixed compaction frequency is provably off the frontier.
//!
//! Vertiorizon takes the good half of each:
//!
//! ```text
//!   ┌──────────────────┐
//!   │  L0              │  horizontal part: ℓ levels, capacity n·B in total.
//!   │  L1              │  Nearly all compactions happen here, on the optimal
//!   │  ...             │  schedule. Full compaction, but the part is small,
//!   │  L(ℓ-1)          │  so a full merge of it is cheap.
//!   ├──────────────────┤
//!   │  L(ℓ)            │  vertical part: exactly two levels, holding the vast
//!   │  L(ℓ+1)          │  majority of the data. Partial compaction, so the
//!   └──────────────────┘  expensive levels never stall or double in size.
//! ```
//!
//! When the horizontal part fills, one full compaction drains **all** of it into
//! the first vertical level. When that overflows, partial compaction moves slices
//! into the largest level. Because the two biggest levels compact partially,
//! space amplification and write stalls fall by roughly a factor of `T` against
//! full compaction throughout.
//!
//! Two vertical levels is the paper's empirical choice: it captures essentially
//! all the benefit, and more would shrink the horizontal part and give back the
//! scheduling advantage.
//!
//! # The size-ratio optimisation
//!
//! The obvious choice is ratio `T` at both vertical steps. The paper does better.
//! Let `T′` be the ratio from the horizontal part into the first vertical level,
//! and `T²/T′` from there into the second — the product is `T²` either way, so
//! total capacity is unchanged.
//!
//! The first vertical level is kept close to capacity, so a compaction into it
//! costs a write amplification of `T′`. The second is filled by *partial*
//! compaction, which leaves the level unevenly dense, and its average write
//! amplification is `(T²/T′ + 1) / 2`. Minimising the sum:
//!
//! ```text
//!   T′ + (T²/T′ + 1)/2  ≥  2·√(T′ · T²/(2T′)) + 1/2  =  √2·T + 1/2
//! ```
//!
//! by AM-GM, with equality exactly at **`T′ = T/√2`**. Using `T` at both steps
//! would cost `T + (T+1)/2`, which is strictly worse.
//!
//! # Ambiguity, flagged rather than guessed
//!
//! §5.1 says `n` is incremented "by a factor of `1/T`" once the largest level
//! fills. That admits two readings: `n ← n·(1 + 1/T)` or `n ← n/T`. The second
//! would *shrink* the horizontal part as data grows, contradicting the stated
//! purpose of letting overall capacity expand, so this implements the first. It
//! is labelled here rather than silently resolved.

use super::{
    saturating_geometric, CompactionRequest, Granularity, GrowthScheme, HorizontalLeveling,
    HorizontalTiering, TreeShape,
};

/// Which merge policy the horizontal part schedules for.
///
/// Paper 01 §5.1 lists this as one of two configurable knobs on the horizontal
/// part (the other being its level count). Under leveling the optimal schedule
/// slows down over time; under tiering it speeds up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HorizontalPolicy {
    Leveling,
    Tiering,
}

/// Levels in the vertical part. Fixed at two by the paper.
const VERTICAL_LEVELS: usize = 2;

/// Horizontal part over a two-level vertical part.
#[derive(Debug)]
pub struct Vertiorizon {
    /// Schedules compactions *within* the horizontal part.
    horizontal: Box<dyn GrowthScheme>,
    /// Number of levels in the horizontal part.
    horizontal_levels: usize,
    /// Memtable flush size, `B`.
    buffer_bytes: u64,
    /// Horizontal part capacity as a multiple of the buffer: capacity is `n·B`.
    n: u64,
    /// Overall size ratio `T`.
    size_ratio: u64,
    /// Ratio into the first vertical level, `T′ = T/√2`.
    first_vertical_ratio: f64,
    /// Set when the largest level has filled. The paper grows `n` only *after*
    /// the next full compaction clears the horizontal part, so the resize waits
    /// for that moment rather than happening mid-flight.
    resize_pending: bool,
}

impl Vertiorizon {
    /// # Panics
    ///
    /// If `horizontal_levels < 2`, `size_ratio < 2`, or `buffer_bytes` is 0.
    pub fn new(
        horizontal_levels: usize,
        size_ratio: u64,
        buffer_bytes: u64,
        initial_n: u64,
        policy: HorizontalPolicy,
    ) -> Self {
        assert!(
            horizontal_levels >= 2,
            "the horizontal part needs at least 2 levels"
        );
        assert!(
            size_ratio >= 2,
            "size ratio must be at least 2, got {size_ratio}"
        );
        assert!(buffer_bytes > 0, "buffer size must be positive");

        let n = initial_n.max(1);
        let horizontal: Box<dyn GrowthScheme> = match policy {
            HorizontalPolicy::Leveling => Box::new(HorizontalLeveling::new(horizontal_levels)),
            HorizontalPolicy::Tiering => Box::new(HorizontalTiering::new(
                horizontal_levels,
                buffer_bytes,
                n.saturating_mul(buffer_bytes),
            )),
        };

        Self {
            horizontal,
            horizontal_levels,
            buffer_bytes,
            n,
            size_ratio,
            // T' = T/√2, the AM-GM minimiser. See the module docs.
            first_vertical_ratio: size_ratio as f64 / std::f64::consts::SQRT_2,
            resize_pending: false,
        }
    }

    /// Index of the first vertical level.
    pub fn first_vertical_level(&self) -> usize {
        self.horizontal_levels
    }

    /// Index of the largest level.
    pub fn largest_level(&self) -> usize {
        self.horizontal_levels + VERTICAL_LEVELS - 1
    }

    /// Capacity of the whole horizontal part: `n·B`.
    pub fn horizontal_capacity_bytes(&self) -> u64 {
        self.n.saturating_mul(self.buffer_bytes)
    }

    /// Capacity of the first vertical level: `n·B·T′`.
    pub fn first_vertical_capacity_bytes(&self) -> u64 {
        let capacity = self.horizontal_capacity_bytes() as f64 * self.first_vertical_ratio;
        capacity.min(u64::MAX as f64) as u64
    }

    /// Capacity of the largest level: `n·B·T²`.
    ///
    /// The second ratio is `T²/T′`, so the product with `T′` is `T²` regardless
    /// of how `T′` was chosen — the optimisation redistributes the ratios
    /// without changing the tree's overall scale.
    pub fn largest_capacity_bytes(&self) -> u64 {
        saturating_geometric(self.horizontal_capacity_bytes(), self.size_ratio, 2)
    }

    /// The current horizontal-part multiplier `n`.
    pub fn n(&self) -> u64 {
        self.n
    }

    /// Bytes currently held by the horizontal part.
    fn horizontal_bytes(&self, tree: &TreeShape) -> u64 {
        (0..self.horizontal_levels)
            .map(|level| tree.level_bytes(level))
            .sum()
    }

    /// Grow the horizontal part, and with it the whole tree's capacity.
    ///
    /// See the module docs on the `1/T` ambiguity.
    fn apply_resize(&mut self) {
        let growth = self.n / self.size_ratio;
        self.n = self.n.saturating_add(growth.max(1));
        self.resize_pending = false;
    }
}

impl GrowthScheme for Vertiorizon {
    fn name(&self) -> &'static str {
        "vertiorizon"
    }

    fn note_flush(&mut self) {
        self.horizontal.note_flush();
    }

    fn next_compaction(&mut self, tree: &TreeShape) -> Option<CompactionRequest> {
        let first_vertical = self.first_vertical_level();
        let largest = self.largest_level();

        // 1. Drain first. Once the horizontal part is full, shuffling data
        //    between its levels achieves nothing — it all has to go down.
        if self.horizontal_bytes(tree) >= self.horizontal_capacity_bytes() {
            // The paper grows `n` after the full compaction clears the part, so
            // apply a pending resize here, as the drain is being scheduled.
            if self.resize_pending {
                self.apply_resize();
            }
            return Some(CompactionRequest {
                first_level: 0,
                last_level: self.horizontal_levels - 1,
                target_level: first_vertical,
                // The horizontal scheme is defined on full compaction, and this
                // is the merge that empties the part outright.
                granularity: Granularity::Full,
            });
        }

        // 2. Otherwise let the horizontal scheme run its own optimal schedule
        //    among its levels. It never returns its own deepest level, so it
        //    cannot spill into the vertical part on its own.
        if let Some(request) = self.horizontal.next_compaction(tree) {
            if request.target_level < first_vertical {
                return Some(request);
            }
        }

        // 3. First vertical level over capacity: move a slice down. Partial
        //    compaction here is the whole point of the vertical part.
        if tree.level_bytes(first_vertical) >= self.first_vertical_capacity_bytes()
            && tree.level_bytes(first_vertical) > 0
        {
            return Some(CompactionRequest::single(
                first_vertical,
                Granularity::Partial,
            ));
        }

        // 4. The largest level has nowhere to go, so a full one means the tree
        //    must grow. Record it; the resize lands with the next drain.
        if tree.level_bytes(largest) >= self.largest_capacity_bytes() {
            self.resize_pending = true;
        }

        None
    }

    fn max_levels(&self) -> Option<usize> {
        Some(self.horizontal_levels + VERTICAL_LEVELS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::shape::LevelShape;

    const KIB: u64 = 1024;

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

    fn scheme() -> Vertiorizon {
        // ℓ = 2 horizontal levels, T = 8, B = 1 KiB, n = 4.
        Vertiorizon::new(2, 8, KIB, 4, HorizontalPolicy::Leveling)
    }

    #[test]
    fn the_layout_is_a_horizontal_part_over_two_vertical_levels() {
        let scheme = scheme();
        assert_eq!(scheme.max_levels(), Some(4), "2 horizontal + 2 vertical");
        assert_eq!(scheme.first_vertical_level(), 2);
        assert_eq!(scheme.largest_level(), 3);
    }

    /// `T′ = T/√2` is the AM-GM minimiser of `T′ + (T²/T′ + 1)/2`.
    #[test]
    fn the_first_vertical_ratio_is_t_over_root_two() {
        let scheme = scheme();
        let expected = 8.0 / std::f64::consts::SQRT_2;
        assert!((scheme.first_vertical_ratio - expected).abs() < 1e-9);

        // And it beats the naive choice of T at both steps.
        let t = 8.0;
        let optimised = |ratio: f64| ratio + (t * t / ratio + 1.0) / 2.0;
        assert!(
            optimised(expected) < optimised(t),
            "T/√2 should cost less than T: {:.3} vs {:.3}",
            optimised(expected),
            optimised(t)
        );
        // The paper's closed form for the minimum.
        assert!((optimised(expected) - (std::f64::consts::SQRT_2 * t + 0.5)).abs() < 1e-9);
    }

    /// Redistributing the ratios must not change the tree's overall scale: the
    /// two vertical ratios still multiply to `T²`.
    #[test]
    fn the_ratios_still_multiply_to_t_squared() {
        let scheme = scheme();
        let horizontal = scheme.horizontal_capacity_bytes() as f64;
        let largest = scheme.largest_capacity_bytes() as f64;
        assert!((largest / horizontal - 64.0).abs() < 0.01, "T² = 64");

        let first = scheme.first_vertical_capacity_bytes() as f64;
        let second_ratio = largest / first;
        assert!(
            (scheme.first_vertical_ratio * second_ratio - 64.0).abs() < 0.1,
            "T′ · (T²/T′) must be T²"
        );
    }

    /// The defining behaviour: a full horizontal part drains **entirely** into
    /// the first vertical level in one merge, rather than level by level.
    #[test]
    fn a_full_horizontal_part_drains_in_one_merge() {
        let mut scheme = scheme();
        // Capacity is n·B = 4 KiB; put 5 KiB across the two horizontal levels.
        let request = scheme
            .next_compaction(&tree(&[2 * KIB, 3 * KIB, 0, 0]))
            .expect("a drain");

        assert_eq!(request.first_level, 0);
        assert_eq!(request.last_level, 1, "every horizontal level goes at once");
        assert_eq!(request.target_level, 2);
        assert!(request.spans_multiple_levels());
        assert_eq!(request.granularity, Granularity::Full);
    }

    /// Below capacity, the horizontal part runs its own optimal schedule and
    /// never spills into the vertical part on its own.
    #[test]
    fn an_unfilled_horizontal_part_uses_its_own_schedule() {
        let mut scheme = scheme();
        let shape = tree(&[KIB, KIB, 0, 0]); // 2 KiB, under the 4 KiB capacity

        let mut saw_internal = false;
        for _ in 0..20 {
            scheme.note_flush();
            while let Some(request) = scheme.next_compaction(&shape) {
                assert!(
                    request.target_level < scheme.first_vertical_level(),
                    "an unfilled horizontal part must not reach the vertical part"
                );
                saw_internal = true;
            }
        }
        assert!(saw_internal, "the horizontal schedule never fired");
    }

    /// The vertical part's reason for existing: partial compaction between the
    /// two largest levels.
    #[test]
    fn the_vertical_part_compacts_partially() {
        let mut scheme = scheme();
        // Horizontal part nearly empty; first vertical level over its capacity.
        let over = scheme.first_vertical_capacity_bytes() + KIB;
        let request = scheme
            .next_compaction(&tree(&[0, 0, over, 0]))
            .expect("a request");

        assert_eq!(request.first_level, 2);
        assert_eq!(request.target_level, 3);
        assert!(!request.spans_multiple_levels());
        assert_eq!(
            request.granularity,
            Granularity::Partial,
            "partial compaction between the largest levels is what cuts space \
             amplification and write stalls"
        );
    }

    /// Draining takes priority: once the horizontal part is full, rearranging it
    /// internally achieves nothing.
    #[test]
    fn draining_takes_priority_over_internal_compaction() {
        let mut scheme = scheme();
        scheme.note_flush();
        let request = scheme
            .next_compaction(&tree(&[4 * KIB, 4 * KIB, 0, 0]))
            .expect("a request");
        assert!(request.spans_multiple_levels(), "expected a drain");
    }

    /// The largest level cannot compact onward, so filling it means the tree
    /// must grow — but only once the next drain clears the horizontal part.
    #[test]
    fn a_full_largest_level_grows_the_tree_at_the_next_drain() {
        let mut scheme = scheme();
        let starting_n = scheme.n();

        // Largest level full, horizontal part not. Nothing to do yet.
        let full_largest = scheme.largest_capacity_bytes() + KIB;
        assert_eq!(
            scheme.next_compaction(&tree(&[0, 0, 0, full_largest])),
            None
        );
        assert_eq!(scheme.n(), starting_n, "the resize must wait for a drain");

        // Now fill the horizontal part: the drain fires and the resize lands.
        let request = scheme
            .next_compaction(&tree(&[4 * KIB, KIB, 0, full_largest]))
            .expect("a drain");
        assert!(request.spans_multiple_levels());
        assert!(
            scheme.n() > starting_n,
            "n should have grown from {starting_n} to something larger"
        );
    }

    /// Growth must be gradual — `n·(1 + 1/T)` — not a jump by a factor of `T`.
    #[test]
    fn resizing_grows_n_gradually() {
        let mut scheme = Vertiorizon::new(2, 10, KIB, 100, HorizontalPolicy::Leveling);
        scheme.resize_pending = true;
        scheme.apply_resize();
        assert_eq!(scheme.n(), 110, "n·(1 + 1/T) with n = 100, T = 10");
    }

    /// A small `n` must still grow: integer division would otherwise round the
    /// increment to zero and stall the tree forever.
    #[test]
    fn a_small_n_still_grows() {
        let mut scheme = Vertiorizon::new(2, 10, KIB, 1, HorizontalPolicy::Leveling);
        scheme.apply_resize();
        assert!(scheme.n() > 1, "n must not be stuck at 1");
    }

    #[test]
    fn the_horizontal_part_can_schedule_for_tiering() {
        let mut scheme = Vertiorizon::new(3, 4, KIB, 8, HorizontalPolicy::Tiering);
        assert_eq!(scheme.max_levels(), Some(5));

        let shape = tree(&[KIB, KIB, KIB, 0, 0]);
        let mut fired = 0;
        for _ in 0..40 {
            scheme.note_flush();
            while let Some(request) = scheme.next_compaction(&shape) {
                assert!(request.target_level <= scheme.first_vertical_level());
                fired += 1;
            }
        }
        assert!(fired > 0, "the tiering schedule never fired");
    }

    #[test]
    fn an_empty_tree_needs_no_compaction() {
        let mut scheme = scheme();
        assert_eq!(scheme.next_compaction(&TreeShape::default()), None);
        assert_eq!(scheme.next_compaction(&tree(&[0, 0, 0, 0])), None);
    }

    #[test]
    #[should_panic(expected = "at least 2 levels")]
    fn a_one_level_horizontal_part_is_rejected() {
        Vertiorizon::new(1, 8, KIB, 4, HorizontalPolicy::Leveling);
    }
}
