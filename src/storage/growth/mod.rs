//! Axis 1: **when** a level compacts into the next.
//!
//! # Why this is a stateful trigger, not a capacity function
//!
//! An earlier version of this module modelled a growth scheme as a pure function
//! from level index to capacity. That works for the vertical scheme, whose
//! trigger really is "level `i` exceeded `B·T^(i+1)`" — and it is wrong for the
//! horizontal scheme, which paper 01 defines with no capacities at all:
//!
//! ```text
//! Algorithm 1 (horizontal, leveling):
//!   C_i ← 0
//!   on each buffer flush:
//!     C_1 ← C_1 + 1
//!     for i in 1..ℓ-1:
//!       if C_i > C_(i+1):  compact i → i+1;  C_(i+1) ← C_(i+1) + 1;  C_i ← 0
//! ```
//!
//! The counters produce a *decreasing compaction frequency*: level `i` merges
//! down often while the level below is small and rarely once it is large. That
//! schedule is the whole point — a compaction costs what the target level
//! currently holds, so the cheap merges should happen while they are cheap. A
//! capacity formula reproduces roughly the right level *sizes* with the wrong
//! *timing*, which is precisely the property the paper is about.
//!
//! # The Bentley-Saxe result
//!
//! Bentley and Saxe (1980) showed this scheme minimises write cost for a fixed
//! number of levels. Under leveling, read cost is proportional to the number of
//! levels, which the horizontal scheme holds fixed — so it sits on the optimal
//! read-write frontier and the vertical scheme does not. Paper 01 generalises
//! this to the tiering merge policy in [`HorizontalTiering`], which had never
//! been done.
//!
//! # The interface
//!
//! [`GrowthScheme::note_flush`] advances the scheme by one memtable flush.
//! [`GrowthScheme::next_compaction`] then yields the levels to compact, one at a
//! time, until the tree is settled. Implementations mutate their own state when
//! they return a level, so a caller that ignores the answer will desynchronise
//! the schedule — the returned compaction must actually be performed.

use std::fmt::Debug;

use super::shape::TreeShape;

pub mod ecotune_scheme;
pub mod horizontal;
pub mod horizontal_tiering;
pub mod vertical;
pub mod vertiorizon;

pub use ecotune_scheme::EcoTune;
pub use horizontal::HorizontalLeveling;
pub use horizontal_tiering::HorizontalTiering;
pub use vertical::Vertical;
pub use vertiorizon::{HorizontalPolicy, Vertiorizon};

/// How much of a level one compaction moves.
///
/// This belongs to the growth scheme rather than the merge policy, because paper
/// 01 makes it a property of the scheme: the horizontal scheme is *defined* on
/// full compaction, the vertical scheme permits partial, and Vertiorizon mixes
/// the two — full in its horizontal part, partial between its bottom two levels.
///
/// The choice is not cosmetic. Full compaction is the mechanism behind the
/// horizontal scheme's space cost: the inputs cannot be freed until a merge of
/// the *entire* level completes, so a large level transiently needs room for
/// both inputs and outputs, and the levels above it cannot drain meanwhile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Granularity {
    /// Merge the whole level at once.
    Full,
    /// Merge a bounded slice: a few files, plus the files they overlap below.
    Partial,
}

/// One compaction the growth scheme has scheduled.
///
/// Sources are a *span* of levels rather than a single one, because Vertiorizon
/// drains its entire horizontal part into the vertical part in one merge. Every
/// other scheme uses [`CompactionRequest::single`], where the span is one level
/// and the target is the next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionRequest {
    /// Shallowest source level, which holds the newest data.
    pub first_level: usize,
    /// Deepest source level, inclusive.
    pub last_level: usize,
    pub target_level: usize,
    pub granularity: Granularity,
    /// Merge only the newest runs whose combined unit count reaches this,
    /// instead of the whole level.
    ///
    /// Only EcoTune sets it. Its schedule specifies widths in *unit runs*, and a
    /// width-4 merge may absorb one previously-merged width-2 run plus two new
    /// ones — so the count is in units, not physical runs. `None` means "take
    /// everything", which is what every other scheme wants.
    pub merge_units: Option<usize>,
}

impl CompactionRequest {
    /// The ordinary case: merge one level into the next.
    pub fn single(level: usize, granularity: Granularity) -> Self {
        Self {
            first_level: level,
            last_level: level,
            target_level: level + 1,
            granularity,
            merge_units: None,
        }
    }

    /// Source levels, shallowest first — the order the merge needs, since
    /// shallower levels hold newer data.
    pub fn source_levels(&self) -> impl Iterator<Item = usize> {
        self.first_level..=self.last_level
    }

    /// Whether this merges more than one level at once.
    pub fn spans_multiple_levels(&self) -> bool {
        self.last_level > self.first_level
    }
}

/// Decides when a level compacts into the next, and at what granularity.
pub trait GrowthScheme: Debug + Send {
    /// Short name for benchmark output and plot legends.
    fn name(&self) -> &'static str;

    /// Advance by one memtable flush landing in level 0.
    fn note_flush(&mut self);

    /// The next compaction to run, or `None` once settled.
    ///
    /// Called repeatedly after a flush until it returns `None`. Implementations
    /// update their internal schedule when they return `Some`, so **the caller
    /// must perform the compaction it is handed**; skipping one leaves the
    /// scheme believing work happened that did not.
    fn next_compaction(&mut self, tree: &TreeShape) -> Option<CompactionRequest>;

    /// Level count, if the scheme fixes one. `None` for schemes that add levels
    /// as data grows.
    fn max_levels(&self) -> Option<usize>;
}

/// The binomial coefficient `C(n, k)`, saturating at [`u64::MAX`].
///
/// Used to size the horizontal-tiering scheme's initial counters, where paper 01
/// picks the smallest `k` satisfying `N/B ≤ C(k+ℓ-1, ℓ)`. Computed
/// multiplicatively in `u128` and dividing as it goes, so intermediate values
/// stay small — the naive factorial form overflows for inputs the real
/// expression handles comfortably.
pub(crate) fn binomial(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    // C(n, k) == C(n, n-k); taking the smaller one keeps the loop short.
    let k = k.min(n - k);
    let mut result: u128 = 1;
    for step in 0..k {
        result = result.saturating_mul((n - step) as u128) / (step as u128 + 1);
        if result > u64::MAX as u128 {
            return u64::MAX;
        }
    }
    result as u64
}

/// `base * ratio^exponent`, saturating instead of overflowing.
///
/// Deep levels in a vertical tree reach enormous nominal capacities. Saturating
/// is correct: a capacity larger than any possible data set means "this level is
/// never full", which is what it should mean.
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
    fn binomial_matches_known_values() {
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 5), 1);
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(10, 3), 120);
        assert_eq!(binomial(52, 5), 2_598_960);
        assert_eq!(binomial(3, 7), 0, "k > n has no combinations");
    }

    /// The naive factorial form overflows here; the multiplicative one must not.
    #[test]
    fn binomial_handles_large_inputs_without_overflowing() {
        assert_eq!(binomial(60, 30), 118_264_581_564_861_424);
        assert_eq!(
            binomial(100, 50),
            u64::MAX,
            "saturates rather than wrapping"
        );
    }

    #[test]
    fn geometric_growth_saturates_rather_than_wrapping() {
        assert_eq!(saturating_geometric(4, 10, 0), 4);
        assert_eq!(saturating_geometric(4, 10, 3), 4000);
        assert_eq!(saturating_geometric(1024, 10, 64), u64::MAX);
    }

    /// All three schemes must satisfy the same interface, so the benchmark
    /// harness can hold the merge policy fixed and swap the growth scheme.
    #[test]
    fn every_scheme_satisfies_the_trait() {
        let schemes: Vec<Box<dyn GrowthScheme>> = vec![
            Box::new(Vertical::new(1024, 10)),
            Box::new(HorizontalLeveling::new(4)),
            Box::new(HorizontalTiering::new(4, 1024, 1024 * 1024)),
        ];
        for mut scheme in schemes {
            assert!(!scheme.name().is_empty());
            scheme.note_flush();
            // An empty tree offers nothing to compact under any scheme.
            assert_eq!(scheme.next_compaction(&TreeShape::default()), None);
        }
    }
}
