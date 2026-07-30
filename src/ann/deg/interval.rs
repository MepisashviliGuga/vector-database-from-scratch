//! Sets of α values, as unions of closed intervals in `[0, 1]`.
//!
//! # Why a set and not a range
//!
//! Paper 05 calls the thing stored on each edge an *active range* (§4.3), and the
//! prose reads as though it were one interval. Algorithm 2 does not permit that:
//! line 8 unions the pruning ranges contributed by every already-selected
//! neighbour, and line 9 takes `[0,1]` minus that union. A union of intervals is
//! not an interval, so its complement is generally several disjoint pieces.
//!
//! Concretely: if one neighbour prunes an edge over `[0, 0.2]` and another over
//! `[0.8, 1]`, the edge is active exactly on `(0.2, 0.8)` — one interval. But if
//! the two pruning ranges are `[0.3, 0.5]` and `[0.7, 0.9]`, the active set is
//! three pieces. Nothing in the paper's construction rules that out.
//!
//! So this is an interval *set*. The discrepancy with the paper's wording is
//! labelled rather than resolved, because collapsing to a hull would change the
//! algorithm: a hull would re-admit α values the pruning rule rejected.
//!
//! # Closed intervals, and which way boundaries err
//!
//! The pruning inequalities (Eq 2, 3) are strict (`<`), but the paper writes the
//! resulting ranges closed — `[0, min(1, B/A)]` in Case 1. Boundaries are a
//! measure-zero set and an α landing exactly on one means two hybrid distances
//! are exactly equal, so this only ever decides a tie.
//!
//! Everything here is closed, matching the paper's notation. The consequence is
//! that a complement can share an endpoint with the set it came from, so an α
//! exactly on a boundary may be treated as *both* pruned and active. That
//! direction is the safe one: RNG pruning is an optimisation, not a correctness
//! requirement, so keeping an edge that could have been pruned costs a little
//! search time, whereas dropping an edge that should have been kept costs
//! recall.

/// A union of disjoint closed intervals within `[0, 1]`.
///
/// Kept normalised: sorted by lower bound, with overlapping or touching pieces
/// merged. Every constructor and combinator restores that invariant, so
/// [`AlphaSet::intervals`] is always canonical and two equal sets compare equal.
#[derive(Debug, Clone, PartialEq)]
pub struct AlphaSet {
    intervals: Vec<(f32, f32)>,
}

impl AlphaSet {
    /// The empty set: no α value at all.
    pub fn empty() -> Self {
        Self {
            intervals: Vec::new(),
        }
    }

    /// All of `[0, 1]`.
    pub fn full() -> Self {
        Self {
            intervals: vec![(0.0, 1.0)],
        }
    }

    /// A single closed interval, clamped to `[0, 1]`.
    ///
    /// Returns the empty set if the interval is reversed or falls entirely
    /// outside `[0, 1]`.
    pub fn interval(low: f32, high: f32) -> Self {
        let low = low.max(0.0);
        let high = high.min(1.0);
        if low > high {
            Self::empty()
        } else {
            Self {
                intervals: vec![(low, high)],
            }
        }
    }

    /// The disjoint pieces, sorted and merged.
    pub fn intervals(&self) -> &[(f32, f32)] {
        &self.intervals
    }

    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    /// Whether this set covers all of `[0, 1]`.
    pub fn is_full(&self) -> bool {
        matches!(self.intervals.as_slice(), [(low, high)] if *low <= 0.0 && *high >= 1.0)
    }

    /// Whether `alpha` falls in the set.
    pub fn contains(&self, alpha: f32) -> bool {
        self.intervals
            .iter()
            .any(|&(low, high)| alpha >= low && alpha <= high)
    }

    /// Total length covered.
    ///
    /// This is the `|u|` of Algorithm 2 line 10, which the `th` threshold tests
    /// to discard edges useful over too narrow a band of α to be worth storing.
    pub fn measure(&self) -> f32 {
        self.intervals.iter().map(|&(low, high)| high - low).sum()
    }

    /// Sort and merge overlapping or touching pieces.
    ///
    /// Touching pieces are merged as well as overlapping ones: with closed
    /// intervals `[0, 0.5]` and `[0.5, 1]` cover `[0, 1]` continuously, and
    /// leaving them apart would make [`is_full`] miss a set that is in fact full.
    fn normalise(mut pieces: Vec<(f32, f32)>) -> Self {
        pieces.retain(|&(low, high)| low <= high);
        pieces.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));

        let mut merged: Vec<(f32, f32)> = Vec::with_capacity(pieces.len());
        for (low, high) in pieces {
            match merged.last_mut() {
                Some(last) if low <= last.1 => last.1 = last.1.max(high),
                _ => merged.push((low, high)),
            }
        }
        Self { intervals: merged }
    }

    /// Everything in either set.
    pub fn union(&self, other: &Self) -> Self {
        let mut pieces = self.intervals.clone();
        pieces.extend_from_slice(&other.intervals);
        Self::normalise(pieces)
    }

    /// Everything in both sets.
    pub fn intersect(&self, other: &Self) -> Self {
        let mut pieces = Vec::new();
        for &(a_low, a_high) in &self.intervals {
            for &(b_low, b_high) in &other.intervals {
                let low = a_low.max(b_low);
                let high = a_high.min(b_high);
                if low <= high {
                    pieces.push((low, high));
                }
            }
        }
        Self::normalise(pieces)
    }

    /// `[0, 1]` minus this set — the active range of Algorithm 2 line 9.
    ///
    /// Endpoints are shared with the original set rather than nudged off it; see
    /// the module docs for why that direction is the safe one.
    pub fn complement(&self) -> Self {
        if self.intervals.is_empty() {
            return Self::full();
        }

        let mut pieces = Vec::with_capacity(self.intervals.len() + 1);
        let mut cursor = 0.0f32;
        for &(low, high) in &self.intervals {
            if low > cursor {
                pieces.push((cursor, low));
            }
            cursor = cursor.max(high);
        }
        if cursor < 1.0 {
            pieces.push((cursor, 1.0));
        }
        Self::normalise(pieces)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_set_contains_nothing_and_measures_zero() {
        let set = AlphaSet::empty();
        assert!(set.is_empty());
        assert_eq!(set.measure(), 0.0);
        assert!(!set.contains(0.0));
        assert!(!set.contains(0.5));
        assert!(!set.contains(1.0));
    }

    #[test]
    fn a_full_set_contains_both_endpoints() {
        let set = AlphaSet::full();
        assert!(set.is_full());
        assert_eq!(set.measure(), 1.0);
        assert!(set.contains(0.0));
        assert!(set.contains(1.0));
    }

    #[test]
    fn an_interval_is_clamped_to_the_unit_range() {
        // Case 1 and Case 4 of the pruning solver can both produce bounds
        // outside [0, 1] before clamping, so this is the guard that keeps a
        // ratio like B/A = 7.5 from widening the set past α = 1.
        let set = AlphaSet::interval(-0.5, 7.5);
        assert_eq!(set.intervals(), &[(0.0, 1.0)]);
    }

    #[test]
    fn a_reversed_interval_is_empty() {
        assert!(AlphaSet::interval(0.7, 0.3).is_empty());
    }

    #[test]
    fn overlapping_pieces_merge() {
        let set = AlphaSet::interval(0.1, 0.4).union(&AlphaSet::interval(0.3, 0.6));
        assert_eq!(set.intervals(), &[(0.1, 0.6)]);
    }

    #[test]
    fn touching_pieces_merge_because_the_intervals_are_closed() {
        let set = AlphaSet::interval(0.0, 0.5).union(&AlphaSet::interval(0.5, 1.0));
        assert_eq!(set.intervals(), &[(0.0, 1.0)]);
        assert!(set.is_full());
    }

    #[test]
    fn disjoint_pieces_stay_apart_and_sort() {
        let set = AlphaSet::interval(0.7, 0.9).union(&AlphaSet::interval(0.1, 0.2));
        assert_eq!(set.intervals(), &[(0.1, 0.2), (0.7, 0.9)]);
        assert!((set.measure() - 0.3).abs() < 1e-6);
        assert!(!set.contains(0.5));
    }

    #[test]
    fn intersection_keeps_only_the_shared_part() {
        let set = AlphaSet::interval(0.2, 0.8).intersect(&AlphaSet::interval(0.6, 1.0));
        assert_eq!(set.intervals(), &[(0.6, 0.8)]);
    }

    #[test]
    fn intersection_of_disjoint_sets_is_empty() {
        assert!(AlphaSet::interval(0.0, 0.3)
            .intersect(&AlphaSet::interval(0.7, 1.0))
            .is_empty());
    }

    #[test]
    fn intersecting_a_multi_piece_set_can_yield_several_pieces() {
        let left = AlphaSet::interval(0.0, 0.3).union(&AlphaSet::interval(0.6, 1.0));
        let got = left.intersect(&AlphaSet::interval(0.2, 0.7));
        assert_eq!(got.intervals(), &[(0.2, 0.3), (0.6, 0.7)]);
    }

    #[test]
    fn complement_of_empty_is_full_and_back() {
        assert!(AlphaSet::empty().complement().is_full());
        assert!(AlphaSet::full().complement().is_empty());
    }

    #[test]
    fn complement_of_an_interior_interval_is_two_pieces() {
        let got = AlphaSet::interval(0.3, 0.7).complement();
        assert_eq!(got.intervals(), &[(0.0, 0.3), (0.7, 1.0)]);
    }

    #[test]
    fn complement_of_two_pruning_ranges_can_be_three_pieces() {
        // This is the case the paper's wording does not cover: the active set of
        // an edge pruned over [0.3, 0.5] and [0.7, 0.9] is genuinely three
        // disjoint bands of α, not one range. See the module docs.
        let pruned = AlphaSet::interval(0.3, 0.5).union(&AlphaSet::interval(0.7, 0.9));
        let active = pruned.complement();
        assert_eq!(
            active.intervals(),
            &[(0.0, 0.3), (0.5, 0.7), (0.9, 1.0)],
            "an active set must be allowed more than one piece"
        );
        assert!(active.contains(0.6));
        assert!(!pruned.contains(0.6));
    }

    #[test]
    fn a_set_and_its_complement_partition_the_unit_range() {
        let set = AlphaSet::interval(0.1, 0.2).union(&AlphaSet::interval(0.55, 0.8));
        let total = set.measure() + set.complement().measure();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "measures should sum to 1, got {total}"
        );
    }

    #[test]
    fn nested_pruning_ranges_collapse_to_the_widest() {
        let set = AlphaSet::interval(0.2, 0.9).union(&AlphaSet::interval(0.4, 0.5));
        assert_eq!(set.intervals(), &[(0.2, 0.9)]);
    }

    #[test]
    fn unioning_full_saturates() {
        let set = AlphaSet::interval(0.4, 0.6).union(&AlphaSet::full());
        assert!(set.is_full());
        assert!(set.complement().is_empty());
    }
}
