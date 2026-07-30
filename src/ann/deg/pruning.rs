//! Solving the RNG pruning condition for *which* α values it holds at.
//!
//! Paper 05 §4.3. This is the heart of DEG, and it rests on one observation:
//! the hybrid distance
//!
//! ```text
//! Dist(q,o) = α·δe(q,o) + (1−α)·δs(q,o) = δs + α·(δe − δs)
//! ```
//!
//! is **affine in α**. For a fixed pair of objects it is a straight line. So
//! "is edge (x,y) longer than edge (x,z)?" compares two lines, and the answer
//! flips at a single crossing point — which turns an infinite family of graphs,
//! one per α, into a bound on each edge that can be stored and tested.
//!
//! # The reduction
//!
//! A Relative Neighbourhood Graph prunes edge (x,y) when some third node z makes
//! it the longest side of the triangle (x,y,z). Under the hybrid distance that is
//! the paper's Eq 2 and Eq 3, both of which must hold:
//!
//! ```text
//! α·δe(x,z) + (1−α)·δs(x,z)  <  α·δe(x,y) + (1−α)·δs(x,y)      (2)
//! α·δe(y,z) + (1−α)·δs(y,z)  <  α·δe(x,y) + (1−α)·δs(x,y)      (3)
//! ```
//!
//! Collecting the α terms on the left (Eq 4 and Eq 5) puts each into the form
//!
//! ```text
//! α·A < B
//! ```
//!
//! with, for Eq 4,
//!
//! ```text
//! A = δe(x,z) − δe(x,y) + δs(x,y) − δs(x,z)
//! B = δs(x,y) − δs(x,z)
//! ```
//!
//! and Eq 5 the same with `y,z` in place of `x,z`. Since `A` and `B` are
//! constants once the three objects are fixed, the solution set is decided
//! entirely by their signs — [`solve`] is that case analysis.

use super::interval::AlphaSet;

/// The distance between one pair of objects, in both modalities.
///
/// Both components are already normalised by the dataset maxima `emax` and
/// `smax` of §3.1, so each lies in `[0, 1]` and the two are commensurable. The
/// algebra below does not depend on that, but the `th` threshold on active-range
/// width in Algorithm 2 does — it compares against a fraction of the α range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridDistance {
    /// Normalised distance in the first modality, `δe`.
    pub e: f32,
    /// Normalised distance in the second modality, `δs`.
    pub s: f32,
}

impl HybridDistance {
    pub fn new(e: f32, s: f32) -> Self {
        Self { e, s }
    }

    /// The hybrid distance at one α — Eq 1.
    ///
    /// Written as `s + α(e − s)` rather than `α·e + (1−α)·s`: the two agree
    /// algebraically, and this form is the line whose slope the pruning analysis
    /// reasons about.
    pub fn at(&self, alpha: f32) -> f32 {
        self.s + alpha * (self.e - self.s)
    }
}

/// The α values in `[0, 1]` satisfying `α·A < B`.
///
/// This is §4.3's four-case analysis, plus the case it omits.
///
/// | `B` | `A` | solution | paper |
/// |---|---|---|---|
/// | `> 0` | `> 0` | `[0, min(1, B/A)]` | Case 1 |
/// | `< 0` | `≥ 0` | empty | Case 2 |
/// | `> 0` | `≤ 0` | all of `[0,1]` | Case 3 |
/// | `< 0` | `< 0` | `[B/A, 1]`, or empty if `B/A ≥ 1` | Case 4, **corrected** |
/// | `= 0` | any | see below | **not covered** |
///
/// Case 4 departs from the paper deliberately; the inline comment explains why
/// its `min(1, B/A)` prunes an edge Table 1 says is never pruned.
///
/// # The gap at `B = 0`
///
/// Every case above requires `B` strictly positive or strictly negative, so
/// `B = 0` falls through all four. It is reachable: `B = δs(x,y) − δs(x,z)` is
/// zero whenever two second-modality distances coincide, which is common when
/// that modality is low-dimensional — and two of the paper's own five datasets
/// use `m = 2`. Worse, `min(1, B/A)` divides by zero when `A` is also zero.
///
/// Solving directly: `α·A < 0` holds for every `α > 0` when `A < 0`, and for no
/// `α ≥ 0` when `A ≥ 0`. So `A < 0` gives `(0, 1]` and `A ≥ 0` gives empty.
/// Returned as the closed `[0, 1]` in the first case, per the module-level note
/// in [`super::interval`]: at `α = 0` exactly, the two distances being compared
/// are equal, so this only decides a tie.
pub fn solve(a: f32, b: f32) -> AlphaSet {
    if b > 0.0 {
        if a > 0.0 {
            // Case 1: α < B/A, and B/A > 0.
            AlphaSet::interval(0.0, (b / a).min(1.0))
        } else {
            // Case 3: α·A ≤ 0 < B for every α ≥ 0.
            AlphaSet::full()
        }
    } else if b < 0.0 {
        if a < 0.0 {
            // Case 4: dividing by a negative flips the inequality to α > B/A,
            // with B/A > 0.
            //
            // The paper writes this range as `[min(1, B/A), 1]`, which is wrong
            // when B/A ≥ 1: there is then no α ≤ 1 with α > B/A, so the answer
            // is empty, but the clamp reports the single point [1,1] instead.
            // That is not the harmless measure-zero slip it looks like, because
            // a non-empty pruning range prunes an edge that should have been
            // kept. It also fires on ordinary input rather than only in
            // contrived cases: Table 1's own first example has A = 0 exactly in
            // real arithmetic but −4·10⁻⁸ in f32, which routes it here with
            // B/A = 1.25·10⁷ — the paper's form prunes an edge it states is
            // never pruned.
            let crossing = b / a;
            if crossing >= 1.0 {
                AlphaSet::empty()
            } else {
                AlphaSet::interval(crossing, 1.0)
            }
        } else {
            // Case 2: α·A ≥ 0 > B is unsatisfiable.
            AlphaSet::empty()
        }
    } else if a < 0.0 {
        // Uncovered by the paper: α·A < 0 for all α > 0.
        AlphaSet::full()
    } else {
        // Uncovered by the paper: α·A ≥ 0, never < 0.
        AlphaSet::empty()
    }
}

/// The α values at which node `z` prunes edge (x,y), per the RNG rule.
///
/// Both Eq 2 and Eq 3 must hold for `z` to prune, so this is the intersection of
/// their solution sets — Lemma 4.2.
///
/// Arguments are the three pairwise distances of the triangle.
pub fn pruned_by(
    xy: HybridDistance,
    xz: HybridDistance,
    yz: HybridDistance,
) -> AlphaSet {
    // Eq 4: the rearrangement of Eq 2.
    let first = solve(xz.e - xy.e + xy.s - xz.s, xy.s - xz.s);
    // Eq 5: the same with (y,z) in place of (x,z).
    let second = solve(yz.e - xy.e + xy.s - yz.s, xy.s - yz.s);
    first.intersect(&second)
}

/// Whether `z` prunes edge (x,y) at one specific α, evaluated directly.
///
/// This is the definition — Eq 2 and Eq 3 as written, with no algebra. It exists
/// so the closed-form [`pruned_by`] can be checked against it pointwise, which
/// is what catches a sign error in the case analysis that golden values alone
/// would not.
pub fn prunes_at(
    alpha: f32,
    xy: HybridDistance,
    xz: HybridDistance,
    yz: HybridDistance,
) -> bool {
    let longest = xy.at(alpha);
    xz.at(alpha) < longest && yz.at(alpha) < longest
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three examples of the paper's Table 1, as
    /// `(δs(x,y), δe(x,y), δs(x,z), δe(x,z), δs(y,z), δe(y,z))`.
    ///
    /// Note the column order in the paper is δs before δe, while
    /// [`HybridDistance::new`] takes `e` first.
    fn table_1(row: usize) -> (HybridDistance, HybridDistance, HybridDistance) {
        let (sxy, exy, sxz, exz, syz, eyz) = match row {
            1 => (0.3, 0.4, 0.8, 0.9, 0.1, 0.7),
            2 => (0.5, 0.7, 0.2, 0.4, 0.3, 0.5),
            3 => (0.2, 0.6, 0.4, 0.5, 0.3, 0.4),
            _ => panic!("Table 1 has three rows"),
        };
        (
            HybridDistance::new(exy, sxy),
            HybridDistance::new(exz, sxz),
            HybridDistance::new(eyz, syz),
        )
    }

    #[test]
    fn table_1_example_1_is_never_pruned() {
        // §4.3 Example 1: "Equation 2 does not hold for any α value. This means
        // that the edge (x,y) will not be pruned due to the presence of node z."
        // Example 2 adds that this row "falls under case 2, resulting in r₁ = ∅".
        let (xy, xz, yz) = table_1(1);
        assert_eq!(solve(xz.e - xy.e + xy.s - xz.s, xy.s - xz.s), AlphaSet::empty());
        assert!(pruned_by(xy, xz, yz).is_empty());
    }

    #[test]
    fn table_1_example_2_is_always_pruned() {
        // Example 1: "both equations are satisfied for any α ∈ [0,1], indicating
        // that the edge (x₂,y₂) is consistently pruned by node z₂, regardless of
        // the α value." Example 2: "case 3, leading to r₁ = [0,1]".
        let (xy, xz, yz) = table_1(2);
        assert!(solve(xz.e - xy.e + xy.s - xz.s, xy.s - xz.s).is_full());
        assert!(pruned_by(xy, xz, yz).is_full());
    }

    #[test]
    fn table_1_example_3_is_pruned_on_two_thirds_to_one() {
        // Example 1: "the first equation holds for α ∈ [2/3, 1] and the second
        // equation holds for α ∈ [1/3, 1]. This means that the edge (x₃,y₃) will
        // be pruned due to the presence of node z when α ∈ [2/3, 1]."
        let (xy, xz, yz) = table_1(3);

        let first = solve(xz.e - xy.e + xy.s - xz.s, xy.s - xz.s);
        let second = solve(yz.e - xy.e + xy.s - yz.s, xy.s - yz.s);
        assert_eq!(first.intervals().len(), 1);
        assert_eq!(second.intervals().len(), 1);
        assert!((first.intervals()[0].0 - 2.0 / 3.0).abs() < 1e-6, "{first:?}");
        assert_eq!(first.intervals()[0].1, 1.0);
        assert!((second.intervals()[0].0 - 1.0 / 3.0).abs() < 1e-6, "{second:?}");
        assert_eq!(second.intervals()[0].1, 1.0);

        let pruned = pruned_by(xy, xz, yz);
        assert_eq!(pruned.intervals().len(), 1);
        assert!((pruned.intervals()[0].0 - 2.0 / 3.0).abs() < 1e-6, "{pruned:?}");
        assert_eq!(pruned.intervals()[0].1, 1.0);
    }

    #[test]
    fn table_1_example_3_is_the_case_the_paper_calls_hard() {
        // The point of Example 1: an edge can be pruned for some α and kept for
        // others. A fixed-α index has to pick one answer and be wrong elsewhere,
        // which is the entire motivation for storing a range.
        let (xy, xz, yz) = table_1(3);
        let pruned = pruned_by(xy, xz, yz);
        assert!(!pruned.is_empty() && !pruned.is_full());
        assert!(!pruned.contains(0.0), "kept when the second modality dominates");
        assert!(pruned.contains(1.0), "pruned when the first modality dominates");
    }

    /// Every row of Table 1, checked pointwise against the definition.
    #[test]
    fn the_closed_form_agrees_with_evaluating_the_inequalities_directly() {
        for row in 1..=3 {
            let (xy, xz, yz) = table_1(row);
            let pruned = pruned_by(xy, xz, yz);

            for step in 0..=1000 {
                let alpha = step as f32 / 1000.0;
                let direct = prunes_at(alpha, xy, xz, yz);
                let closed = pruned.contains(alpha);
                // Boundaries are ties: the closed form includes its endpoints
                // while the strict inequality excludes them, so allow a mismatch
                // only within a hair of a boundary.
                let near_boundary = pruned
                    .intervals()
                    .iter()
                    .any(|&(low, high)| (alpha - low).abs() < 2e-3 || (alpha - high).abs() < 2e-3);
                assert!(
                    direct == closed || near_boundary,
                    "row {row} at α={alpha}: direct={direct} closed={closed}"
                );
            }
        }
    }

    #[test]
    fn the_closed_form_agrees_with_the_definition_on_random_triangles() {
        // Table 1 exercises three of the five cases. This sweeps arbitrary
        // triangles so Case 1, Case 4 and the B = 0 gap all get hit, and checks
        // the closed form against Eq 2/3 evaluated directly at each α.
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Quantised to few values so exact ties, and therefore B = 0, occur
            // often rather than never.
            ((state >> 33) % 11) as f32 / 10.0
        };

        for trial in 0..2000 {
            let xy = HybridDistance::new(next(), next());
            let xz = HybridDistance::new(next(), next());
            let yz = HybridDistance::new(next(), next());
            let pruned = pruned_by(xy, xz, yz);

            for step in 0..=100 {
                let alpha = step as f32 / 100.0;
                let direct = prunes_at(alpha, xy, xz, yz);
                let closed = pruned.contains(alpha);
                if direct == closed {
                    continue;
                }
                // A disagreement is only tolerable where the two lines meet,
                // i.e. where one of the inequalities is an exact tie.
                let tie = (xz.at(alpha) - xy.at(alpha)).abs() < 1e-6
                    || (yz.at(alpha) - xy.at(alpha)).abs() < 1e-6;
                assert!(
                    tie,
                    "trial {trial} at α={alpha}: direct={direct} closed={closed}\n\
                     xy={xy:?} xz={xz:?} yz={yz:?}\npruned={pruned:?}"
                );
            }
        }
    }

    #[test]
    fn case_1_bounds_the_range_above() {
        // B > 0, A > 0 → α < B/A. With B/A = 0.25 the edge survives above it.
        let got = solve(0.8, 0.2);
        assert_eq!(got.intervals(), &[(0.0, 0.25)]);
    }

    #[test]
    fn case_4_bounds_the_range_below() {
        let got = solve(-0.8, -0.2);
        assert_eq!(got.intervals(), &[(0.25, 1.0)]);
    }

    #[test]
    fn case_4_is_empty_when_the_crossing_is_past_one() {
        // The correction to the paper's `[min(1, B/A), 1]`. Here B/A = 4, so
        // α > 4 has no solution in [0,1] and the range must be empty. The
        // paper's form would return the point [1,1] and prune the edge.
        let (a, b) = (-0.1f32, -0.4f32);
        let got = solve(a, b);
        assert!(got.is_empty(), "got {got:?}");
        // The inequality at the only α that `min(1, B/A)` would have admitted:
        // 1·(−0.1) < −0.4 is false, so [1,1] would be wrong.
        assert!(1.0 * a >= b, "sanity: α=1 is not a solution");
    }

    #[test]
    fn a_float_zero_coefficient_does_not_invent_a_pruning_range() {
        // Table 1's first example has A = 0 in exact arithmetic. In f32 the sum
        // 0.9 − 0.4 + 0.3 − 0.8 lands a hair below zero, which routes it to
        // Case 4 rather than Case 2. The corrected Case 4 still returns empty,
        // so the sign noise cannot resurrect an edge the paper says survives.
        let (xy, xz, _) = table_1(1);
        let a = xz.e - xy.e + xy.s - xz.s;
        assert!(a.abs() < 1e-6, "A should be within rounding of zero, got {a}");
        assert!(solve(a, xy.s - xz.s).is_empty());
    }

    #[test]
    fn case_1_saturates_when_the_crossing_is_past_one() {
        // B/A = 4, so the inequality holds across all of [0,1].
        assert!(solve(0.2, 0.8).is_full());
    }

    #[test]
    fn the_b_equals_zero_gap_is_handled_both_ways() {
        // The paper's four cases all require B ≠ 0. Neither of these divides by
        // A, so neither can produce a NaN bound.
        assert!(solve(-0.5, 0.0).is_full(), "A < 0: holds for every α > 0");
        assert!(solve(0.5, 0.0).is_empty(), "A > 0: holds for no α ≥ 0");
        assert!(solve(0.0, 0.0).is_empty(), "A = 0: reduces to 0 < 0");
    }

    #[test]
    fn no_case_ever_produces_a_nan_bound() {
        // A = 0 with B ≠ 0 must route to Case 2 or Case 3 rather than dividing.
        for &(a, b) in &[(0.0, 0.5), (0.0, -0.5), (0.0, 0.0)] {
            let got = solve(a, b);
            for &(low, high) in got.intervals() {
                assert!(low.is_finite() && high.is_finite(), "solve({a},{b}) = {got:?}");
            }
        }
    }

    #[test]
    fn hybrid_distance_interpolates_between_the_modalities() {
        let d = HybridDistance::new(0.9, 0.1);
        assert!((d.at(0.0) - 0.1).abs() < 1e-6, "α=0 is the second modality");
        assert!((d.at(1.0) - 0.9).abs() < 1e-6, "α=1 is the first modality");
        assert!((d.at(0.5) - 0.5).abs() < 1e-6);
    }
}
