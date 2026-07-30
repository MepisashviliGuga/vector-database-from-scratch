//! Objects carrying two feature vectors — paper 05 §3.1.
//!
//! Each object `o` has `o.e` in one modality (say a 128-dimensional image
//! embedding) and `o.s` in another (say a 2-dimensional coordinate). A query
//! supplies its own vectors plus a weight α, and the distance is
//!
//! ```text
//! Dist(q,o) = α·δe(q,o) + (1−α)·δs(q,o)
//! ```
//!
//! # Why the components are normalised
//!
//! `δe` and `δs` are Euclidean distances divided by `emax` and `smax`, the
//! largest distance between any two objects in each modality. Without that, α
//! would not mean anything: a 128-dimensional embedding and a 2-dimensional
//! coordinate produce distances on wildly different scales, so α = 0.5 would
//! silently favour whichever modality happens to have larger numbers.
//!
//! # `emax` and `smax` are estimated — a labelled simplification
//!
//! §3.1 defines them as the maximum distance between *any two* objects, which is
//! O(N²) to compute exactly — 500 billion pairs at the paper's 1M-object scale.
//! The paper does not say how it obtains them. Here they are estimated from a
//! seeded sample of pairs ([`estimate_maxima`]).
//!
//! This is safe, and it is worth being precise about why: the two maxima only
//! *rescale* the modalities. Every distance in the index and every query
//! distance divides by the same pair of constants, so nothing is inconsistent —
//! what shifts is the exchange rate α expresses. An underestimate pushes some
//! normalised distances above 1, which no part of the algorithm requires; the
//! pruning algebra in [`super::pruning`] never assumes a bound, and the only
//! consumer of absolute scale is the `th` threshold on active-range *width*,
//! which lives in α-space rather than distance-space.
//!
//! What would be unsafe is estimating them differently for the index and the
//! queries, so [`HybridSet`] owns both values and every distance goes through it.

use crate::ann::squared_l2;
use crate::workload::Rng;

use super::pruning::HybridDistance;

/// A dataset of objects with two feature vectors each.
#[derive(Debug, Clone)]
pub struct HybridSet {
    dim_e: usize,
    dim_s: usize,
    primary: Vec<f32>,
    secondary: Vec<f32>,
    emax: f32,
    smax: f32,
}

impl HybridSet {
    /// Build from flat per-modality arrays, estimating the normalisation maxima.
    ///
    /// # Panics
    ///
    /// If either dimension is 0, either array is not a whole number of vectors,
    /// the two arrays disagree on object count, or the set is empty.
    pub fn new(
        primary: Vec<f32>,
        dim_e: usize,
        secondary: Vec<f32>,
        dim_s: usize,
        sample_pairs: usize,
        seed: u64,
    ) -> Self {
        assert!(dim_e > 0 && dim_s > 0, "both modalities need dimensions");
        assert_eq!(
            primary.len() % dim_e,
            0,
            "{} primary values do not divide into {dim_e}-dimensional vectors",
            primary.len()
        );
        assert_eq!(
            secondary.len() % dim_s,
            0,
            "{} secondary values do not divide into {dim_s}-dimensional vectors",
            secondary.len()
        );
        let count = primary.len() / dim_e;
        assert_eq!(
            count,
            secondary.len() / dim_s,
            "the two modalities describe different numbers of objects"
        );
        assert!(count > 0, "cannot index an empty set");

        let mut set = Self {
            dim_e,
            dim_s,
            primary,
            secondary,
            // Provisional: `estimate_maxima` needs un-normalised access, and
            // dividing by 1 is the identity.
            emax: 1.0,
            smax: 1.0,
        };
        let (emax, smax) = set.estimate_maxima(sample_pairs, seed);
        set.emax = emax;
        set.smax = smax;
        set
    }

    pub fn len(&self) -> usize {
        self.primary.len() / self.dim_e
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dim_e(&self) -> usize {
        self.dim_e
    }

    pub fn dim_s(&self) -> usize {
        self.dim_s
    }

    /// The normalisation constants actually in use.
    pub fn maxima(&self) -> (f32, f32) {
        (self.emax, self.smax)
    }

    pub fn primary(&self, id: usize) -> &[f32] {
        &self.primary[id * self.dim_e..(id + 1) * self.dim_e]
    }

    pub fn secondary(&self, id: usize) -> &[f32] {
        &self.secondary[id * self.dim_s..(id + 1) * self.dim_s]
    }

    /// Bytes of raw vectors held across both modalities.
    pub fn data_bytes(&self) -> usize {
        (self.primary.len() + self.secondary.len()) * std::mem::size_of::<f32>()
    }

    /// Normalised distance between two indexed objects.
    pub fn distance(&self, a: usize, b: usize) -> HybridDistance {
        HybridDistance::new(
            squared_l2(self.primary(a), self.primary(b)).sqrt() / self.emax,
            squared_l2(self.secondary(a), self.secondary(b)).sqrt() / self.smax,
        )
    }

    /// Normalised distance from a query's two vectors to an indexed object.
    ///
    /// # Panics
    ///
    /// If either query vector has the wrong length.
    pub fn query_distance(&self, query_e: &[f32], query_s: &[f32], id: usize) -> HybridDistance {
        assert_eq!(
            query_e.len(),
            self.dim_e,
            "primary query has the wrong length"
        );
        assert_eq!(
            query_s.len(),
            self.dim_s,
            "secondary query has the wrong length"
        );
        HybridDistance::new(
            squared_l2(query_e, self.primary(id)).sqrt() / self.emax,
            squared_l2(query_s, self.secondary(id)).sqrt() / self.smax,
        )
    }

    /// Only the first modality's normalised distance.
    ///
    /// The search's early exit (§4.5) needs this alone: since `δs ≥ 0`,
    /// `α·δe` is a lower bound on the full hybrid distance, so a node can be
    /// rejected without ever touching the second modality.
    pub fn query_primary(&self, query_e: &[f32], id: usize) -> f32 {
        squared_l2(query_e, self.primary(id)).sqrt() / self.emax
    }

    /// Largest distance found in each modality over a seeded sample of pairs.
    ///
    /// Returns `(emax, smax)`, each floored at a tiny positive value so the
    /// division can never produce an infinity when every object coincides.
    fn estimate_maxima(&self, sample_pairs: usize, seed: u64) -> (f32, f32) {
        let count = self.len();
        let mut emax = 0.0f32;
        let mut smax = 0.0f32;

        if count < 2 {
            return (1.0, 1.0);
        }

        // Exhaustive when the set is small enough that O(N²) is free; the
        // threshold is set so the exact answer is used wherever a test might
        // reasonably check it.
        let exhaustive = count * (count - 1) / 2 <= sample_pairs.max(1);
        if exhaustive {
            for a in 0..count {
                for b in (a + 1)..count {
                    let d = self.distance(a, b);
                    emax = emax.max(d.e);
                    smax = smax.max(d.s);
                }
            }
        } else {
            let mut rng = Rng::new(seed);
            for _ in 0..sample_pairs {
                let a = rng.below(count as u64) as usize;
                let b = rng.below(count as u64) as usize;
                if a == b {
                    continue;
                }
                let d = self.distance(a, b);
                emax = emax.max(d.e);
                smax = smax.max(d.s);
            }
        }

        (emax.max(f32::MIN_POSITIVE), smax.max(f32::MIN_POSITIVE))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four objects: 2-D primary, 1-D secondary, chosen so distances are exact
    /// in f32 and hand-checkable.
    fn tiny() -> HybridSet {
        HybridSet::new(
            vec![0.0, 0.0, 3.0, 4.0, 0.0, 8.0, 6.0, 8.0],
            2,
            vec![0.0, 1.0, 2.0, 4.0],
            1,
            usize::MAX, // exhaustive
            7,
        )
    }

    #[test]
    fn maxima_are_exact_on_a_small_set() {
        let set = tiny();
        let (emax, smax) = set.maxima();
        // Primary: the farthest pair is (0,0) to (6,8), distance 10.
        assert!((emax - 10.0).abs() < 1e-6, "emax {emax}");
        // Secondary: 0 to 4, distance 4.
        assert!((smax - 4.0).abs() < 1e-6, "smax {smax}");
    }

    #[test]
    fn normalised_components_stay_within_the_unit_range() {
        let set = tiny();
        for a in 0..set.len() {
            for b in 0..set.len() {
                let d = set.distance(a, b);
                assert!(d.e >= 0.0 && d.e <= 1.0 + 1e-6, "δe out of range: {d:?}");
                assert!(d.s >= 0.0 && d.s <= 1.0 + 1e-6, "δs out of range: {d:?}");
            }
        }
    }

    #[test]
    fn the_farthest_pair_normalises_to_one() {
        let set = tiny();
        let d = set.distance(0, 3);
        assert!((d.e - 1.0).abs() < 1e-6, "{d:?}");
    }

    #[test]
    fn distance_to_self_is_zero() {
        let set = tiny();
        let d = set.distance(2, 2);
        assert_eq!(d.e, 0.0);
        assert_eq!(d.s, 0.0);
        assert_eq!(d.at(0.5), 0.0);
    }

    #[test]
    fn a_query_at_an_object_matches_that_objects_own_distance() {
        let set = tiny();
        let d = set.query_distance(set.primary(1), set.secondary(1), 3);
        let expected = set.distance(1, 3);
        assert!((d.e - expected.e).abs() < 1e-6);
        assert!((d.s - expected.s).abs() < 1e-6);
    }

    #[test]
    fn query_primary_agrees_with_the_full_query_distance() {
        // The early exit relies on these being the same number.
        let set = tiny();
        let q_e = vec![1.0, 1.0];
        let q_s = vec![0.5];
        for id in 0..set.len() {
            let full = set.query_distance(&q_e, &q_s, id);
            let only = set.query_primary(&q_e, id);
            assert!((full.e - only).abs() < 1e-6, "id {id}: {full:?} vs {only}");
        }
    }

    #[test]
    fn the_early_exit_bound_is_valid() {
        // α·δe ≤ Dist for every α, because δs ≥ 0. This is what licenses
        // rejecting a node before computing the second modality.
        let set = tiny();
        let q_e = vec![2.0, 3.0];
        let q_s = vec![1.5];
        for id in 0..set.len() {
            let full = set.query_distance(&q_e, &q_s, id);
            for step in 0..=10 {
                let alpha = step as f32 / 10.0;
                let bound = alpha * set.query_primary(&q_e, id);
                assert!(
                    bound <= full.at(alpha) + 1e-6,
                    "id {id} α={alpha}: bound {bound} exceeds {}",
                    full.at(alpha)
                );
            }
        }
    }

    /// The hybrid distance at a fixed α is a metric, so RNG pruning is
    /// well-founded — §3.2 notes the RNG property needs metric distances.
    ///
    /// A non-negative weighted sum of two metrics is a metric: the triangle
    /// inequality holds in each modality and survives scaling and addition. This
    /// checks it rather than asserting it, since the whole pruning strategy rests
    /// on it.
    #[test]
    fn the_hybrid_distance_obeys_the_triangle_inequality_at_every_alpha() {
        let mut rng = Rng::new(0xD_E9);
        let count = 24;
        let dim_e = 5;
        let dim_s = 2;
        let primary: Vec<f32> = (0..count * dim_e)
            .map(|_| rng.next_f64() as f32 * 10.0)
            .collect();
        let secondary: Vec<f32> = (0..count * dim_s)
            .map(|_| rng.next_f64() as f32 * 3.0)
            .collect();
        let set = HybridSet::new(primary, dim_e, secondary, dim_s, usize::MAX, 1);

        for step in 0..=10 {
            let alpha = step as f32 / 10.0;
            for a in 0..count {
                for b in 0..count {
                    for c in 0..count {
                        let ab = set.distance(a, b).at(alpha);
                        let bc = set.distance(b, c).at(alpha);
                        let ac = set.distance(a, c).at(alpha);
                        assert!(
                            ac <= ab + bc + 1e-4,
                            "α={alpha}: d({a},{c})={ac} > d({a},{b})+d({b},{c})={}",
                            ab + bc
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn distance_is_symmetric() {
        let set = tiny();
        for a in 0..set.len() {
            for b in 0..set.len() {
                let ab = set.distance(a, b);
                let ba = set.distance(b, a);
                assert!((ab.e - ba.e).abs() < 1e-6 && (ab.s - ba.s).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn sampled_maxima_are_reproducible_from_a_seed() {
        let mut rng = Rng::new(99);
        let count = 200;
        let primary: Vec<f32> = (0..count * 4).map(|_| rng.next_f64() as f32).collect();
        let secondary: Vec<f32> = (0..count * 2).map(|_| rng.next_f64() as f32).collect();

        // Few enough pairs to force the sampling path rather than exhaustive.
        let a = HybridSet::new(primary.clone(), 4, secondary.clone(), 2, 64, 12345);
        let b = HybridSet::new(primary.clone(), 4, secondary.clone(), 2, 64, 12345);
        assert_eq!(
            a.maxima(),
            b.maxima(),
            "same seed must give the same maxima"
        );

        let c = HybridSet::new(primary, 4, secondary, 2, 64, 54321);
        // Not asserting inequality — a different seed *may* find the same pair —
        // only that the estimate never exceeds the true maximum.
        let (exact_e, exact_s) = {
            let full = HybridSet::new(a.primary.clone(), 4, a.secondary.clone(), 2, usize::MAX, 0);
            full.maxima()
        };
        for set in [&a, &c] {
            let (e, s) = set.maxima();
            assert!(e <= exact_e + 1e-6, "sample {e} exceeded exact {exact_e}");
            assert!(s <= exact_s + 1e-6, "sample {s} exceeded exact {exact_s}");
        }
    }

    #[test]
    #[should_panic(expected = "different numbers of objects")]
    fn mismatched_object_counts_are_rejected() {
        HybridSet::new(vec![0.0; 8], 2, vec![0.0; 3], 1, 16, 0);
    }

    #[test]
    #[should_panic(expected = "cannot index an empty set")]
    fn an_empty_set_is_rejected() {
        HybridSet::new(Vec::new(), 2, Vec::new(), 1, 16, 0);
    }
}
