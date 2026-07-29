//! Extended RaBitQ: quantization with a provable error bound.
//!
//! **Faithful reproduction of paper 03 §3** (extended RaBitQ, SIGMOD 2025).
//!
//! # The reduction everything rests on
//!
//! With a centroid `c`, write `o = (o_r−c)/‖o_r−c‖` and `q = (q_r−c)/‖q_r−c‖`.
//! Then
//!
//! ```text
//!   ‖o_r − q_r‖² = ‖o_r−c‖² + ‖q_r−c‖² − 2·‖o_r−c‖·‖q_r−c‖·⟨q,o⟩
//! ```
//!
//! `‖o_r−c‖` is stored per vector at index time; `‖q_r−c‖` is computed once per
//! query and shared across every candidate. So the whole problem becomes
//! estimating **one inner product of two unit vectors** from a handful of bits.
//!
//! # The codebook, and why it is a *normalized* grid
//!
//! ```text
//!   G   = { −(2^B − 1)/2 + u  |  u = 0 … 2^B − 1 }^D      a uniform integer grid
//!   G_r = { P·(y/‖y‖)  |  y ∈ G }                          normalized, then rotated
//! ```
//!
//! Two requirements pull in opposite directions, and normalizing a grid point
//! satisfies both:
//!
//! - The unbiasedness and error bound hold **only** for a codebook of randomly
//!   rotated *unit* vectors.
//! - Fast estimation needs codes that are plain *integers*, so no decompression
//!   step stands between the code and the arithmetic.
//!
//! A normalized grid point is a unit vector whose direction is fixed entirely by
//! integers. The code stores the integers; `‖ȳ‖` folds into the estimator.
//!
//! At `B = 1` this collapses to the original RaBitQ — the codebook becomes the
//! hypercube vertices `{±1/√D}^D` and encoding is just a sign bit per dimension.
//!
//! # Why the random rotation
//!
//! It moves the randomness out of the data and into `P`, so the error bound
//! holds **whatever the data distribution is**. That is precisely what product
//! quantization lacks.
//!
//! # Guarantees
//!
//! Theorem 3.2 gives `B = Θ(log((1/D)·(1/ε²)·log(1/δ)))` — matching the
//! theoretical lower bound. Note `B` is *logarithmic* in `ε⁻²`, and *negatively*
//! related to `D`: higher-dimensional vectors need fewer bits per dimension,
//! because concentration is stronger there. Empirically the paper reports
//!
//! ```text
//!   ε  <  2^(−B) · 5.75 / √D        with probability > 99.9%
//! ```
//!
//! so the error roughly **halves with every additional bit**. This module's
//! tests check that behaviour rather than assuming it.
//!
//! # Not reproduced
//!
//! The SIMD `FastScan` path and the two-stage most-significant-bit query, which
//! paper 03 uses for speed. Both are labelled in the project README as out of
//! scope; the arithmetic here is scalar, so **absolute query speed is not
//! comparable to the paper's**. Accuracy is.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::rotation::Rotation;

/// Largest supported bit width, so a code fits in a `u8`.
pub const MAX_BITS: u32 = 8;

/// A quantized vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Code {
    /// `ȳ_u`: the unsigned integer code, one value per dimension in
    /// `0 ..= 2^B − 1`.
    pub codes: Vec<u8>,
    /// `‖ȳ‖`, the norm of the *signed* grid point the code represents.
    pub grid_norm: f32,
    /// `⟨ō, o⟩`, the estimator's denominator. Computed during encoding at no
    /// extra cost, since it is exactly the quantity encoding maximises.
    pub cosine_to_original: f32,
    /// `‖o_r − c‖`, the distance from the raw vector to the centroid.
    pub distance_to_centroid: f32,
}

/// Per-query values shared across every candidate.
#[derive(Debug, Clone)]
pub struct Query {
    /// `q' = P⁻¹q`, the rotated normalized query.
    rotated: Vec<f32>,
    /// `Σᵢ q'[i]`, the correction term for the unsigned code offset. Depends
    /// only on the query, so it is computed once rather than per candidate.
    sum: f32,
    /// `‖q_r − c‖`.
    distance_to_centroid: f32,
}

/// The extended RaBitQ quantizer.
#[derive(Debug, Clone)]
pub struct RaBitQ {
    dimension: usize,
    bits: u32,
    /// `(2^B − 1)/2`, the offset between signed grid values and unsigned codes.
    ///
    /// Also the largest magnitude a grid coordinate takes, which is why the
    /// sweep needs no explicit clamp: it enumerates exactly `2^(B−1) − 1`
    /// critical values per dimension and so cannot step past it.
    offset: f32,
    rotation: Rotation,
}

impl RaBitQ {
    /// # Panics
    ///
    /// If `dimension` is 0, or `bits` is outside `1 ..= 8`.
    pub fn new(dimension: usize, bits: u32, seed: u64) -> Self {
        assert!(dimension > 0, "vectors need at least one dimension");
        assert!(
            (1..=MAX_BITS).contains(&bits),
            "bits must be between 1 and {MAX_BITS}, got {bits}"
        );

        Self {
            dimension,
            bits,
            offset: ((1u32 << bits) - 1) as f32 / 2.0,
            rotation: Rotation::new(dimension, seed),
        }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// The random rotation `P`.
    ///
    /// Exposed for SymphonyQG §3.1.1, which rotates a *raw* query once and
    /// reuses it across every centroid in a graph traversal, rather than
    /// rotating a normalized query per centroid as [`Self::prepare_query`] does.
    pub fn rotation(&self) -> &Rotation {
        &self.rotation
    }

    /// `(2^B − 1)/2`, the offset between signed grid values and unsigned codes.
    ///
    /// Exposed for the same decomposition: a caller reconstructing the estimator
    /// from its parts needs to subtract this offset itself.
    pub fn offset(&self) -> f32 {
        self.offset
    }

    /// Bytes a single code occupies.
    ///
    /// One byte per dimension here, since the codes are not bit-packed. Paper 03
    /// packs them to `B` bits each; packing changes memory but not accuracy, so
    /// this figure should be scaled by `B/8` when comparing against the paper's
    /// compression rates.
    pub fn code_bytes(&self) -> usize {
        self.dimension
    }

    /// Bytes a code would occupy bit-packed, which is the paper's figure.
    pub fn packed_code_bytes(&self) -> usize {
        (self.dimension * self.bits as usize).div_ceil(8)
    }

    /// Quantize `raw` relative to `centroid`.
    ///
    /// # Panics
    ///
    /// If either slice is the wrong length.
    pub fn encode(&self, raw: &[f32], centroid: &[f32]) -> Code {
        assert_eq!(raw.len(), self.dimension, "data vector has the wrong length");
        assert_eq!(centroid.len(), self.dimension, "centroid has the wrong length");

        let residual: Vec<f32> = raw
            .iter()
            .zip(centroid.iter())
            .map(|(value, centre)| value - centre)
            .collect();
        let distance_to_centroid = residual.iter().map(|v| v * v).sum::<f32>().sqrt();

        // A vector sitting exactly on the centroid has no direction to encode.
        // Its distance is fully described by `distance_to_centroid = 0`, so any
        // code works; a mid-grid one keeps the estimator finite.
        if distance_to_centroid <= f32::MIN_POSITIVE {
            let middle = ((1u32 << self.bits) - 1) as u8 / 2;
            return Code {
                codes: vec![middle; self.dimension],
                grid_norm: 1.0,
                cosine_to_original: 1.0,
                distance_to_centroid: 0.0,
            };
        }

        let unit: Vec<f32> = residual.iter().map(|v| v / distance_to_centroid).collect();
        // Pull the vector into the codebook's frame rather than rotating the
        // codebook, which would cost a matrix multiply per codeword.
        let rotated = self.rotation.apply_inverse(&unit);

        let (magnitudes, inner, norm_squared) = self.best_grid_point(&rotated);
        let grid_norm = norm_squared.sqrt();

        // ⟨ō,o⟩ = ⟨ȳ/‖ȳ‖, o'⟩, which is exactly what the search maximised.
        let cosine_to_original = if grid_norm > 0.0 { inner / grid_norm } else { 0.0 };

        // Signed grid value, then shifted into an unsigned code.
        let codes = magnitudes
            .iter()
            .zip(rotated.iter())
            .map(|(&magnitude, &component)| {
                let signed = if component < 0.0 { -magnitude } else { magnitude };
                (signed + self.offset).round() as u8
            })
            .collect();

        Code {
            codes,
            grid_norm,
            cosine_to_original,
            distance_to_centroid,
        }
    }

    /// Algorithm 1: the grid point maximising `⟨y/‖y‖, o'⟩`.
    ///
    /// Returns per-dimension magnitudes, `⟨ȳ, o'⟩`, and `‖ȳ‖²`.
    ///
    /// Enumerating `G` directly is impossible — it holds `2^(B·D)` points.
    /// Lemma 3.1 says the maximizer is what rounding `t·o'` produces for *some*
    /// scale `t`, so it is enough to sweep `t`. And only the **critical values**
    /// matter, where some coordinate's rounding flips: at most `D·(2^(B−1) − 1)`
    /// of them. A min-heap walks them in order, and because each step changes
    /// exactly one coordinate, `⟨y,o'⟩` and `‖y‖²` update in `O(1)`.
    ///
    /// Cost: `O(2^B · D log D)`.
    ///
    /// Signs are handled separately — the maximizer lies in the same orthant as
    /// `o'` — so this works with `|o'[i]|` and returns magnitudes only. That is
    /// also why there are `2^(B−1)` magnitudes per dimension rather than `2^B`.
    fn best_grid_point(&self, rotated: &[f32]) -> (Vec<f32>, f32, f32) {
        let magnitudes_per_dimension = 1u32 << (self.bits - 1);
        let absolute: Vec<f32> = rotated.iter().map(|v| v.abs()).collect();

        // t = 0 rounds every coordinate to the smallest magnitude, 0.5.
        let mut magnitudes = vec![0.5f32; self.dimension];
        let mut inner: f32 = absolute.iter().map(|a| 0.5 * a).sum();
        let mut norm_squared: f32 = 0.25 * self.dimension as f32;

        let mut best_objective = if norm_squared > 0.0 {
            inner / norm_squared.sqrt()
        } else {
            f32::NEG_INFINITY
        };
        let mut best_magnitudes = magnitudes.clone();

        // Critical values: t·|o'[i]| crossing an integer k moves coordinate i
        // from magnitude k−0.5 up to k+0.5.
        let mut pending: BinaryHeap<Critical> = BinaryHeap::new();
        for (dimension, &value) in absolute.iter().enumerate() {
            if value > 0.0 && magnitudes_per_dimension > 1 {
                pending.push(Critical {
                    t: 1.0 / value,
                    dimension,
                    step: 1,
                });
            }
        }

        while let Some(Critical { dimension, step, .. }) = pending.pop() {
            let magnitude = &mut magnitudes[dimension];
            // ‖y‖² gains (m+1)² − m² = 2m + 1; ⟨y,o'⟩ gains one unit of |o'[i]|.
            norm_squared += 2.0 * *magnitude + 1.0;
            inner += absolute[dimension];
            *magnitude += 1.0;

            if step + 1 < magnitudes_per_dimension {
                pending.push(Critical {
                    t: (step + 1) as f32 / absolute[dimension],
                    dimension,
                    step: step + 1,
                });
            }

            let objective = inner / norm_squared.sqrt();
            if objective > best_objective {
                best_objective = objective;
                // Copying the whole vector on improvement, rather than
                // recomputing from the winning `t` afterwards. Rounding exactly
                // at a critical value is ambiguous in floating point, and this
                // sidesteps that entirely for a bounded cost.
                best_magnitudes.copy_from_slice(&magnitudes);
            }
        }

        let inner = best_magnitudes
            .iter()
            .zip(absolute.iter())
            .map(|(m, a)| m * a)
            .sum();
        let norm_squared = best_magnitudes.iter().map(|m| m * m).sum();
        (best_magnitudes, inner, norm_squared)
    }

    /// Prepare a query. Do this once, then estimate against many codes.
    ///
    /// # Panics
    ///
    /// If either slice is the wrong length.
    pub fn prepare_query(&self, raw: &[f32], centroid: &[f32]) -> Query {
        assert_eq!(raw.len(), self.dimension, "query has the wrong length");
        assert_eq!(centroid.len(), self.dimension, "centroid has the wrong length");

        let residual: Vec<f32> = raw
            .iter()
            .zip(centroid.iter())
            .map(|(value, centre)| value - centre)
            .collect();
        let distance_to_centroid = residual.iter().map(|v| v * v).sum::<f32>().sqrt();

        let unit: Vec<f32> = if distance_to_centroid > f32::MIN_POSITIVE {
            residual.iter().map(|v| v / distance_to_centroid).collect()
        } else {
            vec![0.0; self.dimension]
        };
        let rotated = self.rotation.apply_inverse(&unit);
        let sum = rotated.iter().sum();

        Query {
            rotated,
            sum,
            distance_to_centroid,
        }
    }

    /// Estimate `⟨o, q⟩`, the inner product of the two normalized vectors.
    ///
    /// Equation 12: `⟨ō,q⟩ = (1/‖ȳ‖)·(⟨ȳ_u, q'⟩ − ((2^B−1)/2)·Σq'[i])`, then
    /// divided by `⟨ō,o⟩` to remove the estimator's bias.
    pub fn estimate_inner_product(&self, code: &Code, query: &Query) -> f32 {
        let dot: f32 = code
            .codes
            .iter()
            .zip(query.rotated.iter())
            .map(|(&c, &q)| c as f32 * q)
            .sum();

        let quantized_inner = (dot - self.offset * query.sum) / code.grid_norm;
        if code.cosine_to_original.abs() < f32::MIN_POSITIVE {
            return 0.0;
        }
        quantized_inner / code.cosine_to_original
    }

    /// Estimate the squared Euclidean distance between the raw vectors.
    pub fn estimate_squared_distance(&self, code: &Code, query: &Query) -> f32 {
        let inner = self.estimate_inner_product(code, query);
        let estimate = code.distance_to_centroid * code.distance_to_centroid
            + query.distance_to_centroid * query.distance_to_centroid
            - 2.0 * code.distance_to_centroid * query.distance_to_centroid * inner;
        // A squared distance cannot be negative; estimation noise can push it
        // just below zero when the true distance is near zero.
        estimate.max(0.0)
    }

    /// The paper's empirical error bound: `2^(−B) · 5.75 / √D`, holding with
    /// probability above 99.9% for the inner product of unit vectors.
    pub fn empirical_error_bound(&self) -> f32 {
        let scale = 2.0f32.powi(-(self.bits as i32));
        scale * 5.75 / (self.dimension as f32).sqrt()
    }
}

/// A pending critical value, ordered so [`BinaryHeap`] yields the smallest `t`.
#[derive(Debug)]
struct Critical {
    t: f32,
    dimension: usize,
    step: u32,
}

impl Ord for Critical {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed: BinaryHeap is a max-heap and the sweep needs ascending `t`.
        // `total_cmp` rather than `partial_cmp` so a stray NaN cannot make the
        // ordering non-transitive and panic inside the heap.
        other
            .t
            .total_cmp(&self.t)
            .then_with(|| other.dimension.cmp(&self.dimension))
            .then_with(|| other.step.cmp(&self.step))
    }
}

impl PartialOrd for Critical {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Critical {
    fn eq(&self, other: &Self) -> bool {
        self.dimension == other.dimension && self.step == other.step
    }
}

impl Eq for Critical {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ann::inner_product;
    use crate::workload::Rng;

    fn random_unit(dimension: usize, rng: &mut Rng) -> Vec<f32> {
        let raw: Vec<f32> = (0..dimension)
            .map(|_| rng.next_f64() as f32 * 2.0 - 1.0)
            .collect();
        let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
        raw.iter().map(|v| v / norm).collect()
    }

    fn zero_centroid(dimension: usize) -> Vec<f32> {
        vec![0.0; dimension]
    }

    // -----------------------------------------------------------------
    // The codebook
    // -----------------------------------------------------------------

    /// At `B = 1` extended RaBitQ must reduce exactly to the original: a sign
    /// bit per dimension, and nothing else.
    #[test]
    fn one_bit_reduces_to_sign_quantization() {
        let dimension = 64;
        let quantizer = RaBitQ::new(dimension, 1, 7);
        let mut rng = Rng::new(11);

        for _ in 0..20 {
            let vector = random_unit(dimension, &mut rng);
            let code = quantizer.encode(&vector, &zero_centroid(dimension));

            assert!(
                code.codes.iter().all(|&c| c <= 1),
                "a 1-bit code must be 0 or 1: {:?}",
                &code.codes[..8]
            );

            // The code should be the sign of the rotated vector.
            let rotated = quantizer.rotation.apply_inverse(&vector);
            for (&c, &component) in code.codes.iter().zip(rotated.iter()) {
                let expected = u8::from(component >= 0.0);
                assert_eq!(c, expected, "sign disagreement");
            }
        }
    }

    #[test]
    fn codes_stay_within_their_bit_width() {
        let dimension = 32;
        let mut rng = Rng::new(13);
        for bits in 1..=MAX_BITS {
            let quantizer = RaBitQ::new(dimension, bits, 3);
            let limit = ((1u32 << bits) - 1) as u8;
            for _ in 0..10 {
                let vector = random_unit(dimension, &mut rng);
                let code = quantizer.encode(&vector, &zero_centroid(dimension));
                assert!(
                    code.codes.iter().all(|&c| c <= limit),
                    "B = {bits} produced a code above {limit}"
                );
                assert_eq!(code.codes.len(), dimension);
            }
        }
    }

    /// `⟨ō,o⟩` is the estimator's denominator. Paper 03 notes it concentrates
    /// around 0.8 at one bit, and rises with `B` — more bits means the codeword
    /// sits closer to the original direction.
    #[test]
    fn the_quantized_vector_gets_closer_as_bits_increase() {
        let dimension = 128;
        let mut rng = Rng::new(17);
        let vectors: Vec<Vec<f32>> = (0..30).map(|_| random_unit(dimension, &mut rng)).collect();

        let mut previous = 0.0f32;
        for bits in 1..=6 {
            let quantizer = RaBitQ::new(dimension, bits, 5);
            let mean: f32 = vectors
                .iter()
                .map(|v| {
                    quantizer
                        .encode(v, &zero_centroid(dimension))
                        .cosine_to_original
                })
                .sum::<f32>()
                / vectors.len() as f32;

            assert!(
                mean > previous,
                "B = {bits} gave cosine {mean}, not better than {previous}"
            );
            assert!(mean <= 1.0 + 1e-5, "a cosine above 1 is impossible: {mean}");
            if bits == 1 {
                assert!(
                    (0.7..0.9).contains(&mean),
                    "paper 03 reports about 0.8 at one bit; got {mean}"
                );
            }
            previous = mean;
        }
    }

    // -----------------------------------------------------------------
    // The estimator
    // -----------------------------------------------------------------

    /// The property the whole method rests on. The estimator may be noisy, but
    /// its *mean* error must be ~0 — a biased estimator would systematically
    /// misrank neighbours no matter how many bits were spent.
    #[test]
    fn the_estimator_is_unbiased() {
        let dimension = 128;
        let quantizer = RaBitQ::new(dimension, 4, 19);
        let centroid = zero_centroid(dimension);
        let mut rng = Rng::new(23);

        let mut total_error = 0.0f64;
        let mut samples = 0usize;

        for _ in 0..200 {
            let data = random_unit(dimension, &mut rng);
            let query = random_unit(dimension, &mut rng);

            let code = quantizer.encode(&data, &centroid);
            let prepared = quantizer.prepare_query(&query, &centroid);

            let estimated = quantizer.estimate_inner_product(&code, &prepared);
            let truth = inner_product(&data, &query);
            total_error += (estimated - truth) as f64;
            samples += 1;
        }

        let mean_error = total_error / samples as f64;
        assert!(
            mean_error.abs() < 0.01,
            "mean error {mean_error:.5} indicates bias, not noise"
        );
    }

    /// The paper's headline claim: error decays exponentially in `B`, roughly
    /// halving per extra bit.
    #[test]
    fn error_falls_as_bits_increase() {
        let dimension = 128;
        let centroid = zero_centroid(dimension);

        let mut errors = Vec::new();
        for bits in 1..=6 {
            let quantizer = RaBitQ::new(dimension, bits, 29);
            let mut rng = Rng::new(31);
            let mut total = 0.0f64;
            let mut samples = 0usize;

            for _ in 0..150 {
                let data = random_unit(dimension, &mut rng);
                let query = random_unit(dimension, &mut rng);
                let code = quantizer.encode(&data, &centroid);
                let prepared = quantizer.prepare_query(&query, &centroid);
                total += (quantizer.estimate_inner_product(&code, &prepared)
                    - inner_product(&data, &query))
                .abs() as f64;
                samples += 1;
            }
            errors.push(total / samples as f64);
        }

        assert!(
            errors.windows(2).all(|pair| pair[1] < pair[0]),
            "error must fall with every extra bit: {errors:?}"
        );
        assert!(
            errors[0] / errors[5] > 8.0,
            "five extra bits should cut the error by far more than {:.1}x",
            errors[0] / errors[5]
        );
    }

    /// Against the paper's own empirical formula, `ε < 2^(−B)·5.75/√D`, which
    /// it states holds with probability above 99.9%. Checking our error against
    /// *their* published bound is the closest thing to an external validation
    /// available here.
    #[test]
    fn errors_respect_the_papers_empirical_bound() {
        let dimension = 128;
        let centroid = zero_centroid(dimension);

        for bits in [1u32, 2, 3, 4] {
            let quantizer = RaBitQ::new(dimension, bits, 37);
            let bound = quantizer.empirical_error_bound();
            let mut rng = Rng::new(41);

            let trials = 2000;
            let mut violations = 0usize;
            for _ in 0..trials {
                let data = random_unit(dimension, &mut rng);
                let query = random_unit(dimension, &mut rng);
                let code = quantizer.encode(&data, &centroid);
                let prepared = quantizer.prepare_query(&query, &centroid);
                let error = (quantizer.estimate_inner_product(&code, &prepared)
                    - inner_product(&data, &query))
                .abs();
                if error > bound {
                    violations += 1;
                }
            }

            let rate = violations as f64 / trials as f64;
            assert!(
                rate < 0.01,
                "B = {bits}: {:.2}% of estimates exceeded the bound of {bound:.5}, \
                 against the paper's stated <0.1%",
                rate * 100.0
            );
        }
    }

    /// Recovering an actual distance, not just an inner product — this is what
    /// the index will call.
    #[test]
    fn estimated_distances_track_true_distances() {
        let dimension = 128;
        let quantizer = RaBitQ::new(dimension, 6, 43);
        let mut rng = Rng::new(47);

        // A realistic setup: vectors scattered around a non-zero centroid.
        let centroid: Vec<f32> = (0..dimension).map(|_| rng.next_f64() as f32).collect();
        let data: Vec<Vec<f32>> = (0..100)
            .map(|_| {
                let direction = random_unit(dimension, &mut rng);
                let scale = 1.0 + rng.next_f64() as f32;
                centroid
                    .iter()
                    .zip(direction.iter())
                    .map(|(c, d)| c + d * scale)
                    .collect()
            })
            .collect();

        let query: Vec<f32> = {
            let direction = random_unit(dimension, &mut rng);
            centroid
                .iter()
                .zip(direction.iter())
                .map(|(c, d)| c + d * 1.5)
                .collect()
        };

        let prepared = quantizer.prepare_query(&query, &centroid);
        let mut worst_relative_error = 0.0f32;

        for vector in &data {
            let code = quantizer.encode(vector, &centroid);
            let estimated = quantizer.estimate_squared_distance(&code, &prepared);
            let truth = crate::ann::squared_l2(vector, &query);
            worst_relative_error = worst_relative_error.max((estimated - truth).abs() / truth.max(1e-6));
        }

        assert!(
            worst_relative_error < 0.15,
            "worst relative distance error was {worst_relative_error:.4}"
        );
    }

    /// Ranking is what an index actually needs: the estimate has to order
    /// neighbours the way exact distances do.
    #[test]
    fn ranking_by_estimate_recovers_most_true_neighbours() {
        use crate::ann::squared_l2;

        let dimension = 128;
        let quantizer = RaBitQ::new(dimension, 7, 53);
        let mut rng = Rng::new(59);
        let centroid = zero_centroid(dimension);

        let data: Vec<Vec<f32>> = (0..2000).map(|_| random_unit(dimension, &mut rng)).collect();
        let codes: Vec<Code> = data.iter().map(|v| quantizer.encode(v, &centroid)).collect();

        let mut total_recall = 0.0;
        let queries = 20;
        for _ in 0..queries {
            let query = random_unit(dimension, &mut rng);
            let prepared = quantizer.prepare_query(&query, &centroid);

            let mut estimated: Vec<(f32, usize)> = codes
                .iter()
                .enumerate()
                .map(|(i, code)| (quantizer.estimate_squared_distance(code, &prepared), i))
                .collect();
            estimated.sort_by(|a, b| a.0.total_cmp(&b.0));

            let mut exact: Vec<(f32, usize)> = data
                .iter()
                .enumerate()
                .map(|(i, v)| (squared_l2(v, &query), i))
                .collect();
            exact.sort_by(|a, b| a.0.total_cmp(&b.0));

            let truth: std::collections::HashSet<usize> =
                exact.iter().take(10).map(|&(_, i)| i).collect();
            let hits = estimated
                .iter()
                .take(10)
                .filter(|(_, i)| truth.contains(i))
                .count();
            total_recall += hits as f64 / 10.0;
        }

        let recall = total_recall / queries as f64;
        assert!(
            recall > 0.85,
            "recall@10 from 7-bit codes was {recall:.3}; the paper reports >99% \
             at this rate"
        );
    }

    // -----------------------------------------------------------------
    // Edges
    // -----------------------------------------------------------------

    #[test]
    fn a_vector_on_the_centroid_is_handled() {
        let dimension = 16;
        let quantizer = RaBitQ::new(dimension, 4, 61);
        let centroid: Vec<f32> = (0..dimension).map(|i| i as f32).collect();

        let code = quantizer.encode(&centroid, &centroid);
        assert_eq!(code.distance_to_centroid, 0.0);

        let prepared = quantizer.prepare_query(&centroid, &centroid);
        let distance = quantizer.estimate_squared_distance(&code, &prepared);
        assert!(distance.is_finite(), "got {distance}");
        assert!(distance < 1e-3, "a point at the centroid is zero from itself");
    }

    #[test]
    fn estimated_squared_distances_are_never_negative() {
        let dimension = 32;
        let quantizer = RaBitQ::new(dimension, 2, 67);
        let centroid = zero_centroid(dimension);
        let mut rng = Rng::new(71);

        for _ in 0..200 {
            let vector = random_unit(dimension, &mut rng);
            let code = quantizer.encode(&vector, &centroid);
            // Query the vector itself: the true distance is 0, so estimation
            // noise is most likely to push the estimate below zero here.
            let prepared = quantizer.prepare_query(&vector, &centroid);
            assert!(quantizer.estimate_squared_distance(&code, &prepared) >= 0.0);
        }
    }

    #[test]
    fn compression_rates_match_the_bit_width() {
        let quantizer = RaBitQ::new(128, 4, 1);
        // 128 dimensions of f32 is 512 bytes; 4-bit packed codes are 64.
        assert_eq!(quantizer.packed_code_bytes(), 64);
        assert_eq!(quantizer.code_bytes(), 128, "unpacked, one byte per dimension");

        assert_eq!(RaBitQ::new(128, 1, 1).packed_code_bytes(), 16, "32x");
        assert_eq!(RaBitQ::new(128, 8, 1).packed_code_bytes(), 128, "4x");
    }

    #[test]
    fn encoding_is_deterministic() {
        let dimension = 64;
        let quantizer = RaBitQ::new(dimension, 5, 73);
        let mut rng = Rng::new(79);
        let vector = random_unit(dimension, &mut rng);
        let centroid = zero_centroid(dimension);

        assert_eq!(
            quantizer.encode(&vector, &centroid),
            quantizer.encode(&vector, &centroid)
        );
    }

    #[test]
    #[should_panic(expected = "bits must be between 1 and 8")]
    fn an_unsupported_bit_width_is_rejected() {
        RaBitQ::new(16, 9, 1);
    }
}
