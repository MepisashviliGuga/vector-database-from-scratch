//! Random orthogonal transformation.
//!
//! # Why RaBitQ needs this
//!
//! The codebook is a *fixed*, data-independent set of directions. Quantizing to
//! it directly would work well for vectors that happen to align with those
//! directions and badly for the rest, and the error would depend on the data
//! distribution — exactly the weakness that leaves product quantization without
//! a theoretical guarantee.
//!
//! Rotating the codebook by a random orthogonal matrix moves the randomness out
//! of the data and into `P`. The error bound is then a statement about the
//! sampled rotation, and it holds **whatever the data looks like**. That is the
//! whole reason RaBitQ has a bound and PQ does not.
//!
//! An orthogonal matrix also preserves norms and inner products exactly, so
//! rotating both the data and the query leaves every distance unchanged. The
//! transformation is free in the mathematics and costs one matrix-vector product
//! in practice.
//!
//! # Construction
//!
//! Modified Gram-Schmidt on a matrix of independent standard normals. The
//! *modified* form subtracts each projection as it goes rather than computing
//! them all against the original column; classical Gram-Schmidt loses
//! orthogonality badly at these dimensions in `f32`, which would silently break
//! the guarantee the rotation exists to provide.

use crate::workload::Rng;

/// A `D × D` orthogonal matrix, stored row-major.
#[derive(Debug, Clone)]
pub struct Rotation {
    dimension: usize,
    /// `matrix[i·D + j]` is row `i`, column `j`.
    matrix: Vec<f32>,
}

impl Rotation {
    /// Sample a random orthogonal matrix.
    ///
    /// Deterministic given `seed`, so an index can be rebuilt exactly — the
    /// codes are meaningless without the rotation that produced them.
    ///
    /// # Panics
    ///
    /// If `dimension` is 0.
    pub fn new(dimension: usize, seed: u64) -> Self {
        assert!(dimension > 0, "a rotation needs at least one dimension");
        let mut rng = Rng::new(seed);

        // Columns of independent standard normals; orthonormalising them gives a
        // matrix distributed uniformly over the orthogonal group.
        let mut columns: Vec<Vec<f32>> = (0..dimension)
            .map(|_| (0..dimension).map(|_| standard_normal(&mut rng)).collect())
            .collect();

        for j in 0..dimension {
            // Modified Gram-Schmidt: subtract against the *running* vector, not
            // the original, so rounding errors do not accumulate into a matrix
            // that is only approximately orthogonal.
            for k in 0..j {
                let projection: f32 = columns[j]
                    .iter()
                    .zip(columns[k].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let (earlier, later) = columns.split_at_mut(j);
                for (target, basis) in later[0].iter_mut().zip(earlier[k].iter()) {
                    *target -= projection * basis;
                }
            }

            let norm: f32 = columns[j].iter().map(|v| v * v).sum::<f32>().sqrt();
            // A column that collapsed to nothing means the sampled matrix was
            // near-singular. Astronomically unlikely with continuous normals,
            // but silently emitting a non-orthogonal matrix would invalidate
            // every distance estimate downstream.
            assert!(
                norm > 1e-6,
                "Gram-Schmidt produced a degenerate column; reseed the rotation"
            );
            for value in &mut columns[j] {
                *value /= norm;
            }
        }

        let mut matrix = vec![0.0f32; dimension * dimension];
        for (j, column) in columns.iter().enumerate() {
            for (i, &value) in column.iter().enumerate() {
                matrix[i * dimension + j] = value;
            }
        }

        Self { dimension, matrix }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// `P · x`.
    ///
    /// # Panics
    ///
    /// If `x` is the wrong length.
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.dimension, "dimension mismatch");
        (0..self.dimension)
            .map(|i| {
                let row = &self.matrix[i * self.dimension..(i + 1) * self.dimension];
                row.iter().zip(x.iter()).map(|(a, b)| a * b).sum()
            })
            .collect()
    }

    /// `Pᵀ · x`, which for an orthogonal matrix is `P⁻¹ · x`.
    ///
    /// This is the direction RaBitQ actually uses: both the data vector and the
    /// query are pulled into the codebook's frame by `P⁻¹` rather than the
    /// codebook being rotated per query.
    ///
    /// # Panics
    ///
    /// If `x` is the wrong length.
    pub fn apply_inverse(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.dimension, "dimension mismatch");
        let mut out = vec![0.0f32; self.dimension];
        for (i, &value) in x.iter().enumerate() {
            let row = &self.matrix[i * self.dimension..(i + 1) * self.dimension];
            for (j, &entry) in row.iter().enumerate() {
                out[j] += entry * value;
            }
        }
        out
    }
}

/// One standard normal sample, via the Box-Muller transform.
fn standard_normal(rng: &mut Rng) -> f32 {
    // `u1` must be strictly positive: `ln(0)` is negative infinity.
    let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
    let u2 = rng.next_f64();
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ann::{inner_product, squared_l2};

    /// The defining property. If this drifts, every distance estimate built on
    /// the rotation is wrong by an amount nothing downstream would reveal.
    #[test]
    fn the_matrix_is_orthogonal() {
        for dimension in [2usize, 8, 64, 128] {
            let rotation = Rotation::new(dimension, 42);

            // PᵀP = I, checked column against column.
            for i in 0..dimension {
                for j in 0..dimension {
                    let dot: f32 = (0..dimension)
                        .map(|k| {
                            rotation.matrix[k * dimension + i] * rotation.matrix[k * dimension + j]
                        })
                        .sum();
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (dot - expected).abs() < 1e-4,
                        "dimension {dimension}: columns {i},{j} gave {dot}, expected {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn norms_are_preserved() {
        let rotation = Rotation::new(64, 7);
        let mut rng = Rng::new(11);
        for _ in 0..50 {
            let x: Vec<f32> = (0..64).map(|_| standard_normal(&mut rng)).collect();
            let rotated = rotation.apply(&x);

            let before: f32 = x.iter().map(|v| v * v).sum();
            let after: f32 = rotated.iter().map(|v| v * v).sum();
            assert!(
                (before - after).abs() / before < 1e-4,
                "norm changed from {before} to {after}"
            );
        }
    }

    /// Rotating both sides must leave every inner product — and therefore every
    /// distance — exactly where it was.
    #[test]
    fn inner_products_and_distances_are_preserved() {
        let rotation = Rotation::new(32, 13);
        let mut rng = Rng::new(17);
        for _ in 0..50 {
            let a: Vec<f32> = (0..32).map(|_| standard_normal(&mut rng)).collect();
            let b: Vec<f32> = (0..32).map(|_| standard_normal(&mut rng)).collect();

            let before = inner_product(&a, &b);
            let after = inner_product(&rotation.apply(&a), &rotation.apply(&b));
            assert!(
                (before - after).abs() < 1e-3,
                "inner product changed from {before} to {after}"
            );

            let distance_before = squared_l2(&a, &b);
            let distance_after = squared_l2(&rotation.apply(&a), &rotation.apply(&b));
            assert!((distance_before - distance_after).abs() / distance_before < 1e-3);
        }
    }

    /// `apply_inverse` must actually invert `apply`, or the query and the data
    /// end up in different frames and every estimate is meaningless.
    #[test]
    fn the_inverse_undoes_the_rotation() {
        let rotation = Rotation::new(48, 23);
        let mut rng = Rng::new(29);
        for _ in 0..25 {
            let x: Vec<f32> = (0..48).map(|_| standard_normal(&mut rng)).collect();
            let round_trip = rotation.apply_inverse(&rotation.apply(&x));
            for (original, recovered) in x.iter().zip(round_trip.iter()) {
                assert!(
                    (original - recovered).abs() < 1e-3,
                    "{original} came back as {recovered}"
                );
            }
        }
    }

    /// The codes are meaningless without the rotation that produced them, so an
    /// index must be able to rebuild the exact same matrix from its seed.
    #[test]
    fn a_seed_reproduces_the_matrix() {
        let first = Rotation::new(32, 99);
        let second = Rotation::new(32, 99);
        assert_eq!(first.matrix, second.matrix);

        let different = Rotation::new(32, 100);
        assert_ne!(first.matrix, different.matrix);
    }

    /// A rotation should actually mix the coordinates; an identity matrix would
    /// pass the orthogonality test while providing none of the guarantee.
    #[test]
    fn the_rotation_actually_mixes_coordinates() {
        let rotation = Rotation::new(64, 5);
        let mut basis = vec![0.0f32; 64];
        basis[0] = 1.0;

        let rotated = rotation.apply(&basis);
        let significant = rotated.iter().filter(|v| v.abs() > 0.01).count();
        assert!(
            significant > 20,
            "a basis vector spread over only {significant} coordinates; this is \
             close to the identity and provides no randomisation"
        );
    }

    #[test]
    fn box_muller_produces_roughly_standard_normals() {
        let mut rng = Rng::new(31);
        let samples: Vec<f32> = (0..20_000).map(|_| standard_normal(&mut rng)).collect();

        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        let variance: f32 =
            samples.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / samples.len() as f32;

        assert!(mean.abs() < 0.05, "mean {mean} is not near 0");
        assert!(
            (variance - 1.0).abs() < 0.1,
            "variance {variance} is not near 1"
        );
    }

    #[test]
    #[should_panic(expected = "at least one dimension")]
    fn a_zero_dimensional_rotation_is_rejected() {
        Rotation::new(0, 1);
    }
}
