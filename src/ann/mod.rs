//! Approximate (and exact) nearest-neighbour search.
//!
//! Built bottom-up in the order the papers depend on each other:
//!
//! 1. [`brute_force`] — exact k-NN by scanning every vector. Slow by design.
//!    It is the **ground truth**: recall for every later index is measured
//!    against it, so if it is wrong, every subsequent number is wrong and
//!    nothing downstream will reveal it.
//! 2. Quantization (paper 03, extended RaBitQ) — not started.
//! 3. Graph index (paper 04, SymphonyQG) — not started.
//!
//! # Distances
//!
//! Squared Euclidean throughout. The square root is monotonic, so omitting it
//! changes no ordering and no recall figure, and it removes a `sqrt` from the
//! innermost loop of every scan. Any *reported* distance must say which it is;
//! within the engine it is always squared.

pub mod brute_force;
pub mod deg;
pub mod fvecs;
pub mod graph;
pub mod ivf;
pub mod kmeans;
pub mod rabitq;
pub mod rotation;
pub mod symphony;

pub use brute_force::{BruteForceIndex, Neighbor};
pub use deg::{AlphaSet, HybridDistance};
pub use graph::{GraphConfig, GraphIndex};
pub use ivf::{IvfConfig, IvfIndex};
pub use kmeans::KMeans;
pub use rabitq::{Code, RaBitQ};
pub use rotation::Rotation;
pub use symphony::{SymphonyConfig, SymphonyIndex};

/// Squared Euclidean distance between two equal-length vectors.
///
/// # Panics
///
/// If the lengths differ. A silent truncation here would produce plausible but
/// wrong ground truth, which is the worst possible failure for an oracle.
pub fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "cannot compare a {}-dimensional vector with a {}-dimensional one",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

/// Inner product, for the maximum-inner-product formulation some datasets use.
///
/// # Panics
///
/// If the lengths differ.
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Fraction of the true `k` nearest neighbours that `found` contains.
///
/// The standard ANN measure. Both sides are treated as *sets*: an index that
/// returns the right neighbours in the wrong order has full recall, because
/// recall@k asks what was retrieved, not how it was ranked.
///
/// `truth` is truncated to `k` first, so passing a longer ground-truth list
/// (as the SIFT files supply — 100 neighbours per query) measures recall@k
/// rather than silently inflating the denominator.
pub fn recall_at_k(found: &[Neighbor], truth: &[u32], k: usize) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let wanted: std::collections::HashSet<u32> = truth.iter().take(k).copied().collect();
    if wanted.is_empty() {
        return 1.0;
    }
    let hits = found
        .iter()
        .take(k)
        .filter(|neighbour| wanted.contains(&(neighbour.id as u32)))
        .count();
    hits as f64 / wanted.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squared_l2_matches_hand_computed_values() {
        assert_eq!(squared_l2(&[0.0, 0.0], &[3.0, 4.0]), 25.0);
        assert_eq!(squared_l2(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
        assert_eq!(squared_l2(&[-1.0], &[1.0]), 4.0);
        assert_eq!(squared_l2(&[], &[]), 0.0);
    }

    /// Omitting the square root must not change any ordering, which is the
    /// entire justification for omitting it.
    #[test]
    fn squared_distance_orders_the_same_as_euclidean() {
        let query = [0.0f32, 0.0];
        let points = [[1.0f32, 1.0], [3.0, 0.0], [0.5, 0.5], [2.0, 2.0]];

        let mut by_squared: Vec<usize> = (0..points.len()).collect();
        by_squared.sort_by(|&a, &b| {
            squared_l2(&query, &points[a]).total_cmp(&squared_l2(&query, &points[b]))
        });

        let mut by_euclidean: Vec<usize> = (0..points.len()).collect();
        by_euclidean.sort_by(|&a, &b| {
            squared_l2(&query, &points[a])
                .sqrt()
                .total_cmp(&squared_l2(&query, &points[b]).sqrt())
        });

        assert_eq!(by_squared, by_euclidean);
    }

    #[test]
    #[should_panic(expected = "cannot compare a 2-dimensional vector")]
    fn mismatched_dimensions_panic_rather_than_truncate() {
        squared_l2(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn inner_product_matches_hand_computed_values() {
        assert_eq!(inner_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
        assert_eq!(inner_product(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    }

    // -----------------------------------------------------------------
    // recall@k
    // -----------------------------------------------------------------

    fn neighbours(ids: &[u64]) -> Vec<Neighbor> {
        ids.iter()
            .enumerate()
            .map(|(rank, &id)| Neighbor {
                id,
                distance: rank as f32,
            })
            .collect()
    }

    #[test]
    fn perfect_retrieval_is_full_recall() {
        let found = neighbours(&[7, 3, 9]);
        assert_eq!(recall_at_k(&found, &[7, 3, 9], 3), 1.0);
    }

    /// Recall asks what was retrieved, not in what order.
    #[test]
    fn ordering_does_not_affect_recall() {
        let found = neighbours(&[9, 7, 3]);
        assert_eq!(recall_at_k(&found, &[7, 3, 9], 3), 1.0);
    }

    #[test]
    fn partial_retrieval_is_counted_proportionally() {
        let found = neighbours(&[7, 100, 9]);
        assert!((recall_at_k(&found, &[7, 3, 9], 3) - 2.0 / 3.0).abs() < 1e-9);

        let found = neighbours(&[100, 200, 300]);
        assert_eq!(recall_at_k(&found, &[7, 3, 9], 3), 0.0);
    }

    /// The SIFT ground-truth files hold 100 neighbours per query. Measuring
    /// recall@10 against all 100 would divide by the wrong denominator and
    /// report an implausibly low figure.
    #[test]
    fn a_longer_ground_truth_list_is_truncated_to_k() {
        let truth: Vec<u32> = (0..100).collect();
        let found = neighbours(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            recall_at_k(&found, &truth, 10),
            1.0,
            "the first ten were all retrieved, so recall@10 is 1.0"
        );
    }

    /// Only the first `k` results count. Returning a longer list cannot buy
    /// recall — here two junk results push two true neighbours out of the top 3,
    /// so recall falls from 1.0 to 1/3 even though all three were "returned".
    #[test]
    fn returning_more_than_k_does_not_inflate_recall() {
        let padded = neighbours(&[100, 200, 7, 3, 9]);
        assert!((recall_at_k(&padded, &[7, 3, 9], 3) - 1.0 / 3.0).abs() < 1e-9);

        let unpadded = neighbours(&[7, 3, 9]);
        assert_eq!(recall_at_k(&unpadded, &[7, 3, 9], 3), 1.0);
    }

    #[test]
    fn degenerate_inputs_are_handled() {
        assert_eq!(recall_at_k(&[], &[1, 2, 3], 0), 1.0);
        assert_eq!(recall_at_k(&[], &[], 5), 1.0);
        assert_eq!(recall_at_k(&[], &[1, 2, 3], 3), 0.0);
    }
}
