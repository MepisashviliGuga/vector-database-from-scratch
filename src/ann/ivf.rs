//! IVF: an inverted file index over RaBitQ-quantized residuals.
//!
//! This is the configuration paper 03 actually targets. It does two things at
//! once, and they are easy to conflate:
//!
//! 1. **Accuracy.** Each vector is encoded as a residual from *its own cluster's*
//!    centroid rather than a single global one. Residuals are then small and
//!    locally concentrated, so the same number of bits describes a much easier
//!    quantity. Measured on SIFT1M with one global centroid, recall@10 at 6.4x
//!    compression was 0.921 against the paper's >95% — this is the gap that
//!    closes.
//! 2. **Speed.** A query scans only the `nprobe` nearest clusters instead of
//!    every vector, so cost falls by roughly `nprobe / clusters`.
//!
//! Both matter, but only the first is a *correctness* claim. The second is a
//! recall/speed trade-off that `nprobe` moves along, and the tests below pin
//! both directions of it.
//!
//! # The unavoidable approximation
//!
//! Restricting a search to `nprobe` clusters means a true neighbour sitting just
//! across a cluster boundary can be missed no matter how good the quantizer is.
//! That loss is **independent of the bit width**, so recall has a ceiling set by
//! `nprobe` that more bits cannot raise. Reporting a recall figure without the
//! `nprobe` it came from is therefore meaningless, and every figure this module
//! produces carries it.
//!
//! # Training on a sample
//!
//! k-means over a million vectors costs `O(N·k·D)` per iteration — around
//! 1.3e11 operations at `k = 1000`, which is minutes per iteration scalar. Like
//! FAISS, this trains on a **sample** and then assigns the full set once. That
//! is a labelled deviation from training on everything; the sample is
//! deterministic from the seed, and [`IvfConfig::training_sample`] controls it.

use crate::ann::kmeans::{self, KMeans};
use crate::ann::rabitq::{Code, RaBitQ};
use crate::ann::Neighbor;
use crate::workload::Rng;

/// How to build the index.
#[derive(Debug, Clone, Copy)]
pub struct IvfConfig {
    /// Number of inverted lists. The usual rule of thumb is around `sqrt(N)`.
    pub clusters: usize,
    /// Bits per dimension for the RaBitQ codes.
    pub bits: u32,
    pub kmeans_iterations: usize,
    /// Vectors sampled to train k-means. `None` trains on everything, which is
    /// exact but `O(N·k·D)` per iteration.
    pub training_sample: Option<usize>,
    pub seed: u64,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            clusters: 256,
            bits: 5,
            kmeans_iterations: 25,
            training_sample: Some(50_000),
            seed: 0x5EED,
        }
    }
}

/// One inverted list: the members of a single cluster.
#[derive(Debug, Clone, Default)]
struct PostingList {
    ids: Vec<u32>,
    codes: Vec<Code>,
}

/// An IVF index over quantized residuals.
#[derive(Debug)]
pub struct IvfIndex {
    quantizer: RaBitQ,
    centroids: KMeans,
    lists: Vec<PostingList>,
    dimension: usize,
    vectors: usize,
}

impl IvfIndex {
    /// Build an index over `data`, a flat buffer of `dimension`-sized vectors.
    ///
    /// Ids are assigned by position, matching the SIFT ground-truth convention.
    ///
    /// # Panics
    ///
    /// If `dimension` is 0 or `data` is not a whole number of vectors.
    pub fn build(data: &[f32], dimension: usize, config: IvfConfig) -> Self {
        assert!(dimension > 0, "vectors need at least one dimension");
        assert_eq!(
            data.len() % dimension,
            0,
            "{} values do not divide into {dimension}-dimensional vectors",
            data.len()
        );
        let vectors = data.len() / dimension;
        assert!(vectors > 0, "cannot index an empty set");

        let training = training_set(data, dimension, vectors, &config);
        let centroids = kmeans::train(
            &training,
            dimension,
            config.clusters,
            config.kmeans_iterations,
            config.seed,
        );

        let quantizer = RaBitQ::new(dimension, config.bits, config.seed);
        let mut lists = vec![PostingList::default(); centroids.clusters()];

        for index in 0..vectors {
            let vector = &data[index * dimension..(index + 1) * dimension];
            let (cluster, _) = centroids.assign(vector);
            // The residual is taken against this cluster's centroid, which is
            // the entire point: a small, locally concentrated vector to encode.
            let centroid = centroids
                .centroid(cluster)
                .expect("assign returned a valid cluster");
            lists[cluster].ids.push(index as u32);
            lists[cluster].codes.push(quantizer.encode(vector, centroid));
        }

        Self {
            quantizer,
            centroids,
            lists,
            dimension,
            vectors,
        }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn len(&self) -> usize {
        self.vectors
    }

    pub fn is_empty(&self) -> bool {
        self.vectors == 0
    }

    pub fn clusters(&self) -> usize {
        self.lists.len()
    }

    /// Members per cluster, for checking how evenly the data divided.
    ///
    /// A badly skewed distribution means `nprobe` buys less than it appears to:
    /// probing 10 of 256 lists is not 4% of the data if one list holds half of
    /// it.
    pub fn list_sizes(&self) -> Vec<usize> {
        self.lists.iter().map(|list| list.ids.len()).collect()
    }

    /// Resident bytes: codes plus centroids.
    ///
    /// Codes are one byte per dimension here rather than bit-packed. See
    /// [`Self::packed_bytes`] for the figure comparable to paper 03.
    pub fn memory_bytes(&self) -> usize {
        let codes = self.vectors * self.quantizer.code_bytes();
        let centroids = self.centroids.centroids.len() * std::mem::size_of::<f32>();
        codes + centroids
    }

    /// Bytes if codes were bit-packed to `B` bits per dimension.
    pub fn packed_bytes(&self) -> usize {
        let codes = self.vectors * self.quantizer.packed_code_bytes();
        let centroids = self.centroids.centroids.len() * std::mem::size_of::<f32>();
        codes + centroids
    }

    /// The `k` nearest neighbours, searching the `nprobe` closest clusters.
    ///
    /// Returns fewer than `k` when the probed lists hold fewer vectors. Results
    /// are ordered by *estimated* distance, so their `distance` fields are
    /// estimates, not exact values.
    ///
    /// # Panics
    ///
    /// If `query` is the wrong length.
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<Neighbor> {
        assert_eq!(query.len(), self.dimension, "query has the wrong length");
        if k == 0 || nprobe == 0 {
            return Vec::new();
        }

        // A bounded max-heap of the best k, as the brute-force scan uses.
        // Collecting every candidate and sorting would cost O(n log n) in the
        // number *scanned* rather than O(n log k), which at a high `nprobe`
        // makes the sort rival the distance estimation itself.
        let mut best: std::collections::BinaryHeap<Neighbor> =
            std::collections::BinaryHeap::with_capacity(k + 1);

        for (cluster, _) in self.centroids.nearest_centroids(query, nprobe) {
            let centroid = self
                .centroids
                .centroid(cluster)
                .expect("nearest_centroids returned a valid cluster");
            // Prepared once per *cluster*, not per candidate: the rotation and
            // the offset-correction sum depend only on the query and the
            // centroid, so sharing them across the list is most of the saving.
            let prepared = self.quantizer.prepare_query(query, centroid);

            let list = &self.lists[cluster];
            for (&id, code) in list.ids.iter().zip(list.codes.iter()) {
                let candidate = Neighbor {
                    id: id as u64,
                    distance: self.quantizer.estimate_squared_distance(code, &prepared),
                };
                // Compare before making room: most candidates lose, and this
                // skips a push/pop pair for each of them.
                if best.len() == k {
                    if best.peek().is_some_and(|worst| candidate >= *worst) {
                        continue;
                    }
                    best.pop();
                }
                best.push(candidate);
            }
        }

        let mut results = best.into_vec();
        results.sort_unstable();
        results
    }

    /// Sizes of the lists a query would probe, for reporting how much of the
    /// dataset a search actually examines.
    pub fn search_probe_sizes(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        self.centroids
            .nearest_centroids(query, nprobe)
            .into_iter()
            .map(|(cluster, _)| self.lists[cluster].ids.len())
            .collect()
    }

    /// Search every cluster — the quantizer's accuracy with no IVF pruning loss.
    ///
    /// Useful for separating the two error sources: the difference between this
    /// and [`Self::search`] at a given `nprobe` is exactly what pruning cost.
    pub fn search_exhaustive(&self, query: &[f32], k: usize) -> Vec<Neighbor> {
        self.search(query, k, self.clusters())
    }
}

/// Vectors to train k-means on: either everything, or a deterministic sample.
fn training_set(data: &[f32], dimension: usize, vectors: usize, config: &IvfConfig) -> Vec<f32> {
    let Some(sample) = config.training_sample else {
        return data.to_vec();
    };
    if sample >= vectors {
        return data.to_vec();
    }

    // A stride rather than random draws: it costs nothing, is deterministic,
    // and cannot accidentally miss a contiguous region the way sampling with
    // replacement can.
    let mut rng = Rng::new(config.seed ^ 0xA5A5_A5A5);
    let offset = rng.below((vectors / sample.max(1)).max(1) as u64) as usize;
    let stride = (vectors / sample).max(1);

    let mut training = Vec::with_capacity(sample * dimension);
    let mut index = offset;
    while index < vectors && training.len() < sample * dimension {
        training.extend_from_slice(&data[index * dimension..(index + 1) * dimension]);
        index += stride;
    }
    training
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ann::squared_l2;

    fn synthetic(count: usize, dimension: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..count * dimension)
            .map(|_| rng.next_f64() as f32 * 2.0 - 1.0)
            .collect()
    }

    /// Exact neighbours, for measuring recall.
    fn exact(data: &[f32], dimension: usize, query: &[f32], k: usize) -> Vec<u32> {
        let count = data.len() / dimension;
        let mut scored: Vec<(f32, u32)> = (0..count)
            .map(|index| {
                (
                    squared_l2(&data[index * dimension..(index + 1) * dimension], query),
                    index as u32,
                )
            })
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        scored.into_iter().take(k).map(|(_, id)| id).collect()
    }

    fn recall(found: &[Neighbor], truth: &[u32]) -> f64 {
        let wanted: std::collections::HashSet<u32> = truth.iter().copied().collect();
        let hits = found
            .iter()
            .filter(|n| wanted.contains(&(n.id as u32)))
            .count();
        hits as f64 / wanted.len().max(1) as f64
    }

    fn config(clusters: usize, bits: u32) -> IvfConfig {
        IvfConfig {
            clusters,
            bits,
            kmeans_iterations: 25,
            training_sample: None,
            seed: 7,
        }
    }

    #[test]
    fn every_vector_lands_in_exactly_one_list() {
        let dimension = 16;
        let data = synthetic(500, dimension, 11);
        let index = IvfIndex::build(&data, dimension, config(16, 4));

        assert_eq!(index.len(), 500);
        assert_eq!(index.list_sizes().iter().sum::<usize>(), 500);

        let mut seen = std::collections::HashSet::new();
        for list in &index.lists {
            for &id in &list.ids {
                assert!(seen.insert(id), "id {id} appears in two lists");
            }
        }
        assert_eq!(seen.len(), 500);
    }

    /// The recall/speed knob. More probes means more of the data examined, so
    /// recall must rise monotonically towards the exhaustive figure.
    #[test]
    fn recall_rises_with_nprobe() {
        let dimension = 32;
        let data = synthetic(3000, dimension, 13);
        let index = IvfIndex::build(&data, dimension, config(32, 7));
        let queries = synthetic(30, dimension, 17);

        let mut previous = 0.0;
        for nprobe in [1usize, 2, 4, 8, 16, 32] {
            let mut total = 0.0;
            for q in 0..30 {
                let query = &queries[q * dimension..(q + 1) * dimension];
                let truth = exact(&data, dimension, query, 10);
                total += recall(&index.search(query, 10, nprobe), &truth);
            }
            let mean = total / 30.0;
            assert!(
                mean >= previous - 1e-9,
                "nprobe {nprobe} gave recall {mean:.4}, below {previous:.4}"
            );
            previous = mean;
        }
        assert!(
            previous > 0.85,
            "probing every cluster should recover most neighbours, got {previous:.4}"
        );
    }

    /// **The claim this module exists for.** Per-cluster centroids give smaller
    /// residuals than one global centroid, so the same bit width is more
    /// accurate. Both sides search every vector, isolating the centroid effect
    /// from the pruning effect.
    #[test]
    fn per_cluster_centroids_beat_a_single_global_one() {
        let dimension = 32;
        let data = synthetic(2000, dimension, 19);
        let queries = synthetic(40, dimension, 23);
        let bits = 3;

        // Many clusters: small, local residuals.
        let clustered = IvfIndex::build(&data, dimension, config(64, bits));
        // One cluster is exactly the single-global-centroid setting.
        let global = IvfIndex::build(&data, dimension, config(1, bits));

        let mut clustered_recall = 0.0;
        let mut global_recall = 0.0;
        for q in 0..40 {
            let query = &queries[q * dimension..(q + 1) * dimension];
            let truth = exact(&data, dimension, query, 10);
            clustered_recall += recall(&clustered.search_exhaustive(query, 10), &truth);
            global_recall += recall(&global.search_exhaustive(query, 10), &truth);
        }

        assert!(
            clustered_recall > global_recall,
            "clustered recall {:.4} did not beat global {:.4}; the residuals are \
             not smaller, which is the whole reason for IVF",
            clustered_recall / 40.0,
            global_recall / 40.0
        );
    }

    #[test]
    fn more_bits_give_better_recall_at_a_fixed_nprobe() {
        let dimension = 32;
        let data = synthetic(2000, dimension, 29);
        let queries = synthetic(30, dimension, 31);

        let mut previous = 0.0;
        for bits in [1u32, 2, 4, 6] {
            let index = IvfIndex::build(&data, dimension, config(16, bits));
            let mut total = 0.0;
            for q in 0..30 {
                let query = &queries[q * dimension..(q + 1) * dimension];
                let truth = exact(&data, dimension, query, 10);
                total += recall(&index.search(query, 10, 16), &truth);
            }
            let mean = total / 30.0;
            assert!(
                mean > previous,
                "B = {bits} gave recall {mean:.4}, not better than {previous:.4}"
            );
            previous = mean;
        }
    }

    /// Probing one cluster examines a fraction of the data, and that is where
    /// the speed comes from. Verified by counting candidates rather than timing.
    #[test]
    fn probing_fewer_clusters_examines_less_data() {
        let dimension = 16;
        let data = synthetic(2000, dimension, 37);
        let index = IvfIndex::build(&data, dimension, config(50, 4));

        let sizes = index.list_sizes();
        let total: usize = sizes.iter().sum();
        assert_eq!(total, 2000);

        let query = &data[0..dimension];
        let probed: usize = index
            .centroids
            .nearest_centroids(query, 5)
            .iter()
            .map(|&(cluster, _)| sizes[cluster])
            .sum();
        assert!(
            probed < total / 3,
            "probing 5 of 50 lists touched {probed} of {total} vectors"
        );
    }

    #[test]
    fn results_are_ordered_by_estimated_distance() {
        let dimension = 16;
        let data = synthetic(800, dimension, 41);
        let index = IvfIndex::build(&data, dimension, config(16, 5));
        let query = &data[0..dimension];

        let results = index.search(query, 20, 8);
        assert!(
            results.windows(2).all(|pair| pair[0].distance <= pair[1].distance),
            "results must ascend by estimate: {results:?}"
        );
    }

    /// Training on a sample is a labelled deviation; it must still produce a
    /// usable index rather than degenerate clusters.
    #[test]
    fn sampled_training_still_indexes_everything() {
        let dimension = 16;
        let data = synthetic(4000, dimension, 43);
        let sampled = IvfIndex::build(
            &data,
            dimension,
            IvfConfig {
                clusters: 32,
                bits: 4,
                kmeans_iterations: 20,
                training_sample: Some(500),
                seed: 47,
            },
        );

        assert_eq!(sampled.len(), 4000, "every vector must still be indexed");
        assert_eq!(sampled.list_sizes().iter().sum::<usize>(), 4000);

        let non_empty = sampled.list_sizes().iter().filter(|&&s| s > 0).count();
        assert!(
            non_empty > 16,
            "only {non_empty} of 32 lists were used; sampling collapsed the clustering"
        );
    }

    #[test]
    fn memory_reflects_the_bit_width() {
        let dimension = 128;
        let data = synthetic(1000, dimension, 53);

        let narrow = IvfIndex::build(&data, dimension, config(8, 2));
        let wide = IvfIndex::build(&data, dimension, config(8, 8));

        assert!(narrow.packed_bytes() < wide.packed_bytes());
        // 1000 vectors x 128 dims at 2 bits = 32 KiB of codes, plus centroids.
        assert_eq!(narrow.packed_bytes(), 1000 * 32 + 8 * 128 * 4);

        let raw = 1000 * dimension * 4;
        assert!(
            (raw as f64 / narrow.packed_bytes() as f64) > 10.0,
            "2-bit codes should compress by more than 10x"
        );
    }

    #[test]
    fn degenerate_searches_return_nothing() {
        let dimension = 8;
        let data = synthetic(100, dimension, 59);
        let index = IvfIndex::build(&data, dimension, config(4, 4));
        let query = &data[0..dimension];

        assert!(index.search(query, 0, 4).is_empty(), "k = 0");
        assert!(index.search(query, 10, 0).is_empty(), "nprobe = 0");
    }

    #[test]
    fn asking_for_more_than_a_probe_holds_returns_what_exists() {
        let dimension = 8;
        let data = synthetic(100, dimension, 61);
        let index = IvfIndex::build(&data, dimension, config(20, 4));
        let query = &data[0..dimension];

        let results = index.search(query, 1000, 1);
        assert!(results.len() < 100, "one list cannot hold every vector");
        assert!(!results.is_empty());
    }

    #[test]
    fn building_is_deterministic() {
        let dimension = 16;
        let data = synthetic(500, dimension, 67);
        let first = IvfIndex::build(&data, dimension, config(8, 4));
        let second = IvfIndex::build(&data, dimension, config(8, 4));

        assert_eq!(first.list_sizes(), second.list_sizes());
        let query = &data[0..dimension];
        assert_eq!(first.search(query, 10, 4), second.search(query, 10, 4));
    }

    #[test]
    #[should_panic(expected = "query has the wrong length")]
    fn a_wrong_sized_query_panics() {
        let data = synthetic(50, 8, 71);
        let index = IvfIndex::build(&data, 8, config(4, 4));
        index.search(&[1.0, 2.0], 5, 2);
    }
}
