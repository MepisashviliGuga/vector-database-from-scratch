//! Exact k-nearest-neighbour search by scanning every vector.
//!
//! Deliberately the slowest possible index. Its job is to be **right**: every
//! recall figure this project reports is measured against it, so an error here
//! propagates into every ANN result and nothing downstream would catch it. It is
//! also the naive baseline the quantization and graph indexes have to beat.
//!
//! # Layout
//!
//! Vectors are stored in one flat `Vec<f32>` rather than a `Vec<Vec<f32>>`. A
//! scan then walks contiguous memory instead of chasing a pointer per vector,
//! which for the one operation this type performs is the difference between
//! streaming and thrashing.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::squared_l2;

/// A search result: an id and its **squared** distance to the query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    pub id: u64,
    /// Squared Euclidean distance. See the module docs on why not the root.
    pub distance: f32,
}

impl Eq for Neighbor {}

impl Ord for Neighbor {
    /// Orders by distance, then by id.
    ///
    /// Uses `total_cmp` rather than `partial_cmp`: a `NaN` distance from a
    /// malformed vector would otherwise make the ordering non-transitive and
    /// could panic inside the heap. Breaking ties by id makes results
    /// deterministic, which matters when a dataset contains duplicate vectors —
    /// otherwise recall would wobble between runs for reasons unrelated to the
    /// index.
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Every vector, scanned on every query.
#[derive(Debug, Clone, Default)]
pub struct BruteForceIndex {
    dimension: usize,
    /// All vectors end to end: vector `i` occupies `[i·d, (i+1)·d)`.
    data: Vec<f32>,
    ids: Vec<u64>,
}

impl BruteForceIndex {
    /// # Panics
    ///
    /// If `dimension` is 0.
    pub fn new(dimension: usize) -> Self {
        assert!(dimension > 0, "vectors need at least one dimension");
        Self {
            dimension,
            data: Vec::new(),
            ids: Vec::new(),
        }
    }

    /// Build from a flat buffer of `count · dimension` values.
    ///
    /// Ids are assigned by position, matching how the SIFT and GIST ground-truth
    /// files identify vectors.
    ///
    /// # Panics
    ///
    /// If `data.len()` is not a multiple of `dimension`.
    pub fn from_flat(dimension: usize, data: Vec<f32>) -> Self {
        assert!(dimension > 0, "vectors need at least one dimension");
        assert_eq!(
            data.len() % dimension,
            0,
            "{} values do not divide into {dimension}-dimensional vectors",
            data.len()
        );
        let count = data.len() / dimension;
        Self {
            dimension,
            data,
            ids: (0..count as u64).collect(),
        }
    }

    /// # Panics
    ///
    /// If `vector` is the wrong length.
    pub fn add(&mut self, id: u64, vector: &[f32]) {
        assert_eq!(
            vector.len(),
            self.dimension,
            "expected a {}-dimensional vector, got {}",
            self.dimension,
            vector.len()
        );
        self.data.extend_from_slice(vector);
        self.ids.push(id);
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The vector stored at `index`.
    pub fn vector(&self, index: usize) -> Option<&[f32]> {
        let start = index.checked_mul(self.dimension)?;
        self.data.get(start..start + self.dimension)
    }

    /// Bytes of vector data held, for comparing against a quantized index.
    pub fn data_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// The exact `k` nearest neighbours, nearest first.
    ///
    /// Keeps a bounded max-heap of the best `k` seen so far, so the scan costs
    /// `O(n log k)` rather than sorting all `n`. Returns fewer than `k` only
    /// when the index holds fewer than `k` vectors.
    ///
    /// # Panics
    ///
    /// If `query` is the wrong length.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<Neighbor> {
        assert_eq!(
            query.len(),
            self.dimension,
            "expected a {}-dimensional query, got {}",
            self.dimension,
            query.len()
        );
        if k == 0 || self.is_empty() {
            return Vec::new();
        }

        let mut best: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(k + 1);
        for (index, &id) in self.ids.iter().enumerate() {
            let start = index * self.dimension;
            let candidate = &self.data[start..start + self.dimension];
            let distance = squared_l2(query, candidate);

            // Compare before allocating heap space: on a large index the vast
            // majority of candidates lose, and this skips a push/pop pair for
            // each of them.
            if best.len() == k {
                if let Some(worst) = best.peek() {
                    let contender = Neighbor { id, distance };
                    if contender >= *worst {
                        continue;
                    }
                }
                best.pop();
            }
            best.push(Neighbor { id, distance });
        }

        let mut results = best.into_vec();
        results.sort_unstable();
        results
    }

    /// Ground truth for a batch of queries: `search` for each, ids only.
    ///
    /// This is the shape the SIFT and GIST ground-truth files take, so recall
    /// can be computed against either this or the published answers.
    pub fn ground_truth(&self, queries: &[Vec<f32>], k: usize) -> Vec<Vec<u32>> {
        queries
            .iter()
            .map(|query| {
                self.search(query, k)
                    .into_iter()
                    .map(|neighbour| neighbour.id as u32)
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random vectors, so failures are reproducible.
    fn synthetic(count: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = crate::workload::Rng::new(seed);
        (0..count)
            .map(|_| {
                (0..dimension)
                    .map(|_| rng.next_f64() as f32 * 2.0 - 1.0)
                    .collect()
            })
            .collect()
    }

    /// An independent, obviously-correct implementation to check against.
    ///
    /// Sorting everything is exactly what the heap exists to avoid, which is
    /// what makes it a useful second opinion.
    fn reference_search(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u64> {
        let mut scored: Vec<(f32, u64)> = vectors
            .iter()
            .enumerate()
            .map(|(index, vector)| (squared_l2(query, vector), index as u64))
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        scored.into_iter().take(k).map(|(_, id)| id).collect()
    }

    fn build(vectors: &[Vec<f32>]) -> BruteForceIndex {
        let mut index = BruteForceIndex::new(vectors[0].len());
        for (id, vector) in vectors.iter().enumerate() {
            index.add(id as u64, vector);
        }
        index
    }

    #[test]
    fn finds_the_obvious_nearest_neighbour() {
        let mut index = BruteForceIndex::new(2);
        index.add(10, &[0.0, 0.0]);
        index.add(20, &[1.0, 0.0]);
        index.add(30, &[5.0, 5.0]);

        let results = index.search(&[0.9, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 20);
        assert!((results[0].distance - 0.01).abs() < 1e-6);
    }

    #[test]
    fn results_come_back_nearest_first() {
        let mut index = BruteForceIndex::new(1);
        for (id, x) in [(1u64, 5.0f32), (2, 1.0), (3, 3.0), (4, 0.0)] {
            index.add(id, &[x]);
        }

        let results = index.search(&[0.0], 4);
        assert_eq!(
            results.iter().map(|n| n.id).collect::<Vec<_>>(),
            vec![4, 2, 3, 1]
        );
        assert!(
            results.windows(2).all(|pair| pair[0].distance <= pair[1].distance),
            "distances must ascend: {results:?}"
        );
    }

    /// The property that matters: this is the oracle, so it must agree exactly
    /// with an independent implementation on non-trivial data.
    #[test]
    fn agrees_with_an_independent_implementation() {
        let vectors = synthetic(500, 16, 7);
        let queries = synthetic(50, 16, 8);
        let index = build(&vectors);

        for query in &queries {
            for k in [1usize, 5, 10, 50] {
                let found: Vec<u64> = index.search(query, k).into_iter().map(|n| n.id).collect();
                let expected = reference_search(&vectors, query, k);
                assert_eq!(found, expected, "disagreement at k = {k}");
            }
        }
    }

    /// Duplicate vectors make distances tie. Without a deterministic tiebreak,
    /// recall would wobble between runs for reasons unrelated to the index.
    #[test]
    fn ties_are_broken_deterministically_by_id() {
        let mut index = BruteForceIndex::new(2);
        for id in [5u64, 1, 9, 3] {
            index.add(id, &[1.0, 1.0]);
        }

        let first: Vec<u64> = index.search(&[0.0, 0.0], 3).into_iter().map(|n| n.id).collect();
        let second: Vec<u64> = index.search(&[0.0, 0.0], 3).into_iter().map(|n| n.id).collect();
        assert_eq!(first, second);
        assert_eq!(first, vec![1, 3, 5], "ties resolve by ascending id");
    }

    #[test]
    fn asking_for_more_than_exists_returns_everything() {
        let mut index = BruteForceIndex::new(1);
        index.add(1, &[0.0]);
        index.add(2, &[1.0]);

        let results = index.search(&[0.0], 100);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn degenerate_searches_return_nothing() {
        let index = BruteForceIndex::new(3);
        assert!(index.search(&[0.0, 0.0, 0.0], 5).is_empty(), "empty index");

        let mut populated = BruteForceIndex::new(3);
        populated.add(1, &[1.0, 2.0, 3.0]);
        assert!(populated.search(&[0.0, 0.0, 0.0], 0).is_empty(), "k = 0");
    }

    #[test]
    fn a_vector_is_its_own_nearest_neighbour() {
        let vectors = synthetic(200, 8, 11);
        let index = build(&vectors);

        for (id, vector) in vectors.iter().enumerate() {
            let nearest = index.search(vector, 1);
            assert_eq!(nearest[0].id, id as u64);
            assert!(nearest[0].distance < 1e-9, "a vector is zero distance from itself");
        }
    }

    #[test]
    fn a_flat_buffer_builds_the_same_index() {
        let vectors = synthetic(100, 4, 13);
        let flat: Vec<f32> = vectors.iter().flatten().copied().collect();

        let from_flat = BruteForceIndex::from_flat(4, flat);
        let from_adds = build(&vectors);

        assert_eq!(from_flat.len(), from_adds.len());
        assert_eq!(from_flat.dimension(), 4);
        let query = &vectors[42];
        assert_eq!(from_flat.search(query, 5), from_adds.search(query, 5));
    }

    #[test]
    fn stored_vectors_can_be_read_back() {
        let vectors = synthetic(20, 5, 17);
        let index = build(&vectors);
        for (position, vector) in vectors.iter().enumerate() {
            assert_eq!(index.vector(position), Some(vector.as_slice()));
        }
        assert_eq!(index.vector(20), None);
        assert_eq!(index.data_bytes(), 20 * 5 * 4);
    }

    #[test]
    fn ground_truth_matches_per_query_search() {
        let vectors = synthetic(300, 8, 19);
        let queries = synthetic(10, 8, 23);
        let index = build(&vectors);

        let truth = index.ground_truth(&queries, 10);
        assert_eq!(truth.len(), 10);
        for (query, expected) in queries.iter().zip(truth.iter()) {
            let direct: Vec<u32> = index
                .search(query, 10)
                .into_iter()
                .map(|n| n.id as u32)
                .collect();
            assert_eq!(&direct, expected);
        }
    }

    /// Recall of the oracle against itself must be exactly 1. If this ever
    /// fails, every recall figure in the project is suspect.
    #[test]
    fn the_oracle_has_perfect_recall_against_itself() {
        use crate::ann::recall_at_k;

        let vectors = synthetic(400, 12, 29);
        let queries = synthetic(25, 12, 31);
        let index = build(&vectors);
        let truth = index.ground_truth(&queries, 10);

        for (query, expected) in queries.iter().zip(truth.iter()) {
            let found = index.search(query, 10);
            assert_eq!(recall_at_k(&found, expected, 10), 1.0);
        }
    }

    #[test]
    #[should_panic(expected = "expected a 3-dimensional query")]
    fn a_wrong_sized_query_panics() {
        let index = BruteForceIndex::new(3);
        index.search(&[1.0, 2.0], 1);
    }

    #[test]
    #[should_panic(expected = "do not divide into")]
    fn a_ragged_flat_buffer_is_rejected() {
        BruteForceIndex::from_flat(3, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
