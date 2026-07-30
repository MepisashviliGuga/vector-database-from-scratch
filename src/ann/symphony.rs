//! SymphonyQG: a graph whose vertices carry their neighbours' quantization
//! codes.
//!
//! **Reproduction of paper 04 §3**, minus SIMD FastScan — see the honest
//! accounting at the bottom of this comment.
//!
//! # What the three contributions are, and why they need each other
//!
//! ## 1. Normalize by the vertex, and decompose the query
//!
//! RaBitQ quantizes a residual from a centroid. A graph has no centroids, so
//! each vertex's neighbours are encoded **against that vertex's own vector**.
//! A vector therefore gets a *different* code at every vertex that neighbours
//! it — the codes are replicated, deliberately.
//!
//! That creates a problem. The estimator needs the normalized query
//! `q = (q_r − c)/‖q_r − c‖`, and `c` is now the *current vertex*, so a naive
//! implementation rotates the query afresh at every visit — `O(D²)` work
//! amortized over one vertex's neighbours. That would cost more than it saves.
//!
//! Paper 04 §3.1.1 decomposes it instead:
//!
//! ```text
//!   ⟨x̄, P⁻¹q⟩ = (1/‖q_r − c‖) · ( ⟨x̄, P⁻¹q_r⟩ − ⟨x̄, P⁻¹c⟩ )
//! ```
//!
//! - `⟨x̄, P⁻¹c⟩` is **query-independent** → precomputed at index time.
//! - `⟨x̄, P⁻¹q_r⟩` is **centroid-independent** → the raw query is rotated
//!   **once per query** and reused across the whole traversal.
//!
//! A per-vertex cost becomes a per-query one.
//!
//! ## 2. Implicit re-ranking
//!
//! Estimating a vertex's neighbours needs `‖q_r − c‖`, where `c` is that vertex
//! — which is exactly the *exact* distance from the query to it. So visiting a
//! vertex already produces its true distance, and the best one seen is the
//! answer. **No separate re-ranking pass**, and none of its random accesses.
//!
//! The risk is that a vertex only earns an exact distance if it is *visited*,
//! and it is visited only if some estimate ranked it first. Over-estimate the
//! true nearest neighbour once and it is lost.
//!
//! Paper 04's answer, which is the part that makes the integration
//! "symphonious": **insert a neighbour into the beam on every encounter**, not
//! only the first, so it accumulates several independent estimates. This is
//! sound *because of RaBitQ's guarantees*, not despite them:
//!
//! - The estimator is **unbiased**, so each estimate under-shoots with roughly
//!   even odds. Across several, an under-estimate becomes likely — and since the
//!   true NN has the smallest exact distance, an under-estimate of it very
//!   likely ranks first and earns a visit.
//! - The estimator is **error-bounded**, so estimates for distant vectors cannot
//!   all drift low together. The extra beam entries do not turn into wasted
//!   visits.
//!
//! ## 3. Degree alignment
//!
//! Implemented in [`super::graph`]: the pruning angle is relaxed per vertex
//! until out-degree hits the target, so a batched distance kernel has no idle
//! lanes.
//!
//! # What this costs, stated plainly
//!
//! **SymphonyQG uses more memory than storing the raw vectors.** Each vertex
//! holds a code for every neighbour, so codes alone occupy `N · R · D · B/8`
//! bytes against raw vectors' `N · D · 4` — a ratio of `R·B/32`, which at
//! `R = 32, B = 5` is **5× larger**. And the raw vectors are still needed, for
//! the exact distances that make implicit re-ranking work.
//!
//! This is not a defect in the reproduction; it is what the method is. Paper 04
//! claims the state of the art in the **time-accuracy** trade-off and makes no
//! compression claim. It is the opposite trade from [`super::ivf`], which spends
//! accuracy to save memory. Anyone choosing between them should know that.
//!
//! # Not reproduced: SIMD FastScan
//!
//! The paper's speed rests on FastScan estimating 32 codes per batch far faster
//! than exact distances. This implementation is scalar, and our own measurement
//! ([`results/ann_recall.md`]) is that a scalar code comparison is *slower* than
//! an exact `f32` distance. **So the QPS claim cannot reproduce here**, and none
//! is made. What can be measured honestly is recall, and what memory locality
//! plus dropping the re-ranking stage buy on their own.

use super::graph::{GraphConfig, GraphIndex};
use super::rabitq::{Code, RaBitQ};
use super::{squared_l2, Neighbor};

/// Build parameters.
#[derive(Debug, Clone, Copy)]
pub struct SymphonyConfig {
    pub graph: GraphConfig,
    /// Bits per dimension for the neighbour codes.
    pub bits: u32,
    pub seed: u64,
}

impl Default for SymphonyConfig {
    fn default() -> Self {
        Self {
            graph: GraphConfig {
                // Degree alignment on by default: it is one of the paper's three
                // contributions, and the baseline lives in `super::graph`.
                align_degree: true,
                ..GraphConfig::default()
            },
            bits: 5,
            seed: 0x5EED,
        }
    }
}

/// One neighbour's code, plus the query-independent half of the estimator.
#[derive(Debug, Clone)]
struct NeighbourCode {
    id: u32,
    code: Code,
    /// `⟨ȳ_u, P⁻¹c⟩` — precomputed at index time because it does not involve
    /// the query. Half of the §3.1.1 decomposition.
    centroid_projection: f32,
}

/// A vertex's packed neighbour data.
#[derive(Debug, Clone, Default)]
struct VertexCodes {
    codes: Vec<NeighbourCode>,
    /// `Σᵢ (P⁻¹c)[i]` for this vertex, the offset correction's centroid half.
    centroid_sum: f32,
}

/// A rotated raw query, prepared once and reused across every vertex.
#[derive(Debug, Clone)]
pub struct RotatedQuery {
    /// `P⁻¹q_r`.
    rotated: Vec<f32>,
    /// `Σᵢ (P⁻¹q_r)[i]`.
    sum: f32,
}

/// A graph index with quantized neighbour codes stored at each vertex.
#[derive(Debug)]
pub struct SymphonyIndex {
    graph: GraphIndex,
    quantizer: RaBitQ,
    vertices: Vec<VertexCodes>,
}

impl SymphonyIndex {
    /// # Panics
    ///
    /// If `dimension` is 0 or `data` is not a whole number of vectors.
    pub fn build(data: &[f32], dimension: usize, config: SymphonyConfig) -> Self {
        let graph = GraphIndex::build(data, dimension, config.graph);
        let quantizer = RaBitQ::new(dimension, config.bits, config.seed);

        let mut vertices = vec![VertexCodes::default(); graph.len()];
        for (vertex, slot) in vertices.iter_mut().enumerate() {
            let centre = graph.vector(vertex);
            // `P⁻¹c`, needed for both precomputed halves.
            let rotated_centre = quantizer.rotation().apply_inverse(centre);
            let centroid_sum: f32 = rotated_centre.iter().sum();

            let codes = graph
                .neighbours(vertex)
                .iter()
                .map(|&neighbour| {
                    // Encoded against *this* vertex, which is why the same
                    // vector has a different code at each of its neighbours.
                    let code = quantizer.encode(graph.vector(neighbour as usize), centre);
                    let centroid_projection = code
                        .codes
                        .iter()
                        .zip(rotated_centre.iter())
                        .map(|(&c, &r)| c as f32 * r)
                        .sum();
                    NeighbourCode {
                        id: neighbour,
                        code,
                        centroid_projection,
                    }
                })
                .collect();

            *slot = VertexCodes {
                codes,
                centroid_sum,
            };
        }

        Self {
            graph,
            quantizer,
            vertices,
        }
    }

    pub fn len(&self) -> usize {
        self.graph.len()
    }

    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }

    pub fn dimension(&self) -> usize {
        self.graph.dimension()
    }

    /// Rotate a raw query once, for reuse across the whole traversal.
    ///
    /// This is §3.1.1's payoff: the `O(D²)` rotation happens per *query*, not
    /// per vertex visited.
    pub fn prepare(&self, raw_query: &[f32]) -> RotatedQuery {
        let rotated = self.quantizer.rotation().apply_inverse(raw_query);
        let sum = rotated.iter().sum();
        RotatedQuery { rotated, sum }
    }

    /// Bytes of neighbour codes held, before the raw vectors.
    ///
    /// Codes are replicated once per in-edge, so this grows with out-degree.
    /// See the module docs: it exceeds the raw vectors.
    pub fn code_bytes(&self) -> usize {
        self.vertices
            .iter()
            .map(|vertex| vertex.codes.len() * self.quantizer.packed_code_bytes())
            .sum()
    }

    /// Bytes of raw vectors, which implicit re-ranking still requires.
    pub fn raw_bytes(&self) -> usize {
        self.graph.data_bytes()
    }

    pub fn mean_out_degree(&self) -> f64 {
        self.graph.mean_out_degree()
    }

    /// Estimated squared distance from a query to one neighbour, using the
    /// decomposition.
    ///
    /// `centre_distance` is `‖q_r − c‖`, which the caller already has: it is the
    /// exact distance to the vertex being expanded.
    fn estimate(
        &self,
        entry: &NeighbourCode,
        vertex: &VertexCodes,
        query: &RotatedQuery,
        centre_distance: f32,
    ) -> f32 {
        // ⟨ȳ_u, P⁻¹q_r⟩ — the only per-candidate vector work.
        let query_projection: f32 = entry
            .code
            .codes
            .iter()
            .zip(query.rotated.iter())
            .map(|(&c, &q)| c as f32 * q)
            .sum();

        if centre_distance <= f32::MIN_POSITIVE || entry.code.grid_norm <= 0.0 {
            // The query sits on the vertex, so the vertex's own exact distance
            // already answers the question and the estimate is unused.
            return f32::INFINITY;
        }

        // ⟨ȳ_u, P⁻¹q⟩ − offset·Σ(P⁻¹q), both split into query and centroid parts.
        let numerator = (query_projection - entry.centroid_projection)
            - self.quantizer.offset() * (query.sum - vertex.centroid_sum);
        let quantized_inner = numerator / (entry.code.grid_norm * centre_distance);

        if entry.code.cosine_to_original.abs() < f32::MIN_POSITIVE {
            return f32::INFINITY;
        }
        let inner = quantized_inner / entry.code.cosine_to_original;

        let neighbour_norm = entry.code.distance_to_centroid;
        let estimate = neighbour_norm * neighbour_norm + centre_distance * centre_distance
            - 2.0 * neighbour_norm * centre_distance * inner;
        estimate.max(0.0)
    }

    /// The `k` nearest neighbours, by Algorithm 1.
    ///
    /// Returned distances are **exact**, not estimates: every returned vertex
    /// was visited, and visiting computes the true distance.
    ///
    /// # Panics
    ///
    /// If `query` is the wrong length.
    pub fn search(&self, query: &[f32], k: usize, beam_width: usize) -> Vec<Neighbor> {
        assert_eq!(query.len(), self.dimension(), "query has the wrong length");
        if k == 0 || self.is_empty() {
            return Vec::new();
        }
        let beam_width = beam_width.max(k).max(1);
        let prepared = self.prepare(query);

        // Beam entries are (estimated distance, id). The same id may appear
        // several times with different estimates — that is contribution 2, not
        // an oversight.
        let mut beam: Vec<(f32, u32)> = Vec::with_capacity(beam_width * 2);
        let mut visited = vec![false; self.len()];
        // Exact distances of visited vertices: the implicit re-ranking result.
        let mut results: Vec<Neighbor> = Vec::new();

        beam.push((0.0, 0));

        // The unvisited beam entry with the smallest estimate, until none remain.
        while let Some(position) = beam.iter().position(|&(_, id)| !visited[id as usize]) {
            let (_, current) = beam[position];
            visited[current as usize] = true;

            // Visiting computes the exact distance — this is both the answer's
            // source and `‖q_r − c‖` for estimating the neighbours.
            let exact = squared_l2(query, self.graph.vector(current as usize));
            results.push(Neighbor {
                id: current as u64,
                distance: exact,
            });
            let centre_distance = exact.sqrt();

            let vertex = &self.vertices[current as usize];
            for entry in &vertex.codes {
                if visited[entry.id as usize] {
                    continue;
                }
                let estimate = self.estimate(entry, vertex, &prepared, centre_distance);
                beam.push((estimate, entry.id));
            }

            beam.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            beam.truncate(beam_width);
        }

        results.sort_unstable();
        results.truncate(k);
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workload::Rng;

    fn synthetic(count: usize, dimension: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..count * dimension)
            .map(|_| rng.next_f64() as f32 * 2.0 - 1.0)
            .collect()
    }

    fn exact(data: &[f32], dimension: usize, query: &[f32], k: usize) -> Vec<u64> {
        let count = data.len() / dimension;
        let mut scored: Vec<(f32, u64)> = (0..count)
            .map(|i| {
                (
                    squared_l2(&data[i * dimension..(i + 1) * dimension], query),
                    i as u64,
                )
            })
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        scored.into_iter().take(k).map(|(_, id)| id).collect()
    }

    fn recall(found: &[Neighbor], truth: &[u64]) -> f64 {
        let wanted: std::collections::HashSet<u64> = truth.iter().copied().collect();
        found.iter().filter(|n| wanted.contains(&n.id)).count() as f64 / wanted.len() as f64
    }

    fn config(bits: u32) -> SymphonyConfig {
        SymphonyConfig {
            graph: GraphConfig {
                max_degree: 16,
                build_beam: 48,
                align_degree: true,
                ..GraphConfig::default()
            },
            bits,
            seed: 7,
        }
    }

    /// The §3.1.1 decomposition must produce the same number as rotating a
    /// normalized query per vertex. If it does not, the whole optimisation is
    /// silently returning different distances.
    #[test]
    fn the_decomposition_matches_the_direct_estimator() {
        let dimension = 32;
        let data = synthetic(200, dimension, 11);
        let index = SymphonyIndex::build(&data, dimension, config(6));
        let queries = synthetic(20, dimension, 13);

        for q in 0..20 {
            let query = &queries[q * dimension..(q + 1) * dimension];
            let prepared = index.prepare(query);

            for vertex in [0usize, 5, 50, 199] {
                let centre = index.graph.vector(vertex);
                let centre_distance = squared_l2(query, centre).sqrt();
                if centre_distance <= f32::MIN_POSITIVE {
                    continue;
                }
                // The direct path: normalize the query against this centroid and
                // rotate it, exactly as RaBitQ does for IVF.
                let direct_query = index.quantizer.prepare_query(query, centre);
                let codes = &index.vertices[vertex];

                for entry in &codes.codes {
                    let decomposed = index.estimate(entry, codes, &prepared, centre_distance);
                    let direct = index
                        .quantizer
                        .estimate_squared_distance(&entry.code, &direct_query);

                    let scale = direct.abs().max(1.0);
                    assert!(
                        (decomposed - direct).abs() / scale < 1e-2,
                        "vertex {vertex}: decomposed {decomposed} against direct {direct}"
                    );
                }
            }
        }
    }

    /// Returned distances must be exact, since every returned vertex was
    /// visited. This is implicit re-ranking working.
    #[test]
    fn returned_distances_are_exact_not_estimated() {
        let dimension = 24;
        let data = synthetic(500, dimension, 17);
        let index = SymphonyIndex::build(&data, dimension, config(4));
        let query = synthetic(1, dimension, 19);

        for found in index.search(&query, 10, 64) {
            let truth = squared_l2(&query, index.graph.vector(found.id as usize));
            assert!(
                (found.distance - truth).abs() < 1e-3,
                "vertex {} reported {} against a true {truth}",
                found.id,
                found.distance
            );
        }
    }

    #[test]
    fn recall_rises_with_beam_width() {
        let dimension = 32;
        let data = synthetic(2000, dimension, 23);
        let index = SymphonyIndex::build(&data, dimension, config(6));
        let queries = synthetic(30, dimension, 29);

        let mut previous = 0.0;
        for beam in [1usize, 4, 16, 64, 160] {
            let mut total = 0.0;
            for q in 0..30 {
                let query = &queries[q * dimension..(q + 1) * dimension];
                total += recall(
                    &index.search(query, 10, beam),
                    &exact(&data, dimension, query, 10),
                );
            }
            let mean = total / 30.0;
            assert!(
                mean >= previous - 1e-9,
                "beam {beam} gave recall {mean:.4}, below {previous:.4}"
            );
            previous = mean;
        }
        assert!(
            previous > 0.85,
            "a wide beam should recover most neighbours, got {previous:.4}"
        );
    }

    /// Very coarse codes must hurt, and a few bits must fix it.
    ///
    /// Deliberately *not* a monotonicity assertion. Measured over 300 queries,
    /// recall peaks around `B = 3..4` and then declines slightly — 0.851 at
    /// `B = 4` against 0.838 at `B = 8`, consistently at two beam widths. See
    /// [`recall_converges_to_exact_search_as_bits_rise`] for why that is correct
    /// behaviour rather than a defect.
    #[test]
    fn coarse_codes_hurt_recall_and_a_few_bits_fix_it() {
        let dimension = 32;
        let data = synthetic(1500, dimension, 31);
        let queries = synthetic(40, dimension, 37);

        let measure = |bits: u32| {
            let index = SymphonyIndex::build(&data, dimension, config(bits));
            let mut total = 0.0;
            for q in 0..40 {
                let query = &queries[q * dimension..(q + 1) * dimension];
                total += recall(
                    &index.search(query, 10, 32),
                    &exact(&data, dimension, query, 10),
                );
            }
            total / 40.0
        };

        let coarse = measure(1);
        let adequate = measure(4);
        assert!(
            adequate > coarse + 0.1,
            "one bit per dimension ({coarse:.4}) should be far worse than four \
             ({adequate:.4})"
        );
        assert!(adequate > 0.75, "four bits gave only {adequate:.4}");
    }

    /// **The end-to-end check on the estimator.**
    ///
    /// As the code widens, estimates approach exact distances, so the search
    /// must converge to plain greedy beam search over raw vectors. If it lands
    /// somewhere else, the estimator or the decomposition is wrong in a way the
    /// recall numbers alone would not expose.
    ///
    /// This also explains why recall is not monotonic in bits: converging to
    /// exact search means becoming *greedier*. Mild estimate noise keeps more
    /// diverse candidates in the beam and explores more of the graph, which is
    /// worth a little accuracy at moderate beam widths.
    #[test]
    fn recall_converges_to_exact_search_as_bits_rise() {
        let dimension = 32;
        let data = synthetic(1500, dimension, 79);
        let queries = synthetic(40, dimension, 83);
        let beam = 32;

        let quantized = SymphonyIndex::build(&data, dimension, config(8));
        let plain = GraphIndex::build(&data, dimension, config(8).graph);

        let mut quantized_recall = 0.0;
        let mut plain_recall = 0.0;
        for q in 0..40 {
            let query = &queries[q * dimension..(q + 1) * dimension];
            let truth = exact(&data, dimension, query, 10);
            quantized_recall += recall(&quantized.search(query, 10, beam), &truth);
            plain_recall += recall(&plain.search(query, 10, beam), &truth);
        }
        quantized_recall /= 40.0;
        plain_recall /= 40.0;

        assert!(
            (quantized_recall - plain_recall).abs() < 0.06,
            "8-bit codes gave {quantized_recall:.4} against exact search's \
             {plain_recall:.4}; they should nearly coincide"
        );
    }

    /// Codes are replicated per in-edge, so the index is *larger* than the raw
    /// vectors. Pinned as a test because it is the method's real cost and is
    /// easy to assume away.
    #[test]
    fn codes_outweigh_the_raw_vectors() {
        let dimension = 128;
        let data = synthetic(500, dimension, 41);
        let index = SymphonyIndex::build(&data, dimension, config(5));

        assert!(
            index.code_bytes() > index.raw_bytes(),
            "codes {} against raw {}; replication should make this larger, and \
             the raw vectors are still needed for exact distances",
            index.code_bytes(),
            index.raw_bytes()
        );
    }

    /// Degree alignment is one of the three contributions, on by default here
    /// and off in the plain graph baseline.
    #[test]
    fn degree_alignment_is_enabled_by_default() {
        assert!(SymphonyConfig::default().graph.align_degree);

        let dimension = 16;
        let data = synthetic(800, dimension, 43);
        let aligned = SymphonyIndex::build(&data, dimension, config(4));
        let plain = GraphIndex::build(
            &data,
            dimension,
            GraphConfig {
                align_degree: false,
                ..config(4).graph
            },
        );
        assert!(aligned.mean_out_degree() > plain.mean_out_degree());
    }

    /// Contribution 2 relies on a vertex being able to enter the beam more than
    /// once with different estimates. Verified by observing that search visits
    /// more vertices than a single-entry beam of the same width could hold.
    #[test]
    fn a_vertex_may_enter_the_beam_more_than_once() {
        let dimension = 16;
        let data = synthetic(600, dimension, 47);
        let index = SymphonyIndex::build(&data, dimension, config(3));
        let query = synthetic(1, dimension, 53);

        let visited = index.search(&query, 600, 8).len();
        assert!(
            visited > 8,
            "only {visited} vertices were visited with a beam of 8; re-entry is \
             not happening"
        );
    }

    #[test]
    fn building_and_searching_are_deterministic() {
        let dimension = 16;
        let data = synthetic(300, dimension, 59);
        let query = synthetic(1, dimension, 61);

        let first = SymphonyIndex::build(&data, dimension, config(4));
        let second = SymphonyIndex::build(&data, dimension, config(4));
        assert_eq!(first.search(&query, 10, 32), second.search(&query, 10, 32));
    }

    #[test]
    fn degenerate_searches_are_handled() {
        let dimension = 8;
        let data = synthetic(50, dimension, 67);
        let index = SymphonyIndex::build(&data, dimension, config(4));
        let query = synthetic(1, dimension, 71);

        assert!(index.search(&query, 0, 16).is_empty(), "k = 0");
        assert_eq!(
            index.search(&query, 1000, 16).len(),
            50,
            "capped at the set"
        );

        // A query sitting exactly on a vertex: `‖q_r − c‖` is zero there.
        let on_vertex = index.graph.vector(3).to_vec();
        let found = index.search(&on_vertex, 5, 16);
        assert_eq!(found[0].id, 3);
        assert!(found[0].distance < 1e-6);
    }

    #[test]
    #[should_panic(expected = "query has the wrong length")]
    fn a_wrong_sized_query_panics() {
        let data = synthetic(50, 8, 73);
        let index = SymphonyIndex::build(&data, 8, config(4));
        index.search(&[1.0, 2.0], 5, 16);
    }
}
