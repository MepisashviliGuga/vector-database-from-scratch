//! A navigable proximity graph with greedy beam search.
//!
//! The baseline SymphonyQG is built on and measured against. Vertices are
//! vectors; edges connect a vertex to a diverse set of near neighbours; a query
//! walks the graph towards itself.
//!
//! # Why diversity pruning, not just "nearest M"
//!
//! Connecting each vertex to its `M` nearest neighbours produces a graph that
//! greedy search gets *stuck* in: all `M` edges point into the same dense pocket,
//! so once search enters a cluster it cannot leave. What is needed is edges that
//! are near **and spread out in direction**.
//!
//! The relative-neighbourhood rule does this. Walking candidates nearest-first,
//! a candidate `c` is kept unless some already-kept neighbour `r` sits closer to
//! `c` than `c` is to the vertex — meaning `r` already covers that direction.
//! Equivalently, kept edges are separated by at least 60°.
//!
//! This module states the rule as an explicit **angle threshold** rather than
//! the distance comparison, because SymphonyQG's third contribution is to
//! *relax* that threshold to hit a target out-degree, and an explicit angle is
//! the thing it relaxes. At the default 60° the two formulations agree.
//!
//! # Construction
//!
//! Incremental insertion: each vector searches the graph built so far, and the
//! result becomes its candidate set. This is the NSW/HNSW approach rather than
//! paper 04's NSG two-stage build, and it is a **labelled deviation** — NSG
//! builds an approximate kNN graph first and then prunes globally. The pruning
//! rule, which is the part SymphonyQG modifies, is the same either way.
//!
//! # Cost
//!
//! Building is `O(N · beam · degree · D)`. At a million 128-dimensional vectors
//! that is roughly `4e11` scalar operations — impractical single-threaded, so
//! measurements here use smaller sets and say so.

use std::collections::{BinaryHeap, HashSet};

use super::{squared_l2, Neighbor};

/// Build parameters.
#[derive(Debug, Clone, Copy)]
pub struct GraphConfig {
    /// Target out-degree `R`.
    pub max_degree: usize,
    /// Beam width used to generate candidates during construction. Wider gives
    /// a better graph and a slower build.
    pub build_beam: usize,
    /// Minimum angle, in degrees, between two kept edges. 60 reproduces the
    /// classic relative-neighbourhood rule; lower keeps more edges.
    pub prune_angle_degrees: f32,
    /// Relax the angle threshold per vertex until out-degree reaches
    /// `max_degree` — SymphonyQG §3.2.2. Off by default so the baseline is the
    /// unmodified rule.
    pub align_degree: bool,
    pub seed: u64,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_degree: 32,
            build_beam: 64,
            prune_angle_degrees: 60.0,
            align_degree: false,
            seed: 0x5EED,
        }
    }
}

/// A proximity graph over raw vectors.
#[derive(Debug, Clone)]
pub struct GraphIndex {
    dimension: usize,
    data: Vec<f32>,
    adjacency: Vec<Vec<u32>>,
    entry: u32,
    config: GraphConfig,
}

impl GraphIndex {
    /// # Panics
    ///
    /// If `dimension` is 0, `data` is not a whole number of vectors, or the set
    /// is empty.
    pub fn build(data: &[f32], dimension: usize, config: GraphConfig) -> Self {
        assert!(dimension > 0, "vectors need at least one dimension");
        assert_eq!(
            data.len() % dimension,
            0,
            "{} values do not divide into {dimension}-dimensional vectors",
            data.len()
        );
        let count = data.len() / dimension;
        assert!(count > 0, "cannot index an empty set");

        let mut index = Self {
            dimension,
            data: data.to_vec(),
            adjacency: vec![Vec::new(); count],
            // The medoid would be a better entry point; vertex 0 is adequate
            // once the graph is navigable and costs nothing to choose.
            entry: 0,
            config,
        };

        for id in 1..count {
            let vector = index.vector(id).to_vec();
            // Candidates come from searching the graph built so far.
            let candidates = index.search_internal(&vector, config.build_beam, id);
            let neighbours = index.prune(id, &candidates, config.max_degree);

            index.adjacency[id] = neighbours.clone();

            // Reverse edges, re-pruned only when a vertex overflows. Without
            // these the graph points towards older vertices and search cannot
            // descend into newly added regions.
            //
            // Note this means the finished adjacency does **not** satisfy the
            // diversity rule strictly — a reverse edge that fits under the
            // degree cap is kept whatever its angle. HNSW makes the same trade,
            // buying connectivity at the cost of some redundant edges.
            for &neighbour in &neighbours {
                let target = neighbour as usize;
                index.adjacency[target].push(id as u32);
                if index.adjacency[target].len() > config.max_degree {
                    let existing: Vec<Neighbor> = index.adjacency[target]
                        .iter()
                        .map(|&other| Neighbor {
                            id: other as u64,
                            distance: squared_l2(index.vector(target), index.vector(other as usize)),
                        })
                        .collect();
                    let mut sorted = existing;
                    sorted.sort_unstable();
                    index.adjacency[target] = index.prune(target, &sorted, config.max_degree);
                }
            }
        }

        if config.align_degree {
            index.align_out_degree();
        }
        index
    }

    pub fn len(&self) -> usize {
        self.adjacency.len()
    }

    pub fn is_empty(&self) -> bool {
        self.adjacency.is_empty()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn vector(&self, id: usize) -> &[f32] {
        &self.data[id * self.dimension..(id + 1) * self.dimension]
    }

    pub fn neighbours(&self, id: usize) -> &[u32] {
        &self.adjacency[id]
    }

    /// Mean out-degree. SymphonyQG's degree alignment exists because this sits
    /// below `max_degree` under the unmodified pruning rule.
    pub fn mean_out_degree(&self) -> f64 {
        if self.adjacency.is_empty() {
            return 0.0;
        }
        self.adjacency.iter().map(Vec::len).sum::<usize>() as f64 / self.adjacency.len() as f64
    }

    /// Bytes of raw vectors held. A graph over raw vectors keeps all of them,
    /// which is the memory cost quantization removes.
    pub fn data_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// The `k` nearest neighbours by greedy beam search over **exact**
    /// distances.
    ///
    /// # Panics
    ///
    /// If `query` is the wrong length.
    pub fn search(&self, query: &[f32], k: usize, beam: usize) -> Vec<Neighbor> {
        assert_eq!(query.len(), self.dimension, "query has the wrong length");
        let mut results = self.search_internal(query, beam.max(k), self.len());
        results.truncate(k);
        results
    }

    /// Beam search, returning the beam sorted nearest-first.
    ///
    /// `limit` bounds which vertices exist yet, so construction can search a
    /// partially built graph.
    fn search_internal(&self, query: &[f32], beam: usize, limit: usize) -> Vec<Neighbor> {
        let beam = beam.max(1);
        let entry = self.entry as usize;
        if limit == 0 || entry >= limit {
            return Vec::new();
        }

        let mut visited: HashSet<u32> = HashSet::new();
        // Frontier ordered nearest-first; `Reverse` turns the max-heap around.
        let mut frontier: BinaryHeap<std::cmp::Reverse<Neighbor>> = BinaryHeap::new();
        // Results ordered farthest-first, so the worst is cheap to evict.
        let mut results: BinaryHeap<Neighbor> = BinaryHeap::new();

        let start = Neighbor {
            id: entry as u64,
            distance: squared_l2(query, self.vector(entry)),
        };
        visited.insert(entry as u32);
        frontier.push(std::cmp::Reverse(start));
        results.push(start);

        while let Some(std::cmp::Reverse(current)) = frontier.pop() {
            // Everything left in the frontier is farther than the worst result,
            // and neighbours only get farther, so the search is done.
            if let Some(worst) = results.peek() {
                if results.len() >= beam && current.distance > worst.distance {
                    break;
                }
            }

            for &neighbour in &self.adjacency[current.id as usize] {
                if neighbour as usize >= limit || !visited.insert(neighbour) {
                    continue;
                }
                let candidate = Neighbor {
                    id: neighbour as u64,
                    distance: squared_l2(query, self.vector(neighbour as usize)),
                };

                let worst = results.peek().map_or(f32::INFINITY, |n| n.distance);
                if results.len() < beam || candidate.distance < worst {
                    frontier.push(std::cmp::Reverse(candidate));
                    results.push(candidate);
                    if results.len() > beam {
                        results.pop();
                    }
                }
            }
        }

        let mut ordered = results.into_vec();
        ordered.sort_unstable();
        ordered
    }

    /// Apply the diversity rule to `candidates` (nearest-first), keeping at most
    /// `limit`.
    fn prune(&self, vertex: usize, candidates: &[Neighbor], limit: usize) -> Vec<u32> {
        self.prune_with_angle(vertex, candidates, limit, self.config.prune_angle_degrees)
    }

    /// The rule, with an explicit minimum angle between kept edges.
    ///
    /// Lowering `angle_degrees` keeps more candidates, monotonically — which is
    /// what makes SymphonyQG's binary search for a target out-degree valid.
    fn prune_with_angle(
        &self,
        vertex: usize,
        candidates: &[Neighbor],
        limit: usize,
        angle_degrees: f32,
    ) -> Vec<u32> {
        let cos_limit = angle_degrees.to_radians().cos();
        let origin = self.vector(vertex);
        let mut kept: Vec<u32> = Vec::with_capacity(limit);
        let mut directions: Vec<Vec<f32>> = Vec::with_capacity(limit);

        for candidate in candidates {
            if kept.len() >= limit {
                break;
            }
            let id = candidate.id as u32;
            if id as usize == vertex {
                continue;
            }

            let point = self.vector(id as usize);
            let direction = unit_direction(origin, point);
            let Some(direction) = direction else {
                // A candidate identical to the vertex has no direction; keeping
                // it would add an edge that teaches search nothing.
                continue;
            };

            // Candidates arrive nearest-first, so any kept edge is at least as
            // close. Rejecting on angle alone therefore reproduces the
            // relative-neighbourhood rule at 60 degrees.
            let too_close_in_direction = directions.iter().any(|existing| {
                let cosine: f32 = existing
                    .iter()
                    .zip(direction.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                cosine > cos_limit
            });

            if !too_close_in_direction {
                kept.push(id);
                directions.push(direction);
            }
        }
        kept
    }

    /// SymphonyQG §3.2.2: relax the pruning angle per vertex until out-degree
    /// reaches the target.
    ///
    /// The unmodified rule leaves mean out-degree below `R` — the paper measures
    /// 19.8 at `R = 32` — which wastes lanes in a batched distance kernel that
    /// processes a fixed number of codes at a time.
    ///
    /// Because the kept-set size is **monotonic** in the threshold, a binary
    /// search finds the most restrictive angle that still yields `R` edges. Only
    /// vertices short of the target are touched, so a vertex that already has
    /// enough diverse edges keeps them.
    fn align_out_degree(&mut self) {
        let target = self.config.max_degree;
        let strict = self.config.prune_angle_degrees;

        for vertex in 0..self.len() {
            if self.adjacency[vertex].len() >= target {
                continue;
            }

            // Re-derive candidates by searching, so relaxing has something to
            // choose from beyond the already-pruned set.
            let vector = self.vector(vertex).to_vec();
            let candidates = self.search_internal(&vector, target * 4, self.len());
            if candidates.len() <= self.adjacency[vertex].len() {
                continue;
            }

            let mut low = 0.0f32;
            let mut high = strict;
            let mut best = self.adjacency[vertex].clone();

            // Twenty halvings resolve the angle to well under a degree, which is
            // far finer than the kept-set size can distinguish.
            for _ in 0..20 {
                let mid = 0.5 * (low + high);
                let kept = self.prune_with_angle(vertex, &candidates, target, mid);
                if kept.len() >= target {
                    // Enough edges: try to stay more restrictive.
                    best = kept;
                    low = mid;
                } else {
                    high = mid;
                }
                if high - low < 0.05 {
                    break;
                }
            }

            if best.len() > self.adjacency[vertex].len() {
                self.adjacency[vertex] = best;
            }
        }
    }
}

/// The unit vector from `origin` to `point`, or `None` if they coincide.
fn unit_direction(origin: &[f32], point: &[f32]) -> Option<Vec<f32>> {
    let difference: Vec<f32> = point
        .iter()
        .zip(origin.iter())
        .map(|(p, o)| p - o)
        .collect();
    let norm = difference.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm <= f32::MIN_POSITIVE {
        return None;
    }
    Some(difference.into_iter().map(|v| v / norm).collect())
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

    fn small_config() -> GraphConfig {
        GraphConfig {
            max_degree: 16,
            build_beam: 48,
            ..Default::default()
        }
    }

    #[test]
    fn out_degree_respects_the_cap() {
        let dimension = 16;
        let data = synthetic(600, dimension, 11);
        let index = GraphIndex::build(&data, dimension, small_config());

        for id in 0..index.len() {
            assert!(
                index.neighbours(id).len() <= small_config().max_degree,
                "vertex {id} has {} edges",
                index.neighbours(id).len()
            );
        }
    }

    #[test]
    fn no_vertex_links_to_itself_or_duplicates() {
        let dimension = 16;
        let data = synthetic(400, dimension, 13);
        let index = GraphIndex::build(&data, dimension, small_config());

        for id in 0..index.len() {
            let neighbours = index.neighbours(id);
            let unique: std::collections::HashSet<u32> = neighbours.iter().copied().collect();
            assert_eq!(unique.len(), neighbours.len(), "vertex {id} has a duplicate edge");
            assert!(!neighbours.contains(&(id as u32)), "vertex {id} links to itself");
        }
    }

    /// The knob that trades recall for latency. Wider beams examine more of the
    /// graph, so recall must rise.
    #[test]
    fn recall_rises_with_beam_width() {
        let dimension = 24;
        let data = synthetic(2000, dimension, 17);
        let index = GraphIndex::build(&data, dimension, small_config());
        let queries = synthetic(30, dimension, 19);

        let mut previous = 0.0;
        for beam in [1usize, 4, 16, 64, 200] {
            let mut total = 0.0;
            for q in 0..30 {
                let query = &queries[q * dimension..(q + 1) * dimension];
                let truth = exact(&data, dimension, query, 10);
                total += recall(&index.search(query, 10, beam), &truth);
            }
            let mean = total / 30.0;
            assert!(
                mean >= previous - 1e-9,
                "beam {beam} gave recall {mean:.4}, below {previous:.4}"
            );
            previous = mean;
        }
        assert!(
            previous > 0.9,
            "a wide beam should recover almost every neighbour, got {previous:.4}"
        );
    }

    /// The property diversity pruning exists for. A graph of pure nearest
    /// neighbours traps greedy search inside a cluster; a navigable one lets it
    /// cross between them.
    #[test]
    fn search_escapes_a_distant_cluster() {
        let dimension = 8;
        let mut rng = Rng::new(23);
        let mut data = Vec::new();
        // Two tight, far-apart blobs. Vertex 0 — the entry point — is in the
        // first, so reaching the second requires traversing between them.
        for centre in [0.0f32, 100.0] {
            for _ in 0..300 {
                for _ in 0..dimension {
                    data.push(centre + (rng.next_f64() as f32 - 0.5));
                }
            }
        }

        let index = GraphIndex::build(&data, dimension, small_config());
        let query = vec![100.0f32; dimension];
        let found = index.search(&query, 10, 64);

        assert!(
            found.iter().all(|n| n.id >= 300),
            "search failed to cross into the far cluster: {:?}",
            found.iter().map(|n| n.id).collect::<Vec<_>>()
        );
    }

    /// Pruned edges must be spread in direction, not merely near. This is what
    /// separates the rule from "keep the M closest".
    ///
    /// Checked on `prune`'s output rather than the final adjacency, because the
    /// graph deliberately adds **reverse edges** without re-pruning unless the
    /// target overflows — the same choice HNSW makes, trading strict diversity
    /// for connectivity. An earlier version of this test asserted the property
    /// of the finished graph and failed on exactly those reverse edges, which is
    /// the construction working as intended rather than a defect.
    #[test]
    fn pruning_keeps_angularly_diverse_edges() {
        let dimension = 12;
        let data = synthetic(500, dimension, 29);
        let index = GraphIndex::build(&data, dimension, small_config());

        let cos_limit = 60.0f32.to_radians().cos();
        for vertex in [0usize, 7, 42, 199, 499] {
            let vector = index.vector(vertex).to_vec();
            let candidates = index.search_internal(&vector, 64, index.len());
            let kept = index.prune(vertex, &candidates, small_config().max_degree);

            let origin = index.vector(vertex);
            let directions: Vec<Vec<f32>> = kept
                .iter()
                .filter_map(|&n| unit_direction(origin, index.vector(n as usize)))
                .collect();

            for a in 0..directions.len() {
                for b in (a + 1)..directions.len() {
                    let cosine: f32 = directions[a]
                        .iter()
                        .zip(directions[b].iter())
                        .map(|(x, y)| x * y)
                        .sum();
                    assert!(
                        cosine <= cos_limit + 1e-3,
                        "vertex {vertex} kept two edges only {:.1} degrees apart",
                        cosine.clamp(-1.0, 1.0).acos().to_degrees()
                    );
                }
            }
        }
    }

    /// SymphonyQG §3.2.2. The unmodified rule leaves mean out-degree short of
    /// the target; relaxing the angle per vertex closes the gap.
    #[test]
    fn degree_alignment_raises_mean_out_degree() {
        let dimension = 16;
        let data = synthetic(1200, dimension, 31);

        let baseline = GraphIndex::build(&data, dimension, small_config());
        let aligned = GraphIndex::build(
            &data,
            dimension,
            GraphConfig {
                align_degree: true,
                ..small_config()
            },
        );

        assert!(
            aligned.mean_out_degree() > baseline.mean_out_degree(),
            "alignment did not raise mean out-degree: {:.2} against {:.2}",
            aligned.mean_out_degree(),
            baseline.mean_out_degree()
        );
        // And it must not exceed the cap, which is the point of aligning *to* it.
        for id in 0..aligned.len() {
            assert!(aligned.neighbours(id).len() <= small_config().max_degree);
        }
    }

    /// Relaxing the angle must never *remove* candidates, or the binary search
    /// SymphonyQG relies on would not be well founded.
    #[test]
    fn a_looser_angle_keeps_at_least_as_many_edges() {
        let dimension = 12;
        let data = synthetic(400, dimension, 37);
        let index = GraphIndex::build(&data, dimension, small_config());

        let vertex = 5;
        let vector = index.vector(vertex).to_vec();
        let candidates = index.search_internal(&vector, 64, index.len());

        let mut previous = 0usize;
        for angle in [10.0f32, 20.0, 40.0, 60.0, 90.0] {
            let kept = index.prune_with_angle(vertex, &candidates, 64, angle);
            // Stricter (larger) angles keep fewer, so walking upward must not
            // increase the count.
            if previous > 0 {
                assert!(
                    kept.len() <= previous,
                    "angle {angle} kept {} edges, more than the looser rule's {previous}",
                    kept.len()
                );
            }
            previous = kept.len();
        }
    }

    #[test]
    fn building_is_deterministic() {
        let dimension = 12;
        let data = synthetic(300, dimension, 41);
        let first = GraphIndex::build(&data, dimension, small_config());
        let second = GraphIndex::build(&data, dimension, small_config());

        for id in 0..first.len() {
            assert_eq!(first.neighbours(id), second.neighbours(id));
        }
    }

    #[test]
    fn degenerate_inputs_are_handled() {
        let index = GraphIndex::build(&[1.0, 2.0], 2, small_config());
        assert_eq!(index.len(), 1);
        let found = index.search(&[1.0, 2.0], 5, 10);
        assert_eq!(found.len(), 1, "a one-vertex graph returns its only vertex");

        let identical = vec![7.0f32; 40];
        let index = GraphIndex::build(&identical, 4, small_config());
        assert_eq!(index.len(), 10);
        assert!(!index.search(&[7.0, 7.0, 7.0, 7.0], 3, 10).is_empty());
    }

    #[test]
    #[should_panic(expected = "query has the wrong length")]
    fn a_wrong_sized_query_panics() {
        let data = synthetic(50, 8, 43);
        let index = GraphIndex::build(&data, 8, small_config());
        index.search(&[1.0, 2.0], 5, 10);
    }
}
