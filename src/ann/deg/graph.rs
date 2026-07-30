//! The Dynamic Edge Navigation Graph — paper 05 Algorithm 3 and §4.4–4.5.
//!
//! # One builder, three indexes
//!
//! This type builds DEG *and* both of the baselines §3.3 measures against it,
//! selected by [`PruningPolicy`]. That is deliberate: the experiment this exists
//! to run compares how the three behave as query α moves, and a comparison is
//! only worth reporting if the three share a beam search, a reverse-edge rule,
//! and a candidate-acquisition path. Only the pruning differs.
//!
//! [`super::super::graph::GraphIndex`] could not be reused for any of them: it
//! holds one vector array and hardcodes `squared_l2`, whereas the hybrid distance
//! combines two *distances* linearly, which is not Euclidean distance on a
//! concatenated vector.
//!
//! # Why an edge carries a set of α
//!
//! An RNG prunes an edge when some third node makes it the longest side of a
//! triangle. With α unknown at build time that condition holds for *some* α and
//! not others, so DEG stores the α-set where it does *not* hold and consults it
//! per hop. The graph the search actually walks is therefore a different graph
//! for every query, and each one satisfies the RNG property (Lemma 4.3).
//!
//! # Edge seeds, not centroids
//!
//! §4.4: seeding from nodes nearest the centre goes wrong here, because the
//! centre moves with α, and seeding from several centres makes greedy search run
//! several searches at once and revisit nodes. Seeding from the *farthest* nodes
//! instead — the inverse Pareto frontier of the centroid — leaves the irrelevant
//! seeds sitting at the bottom of the candidate heap, so the search effectively
//! starts from whichever seed is closest to the query.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use crate::ann::Neighbor;

use super::hybrid::HybridSet;
use super::interval::AlphaSet;
use super::pareto::{split_frontier, ParetoPoint};
use super::pruning::HybridDistance;
use super::select::{select_neighbours, DegEdge};

/// How edges are pruned at build time — DEG against the two §3.3 baselines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PruningPolicy {
    /// DEG: Algorithm 2, each edge keeping the α-set where it survives.
    Dynamic,
    /// **Fusion** (§3.3): prune with the RNG rule at one fixed α, then treat every
    /// surviving edge as always active. This is what an ordinary graph index does
    /// when pointed at a hybrid distance, and the α it commits to is the α it is
    /// good at.
    Fixed(f32),
    /// **Merging** (§3.3): ignore one modality entirely. Two such graphs, one per
    /// modality, are what the Merging baseline searches before re-ranking by
    /// Eq 1. `Primary` weights `δe` alone (α = 1), `Secondary` weights `δs`
    /// (α = 0) — the same as `Fixed(1.0)` / `Fixed(0.0)`, named separately
    /// because the baseline is a different *strategy*, not a different weight.
    SingleModality { primary: bool },
}

impl PruningPolicy {
    /// The α this policy prunes at, if it prunes at a single one.
    fn build_alpha(&self) -> Option<f32> {
        match self {
            PruningPolicy::Dynamic => None,
            PruningPolicy::Fixed(alpha) => Some(alpha.clamp(0.0, 1.0)),
            PruningPolicy::SingleModality { primary } => Some(if *primary { 1.0 } else { 0.0 }),
        }
    }
}

/// Where candidate neighbours come from at build time.
///
/// Exists to isolate a measured defect, not as a tuning knob: the α sweep in
/// `results/deg.md` found the hybrid graphs reaching a third of the recall a
/// plain [`GraphIndex`](crate::ann::graph::GraphIndex) gets on the same vectors
/// at α = 1, where the two solve an identical problem. Swapping one component at
/// a time is what identifies the cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateSource {
    /// Algorithm 1. Expands along Pareto frontier layers, which is what lets one
    /// candidate set serve every α.
    Gps,
    /// Ordinary greedy beam search at the policy's fixed α — what every
    /// single-metric graph index does.
    ///
    /// Only meaningful for [`PruningPolicy::Fixed`] and
    /// [`PruningPolicy::SingleModality`]; [`PruningPolicy::Dynamic`] has no single
    /// α to search at, which is the whole reason GPS exists.
    Beam,
}

/// Where a search starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryPolicy {
    /// §4.4's edge seeds: the nodes *farthest* from the centroid.
    EdgeSeeds,
    /// A single interior vertex, as an ordinary graph index uses.
    Interior,
}

/// Build parameters.
#[derive(Debug, Clone, Copy)]
pub struct DegConfig {
    /// `M`: cap on out-degree.
    pub max_degree: usize,
    /// `ef_construction`: candidate pool size for GPS.
    pub build_pool: usize,
    /// `th`: the narrowest active range worth storing, as a fraction of the α
    /// range. Ignored by the fixed-α policies, whose edges are always active.
    pub min_active_width: f32,
    /// How many edge seeds to keep.
    pub max_seeds: usize,
    pub policy: PruningPolicy,
    pub candidates: CandidateSource,
    pub entry: EntryPolicy,
}

impl Default for DegConfig {
    fn default() -> Self {
        Self {
            max_degree: 32,
            build_pool: 64,
            // The paper suggests discarding ranges as narrow as [0, 0.05].
            min_active_width: 0.05,
            max_seeds: 16,
            policy: PruningPolicy::Dynamic,
            candidates: CandidateSource::Gps,
            entry: EntryPolicy::EdgeSeeds,
        }
    }
}

/// A graph over two-modality objects, navigable at any query α.
#[derive(Debug, Clone)]
pub struct DegIndex {
    set: HybridSet,
    adjacency: Vec<Vec<DegEdge>>,
    seeds: Vec<u32>,
    config: DegConfig,
}

impl DegIndex {
    /// Algorithm 3: insert nodes one at a time into a growing graph.
    pub fn build(set: HybridSet, config: DegConfig) -> Self {
        let count = set.len();
        let mut index = Self {
            set,
            adjacency: vec![Vec::new(); count],
            seeds: vec![0],
            config,
        };

        for id in 1..count {
            // Candidates from the graph so far. Both sources run before
            // insertion, so `id` is not reachable and cannot dominate its own
            // frontier.
            let entries = index.entry_points();
            let layers = match config.candidates {
                CandidateSource::Gps => super::gps::gps(
                    &entries,
                    config.build_pool,
                    id,
                    |other| index.set.distance(id, other as usize),
                    |vertex, out| {
                        out.extend(index.adjacency[vertex as usize].iter().map(|e| e.target));
                    },
                ),
                CandidateSource::Beam => {
                    let alpha = config.policy.build_alpha().expect(
                        "CandidateSource::Beam needs a fixed-α policy; \
                         Dynamic has no single α to search at",
                    );
                    vec![index.beam_candidates(id, alpha, config.build_pool, &entries)]
                }
            };

            let edges = index.select(id, &layers);
            index.adjacency[id] = edges.clone();

            // Reverse edges. Without them the graph points only backwards
            // towards older vertices and search cannot descend into regions
            // added later — the same reason `GraphIndex` adds them.
            //
            // Algorithm 3 line 10 re-prunes the whole neighbour set of each
            // target — `E(y) ← DRNGPrune(E(y) ∪ {x}, M, th)` — and that is not
            // an optimisation detail. An active range is only meaningful for the
            // vertex it was computed at: the range on edge `id → target` came
            // from the triangle around `id`, tested against `id`'s other
            // neighbours, and says nothing about when `target → id` should be
            // live. Copying it across, or keeping it until the vertex happens to
            // overflow, leaves most reverse edges carrying a range for the wrong
            // triangle — which skips them at α where they should be traversed and
            // quietly under-connects the graph. The placeholder below is
            // overwritten by the re-prune on the next line.
            for edge in &edges {
                let target = edge.target as usize;
                index.adjacency[target].push(DegEdge {
                    target: id as u32,
                    active: AlphaSet::full(),
                });
                index.reprune(target);
            }

            index.update_seeds(id);
        }

        index
    }

    pub fn len(&self) -> usize {
        self.adjacency.len()
    }

    pub fn is_empty(&self) -> bool {
        self.adjacency.is_empty()
    }

    pub fn set(&self) -> &HybridSet {
        &self.set
    }

    pub fn seeds(&self) -> &[u32] {
        &self.seeds
    }

    pub fn edges(&self, id: usize) -> &[DegEdge] {
        &self.adjacency[id]
    }

    pub fn mean_out_degree(&self) -> f64 {
        if self.adjacency.is_empty() {
            return 0.0;
        }
        self.adjacency.iter().map(Vec::len).sum::<usize>() as f64 / self.adjacency.len() as f64
    }

    /// Bytes held: the vectors, plus the adjacency and its stored α-ranges.
    ///
    /// The active ranges are the space DEG spends that an ordinary graph does
    /// not; §4.5 calls the overhead negligible against the vectors, and this
    /// makes it measurable rather than assumed.
    pub fn memory_bytes(&self) -> (usize, usize) {
        let edges: usize = self
            .adjacency
            .iter()
            .map(|list| {
                list.iter()
                    .map(|edge| {
                        std::mem::size_of::<u32>()
                            + edge.active.intervals().len() * 2 * std::mem::size_of::<f32>()
                    })
                    .sum::<usize>()
            })
            .sum();
        (self.set.data_bytes(), edges)
    }

    /// The vertices a search starts from, per [`EntryPolicy`].
    fn entry_points(&self) -> Vec<u32> {
        match self.config.entry {
            EntryPolicy::EdgeSeeds => self.seeds.clone(),
            EntryPolicy::Interior => vec![0],
        }
    }

    /// Ordinary greedy beam search over the partially built graph at one α,
    /// returning the beam nearest-first as a single candidate layer.
    ///
    /// This is deliberately the same shape as
    /// `GraphIndex::search_internal` — a min-heap frontier, a bounded max-heap of
    /// results, and a `limit` bounding which vertices exist — so that when it
    /// replaces GPS the *only* thing that changed is how candidates are found.
    fn beam_candidates(
        &self,
        inserting: usize,
        alpha: f32,
        beam: usize,
        entries: &[u32],
    ) -> Vec<ParetoPoint> {
        let beam = beam.max(1);
        let limit = inserting;
        let mut visited: HashSet<u32> = HashSet::new();
        let mut frontier: BinaryHeap<Reverse<Neighbor>> = BinaryHeap::new();
        let mut results: BinaryHeap<Neighbor> = BinaryHeap::new();

        for &entry in entries {
            if (entry as usize) >= limit || !visited.insert(entry) {
                continue;
            }
            let start = Neighbor {
                id: entry as u64,
                distance: self.set.distance(inserting, entry as usize).at(alpha),
            };
            frontier.push(Reverse(start));
            results.push(start);
        }
        while results.len() > beam {
            results.pop();
        }

        while let Some(Reverse(current)) = frontier.pop() {
            if let Some(worst) = results.peek() {
                if results.len() >= beam && current.distance > worst.distance {
                    break;
                }
            }
            for edge in &self.adjacency[current.id as usize] {
                let target = edge.target as usize;
                if target >= limit || !visited.insert(edge.target) {
                    continue;
                }
                let candidate = Neighbor {
                    id: edge.target as u64,
                    distance: self.set.distance(inserting, target).at(alpha),
                };
                let worst = results.peek().map_or(f32::INFINITY, |n| n.distance);
                if results.len() < beam || candidate.distance < worst {
                    frontier.push(Reverse(candidate));
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
            .into_iter()
            .map(|n| ParetoPoint::new(n.id as u32, self.set.distance(inserting, n.id as usize)))
            .collect()
    }

    /// Turn candidate layers into edges, per the configured policy.
    fn select(&self, id: usize, layers: &[Vec<ParetoPoint>]) -> Vec<DegEdge> {
        match self.config.policy.build_alpha() {
            // DEG: the full active-range computation.
            None => select_neighbours(
                layers,
                self.config.max_degree,
                self.config.min_active_width,
                |a, b| self.set.distance(a as usize, b as usize),
            ),
            // The baselines: the classic RNG rule at one α, and every surviving
            // edge is unconditionally active.
            Some(alpha) => self.select_fixed(id, layers, alpha),
        }
    }

    /// RNG pruning at a single α — the Fusion and Merging baselines.
    ///
    /// Keeps a candidate unless an already-selected neighbour makes the edge the
    /// longest side of their triangle, evaluated at `alpha` only. This is the
    /// same rule DEG generalises, which is what makes the comparison meaningful.
    fn select_fixed(&self, id: usize, layers: &[Vec<ParetoPoint>], alpha: f32) -> Vec<DegEdge> {
        // Nearest-first at this α, since a fixed α gives a total order and the
        // RNG heuristic depends on considering near candidates first.
        let mut candidates: Vec<ParetoPoint> = layers.iter().flatten().copied().collect();
        candidates.sort_by(|a, b| {
            a.distance
                .at(alpha)
                .total_cmp(&b.distance.at(alpha))
                .then_with(|| a.id.cmp(&b.id))
        });

        let mut kept: Vec<DegEdge> = Vec::new();
        for candidate in candidates {
            if kept.len() >= self.config.max_degree {
                break;
            }
            if candidate.id as usize == id {
                continue;
            }
            let edge_length = candidate.distance.at(alpha);
            let pruned = kept.iter().any(|existing| {
                let to_existing = self.set.distance(id, existing.target as usize).at(alpha);
                let between = self
                    .set
                    .distance(candidate.id as usize, existing.target as usize)
                    .at(alpha);
                to_existing < edge_length && between < edge_length
            });
            if !pruned {
                kept.push(DegEdge {
                    target: candidate.id,
                    active: AlphaSet::full(),
                });
            }
        }
        kept
    }

    /// Re-prune an overflowing vertex through the configured policy.
    fn reprune(&mut self, vertex: usize) {
        let existing: Vec<ParetoPoint> = self.adjacency[vertex]
            .iter()
            .map(|edge| {
                ParetoPoint::new(edge.target, self.set.distance(vertex, edge.target as usize))
            })
            .collect();
        // Re-layer so Algorithm 2 sees the same nearest-first structure it does
        // during insertion.
        let layers = super::pareto::frontier_layers(&existing, existing.len().max(1));
        self.adjacency[vertex] = self.select(vertex, &layers);
    }

    /// §4.4: keep the nodes *farthest* from the centroid under varying α.
    ///
    /// The inverse Pareto frontier is the set dominating nothing — maximal in
    /// both distances-from-centroid. Computed by negating both coordinates and
    /// reusing the same skyline sweep as [`split_frontier`], rather than writing
    /// a second one that could disagree with it.
    fn update_seeds(&mut self, newest: usize) {
        let count = newest + 1;
        let (dim_e, dim_s) = (self.set.dim_e(), self.set.dim_s());

        let mut centre_e = vec![0.0f32; dim_e];
        let mut centre_s = vec![0.0f32; dim_s];
        for id in 0..count {
            for (slot, value) in centre_e.iter_mut().zip(self.set.primary(id)) {
                *slot += *value;
            }
            for (slot, value) in centre_s.iter_mut().zip(self.set.secondary(id)) {
                *slot += *value;
            }
        }
        for slot in centre_e.iter_mut() {
            *slot /= count as f32;
        }
        for slot in centre_s.iter_mut() {
            *slot /= count as f32;
        }

        // Consider the current seeds plus the new node, so the set only has to
        // be recomputed over a handful of points rather than the whole dataset.
        let mut pool: Vec<u32> = self.seeds.clone();
        if !pool.contains(&(newest as u32)) {
            pool.push(newest as u32);
        }

        let points: Vec<ParetoPoint> = pool
            .iter()
            .map(|&id| {
                let d = self.set.query_distance(&centre_e, &centre_s, id as usize);
                // Negated, so "farthest from the centre" becomes "undominated".
                ParetoPoint::new(id, HybridDistance::new(-d.e, -d.s))
            })
            .collect();

        let (frontier, _) = split_frontier(&points);
        let mut seeds: Vec<u32> = frontier.iter().map(|p| p.id).collect();
        seeds.truncate(self.config.max_seeds.max(1));
        if seeds.is_empty() {
            seeds.push(0);
        }
        self.seeds = seeds;
    }

    /// §4.5: greedy beam search that skips edges inactive at this α.
    ///
    /// # Panics
    ///
    /// If either query vector has the wrong length.
    pub fn search(
        &self,
        query_e: &[f32],
        query_s: &[f32],
        alpha: f32,
        k: usize,
        beam: usize,
    ) -> Vec<Neighbor> {
        self.search_inner(query_e, query_s, alpha, k, beam, true)
    }

    /// The same search with the §4.5 early exit disabled.
    ///
    /// Exists so a test can assert the optimisation changes timing and nothing
    /// else — an early exit that is subtly too aggressive would silently cost
    /// recall, which a recall number alone would not localise.
    pub fn search_without_early_exit(
        &self,
        query_e: &[f32],
        query_s: &[f32],
        alpha: f32,
        k: usize,
        beam: usize,
    ) -> Vec<Neighbor> {
        self.search_inner(query_e, query_s, alpha, k, beam, false)
    }

    fn search_inner(
        &self,
        query_e: &[f32],
        query_s: &[f32],
        alpha: f32,
        k: usize,
        beam: usize,
        early_exit: bool,
    ) -> Vec<Neighbor> {
        if self.adjacency.is_empty() || k == 0 {
            return Vec::new();
        }
        let alpha = alpha.clamp(0.0, 1.0);
        let beam = beam.max(k).max(1);

        let mut visited: HashSet<u32> = HashSet::new();
        let mut frontier: BinaryHeap<Reverse<Neighbor>> = BinaryHeap::new();
        let mut results: BinaryHeap<Neighbor> = BinaryHeap::new();

        for &seed in self.entry_points().iter() {
            if (seed as usize) >= self.adjacency.len() || !visited.insert(seed) {
                continue;
            }
            let entry = Neighbor {
                id: seed as u64,
                distance: self
                    .set
                    .query_distance(query_e, query_s, seed as usize)
                    .at(alpha),
            };
            frontier.push(Reverse(entry));
            results.push(entry);
        }
        while results.len() > beam {
            results.pop();
        }

        while let Some(Reverse(current)) = frontier.pop() {
            if let Some(worst) = results.peek() {
                if results.len() >= beam && current.distance > worst.distance {
                    break;
                }
            }

            for edge in &self.adjacency[current.id as usize] {
                // The dynamic part: an edge outside its active range is not in
                // this query's graph at all.
                if !edge.active.contains(alpha) {
                    continue;
                }
                if !visited.insert(edge.target) {
                    continue;
                }
                let target = edge.target as usize;
                let worst = results.peek().map_or(f32::INFINITY, |n| n.distance);
                let full = results.len() >= beam;

                if early_exit && full {
                    // Since δs ≥ 0, α·δe alone is a lower bound on the hybrid
                    // distance. If that already loses, the second modality need
                    // never be read.
                    let bound = alpha * self.set.query_primary(query_e, target);
                    if bound > worst {
                        continue;
                    }
                }

                let candidate = Neighbor {
                    id: edge.target as u64,
                    distance: self.set.query_distance(query_e, query_s, target).at(alpha),
                };
                if !full || candidate.distance < worst {
                    frontier.push(Reverse(candidate));
                    results.push(candidate);
                    if results.len() > beam {
                        results.pop();
                    }
                }
            }
        }

        let mut ordered = results.into_vec();
        ordered.sort_unstable();
        ordered.truncate(k);
        ordered
    }

    /// Exact top-`k` by scanning every object at this α — the oracle.
    pub fn exact_search(
        &self,
        query_e: &[f32],
        query_s: &[f32],
        alpha: f32,
        k: usize,
    ) -> Vec<Neighbor> {
        let mut all: Vec<Neighbor> = (0..self.len())
            .map(|id| Neighbor {
                id: id as u64,
                distance: self.set.query_distance(query_e, query_s, id).at(alpha),
            })
            .collect();
        all.sort_unstable();
        all.truncate(k);
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ann::recall_at_k;
    use crate::workload::Rng;

    fn random_set(count: usize, seed: u64) -> HybridSet {
        let mut rng = Rng::new(seed);
        let primary: Vec<f32> = (0..count * 8).map(|_| rng.next_f64() as f32).collect();
        let secondary: Vec<f32> = (0..count * 2).map(|_| rng.next_f64() as f32).collect();
        HybridSet::new(primary, 8, secondary, 2, 4096, seed)
    }

    fn deg(count: usize, seed: u64) -> DegIndex {
        DegIndex::build(random_set(count, seed), DegConfig::default())
    }

    #[test]
    fn every_stored_active_range_is_non_empty() {
        let index = deg(200, 1);
        for id in 0..index.len() {
            for edge in index.edges(id) {
                assert!(
                    !edge.active.is_empty(),
                    "vertex {id} has an edge to {} active nowhere",
                    edge.target
                );
                assert!(
                    edge.active.measure() > 0.0,
                    "vertex {id} edge to {} has zero-width range",
                    edge.target
                );
            }
        }
    }

    #[test]
    fn no_vertex_points_at_itself() {
        let index = deg(150, 2);
        for id in 0..index.len() {
            for edge in index.edges(id) {
                assert_ne!(edge.target as usize, id, "vertex {id} has a self-loop");
            }
        }
    }

    #[test]
    fn the_graph_is_connected_enough_to_reach_good_neighbours() {
        // Recall against the exact oracle at the three α the paper highlights.
        let index = deg(300, 3);
        let set = index.set();
        let queries: Vec<(Vec<f32>, Vec<f32>)> = (0..20)
            .map(|i| (set.primary(i * 7).to_vec(), set.secondary(i * 7).to_vec()))
            .collect();

        for &alpha in &[0.0f32, 0.5, 1.0] {
            let mut total = 0.0;
            for (q_e, q_s) in &queries {
                let truth: Vec<u32> = index
                    .exact_search(q_e, q_s, alpha, 10)
                    .iter()
                    .map(|n| n.id as u32)
                    .collect();
                let found = index.search(q_e, q_s, alpha, 10, 64);
                total += recall_at_k(&found, &truth, 10);
            }
            let recall = total / queries.len() as f64;
            assert!(
                recall > 0.75,
                "α={alpha}: recall {recall:.3} is too low for a working graph"
            );
        }
    }

    #[test]
    fn the_early_exit_does_not_change_results() {
        // The optimisation rests on α·δe being a lower bound. If that reasoning
        // were wrong it would silently drop good candidates, so the two paths
        // must agree exactly.
        let index = deg(250, 4);
        let set = index.set();
        for i in 0..25 {
            let q_e = set.primary(i * 9).to_vec();
            let q_s = set.secondary(i * 9).to_vec();
            for step in 0..=10 {
                let alpha = step as f32 / 10.0;
                let with = index.search(&q_e, &q_s, alpha, 10, 48);
                let without = index.search_without_early_exit(&q_e, &q_s, alpha, 10, 48);
                assert_eq!(
                    with.iter().map(|n| n.id).collect::<Vec<_>>(),
                    without.iter().map(|n| n.id).collect::<Vec<_>>(),
                    "query {i} α={alpha}: early exit changed the result"
                );
            }
        }
    }

    #[test]
    fn seeds_are_far_from_the_centre_not_near_it() {
        // §4.4's inversion. The mean distance from the centroid to a seed should
        // exceed the mean over all objects, at every α.
        let index = deg(200, 5);
        let set = index.set();
        let count = set.len();

        let mut centre_e = vec![0.0f32; set.dim_e()];
        let mut centre_s = vec![0.0f32; set.dim_s()];
        for id in 0..count {
            for (slot, v) in centre_e.iter_mut().zip(set.primary(id)) {
                *slot += *v;
            }
            for (slot, v) in centre_s.iter_mut().zip(set.secondary(id)) {
                *slot += *v;
            }
        }
        for slot in centre_e.iter_mut() {
            *slot /= count as f32;
        }
        for slot in centre_s.iter_mut() {
            *slot /= count as f32;
        }

        for &alpha in &[0.0f32, 0.5, 1.0] {
            let overall: f32 = (0..count)
                .map(|id| set.query_distance(&centre_e, &centre_s, id).at(alpha))
                .sum::<f32>()
                / count as f32;
            let seeded: f32 = index
                .seeds()
                .iter()
                .map(|&id| {
                    set.query_distance(&centre_e, &centre_s, id as usize)
                        .at(alpha)
                })
                .sum::<f32>()
                / index.seeds().len() as f32;
            assert!(
                seeded > overall,
                "α={alpha}: seeds average {seeded:.4} from the centre, \
                 all objects average {overall:.4} — seeds should be farther"
            );
        }
    }

    #[test]
    fn the_fixed_policy_marks_every_edge_always_active() {
        let index = DegIndex::build(
            random_set(150, 6),
            DegConfig {
                policy: PruningPolicy::Fixed(0.5),
                ..Default::default()
            },
        );
        for id in 0..index.len() {
            for edge in index.edges(id) {
                assert!(
                    edge.active.is_full(),
                    "the Fusion baseline should not store ranges"
                );
            }
        }
    }

    /// Reciprocal edges must carry *independently computed* ranges.
    ///
    /// An active range is only meaningful at the vertex it was derived at: the
    /// range on `a → b` comes from the triangle around `a`, tested against `a`'s
    /// other neighbours, and says nothing about when `b → a` should be live.
    /// Algorithm 3 line 10 therefore re-prunes the target's whole neighbour set
    /// on every insertion rather than copying anything across.
    ///
    /// This is the direct detector for that mistake. If a reverse edge inherits
    /// the forward edge's range, reciprocal pairs come out *identical*; when each
    /// is computed at its own vertex against a different neighbour set, most
    /// differ. An earlier version of `build` copied the range and re-pruned only
    /// on overflow, which left the graph under-connected at query time while
    /// every recall figure still looked merely "a bit low".
    ///
    /// Note this cannot be an equality or containment check in either direction:
    /// Algorithm 2 tests a candidate only against neighbours *already* committed,
    /// so the ranges depend on selection order and are not recoverable from the
    /// final edge set.
    #[test]
    fn reciprocal_edges_carry_independently_computed_ranges() {
        let index = deg(300, 14);
        let mut pairs = 0usize;
        let mut identical = 0usize;

        for id in 0..index.len() {
            for edge in index.edges(id) {
                let target = edge.target as usize;
                // Count each unordered pair once.
                if target <= id {
                    continue;
                }
                let Some(back) = index.edges(target).iter().find(|e| e.target as usize == id)
                else {
                    continue;
                };
                pairs += 1;
                if edge.active == back.active {
                    identical += 1;
                }
            }
        }

        assert!(pairs > 50, "too few reciprocal pairs to judge: {pairs}");
        let differing = pairs - identical;
        assert!(
            differing * 4 >= pairs,
            "only {differing} of {pairs} reciprocal pairs have different ranges; \
             ranges look copied rather than computed per vertex"
        );
    }

    /// How much of the graph is traversable at a given α.
    ///
    /// This is not a "higher is better" check — it records a real property of the
    /// algorithm. At α = 0 the distance is the second modality alone, and an RNG
    /// over a 2-dimensional space is intrinsically sparse (a planar RNG averages
    /// roughly six neighbours), so most edges *should* be inactive there. The
    /// bound is therefore loose; what it guards against is the degenerate case
    /// where nearly every edge is inactive and search has almost no hops.
    #[test]
    fn a_useful_fraction_of_edges_is_traversable_at_every_alpha() {
        let index = deg(300, 15);
        let stored: usize = (0..index.len()).map(|id| index.edges(id).len()).sum();
        assert!(stored > 0);

        for &alpha in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let live: usize = (0..index.len())
                .map(|id| {
                    index
                        .edges(id)
                        .iter()
                        .filter(|e| e.active.contains(alpha))
                        .count()
                })
                .sum();
            let fraction = live as f64 / stored as f64;
            assert!(
                fraction > 0.15,
                "α={alpha}: only {fraction:.2} of edges are traversable, \
                 which is too sparse to navigate"
            );
        }
    }

    #[test]
    fn the_dynamic_policy_actually_stores_partial_ranges() {
        // If every range came out full, DEG would be Fusion with extra steps and
        // the α sweep would show nothing. This is the check that the dynamic
        // pruning is doing something.
        let index = deg(300, 7);
        let partial = (0..index.len())
            .flat_map(|id| index.edges(id))
            .filter(|edge| !edge.active.is_full())
            .count();
        assert!(
            partial > 0,
            "no edge has a restricted active range; dynamic pruning is inert"
        );
    }

    #[test]
    fn a_single_modality_policy_ignores_the_other_one() {
        // Built weighting δe alone, the graph's edges should be the RNG edges of
        // the first modality — so searching at α=1 must work well, and the graph
        // must differ from one built on the second modality.
        let set = random_set(150, 8);
        let primary_graph = DegIndex::build(
            set.clone(),
            DegConfig {
                policy: PruningPolicy::SingleModality { primary: true },
                ..Default::default()
            },
        );
        let secondary_graph = DegIndex::build(
            set,
            DegConfig {
                policy: PruningPolicy::SingleModality { primary: false },
                ..Default::default()
            },
        );

        let differing = (0..primary_graph.len())
            .filter(|&id| {
                let a: Vec<u32> = primary_graph.edges(id).iter().map(|e| e.target).collect();
                let b: Vec<u32> = secondary_graph.edges(id).iter().map(|e| e.target).collect();
                a != b
            })
            .count();
        assert!(
            differing > primary_graph.len() / 4,
            "the two modalities should produce visibly different graphs, \
             only {differing} vertices differ"
        );
    }

    #[test]
    fn out_degree_respects_the_cap_within_one_layer() {
        // Algorithm 2 tests the cap after a layer, so overshoot is expected;
        // what must not happen is unbounded growth from reverse edges.
        let config = DegConfig {
            max_degree: 8,
            ..Default::default()
        };
        let index = DegIndex::build(random_set(200, 9), config);
        for id in 0..index.len() {
            assert!(
                index.edges(id).len() <= config.max_degree * 2,
                "vertex {id} has {} edges against a cap of {}",
                index.edges(id).len(),
                config.max_degree
            );
        }
    }

    #[test]
    fn search_returns_k_results_and_they_are_sorted() {
        let index = deg(120, 10);
        let set = index.set();
        let found = index.search(set.primary(3), set.secondary(3), 0.4, 10, 32);
        assert_eq!(found.len(), 10);
        for pair in found.windows(2) {
            assert!(pair[0].distance <= pair[1].distance, "results not sorted");
        }
    }

    #[test]
    fn a_query_matching_an_object_finds_that_object_first() {
        let index = deg(200, 11);
        let set = index.set();
        for id in [0usize, 57, 101, 199] {
            let found = index.search(set.primary(id), set.secondary(id), 0.5, 5, 64);
            assert_eq!(
                found[0].id as usize, id,
                "querying object {id} should return it first"
            );
            assert!(found[0].distance < 1e-6);
        }
    }

    #[test]
    fn degenerate_searches_are_handled() {
        let index = deg(50, 12);
        let set = index.set();
        assert!(index
            .search(set.primary(0), set.secondary(0), 0.5, 0, 16)
            .is_empty());
        // α outside [0,1] is clamped rather than producing an empty graph.
        let low = index.search(set.primary(0), set.secondary(0), -5.0, 5, 16);
        let high = index.search(set.primary(0), set.secondary(0), 9.0, 5, 16);
        assert_eq!(low.len(), 5);
        assert_eq!(high.len(), 5);
        // A beam narrower than k is widened to k.
        assert_eq!(
            index
                .search(set.primary(0), set.secondary(0), 0.5, 10, 1)
                .len(),
            10
        );
    }

    #[test]
    fn a_two_object_set_builds_and_searches() {
        let index = deg(2, 13);
        assert_eq!(index.len(), 2);
        let set = index.set();
        let found = index.search(set.primary(1), set.secondary(1), 0.5, 2, 8);
        assert_eq!(found[0].id, 1);
    }
}
