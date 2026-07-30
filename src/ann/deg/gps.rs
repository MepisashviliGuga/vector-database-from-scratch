//! Greedy Pareto Frontier Search — paper 05 Algorithm 1.
//!
//! # What it is for
//!
//! Inserting a node needs a candidate set for it. An ordinary graph index gets
//! one by beam-searching for the nearest vertices, but "nearest" needs an α.
//! §4.2 replaces the target: find the node's **Pareto frontier**, which by
//! Theorem 4.1 contains the nearest neighbour at *every* α at once.
//!
//! Computing the true frontier means scanning the whole dataset. GPS
//! approximates it on the graph already built, using the same bet every graph
//! index makes — a neighbour's neighbour is likely a neighbour — but expanding
//! along frontier *layers* rather than along a single distance ordering.
//!
//! # The loop
//!
//! Start from the seeds, organise everything discovered into layered frontiers,
//! then repeatedly expand the nodes of the **nearest layer that still has
//! unexpanded members** and re-layer. Restricting expansion to the nearest such
//! layer is what keeps this cheap: the paper notes it avoids the neighbour
//! explosion of expanding everything found (lines 11–14).
//!
//! Two structural notes, both deliberate:
//!
//! - Re-layering *discards* candidates beyond the pool size, but the visited set
//!   still remembers them, so a discarded node is never re-added and cannot cause
//!   a cycle.
//! - The pool can overshoot `ef_construction` by up to the width of the final
//!   layer, because [`frontier_layers`] keeps its last layer whole. Algorithm 1's
//!   output specification writes `≤ ef_construction`; the reasoning for
//!   overshooting instead is the same as in [`super::select`] — a layer is an
//!   unordered set of mutually non-dominating candidates, so cutting it to fit
//!   discards by array position rather than by any property of the data.

use std::collections::HashSet;

use super::pareto::{frontier_layers, ParetoPoint};
use super::pruning::HybridDistance;

/// Search for a node's approximate multi-layer Pareto frontier.
///
/// - `seeds`: entry points (the edge seeds of §4.4, or vertex 0 early in a build).
/// - `ef_construction`: target candidate-pool size.
/// - `limit`: ids at or above this do not exist yet, so a partially built graph
///   can be searched — the same device as `GraphIndex::search_internal`.
/// - `distance_to`: the hybrid distance from the query node to a candidate.
/// - `neighbours`: fills the buffer with a vertex's out-edges. Takes a buffer
///   rather than returning a `Vec` so an adjacency list can be read without
///   allocating per expansion.
///
/// Returns the layers, nearest first. Empty only if no seed is below `limit`.
pub fn gps<D, N>(
    seeds: &[u32],
    ef_construction: usize,
    limit: usize,
    distance_to: D,
    mut neighbours: N,
) -> Vec<Vec<ParetoPoint>>
where
    D: Fn(u32) -> HybridDistance,
    N: FnMut(u32, &mut Vec<u32>),
{
    let pool = ef_construction.max(1);
    let mut visited: HashSet<u32> = HashSet::new();
    // `Flag` in the paper: vertices whose neighbours have already been pulled in.
    let mut expanded: HashSet<u32> = HashSet::new();
    let mut discovered: Vec<ParetoPoint> = Vec::new();

    for &seed in seeds {
        if (seed as usize) < limit && visited.insert(seed) {
            discovered.push(ParetoPoint::new(seed, distance_to(seed)));
        }
    }
    if discovered.is_empty() {
        return Vec::new();
    }

    let mut layers = frontier_layers(&discovered, pool);
    let mut buffer: Vec<u32> = Vec::new();

    loop {
        let total: usize = layers.iter().map(Vec::len).sum();
        if total >= pool {
            break;
        }

        // The nearest layer with anything left to expand (lines 11–14).
        let mut to_expand: Vec<u32> = Vec::new();
        for layer in &layers {
            to_expand = layer
                .iter()
                .filter(|point| !expanded.contains(&point.id))
                .map(|point| point.id)
                .collect();
            if !to_expand.is_empty() {
                break;
            }
        }
        // Nothing new can improve the frontier (lines 15–16).
        if to_expand.is_empty() {
            break;
        }

        // Carry forward only what survived re-layering, matching the paper's
        // `Res`, then add the newly reachable vertices.
        discovered = layers.iter().flatten().copied().collect();
        for vertex in to_expand {
            expanded.insert(vertex);
            buffer.clear();
            neighbours(vertex, &mut buffer);
            for &next in &buffer {
                if (next as usize) < limit && visited.insert(next) {
                    discovered.push(ParetoPoint::new(next, distance_to(next)));
                }
            }
        }

        layers = frontier_layers(&discovered, pool);
    }

    layers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ann::deg::hybrid::HybridSet;
    use crate::workload::Rng;

    fn distances(set: &HybridSet, from: usize) -> impl Fn(u32) -> HybridDistance + '_ {
        move |id: u32| set.distance(from, id as usize)
    }

    /// A set of `count` objects with a 3-D primary and 2-D secondary modality.
    fn random_set(count: usize, seed: u64) -> HybridSet {
        let mut rng = Rng::new(seed);
        let primary: Vec<f32> = (0..count * 3).map(|_| rng.next_f64() as f32).collect();
        let secondary: Vec<f32> = (0..count * 2).map(|_| rng.next_f64() as f32).collect();
        HybridSet::new(primary, 3, secondary, 2, usize::MAX, seed)
    }

    #[test]
    fn with_no_edges_the_seeds_are_the_whole_result() {
        let set = random_set(10, 1);
        let layers = gps(&[3, 7], 16, set.len(), distances(&set, 0), |_, _| {});
        let ids: Vec<u32> = layers.iter().flatten().map(|p| p.id).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&3) && ids.contains(&7));
    }

    #[test]
    fn an_out_of_range_seed_is_ignored() {
        let set = random_set(10, 2);
        // limit = 5, so seed 9 does not exist yet.
        let layers = gps(&[9], 16, 5, distances(&set, 0), |_, _| {});
        assert!(layers.is_empty());
    }

    #[test]
    fn expansion_never_crosses_the_limit() {
        let set = random_set(30, 3);
        // A complete graph, but only the first 10 vertices are built.
        let layers = gps(&[0], 64, 10, distances(&set, 0), |_, out| {
            out.extend(0..30u32);
        });
        for point in layers.iter().flatten() {
            assert!(point.id < 10, "id {} is beyond the limit", point.id);
        }
    }

    #[test]
    fn every_returned_id_is_distinct() {
        let set = random_set(40, 4);
        let layers = gps(&[0], 32, set.len(), distances(&set, 5), |v, out| {
            // A ring with chords, so most vertices are reachable and many paths
            // revisit the same vertices.
            out.push((v + 1) % 40);
            out.push((v + 7) % 40);
            out.push((v + 39) % 40);
        });
        let ids: Vec<u32> = layers.iter().flatten().map(|p| p.id).collect();
        let unique: HashSet<u32> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len(), "duplicate ids in {ids:?}");
    }

    #[test]
    fn a_disconnected_graph_terminates() {
        let set = random_set(20, 5);
        // Vertex 0 points only at itself, so nothing new is ever discovered and
        // the pool can never fill. This must exit via the no-new-nodes break.
        let layers = gps(&[0], 64, set.len(), distances(&set, 1), |v, out| {
            out.push(v);
        });
        assert_eq!(layers.iter().flatten().count(), 1);
    }

    /// The property GPS exists to deliver: layer 1 of its output contains the
    /// nearest neighbour, at every α.
    ///
    /// Run over a complete graph so GPS can reach everything and its layer 1 is
    /// the true Pareto frontier — this checks the plumbing (visited sets,
    /// re-layering, carry-forward) rather than re-checking Theorem 4.1, which
    /// [`super::pareto`] already verifies directly. Compared on distances rather
    /// than ids so ties cannot produce a false failure.
    ///
    /// The query node is the *last* id and the graph is capped just below it, so
    /// it is not a member of the graph it searches. That mirrors Algorithm 3,
    /// which calls GPS for a node before inserting it — see
    /// [`the_query_node_dominates_its_own_frontier_if_it_is_in_the_graph`] for
    /// what happens otherwise.
    #[test]
    fn layer_one_holds_the_nearest_neighbour_at_every_alpha() {
        for trial in 0..20 {
            let set = random_set(60, 100 + trial);
            let inserting = set.len() - 1;
            let built = inserting; // ids 0..built exist; `inserting` does not yet
            let layers = gps(&[0], 200, built, distances(&set, inserting), |_, out| {
                out.extend(0..built as u32)
            });
            assert!(!layers.is_empty());

            for step in 0..=20 {
                let alpha = step as f32 / 20.0;
                let global = (0..built)
                    .map(|id| set.distance(inserting, id).at(alpha))
                    .fold(f32::INFINITY, f32::min);
                let best = layers[0]
                    .iter()
                    .map(|p| p.distance.at(alpha))
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    best <= global + 1e-6,
                    "trial {trial} α={alpha}: layer 1 best {best} worse than {global}"
                );
            }
        }
    }

    /// Documents why the test above keeps the query node out of the graph.
    ///
    /// A node's distance to itself is `(0, 0)`, which dominates every other
    /// point, so it alone occupies layer 1 and pushes the real neighbours into
    /// layer 2. That is the *correct* Pareto frontier, not a bug — it is simply
    /// not the situation Algorithm 3 ever creates, since GPS runs before the node
    /// is inserted.
    #[test]
    fn the_query_node_dominates_its_own_frontier_if_it_is_in_the_graph() {
        let set = random_set(30, 77);
        let from = 4usize;
        let layers = gps(&[0], 100, set.len(), distances(&set, from), |_, out| {
            out.extend(0..30u32);
        });
        assert_eq!(
            layers[0].iter().map(|p| p.id).collect::<Vec<_>>(),
            vec![from as u32],
            "self-distance (0,0) dominates everything"
        );
        assert!(layers.len() > 1, "the real neighbours land in later layers");
    }

    #[test]
    fn the_pool_size_bounds_the_result_to_within_one_layer() {
        let set = random_set(80, 9);
        let pool = 20;
        let layers = gps(&[0], pool, set.len(), distances(&set, 0), |_, out| {
            out.extend(0..80u32);
        });
        let total: usize = layers.iter().map(Vec::len).sum();
        assert!(total >= pool, "should fill the pool, got {total}");
        let without_last: usize = layers[..layers.len() - 1].iter().map(Vec::len).sum();
        assert!(
            without_last < pool,
            "only the last layer may overshoot: {without_last} of {pool}"
        );
    }

    #[test]
    fn a_sparse_graph_still_reaches_beyond_its_seed() {
        // A plain ring: each expansion adds two vertices, so filling a pool of 16
        // takes several rounds and exercises the carry-forward path.
        let set = random_set(50, 11);
        let layers = gps(&[0], 16, set.len(), distances(&set, 0), |v, out| {
            out.push((v + 1) % 50);
            out.push((v + 49) % 50);
        });
        let total: usize = layers.iter().map(Vec::len).sum();
        assert!(total >= 16, "expected the pool to fill, got {total}");
    }

    #[test]
    fn degenerate_inputs_are_handled() {
        let set = random_set(5, 13);
        assert!(gps(&[], 8, set.len(), distances(&set, 0), |_, _| {}).is_empty());
        assert!(gps(&[0], 8, 0, distances(&set, 0), |_, _| {}).is_empty());
        // A zero pool is floored at 1 rather than returning nothing.
        let layers = gps(&[0], 0, set.len(), distances(&set, 0), |_, _| {});
        assert_eq!(layers.iter().flatten().count(), 1);
    }
}
