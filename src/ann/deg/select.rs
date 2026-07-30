//! Turning a candidate set into edges with active ranges — paper 05 Algorithm 2.
//!
//! This is where [`super::pareto`] and [`super::pruning`] meet. The frontier
//! layers say *which* objects could matter; the solver says *at which α* the RNG
//! rule would discard each edge. Algorithm 2 walks the layers nearest-first and,
//! for each candidate, unions the pruning ranges contributed by every neighbour
//! already selected. What is left over — the complement — is the edge's active
//! range, and the edge is kept only if that range is wide enough to be worth
//! storing.
//!
//! The nearest-first order is what makes this work: a candidate is only ever
//! tested against neighbours that are already committed, exactly as HNSW's
//! selection heuristic tests against already-chosen neighbours rather than all
//! candidates. Reversing the order would let a far neighbour veto a near one.

use super::interval::AlphaSet;
use super::pareto::ParetoPoint;
use super::pruning::{pruned_by, HybridDistance};

/// An outgoing edge, live only while the query's α falls in `active`.
#[derive(Debug, Clone, PartialEq)]
pub struct DegEdge {
    pub target: u32,
    /// The α values at which this edge may be traversed — Algorithm 2 line 9.
    pub active: AlphaSet,
}

/// Select a node's neighbours from its candidate layers — `DRNGPrune(CS, M, th)`.
///
/// - `layers`: the multi-layer Pareto frontier of the node, nearest layer first.
/// - `max_edges`: `M`, the cap on out-degree.
/// - `min_active_width`: `th`. An edge usable over only a sliver of α costs a
///   stored range and a per-hop test while almost never being traversed, so the
///   paper discards it (lines 10–11). A `th` of 0 keeps every edge with any
///   active α at all.
/// - `pair_distance`: distance between two *candidates*. The frontier supplies
///   each candidate's distance to the node being inserted, but the RNG triangle
///   also needs the candidate-to-neighbour side, which only the vectors can give.
///
/// # The `M` cap is checked per layer
///
/// Algorithm 2 tests `|NS| ≥ M` after finishing a layer (lines 12–13), not after
/// each edge, so the result can exceed `M` by up to the width of the last layer.
/// That is the paper's structure and it is kept: a frontier layer is an unordered
/// set of mutually non-dominating candidates, so cutting one mid-way would
/// discard by array position rather than by any property of the data.
pub fn select_neighbours<F>(
    layers: &[Vec<ParetoPoint>],
    max_edges: usize,
    min_active_width: f32,
    pair_distance: F,
) -> Vec<DegEdge>
where
    F: Fn(u32, u32) -> HybridDistance,
{
    let mut selected: Vec<(ParetoPoint, DegEdge)> = Vec::new();

    for layer in layers {
        for candidate in layer {
            // The α values at which some already-selected neighbour prunes this
            // edge. Union, because any one neighbour is enough to prune.
            let mut pruned = AlphaSet::empty();
            for (neighbour, _) in &selected {
                // The triangle is (p, candidate, neighbour) with the edge under
                // test being p→candidate:
                //   Eq 2  δ(p, neighbour)      < δ(p, candidate)
                //   Eq 3  δ(candidate, neighbour) < δ(p, candidate)
                let range = pruned_by(
                    candidate.distance,
                    neighbour.distance,
                    pair_distance(candidate.id, neighbour.id),
                );
                pruned = pruned.union(&range);
                if pruned.is_full() {
                    break;
                }
            }

            let active = pruned.complement();
            if active.measure() >= min_active_width && !active.is_empty() {
                selected.push((
                    *candidate,
                    DegEdge {
                        target: candidate.id,
                        active,
                    },
                ));
            }
        }

        if selected.len() >= max_edges {
            break;
        }
    }

    selected.into_iter().map(|(_, edge)| edge).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ann::deg::pruning::prunes_at;

    fn point(id: u32, e: f32, s: f32) -> ParetoPoint {
        ParetoPoint::new(id, HybridDistance::new(e, s))
    }

    /// No pair distances are needed when a test never reaches the triangle.
    fn unreachable(_: u32, _: u32) -> HybridDistance {
        panic!("pair_distance should not be consulted here");
    }

    #[test]
    fn the_first_candidate_is_always_kept_with_a_full_range() {
        // Nothing is selected yet, so no neighbour can prune it.
        let layers = vec![vec![point(5, 0.3, 0.4)]];
        let edges = select_neighbours(&layers, 8, 0.0, unreachable);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target, 5);
        assert!(edges[0].active.is_full());
    }

    #[test]
    fn table_1_example_3_becomes_an_edge_active_below_two_thirds() {
        // The paper's worked example, driven through selection rather than the
        // solver directly. Mapping Table 1's (x,y,z) onto this API: the node
        // being inserted is x, the candidate is y, the already-selected
        // neighbour is z.
        //
        //   δ(x,y) = (e 0.6, s 0.2)   the candidate's distance to the node
        //   δ(x,z) = (e 0.5, s 0.4)   the neighbour's distance to the node
        //   δ(y,z) = (e 0.4, s 0.3)   candidate-to-neighbour
        //
        // §4.3 Example 1 concludes the edge is pruned on [2/3, 1], so it must
        // remain active on exactly the complement, [0, 2/3].
        //
        // The two are in separate layers only to force the selection order;
        // neither dominates the other, so a real frontier would place both in
        // one layer and the order would depend on the sweep.
        let layers = vec![vec![point(1, 0.5, 0.4)], vec![point(2, 0.6, 0.2)]];
        let edges = select_neighbours(&layers, 8, 0.0, |a, b| {
            assert!(
                (a == 2 && b == 1) || (a == 1 && b == 2),
                "unexpected pair ({a},{b})"
            );
            HybridDistance::new(0.4, 0.3)
        });

        assert_eq!(edges.len(), 2, "both edges survive: {edges:?}");
        let candidate_edge = edges.iter().find(|e| e.target == 2).expect("edge to 2");
        assert_eq!(candidate_edge.active.intervals().len(), 1);
        let (low, high) = candidate_edge.active.intervals()[0];
        assert_eq!(low, 0.0);
        assert!(
            (high - 2.0 / 3.0).abs() < 1e-6,
            "expected active up to 2/3, got {high}"
        );
        assert!(candidate_edge.active.contains(0.0), "kept at α=0");
        assert!(!candidate_edge.active.contains(0.9), "pruned at α=0.9");
    }

    #[test]
    fn an_edge_pruned_at_every_alpha_is_dropped() {
        // Table 1's second example: node z prunes edge (x,y) for all α. Same
        // mapping as above, so the candidate must not survive at all.
        let layers = vec![vec![point(1, 0.4, 0.2)], vec![point(2, 0.7, 0.5)]];
        let edges = select_neighbours(&layers, 8, 0.0, |_, _| HybridDistance::new(0.5, 0.3));
        assert_eq!(edges.len(), 1, "only the neighbour survives: {edges:?}");
        assert_eq!(edges[0].target, 1);
    }

    #[test]
    fn the_threshold_discards_narrow_active_ranges() {
        // Same geometry as the Table 1 example, whose surviving range is
        // [0, 2/3] with measure 0.667. A threshold above that must drop it.
        let layers = vec![vec![point(1, 0.5, 0.4)], vec![point(2, 0.6, 0.2)]];
        let pair = |_: u32, _: u32| HybridDistance::new(0.4, 0.3);

        let kept = select_neighbours(&layers, 8, 0.6, pair);
        assert!(kept.iter().any(|e| e.target == 2), "0.667 ≥ 0.6, kept");

        let dropped = select_neighbours(&layers, 8, 0.7, pair);
        assert!(
            !dropped.iter().any(|e| e.target == 2),
            "0.667 < 0.7, should be dropped: {dropped:?}"
        );
    }

    #[test]
    fn selection_order_decides_which_of_two_rivals_survives() {
        // Reversing the layers swaps which node is already committed when the
        // other is tested, which is why nearest-first ordering matters.
        let pair = |_: u32, _: u32| HybridDistance::new(0.4, 0.3);
        let forward = select_neighbours(
            &[vec![point(1, 0.5, 0.4)], vec![point(2, 0.6, 0.2)]],
            8,
            0.9,
            pair,
        );
        let reversed = select_neighbours(
            &[vec![point(2, 0.6, 0.2)], vec![point(1, 0.5, 0.4)]],
            8,
            0.9,
            pair,
        );
        let forward_ids: Vec<u32> = forward.iter().map(|e| e.target).collect();
        let reversed_ids: Vec<u32> = reversed.iter().map(|e| e.target).collect();
        assert_ne!(
            forward_ids, reversed_ids,
            "order should change the outcome at a strict threshold"
        );
    }

    #[test]
    fn the_active_range_matches_the_pruning_rule_pointwise() {
        // For every kept edge, the stored range must agree at each α with
        // evaluating the RNG condition directly against every earlier neighbour.
        let mut state = 0x5deb_ce2d_5aa1_9e3fu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (((state >> 43) % 16) as f32) / 16.0
        };

        for trial in 0..200 {
            let candidates: Vec<ParetoPoint> = (0..6).map(|i| point(i, next(), next())).collect();
            // A fixed table of pairwise distances, so the closure is consistent.
            let mut pairs = vec![HybridDistance::new(0.0, 0.0); 36];
            for i in 0..6 {
                for j in (i + 1)..6 {
                    let d = HybridDistance::new(next(), next());
                    pairs[i * 6 + j] = d;
                    pairs[j * 6 + i] = d;
                }
            }
            let pair = |a: u32, b: u32| pairs[a as usize * 6 + b as usize];

            // One candidate per layer, so the selection order is explicit and
            // the expected pruning set can be recomputed independently.
            let layers: Vec<Vec<ParetoPoint>> = candidates.iter().map(|c| vec![*c]).collect();
            let edges = select_neighbours(&layers, 64, 0.0, pair);

            // Replay: walk the same order, tracking which are committed.
            let mut committed: Vec<ParetoPoint> = Vec::new();
            for candidate in &candidates {
                let edge = edges.iter().find(|e| e.target == candidate.id);
                for step in 0..=40 {
                    let alpha = step as f32 / 40.0;
                    let pruned_here = committed.iter().any(|n| {
                        prunes_at(
                            alpha,
                            candidate.distance,
                            n.distance,
                            pair(candidate.id, n.id),
                        )
                    });
                    if let Some(edge) = edge {
                        if pruned_here && edge.active.contains(alpha) {
                            // Tolerate boundary ties, as elsewhere.
                            let near = edge.active.intervals().iter().any(|&(lo, hi)| {
                                (alpha - lo).abs() < 1e-3 || (alpha - hi).abs() < 1e-3
                            });
                            assert!(
                                near,
                                "trial {trial}: edge to {} active at α={alpha} but pruned",
                                candidate.id
                            );
                        }
                    }
                }
                if edge.is_some() {
                    committed.push(*candidate);
                }
            }
        }
    }

    #[test]
    fn the_edge_cap_stops_after_a_layer() {
        // Four layers of two candidates each, with pair distances far enough
        // apart that nothing prunes anything, so only the cap can bite.
        let layers: Vec<Vec<ParetoPoint>> = (0..4)
            .map(|l| {
                let f = l as f32;
                vec![
                    point(l * 2, 0.1 + f * 0.2, 0.9 - f * 0.2),
                    point(l * 2 + 1, 0.15 + f * 0.2, 0.85 - f * 0.2),
                ]
            })
            .collect();
        // A huge candidate-to-candidate distance can never be the shortest side,
        // so Eq 3 never holds and nothing is pruned.
        let edges = select_neighbours(&layers, 3, 0.0, |_, _| HybridDistance::new(9.0, 9.0));
        assert_eq!(
            edges.len(),
            4,
            "stops after the layer that reaches M=3, so 4 not 3: {edges:?}"
        );
    }

    #[test]
    fn degenerate_inputs_are_handled() {
        assert!(select_neighbours(&[], 8, 0.0, unreachable).is_empty());
        assert!(select_neighbours(&[vec![]], 8, 0.0, unreachable).is_empty());
        // A zero cap still processes the first layer, since the test is after it.
        let edges = select_neighbours(&[vec![point(1, 0.2, 0.3)]], 0, 0.0, unreachable);
        assert_eq!(edges.len(), 1);
    }
}
