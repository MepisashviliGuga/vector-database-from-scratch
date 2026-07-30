//! Pareto frontiers over the two modalities — paper 05 §4.2.
//!
//! # Why a frontier is the right candidate set
//!
//! When building a graph you must choose each node's candidate neighbours. Every
//! other index picks the nearest ones, but "nearest" is undefined here until α
//! arrives. §4.2 asks a weaker question that *is* answerable: which objects could
//! **ever** be the nearest neighbour, at any α?
//!
//! Treat `(δe, δs)` as two objectives to minimise. If `o'` is at least as close
//! as `o` in both modalities and strictly closer in one, then
//!
//! ```text
//! Dist_α(o') = α·δe(o') + (1−α)·δs(o')  ≤  α·δe(o) + (1−α)·δs(o) = Dist_α(o)
//! ```
//!
//! for every α in `[0,1]`, because both coefficients are non-negative. So `o`
//! can never beat `o'` — it is *dominated* and can be discarded outright.
//!
//! What remains is the Pareto frontier, and Theorem 4.1 is the consequence: for
//! any α, the nearest neighbour of `p` lies in `PF(D, p)`. That is what makes one
//! candidate set serve every α at once.
//!
//! # Why layers
//!
//! A frontier is thin — the paper notes ~10 points where an index wants ~200
//! candidates (§4.2). So after taking a frontier you remove it and take the next
//! one from what is left, peeling *layers* (Figure 4a). Only layer 1 carries the
//! Theorem 4.1 guarantee; the deeper layers are there to fill the candidate quota
//! that edge selection then prunes back down.

use super::pruning::HybridDistance;

/// One candidate, with its distance to the node whose frontier is being built.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParetoPoint {
    pub id: u32,
    pub distance: HybridDistance,
}

impl ParetoPoint {
    pub fn new(id: u32, distance: HybridDistance) -> Self {
        Self { id, distance }
    }
}

/// Whether `a` dominates `b`: no worse in both modalities, strictly better in
/// at least one.
///
/// The strictness matters. Two objects at exactly the same `(δe, δs)` do *not*
/// dominate each other, so both stay on the frontier — dropping one would risk
/// discarding a genuine nearest neighbour whose twin has a different id.
pub fn dominates(a: HybridDistance, b: HybridDistance) -> bool {
    a.e <= b.e && a.s <= b.s && (a.e < b.e || a.s < b.s)
}

/// Peel successive Pareto frontiers until at least `limit` points are collected.
///
/// This is the `FindPF` of Algorithm 1: it returns the first `l` layers whose
/// running total satisfies the candidate-pool size, so the last layer may push
/// the total past `limit` rather than being truncated — truncating a layer would
/// cut it at an arbitrary point in a set whose whole purpose is to be
/// unordered-but-complete.
///
/// Layers are returned nearest-first. Points not reached are simply omitted.
pub fn frontier_layers(points: &[ParetoPoint], limit: usize) -> Vec<Vec<ParetoPoint>> {
    let mut remaining: Vec<ParetoPoint> = points.to_vec();
    let mut layers = Vec::new();
    let mut collected = 0usize;

    while !remaining.is_empty() && collected < limit {
        let (layer, rest) = split_frontier(&remaining);
        if layer.is_empty() {
            // Cannot happen for a non-empty input — the minimum is always
            // undominated — but guard rather than spin.
            break;
        }
        collected += layer.len();
        layers.push(layer);
        remaining = rest;
    }

    layers
}

/// The single Pareto frontier of `points`, and everything it leaves behind.
///
/// Uses the standard 2-D skyline sweep: sort by `δe` ascending, then walk keeping
/// the smallest `δs` seen. A point survives when its `δs` beats every `δs` seen
/// at a smaller-or-equal `δe`, which is exactly "not dominated" — with the
/// duplicate-handling exception described on [`dominates`], implemented by
/// running the sweep over distinct coordinate pairs and re-attaching every id
/// that shares a surviving pair.
pub fn split_frontier(points: &[ParetoPoint]) -> (Vec<ParetoPoint>, Vec<ParetoPoint>) {
    if points.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by(|&i, &j| {
        let a = points[i].distance;
        let b = points[j].distance;
        a.e.total_cmp(&b.e).then_with(|| a.s.total_cmp(&b.s))
    });

    // Sweep over distinct (δe, δs) pairs, deciding each pair once.
    let mut best_s = f32::INFINITY;
    let mut on_frontier = vec![false; points.len()];
    let mut cursor = 0usize;
    while cursor < order.len() {
        let here = points[order[cursor]].distance;
        // Collect the run of points sharing this exact coordinate pair.
        let mut end = cursor + 1;
        while end < order.len() {
            let next = points[order[end]].distance;
            if next.e == here.e && next.s == here.s {
                end += 1;
            } else {
                break;
            }
        }

        if here.s < best_s {
            // Undominated: nothing at a smaller-or-equal δe had a δs this small.
            // Every id at this exact coordinate qualifies equally.
            for slot in &order[cursor..end] {
                on_frontier[*slot] = true;
            }
            best_s = here.s;
        }

        cursor = end;
    }

    let mut frontier = Vec::new();
    let mut rest = Vec::new();
    for (index, point) in points.iter().enumerate() {
        if on_frontier[index] {
            frontier.push(*point);
        } else {
            rest.push(*point);
        }
    }
    (frontier, rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(id: u32, e: f32, s: f32) -> ParetoPoint {
        ParetoPoint::new(id, HybridDistance::new(e, s))
    }

    #[test]
    fn domination_needs_one_strict_improvement() {
        let a = HybridDistance::new(0.2, 0.3);
        assert!(dominates(a, HybridDistance::new(0.4, 0.5)), "better in both");
        assert!(dominates(a, HybridDistance::new(0.2, 0.5)), "tie then better");
        assert!(!dominates(a, a), "identical points do not dominate");
        assert!(
            !dominates(a, HybridDistance::new(0.1, 0.9)),
            "better in one, worse in the other"
        );
    }

    #[test]
    fn a_dominated_point_is_excluded_from_the_first_layer() {
        let points = vec![point(0, 0.2, 0.3), point(1, 0.4, 0.5)];
        let (frontier, rest) = split_frontier(&points);
        assert_eq!(frontier.iter().map(|p| p.id).collect::<Vec<_>>(), vec![0]);
        assert_eq!(rest.iter().map(|p| p.id).collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn a_trade_off_keeps_both_points() {
        // Neither dominates: 0 is better in the first modality, 1 in the second.
        let points = vec![point(0, 0.1, 0.9), point(1, 0.9, 0.1)];
        let (frontier, rest) = split_frontier(&points);
        assert_eq!(frontier.len(), 2);
        assert!(rest.is_empty());
    }

    #[test]
    fn identical_coordinates_both_survive() {
        // Per `dominates`, ties are not domination. Both ids must be kept, since
        // either could be the true nearest neighbour.
        let points = vec![point(7, 0.3, 0.4), point(9, 0.3, 0.4), point(1, 0.8, 0.9)];
        let (frontier, _) = split_frontier(&points);
        let mut ids: Vec<u32> = frontier.iter().map(|p| p.id).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![7, 9]);
    }

    #[test]
    fn equal_second_modality_at_a_larger_first_is_dominated() {
        // Same δs, worse δe → dominated, so only the cheaper δe survives.
        let points = vec![point(0, 0.2, 0.5), point(1, 0.6, 0.5)];
        let (frontier, rest) = split_frontier(&points);
        assert_eq!(frontier.iter().map(|p| p.id).collect::<Vec<_>>(), vec![0]);
        assert_eq!(rest.iter().map(|p| p.id).collect::<Vec<_>>(), vec![1]);
    }

    /// Theorem 4.1, checked directly: for any α, the nearest neighbour is on the
    /// first frontier.
    ///
    /// Compares the best hybrid distance achievable *within layer 1* against the
    /// best over *all* points, at many α. Comparing distances rather than ids
    /// makes the check tie-proof: several objects may share the minimum, and the
    /// theorem only requires that one of them is on the frontier.
    #[test]
    fn theorem_4_1_the_nearest_neighbour_is_always_on_the_first_frontier() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32) / (1u64 << 24) as f32
        };

        for trial in 0..300 {
            let count = 2 + (trial % 40);
            let points: Vec<ParetoPoint> = (0..count)
                .map(|i| point(i as u32, next(), next()))
                .collect();
            let (frontier, _) = split_frontier(&points);
            assert!(!frontier.is_empty());

            for step in 0..=50 {
                let alpha = step as f32 / 50.0;
                let global = points
                    .iter()
                    .map(|p| p.distance.at(alpha))
                    .fold(f32::INFINITY, f32::min);
                let best_on_frontier = frontier
                    .iter()
                    .map(|p| p.distance.at(alpha))
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    best_on_frontier <= global + 1e-6,
                    "trial {trial} at α={alpha}: frontier best {best_on_frontier} \
                     worse than global best {global}"
                );
            }
        }
    }

    #[test]
    fn every_point_off_the_frontier_is_dominated_by_one_on_it() {
        // The converse property: the sweep must not discard anything undominated.
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (((state >> 45) % 20) as f32) / 20.0
        };

        for trial in 0..300 {
            let points: Vec<ParetoPoint> = (0..15).map(|i| point(i, next(), next())).collect();
            let (frontier, rest) = split_frontier(&points);
            for dropped in &rest {
                assert!(
                    frontier.iter().any(|kept| dominates(kept.distance, dropped.distance)),
                    "trial {trial}: {dropped:?} was dropped but nothing on the frontier \
                     dominates it\nfrontier={frontier:?}"
                );
            }
            for kept in &frontier {
                assert!(
                    !points
                        .iter()
                        .any(|other| dominates(other.distance, kept.distance)),
                    "trial {trial}: {kept:?} is on the frontier but is dominated"
                );
            }
        }
    }

    #[test]
    fn layers_partition_the_input() {
        let points: Vec<ParetoPoint> = (0..40)
            .map(|i| {
                let f = i as f32;
                point(i, (f * 7.0 % 13.0) / 13.0, (f * 11.0 % 17.0) / 17.0)
            })
            .collect();

        let layers = frontier_layers(&points, points.len());
        let mut seen: Vec<u32> = layers.iter().flatten().map(|p| p.id).collect();
        seen.sort_unstable();
        let mut expected: Vec<u32> = points.iter().map(|p| p.id).collect();
        expected.sort_unstable();
        assert_eq!(seen, expected, "every point should land in exactly one layer");
    }

    #[test]
    fn layers_are_ordered_nearest_first() {
        // A later layer's points must each be dominated by something earlier;
        // that is what makes "peel and repeat" a meaningful ordering.
        let points: Vec<ParetoPoint> = (0..30)
            .map(|i| {
                let f = i as f32;
                point(i, (f * 5.0 % 11.0) / 11.0, (f * 3.0 % 7.0) / 7.0)
            })
            .collect();

        let layers = frontier_layers(&points, points.len());
        assert!(layers.len() >= 2, "expected several layers, got {}", layers.len());
        for depth in 1..layers.len() {
            for candidate in &layers[depth] {
                let dominated_by_previous = layers[depth - 1]
                    .iter()
                    .any(|earlier| dominates(earlier.distance, candidate.distance));
                assert!(
                    dominated_by_previous,
                    "layer {depth} point {candidate:?} is not dominated by layer {}",
                    depth - 1
                );
            }
        }
    }

    #[test]
    fn peeling_stops_once_the_pool_is_full() {
        let points: Vec<ParetoPoint> = (0..60)
            .map(|i| {
                let f = i as f32;
                point(i, (f * 7.0 % 13.0) / 13.0, (f * 11.0 % 17.0) / 17.0)
            })
            .collect();

        let full = frontier_layers(&points, points.len());
        let capped = frontier_layers(&points, 5);
        assert!(
            capped.len() < full.len(),
            "a small limit should stop early: {} vs {}",
            capped.len(),
            full.len()
        );
        let total: usize = capped.iter().map(|l| l.len()).sum();
        assert!(total >= 5, "should reach the limit, got {total}");
        // The final layer is kept whole rather than cut to fit.
        let without_last: usize = capped[..capped.len() - 1].iter().map(|l| l.len()).sum();
        assert!(without_last < 5, "only the last layer may overshoot");
    }

    #[test]
    fn degenerate_inputs_are_handled() {
        assert!(split_frontier(&[]).0.is_empty());
        assert!(frontier_layers(&[], 10).is_empty());
        assert!(
            frontier_layers(&[point(0, 0.5, 0.5)], 0).is_empty(),
            "a zero limit collects nothing"
        );
        let single = frontier_layers(&[point(3, 0.5, 0.5)], 10);
        assert_eq!(single.len(), 1);
        assert_eq!(single[0][0].id, 3);
    }
}
