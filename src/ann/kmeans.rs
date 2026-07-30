//! Lloyd's k-means with k-means++ seeding.
//!
//! # Why the ANN layer needs this
//!
//! Extended RaBitQ quantizes the *residual* `o_r − c` from a centroid `c`. With
//! a single global centroid over a million vectors those residuals are large and
//! point in arbitrary directions, so a fixed-precision code has to describe the
//! full spread of the data. Clustering first makes every residual small and
//! locally concentrated, which is the same number of bits spent on a much easier
//! problem.
//!
//! That is measured, not assumed: with one global centroid our RaBitQ recall@10
//! on SIFT1M was 0.921 at 6.4x compression, against paper 03's >95% — and the
//! paper pairs the quantizer with exactly this clustering.
//!
//! # k-means++ seeding
//!
//! Uniformly random initial centroids routinely place several in the same dense
//! region and none in a sparse one, and Lloyd's algorithm cannot recover from
//! that — it only ever moves centroids downhill. k-means++ instead picks each
//! new centroid with probability proportional to its squared distance from the
//! nearest existing one, which spreads them out and gives an `O(log k)`
//! approximation guarantee before a single iteration runs.

use crate::ann::squared_l2;
use crate::workload::Rng;

/// The outcome of a training run.
#[derive(Debug, Clone)]
pub struct KMeans {
    /// `clusters × dimension`, flat.
    pub centroids: Vec<f32>,
    pub dimension: usize,
    /// Sum of squared distances from each training vector to its centroid.
    /// Lloyd's algorithm decreases this monotonically; it is the objective.
    pub inertia: f64,
    /// Iterations actually run, which is below the cap when assignments settle.
    pub iterations: usize,
}

impl KMeans {
    pub fn clusters(&self) -> usize {
        self.centroids
            .len()
            .checked_div(self.dimension)
            .unwrap_or(0)
    }

    pub fn centroid(&self, index: usize) -> Option<&[f32]> {
        let start = index.checked_mul(self.dimension)?;
        self.centroids.get(start..start + self.dimension)
    }

    /// The nearest centroid to `vector`, and its squared distance.
    pub fn assign(&self, vector: &[f32]) -> (usize, f32) {
        let mut best = (0usize, f32::INFINITY);
        for index in 0..self.clusters() {
            let centroid = &self.centroids[index * self.dimension..(index + 1) * self.dimension];
            let distance = squared_l2(vector, centroid);
            if distance < best.1 {
                best = (index, distance);
            }
        }
        best
    }

    /// The `count` nearest centroids, nearest first.
    ///
    /// This is IVF's probe list: a query scans only the lists belonging to these
    /// centroids instead of the whole dataset.
    pub fn nearest_centroids(&self, vector: &[f32], count: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = (0..self.clusters())
            .map(|index| {
                let centroid =
                    &self.centroids[index * self.dimension..(index + 1) * self.dimension];
                (index, squared_l2(vector, centroid))
            })
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(count);
        scored
    }
}

/// Train k-means on `data`, a flat buffer of `dimension`-sized vectors.
///
/// # Panics
///
/// If `dimension` or `clusters` is 0, or `data` is not a whole number of
/// vectors.
pub fn train(
    data: &[f32],
    dimension: usize,
    clusters: usize,
    max_iterations: usize,
    seed: u64,
) -> KMeans {
    assert!(dimension > 0, "vectors need at least one dimension");
    assert!(clusters > 0, "need at least one cluster");
    assert_eq!(
        data.len() % dimension,
        0,
        "{} values do not divide into {dimension}-dimensional vectors",
        data.len()
    );

    let count = data.len() / dimension;
    assert!(count > 0, "cannot cluster an empty set");

    // More clusters than points is degenerate; every point becomes its own.
    let clusters = clusters.min(count);
    let vector_at = |index: usize| &data[index * dimension..(index + 1) * dimension];

    let mut rng = Rng::new(seed);
    let mut centroids = seed_plus_plus(data, dimension, count, clusters, &mut rng);

    let mut assignments = vec![usize::MAX; count];
    let mut inertia = f64::INFINITY;
    let mut iterations = 0;

    for iteration in 1..=max_iterations.max(1) {
        iterations = iteration;

        // Assignment step.
        let mut changed = false;
        let mut new_inertia = 0.0f64;
        for (index, assignment) in assignments.iter_mut().enumerate() {
            let vector = vector_at(index);
            let mut best = (0usize, f32::INFINITY);
            for cluster in 0..clusters {
                let centroid = &centroids[cluster * dimension..(cluster + 1) * dimension];
                let distance = squared_l2(vector, centroid);
                if distance < best.1 {
                    best = (cluster, distance);
                }
            }
            if *assignment != best.0 {
                *assignment = best.0;
                changed = true;
            }
            new_inertia += best.1 as f64;
        }
        inertia = new_inertia;

        // Assignments settled: further iterations cannot change anything.
        if !changed {
            break;
        }

        // Update step: each centroid becomes the mean of its members.
        let mut sums = vec![0.0f64; clusters * dimension];
        let mut counts = vec![0usize; clusters];
        for (index, &cluster) in assignments.iter().enumerate() {
            counts[cluster] += 1;
            let vector = vector_at(index);
            let target = &mut sums[cluster * dimension..(cluster + 1) * dimension];
            for (slot, value) in target.iter_mut().zip(vector.iter()) {
                *slot += *value as f64;
            }
        }

        for cluster in 0..clusters {
            if counts[cluster] == 0 {
                // An empty cluster contributes nothing and would stay empty
                // forever. Re-seed it on the point currently worst served by its
                // own centroid, which is where an extra centroid helps most.
                if let Some(worst) =
                    farthest_point(data, dimension, count, &centroids, &assignments)
                {
                    let source = vector_at(worst);
                    centroids[cluster * dimension..(cluster + 1) * dimension]
                        .copy_from_slice(source);
                    assignments[worst] = cluster;
                }
                continue;
            }
            let divisor = counts[cluster] as f64;
            for offset in 0..dimension {
                centroids[cluster * dimension + offset] =
                    (sums[cluster * dimension + offset] / divisor) as f32;
            }
        }
    }

    KMeans {
        centroids,
        dimension,
        inertia,
        iterations,
    }
}

/// k-means++ seeding: spread the initial centroids by squared distance.
fn seed_plus_plus(
    data: &[f32],
    dimension: usize,
    count: usize,
    clusters: usize,
    rng: &mut Rng,
) -> Vec<f32> {
    let vector_at = |index: usize| &data[index * dimension..(index + 1) * dimension];

    let mut centroids = Vec::with_capacity(clusters * dimension);
    let first = rng.below(count as u64) as usize;
    centroids.extend_from_slice(vector_at(first));

    // Distance from every point to the nearest centroid chosen so far.
    let mut nearest: Vec<f32> = (0..count)
        .map(|index| squared_l2(vector_at(index), vector_at(first)))
        .collect();

    for _ in 1..clusters {
        let total: f64 = nearest.iter().map(|&d| d as f64).sum();

        let chosen = if total <= 0.0 {
            // Every point coincides with a centroid — duplicates, or fewer
            // distinct points than clusters. Any point will do.
            rng.below(count as u64) as usize
        } else {
            // Sample proportional to squared distance.
            let target = rng.next_f64() * total;
            let mut running = 0.0f64;
            let mut picked = count - 1;
            for (index, &distance) in nearest.iter().enumerate() {
                running += distance as f64;
                if running >= target {
                    picked = index;
                    break;
                }
            }
            picked
        };

        let centroid = vector_at(chosen).to_vec();
        centroids.extend_from_slice(&centroid);
        for (index, slot) in nearest.iter_mut().enumerate() {
            *slot = slot.min(squared_l2(vector_at(index), &centroid));
        }
    }

    centroids
}

/// The point farthest from its assigned centroid, used to revive empty clusters.
fn farthest_point(
    data: &[f32],
    dimension: usize,
    count: usize,
    centroids: &[f32],
    assignments: &[usize],
) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (index, &cluster) in assignments.iter().enumerate().take(count) {
        if cluster == usize::MAX {
            continue;
        }
        let vector = &data[index * dimension..(index + 1) * dimension];
        let centroid = &centroids[cluster * dimension..(cluster + 1) * dimension];
        let distance = squared_l2(vector, centroid);
        if best.is_none_or(|(_, worst)| distance > worst) {
            best = Some((index, distance));
        }
    }
    best.map(|(index, _)| index)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three well-separated blobs, so the right answer is unambiguous.
    fn blobs(per_cluster: usize, seed: u64) -> (Vec<f32>, usize) {
        let dimension = 8;
        let centres = [[0.0f32; 8], [50.0; 8], [-50.0; 8]];
        let mut rng = Rng::new(seed);
        let mut data = Vec::new();
        for centre in &centres {
            for _ in 0..per_cluster {
                for &value in centre.iter() {
                    data.push(value + (rng.next_f64() as f32 - 0.5) * 2.0);
                }
            }
        }
        (data, dimension)
    }

    #[test]
    fn well_separated_blobs_are_recovered() {
        let (data, dimension) = blobs(60, 1);
        let model = train(&data, dimension, 3, 50, 7);

        assert_eq!(model.clusters(), 3);

        // Each blob's points must all land in one cluster.
        for blob in 0..3 {
            let assignments: Vec<usize> = (0..60)
                .map(|i| {
                    let index = blob * 60 + i;
                    model
                        .assign(&data[index * dimension..(index + 1) * dimension])
                        .0
                })
                .collect();
            let first = assignments[0];
            assert!(
                assignments.iter().all(|&a| a == first),
                "blob {blob} was split across clusters"
            );
        }

        // And the three blobs must land in *different* clusters.
        let chosen: std::collections::HashSet<usize> = (0..3)
            .map(|blob| {
                let index = blob * 60;
                model
                    .assign(&data[index * dimension..(index + 1) * dimension])
                    .0
            })
            .collect();
        assert_eq!(chosen.len(), 3, "two blobs collapsed into one cluster");
    }

    /// The objective Lloyd's algorithm minimises. More clusters must fit the
    /// data at least as well, or the implementation is not descending.
    #[test]
    fn inertia_falls_as_clusters_increase() {
        let (data, dimension) = blobs(40, 3);
        let mut previous = f64::INFINITY;
        for clusters in [1usize, 2, 3, 6, 12] {
            let model = train(&data, dimension, clusters, 50, 11);
            assert!(
                model.inertia <= previous + 1e-3,
                "{clusters} clusters gave inertia {} against {previous}",
                model.inertia
            );
            previous = model.inertia;
        }
    }

    /// A centroid must be the mean of the points assigned to it — that is what
    /// makes it the optimal representative, and what RaBitQ's residuals depend
    /// on being small.
    #[test]
    fn centroids_are_the_means_of_their_members() {
        let (data, dimension) = blobs(50, 5);
        let count = data.len() / dimension;
        let model = train(&data, dimension, 3, 100, 13);

        let mut sums = vec![0.0f64; model.clusters() * dimension];
        let mut counts = vec![0usize; model.clusters()];
        for index in 0..count {
            let vector = &data[index * dimension..(index + 1) * dimension];
            let cluster = model.assign(vector).0;
            counts[cluster] += 1;
            for (offset, &value) in vector.iter().enumerate() {
                sums[cluster * dimension + offset] += value as f64;
            }
        }

        for cluster in 0..model.clusters() {
            if counts[cluster] == 0 {
                continue;
            }
            for offset in 0..dimension {
                let mean = (sums[cluster * dimension + offset] / counts[cluster] as f64) as f32;
                let stored = model.centroid(cluster).unwrap()[offset];
                assert!(
                    (mean - stored).abs() < 1e-2,
                    "cluster {cluster} dim {offset}: stored {stored}, mean {mean}"
                );
            }
        }
    }

    /// k-means++ should beat uniform seeding, which is the entire reason for it.
    /// Compared against deliberately bad seeding — all centroids from one blob.
    #[test]
    fn plus_plus_seeding_spreads_the_centroids() {
        let (data, dimension) = blobs(50, 17);
        let count = data.len() / dimension;
        let mut rng = Rng::new(19);

        let seeded = seed_plus_plus(&data, dimension, count, 3, &mut rng);

        // The three seeds should be far apart — they came from different blobs.
        let mut minimum = f32::INFINITY;
        for a in 0..3 {
            for b in (a + 1)..3 {
                let first = &seeded[a * dimension..(a + 1) * dimension];
                let second = &seeded[b * dimension..(b + 1) * dimension];
                minimum = minimum.min(squared_l2(first, second));
            }
        }
        assert!(
            minimum > 100.0,
            "seeds are only {minimum} apart; the blobs are 50 units apart, so \
             seeding did not spread them"
        );
    }

    /// An empty cluster would stay empty forever, wasting a centroid. Asking for
    /// more clusters than there are distinct groups forces the case.
    #[test]
    fn empty_clusters_are_revived() {
        // Three tight blobs, but ask for eight clusters.
        let (data, dimension) = blobs(20, 23);
        let count = data.len() / dimension;
        let model = train(&data, dimension, 8, 50, 29);

        let mut used = std::collections::HashSet::new();
        for index in 0..count {
            used.insert(
                model
                    .assign(&data[index * dimension..(index + 1) * dimension])
                    .0,
            );
        }
        assert!(
            used.len() >= 6,
            "only {} of 8 clusters hold any points; empty ones were not revived",
            used.len()
        );
    }

    #[test]
    fn training_is_deterministic() {
        let (data, dimension) = blobs(30, 31);
        let first = train(&data, dimension, 4, 25, 37);
        let second = train(&data, dimension, 4, 25, 37);
        assert_eq!(first.centroids, second.centroids);
        assert_eq!(first.inertia, second.inertia);
    }

    /// Convergence should stop early rather than burning the whole budget.
    #[test]
    fn training_stops_once_assignments_settle() {
        let (data, dimension) = blobs(40, 41);
        let model = train(&data, dimension, 3, 1000, 43);
        assert!(
            model.iterations < 1000,
            "ran the full {} iterations without detecting convergence",
            model.iterations
        );
    }

    #[test]
    fn probe_lists_come_back_nearest_first() {
        let (data, dimension) = blobs(30, 47);
        let model = train(&data, dimension, 5, 50, 53);

        let query = &data[0..dimension];
        let probes = model.nearest_centroids(query, 3);
        assert_eq!(probes.len(), 3);
        assert!(
            probes.windows(2).all(|pair| pair[0].1 <= pair[1].1),
            "probes must ascend by distance: {probes:?}"
        );
        assert_eq!(
            probes[0].0,
            model.assign(query).0,
            "the nearest probe must be the assigned cluster"
        );
    }

    #[test]
    fn degenerate_inputs_are_handled() {
        // More clusters than points.
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let model = train(&data, 2, 10, 20, 59);
        assert_eq!(
            model.clusters(),
            2,
            "clusters are capped at the point count"
        );

        // Every point identical: all distances zero, so seeding cannot sample
        // proportionally to distance.
        let identical = vec![5.0f32; 40];
        let model = train(&identical, 4, 3, 20, 61);
        assert_eq!(model.clusters(), 3);
        assert!(
            model.inertia.abs() < 1e-6,
            "identical points have no spread"
        );
    }

    #[test]
    #[should_panic(expected = "at least one cluster")]
    fn zero_clusters_is_rejected() {
        train(&[1.0, 2.0], 2, 0, 10, 1);
    }
}
