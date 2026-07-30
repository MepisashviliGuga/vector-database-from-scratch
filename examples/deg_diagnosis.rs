//! Why are the hybrid graphs so much worse than a plain one? — a controlled swap.
//!
//! ```text
//! cargo run --release --example deg_diagnosis
//! cargo run --release --example deg_diagnosis benchmark/datasets/sift/sift 20000
//! ```
//!
//! # The observation being chased
//!
//! `results/deg.md` records the α sweep reaching a third of the recall a plain
//! [`GraphIndex`] gets on the same vectors. The comparison is exact rather than
//! approximate: **at α = 1 the hybrid distance is `δe` alone**, so a hybrid graph
//! searched at α = 1 and a plain graph over the same vectors are solving an
//! identical problem, and any gap between them is a defect in the hybrid build
//! rather than a property of hybrid search.
//!
//! # The design
//!
//! Everything runs on the **Fusion policy at α = 1**, because that is the arm
//! whose behaviour a plain graph should exactly match — it prunes with the plain
//! RNG rule at a single α, and marks every edge always-active. Two components are
//! then swapped independently:
//!
//! | | edge seeds (§4.4) | interior entry |
//! |---|---|---|
//! | **GPS candidates** (Alg 1) | the paper's design | isolates the seeds |
//! | **beam candidates** | isolates GPS | neither |
//!
//! Against `GraphIndex` as the reference. Whichever swap closes the gap names the
//! cause; if neither does, the defect is in something both share — the pruning,
//! the degree, or the search.
//!
//! Note the four hybrid arms differ *only* in these two switches: same distances,
//! same pruning, same reverse-edge rule, same beam search.

use std::time::Instant;

use vectordb::ann::deg::graph::{CandidateSource, EntryPolicy};
use vectordb::ann::deg::{DegConfig, DegIndex, HybridSet, PruningPolicy};
use vectordb::ann::graph::{GraphConfig, GraphIndex};
use vectordb::ann::{fvecs, recall_at_k, Neighbor};
use vectordb::workload::Rng;

const K: usize = 10;
const SECONDARY_DIM: usize = 2;

fn main() -> std::io::Result<()> {
    let prefix = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "benchmark/datasets/siftsmall/siftsmall".to_string());
    let limit: usize = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(10_000);
    let beam: usize = std::env::args()
        .nth(3)
        .and_then(|a| a.parse().ok())
        .unwrap_or(80);

    let base = fvecs::read_fvecs(format!("{prefix}_base.fvecs"))?;
    let raw_queries = fvecs::read_fvecs(format!("{prefix}_query.fvecs"))?;
    let dim_e = base.dimension;
    let count = base.count().min(limit);
    let primary: Vec<f32> = base.data[..count * dim_e].to_vec();

    // The second modality exists so the types line up; at α = 1 it contributes
    // nothing to any distance, which is exactly what makes this comparison fair.
    let mut rng = Rng::new(0xD_E9);
    let secondary: Vec<f32> = (0..count * SECONDARY_DIM)
        .map(|_| rng.next_f64() as f32)
        .collect();
    let set = HybridSet::new(
        primary.clone(),
        dim_e,
        secondary,
        SECONDARY_DIM,
        100_000,
        17,
    );

    let queries: Vec<Vec<f32>> = raw_queries.rows().into_iter().take(100).collect();
    let dummy_secondary = vec![0.0f32; SECONDARY_DIM];

    println!("DEG diagnosis — everything measured at α = 1, where the hybrid");
    println!("distance is δe alone and a plain graph solves the same problem.");
    println!(
        "  {count} objects × {dim_e}-D, {} queries, recall@{K}, beam {beam}\n",
        queries.len()
    );

    // Ground truth: exact δe nearest neighbours.
    print!("exact neighbours... ");
    let started = Instant::now();
    let truth: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| {
            let mut scored: Vec<(f32, u32)> = (0..count)
                .map(|i| {
                    let d = set.query_distance(q, &dummy_secondary, i);
                    (d.at(1.0), i as u32)
                })
                .collect();
            scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            scored.into_iter().take(K).map(|(_, id)| id).collect()
        })
        .collect();
    println!("{:.1}s\n", started.elapsed().as_secs_f64());

    println!(
        "  {:<34} {:>9} {:>10} {:>9} {:>8}",
        "arm", "recall@10", "build s", "mean deg", "ms/query"
    );
    println!("  {}", "-".repeat(75));

    // Reference: the plain single-modality graph from Phase 5.
    let started = Instant::now();
    let plain = GraphIndex::build(
        &primary,
        dim_e,
        GraphConfig {
            max_degree: 32,
            build_beam: 64,
            ..Default::default()
        },
    );
    let plain_build = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let mut recall = 0.0;
    for (q, want) in queries.iter().zip(truth.iter()) {
        let found: Vec<Neighbor> = plain.search(q, K, beam);
        recall += recall_at_k(&found, want, K);
    }
    let elapsed = started.elapsed().as_secs_f64() / queries.len() as f64;
    println!(
        "  {:<34} {:>9.4} {:>10.1} {:>9.1} {:>8.2}",
        "GraphIndex (reference)",
        recall / queries.len() as f64,
        plain_build,
        plain.mean_out_degree(),
        elapsed * 1000.0,
    );

    // The 2×2. Fusion at α = 1 so the plain RNG rule applies and every edge is
    // always active — the closest a hybrid graph gets to being a plain one.
    // The GPS pool is swept as well. A Pareto pool is spread along the whole
    // (δe, δs) trade-off curve, so only a fraction of it is useful at any one α —
    // which predicts GPS needs a much larger pool than a beam search does to
    // yield the same number of usable candidates. If a bigger pool closes the
    // gap, GPS is under-provisioned rather than wrong.
    let arms = [
        (
            "GPS + edge seeds (paper)",
            CandidateSource::Gps,
            EntryPolicy::EdgeSeeds,
            64,
        ),
        (
            "GPS + interior entry",
            CandidateSource::Gps,
            EntryPolicy::Interior,
            64,
        ),
        (
            "GPS + edge seeds, pool 256",
            CandidateSource::Gps,
            EntryPolicy::EdgeSeeds,
            256,
        ),
        (
            "GPS + edge seeds, pool 1024",
            CandidateSource::Gps,
            EntryPolicy::EdgeSeeds,
            1024,
        ),
        (
            "beam + edge seeds",
            CandidateSource::Beam,
            EntryPolicy::EdgeSeeds,
            64,
        ),
        (
            "beam + interior entry",
            CandidateSource::Beam,
            EntryPolicy::Interior,
            64,
        ),
    ];

    for (label, candidates, entry, build_pool) in arms {
        let config = DegConfig {
            max_degree: 32,
            build_pool,
            policy: PruningPolicy::Fixed(1.0),
            candidates,
            entry,
            ..DegConfig::default()
        };

        let started = Instant::now();
        let index = DegIndex::build(set.clone(), config);
        let build = started.elapsed().as_secs_f64();

        let started = Instant::now();
        let mut recall = 0.0;
        for (q, want) in queries.iter().zip(truth.iter()) {
            let found = index.search(q, &dummy_secondary, 1.0, K, beam);
            recall += recall_at_k(&found, want, K);
        }
        let elapsed = started.elapsed().as_secs_f64() / queries.len() as f64;

        println!(
            "  {:<34} {:>9.4} {:>10.1} {:>9.1} {:>8.2}",
            label,
            recall / queries.len() as f64,
            build,
            index.mean_out_degree(),
            elapsed * 1000.0,
        );
    }

    println!(
        "\n  All four hybrid arms share distances, pruning, reverse edges and beam\n\
         \x20 search, differing only in the two switches. Whichever swap closes the\n\
         \x20 gap to GraphIndex identifies the defect; if neither does, the cause is\n\
         \x20 in something they share."
    );

    Ok(())
}
