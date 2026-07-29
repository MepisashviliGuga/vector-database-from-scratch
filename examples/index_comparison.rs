//! All four index types on one axis: recall@10 against queries per second.
//!
//! ```text
//! cargo run --release --example index_comparison
//! cargo run --release --example index_comparison benchmark/datasets/sift/sift 200
//! ```
//!
//! # What is being compared
//!
//! | index       | what it exploits                     | knob swept  |
//! |-------------|--------------------------------------|-------------|
//! | brute force | nothing; the oracle                  | none        |
//! | IVF+RaBitQ  | pruning by cluster, compressed codes | `nprobe`    |
//! | graph       | proximity structure, exact distances | `beam`      |
//! | SymphonyQG  | both, codes fused into the graph     | `beam`      |
//!
//! Every row shares one process, one dataset, one ground truth, and one timing
//! loop, so the numbers are comparable to each other. They are NOT comparable to
//! published figures: those come from SIMD FastScan kernels we do not have, and
//! the [`vectordb::ann::symphony`] module docs explain what that costs us.
//!
//! Build time and resident bytes are printed per index, because a curve that
//! ignores build cost flatters graphs and a curve that ignores memory flatters
//! SymphonyQG.

use std::time::Instant;

use vectordb::ann::brute_force::{BruteForceIndex, Neighbor};
use vectordb::ann::graph::{GraphConfig, GraphIndex};
use vectordb::ann::ivf::{IvfConfig, IvfIndex};
use vectordb::ann::symphony::{SymphonyConfig, SymphonyIndex};
use vectordb::ann::{fvecs, recall_at_k};

const K: usize = 10;

/// The shortest window a QPS figure may be measured over.
///
/// A fast index answers 100 queries in single-digit milliseconds, which is close
/// enough to the clock's own noise to invert the ordering of two configurations.
/// Repeat the query set until the window is long enough to mean something.
const MIN_TIMED_SECONDS: f64 = 1.0;

/// Passes to run regardless, so the best-of statistic has something to choose from.
const MIN_PASSES: usize = 5;

/// Time `queries` against one search closure and return (recall@K, QPS).
fn measure<F>(queries: &[Vec<f32>], truth: &[Vec<u32>], mut search: F) -> (f64, f64)
where
    F: FnMut(&[f32]) -> Vec<Neighbor>,
{
    // One untimed pass so page faults and branch predictors are not charged to
    // the first configuration swept.
    let found: Vec<Vec<Neighbor>> = queries.iter().map(|q| search(q)).collect();

    let mut recall = 0.0;
    for (got, want) in found.iter().zip(truth.iter()) {
        recall += recall_at_k(got, want, K);
    }
    let recall = recall / queries.len() as f64;

    // Report the fastest pass, not the mean. Every source of noise here — other
    // processes, frequency scaling, thermal limits — can only ever make a pass
    // slower, so the minimum is the closest estimate of what the index actually
    // costs. Averaging instead lets background load reorder two configurations,
    // which is how an earlier version of this table reported a larger `nprobe`
    // as cheaper than a smaller one.
    let overall = Instant::now();
    let mut passes = 0usize;
    let mut best = f64::INFINITY;
    while passes < MIN_PASSES || overall.elapsed().as_secs_f64() < MIN_TIMED_SECONDS {
        let pass = Instant::now();
        for query in queries {
            std::hint::black_box(search(query));
        }
        best = best.min(pass.elapsed().as_secs_f64());
        passes += 1;
    }

    (recall, queries.len() as f64 / best)
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() -> std::io::Result<()> {
    let prefix = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "benchmark/datasets/siftsmall/siftsmall".to_string());
    let query_limit: usize = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(200);

    // Optional cap on the base set. The graph builds are superlinear enough that
    // a full million-vector SymphonyQG build runs for hours; capping lets the
    // comparison cover an order of magnitude more data than siftsmall while
    // still finishing. A capped run is a different dataset, so it is labelled
    // as one rather than compared against uncapped numbers.
    let base_limit: Option<usize> = std::env::args().nth(3).and_then(|a| a.parse().ok());

    let mut base = fvecs::read_fvecs(format!("{prefix}_base.fvecs"))?;
    let raw = fvecs::read_fvecs(format!("{prefix}_query.fvecs"))?;
    let dimension = base.dimension;
    if let Some(limit) = base_limit {
        if limit < base.count() {
            base.data.truncate(limit * dimension);
        }
    }
    let count = base.count();
    let queries: Vec<Vec<f32>> = raw.rows().into_iter().take(query_limit).collect();

    println!("Index comparison on {prefix}");
    println!(
        "  {count} vectors x {dimension} dimensions, {} queries, recall@{K}\n",
        queries.len()
    );

    let flat = BruteForceIndex::from_flat(dimension, base.data.clone());
    print!("ground truth... ");
    let truth_start = Instant::now();
    let truth = flat.ground_truth(&queries, K);
    println!("{:.1}s\n", truth_start.elapsed().as_secs_f64());

    println!(
        "  {:<14} {:>8} {:>10} {:>12} {:>10} {:>10}",
        "index", "knob", "recall@10", "QPS", "build s", "MiB"
    );
    println!("  {}", "-".repeat(68));

    // ---- brute force: the oracle, and the QPS floor everything must beat ----
    let (recall, qps) = measure(&queries, &truth, |q| flat.search(q, K));
    println!(
        "  {:<14} {:>8} {:>10.4} {:>12.1} {:>10.1} {:>10.1}",
        "brute force",
        "-",
        recall,
        qps,
        0.0,
        mib(flat.data_bytes())
    );

    // ---- IVF + RaBitQ: prune by cluster, score compressed codes ----
    let ivf_config = IvfConfig {
        clusters: (count as f64).sqrt() as usize,
        bits: 5,
        training_sample: Some(50_000),
        ..Default::default()
    };
    let started = Instant::now();
    let ivf = IvfIndex::build(&base.data, dimension, ivf_config);
    let ivf_build = started.elapsed().as_secs_f64();

    for nprobe in [1usize, 4, 8, 16, 32, 64, 128] {
        if nprobe > ivf.clusters() {
            break;
        }
        let (recall, qps) = measure(&queries, &truth, |q| ivf.search(q, K, nprobe));
        println!(
            "  {:<14} {:>8} {:>10.4} {:>12.1} {:>10.1} {:>10.1}",
            "IVF+RaBitQ",
            nprobe,
            recall,
            qps,
            ivf_build,
            mib(ivf.packed_bytes())
        );
    }

    // ---- proximity graph: exact distances, structure does the pruning ----
    let graph_config = GraphConfig {
        max_degree: 32,
        build_beam: 64,
        ..Default::default()
    };
    let started = Instant::now();
    let graph = GraphIndex::build(&base.data, dimension, graph_config);
    let graph_build = started.elapsed().as_secs_f64();

    for beam in [10usize, 20, 40, 80, 160] {
        let (recall, qps) = measure(&queries, &truth, |q| graph.search(q, K, beam));
        println!(
            "  {:<14} {:>8} {:>10.4} {:>12.1} {:>10.1} {:>10.1}",
            "graph",
            beam,
            recall,
            qps,
            graph_build,
            mib(graph.data_bytes())
        );
    }

    // ---- SymphonyQG: codes fused into the graph, implicit re-ranking ----
    let symphony_config = SymphonyConfig {
        graph: GraphConfig {
            max_degree: 32,
            build_beam: 64,
            align_degree: true,
            ..Default::default()
        },
        bits: 4,
        ..Default::default()
    };
    let started = Instant::now();
    let symphony = SymphonyIndex::build(&base.data, dimension, symphony_config);
    let symphony_build = started.elapsed().as_secs_f64();

    for beam in [10usize, 20, 40, 80, 160] {
        let (recall, qps) = measure(&queries, &truth, |q| symphony.search(q, K, beam));
        println!(
            "  {:<14} {:>8} {:>10.4} {:>12.1} {:>10.1} {:>10.1}",
            "SymphonyQG",
            beam,
            recall,
            qps,
            symphony_build,
            mib(symphony.code_bytes() + symphony.raw_bytes())
        );
    }

    println!(
        "\n  MiB is what each index must hold resident to answer a query.\n\
         \x20 Brute force and graph hold the raw vectors; IVF holds only packed codes;\n\
         \x20 SymphonyQG holds both, because implicit re-ranking still reads raw vectors\n\
         \x20 and the codes are replicated once per in-edge.\n\
         \x20 Build time is charged once per index, repeated on every row of that index."
    );

    Ok(())
}
