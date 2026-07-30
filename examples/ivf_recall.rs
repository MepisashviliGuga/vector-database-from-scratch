//! IVF + extended RaBitQ recall, against the exact oracle.
//!
//! ```text
//! cargo run --release --example ivf_recall
//! cargo run --release --example ivf_recall benchmark/datasets/sift/sift 100 5 1024
//! ```
//!
//! Arguments: dataset prefix, query count, bits per dimension, cluster count.
//!
//! # The two error sources, separated
//!
//! Recall here loses accuracy in two independent ways, and reporting a single
//! number would hide which one dominates:
//!
//! 1. **Quantization** — the code is an approximation of the vector. Fixed by
//!    spending more bits.
//! 2. **Pruning** — a true neighbour may sit in a cluster the query did not
//!    probe. Fixed by raising `nprobe`, and **not** by more bits.
//!
//! The `nprobe = clusters` row probes everything, so it isolates (1). The
//! difference between it and any smaller `nprobe` is exactly (2).

use std::time::Instant;

use vectordb::ann::{fvecs, squared_l2, IvfConfig, IvfIndex};

fn main() -> std::io::Result<()> {
    let mut args = std::env::args().skip(1);
    let prefix = args
        .next()
        .unwrap_or_else(|| "benchmark/datasets/siftsmall/siftsmall".to_string());
    let query_limit: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(100);
    let bits: u32 = args.next().and_then(|a| a.parse().ok()).unwrap_or(5);
    let clusters: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(256);

    let base = fvecs::read_fvecs(format!("{prefix}_base.fvecs"))?;
    let queries = fvecs::read_fvecs(format!("{prefix}_query.fvecs"))?;
    let dimension = base.dimension;

    println!("IVF + extended RaBitQ on {prefix}");
    println!(
        "  {} vectors x {dimension} dims, B = {bits}, {clusters} clusters",
        base.count()
    );

    let query_rows: Vec<Vec<f32>> = queries.rows().into_iter().take(query_limit).collect();
    println!("  {} queries\n", query_rows.len());

    let build_start = Instant::now();
    let index = IvfIndex::build(
        &base.data,
        dimension,
        IvfConfig {
            clusters,
            bits,
            kmeans_iterations: 20,
            training_sample: Some(100_000),
            seed: 0x5EED,
        },
    );
    println!("  built in {:.1}s", build_start.elapsed().as_secs_f64());

    let sizes = index.list_sizes();
    let non_empty = sizes.iter().filter(|&&s| s > 0).count();
    let largest = sizes.iter().max().copied().unwrap_or(0);
    let mean = index.len() as f64 / non_empty.max(1) as f64;
    println!("  {non_empty}/{clusters} lists used, mean {mean:.0}, largest {largest}");

    let raw_bytes = base.count() * dimension * 4;
    println!(
        "  memory: {:.1} MiB packed against {:.1} MiB raw ({:.1}x)\n",
        index.packed_bytes() as f64 / (1024.0 * 1024.0),
        raw_bytes as f64 / (1024.0 * 1024.0),
        raw_bytes as f64 / index.packed_bytes() as f64,
    );

    // Exact answers over this base set, so recall is ground truth for the data
    // actually indexed rather than a published file.
    print!("computing exact neighbours... ");
    let exact_start = Instant::now();
    let truth: Vec<Vec<u32>> = query_rows
        .iter()
        .map(|query| {
            let mut scored: Vec<(f32, u32)> = (0..base.count())
                .map(|i| {
                    (
                        squared_l2(&base.data[i * dimension..(i + 1) * dimension], query),
                        i as u32,
                    )
                })
                .collect();
            scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            scored.into_iter().take(10).map(|(_, id)| id).collect()
        })
        .collect();
    println!("{:.1}s\n", exact_start.elapsed().as_secs_f64());

    println!(
        "  {:>7}  {:>9}  {:>10}  {:>12}",
        "nprobe", "recall@10", "ms/query", "% scanned"
    );

    let mut probes: Vec<usize> = vec![1, 2, 4, 8, 16, 32, 64]
        .into_iter()
        .filter(|&p| p < clusters)
        .collect();
    probes.push(clusters);

    for nprobe in probes {
        let started = Instant::now();
        let mut total = 0.0;
        for (query, wanted) in query_rows.iter().zip(truth.iter()) {
            let found = index.search(query, 10, nprobe);
            let wanted: std::collections::HashSet<u32> = wanted.iter().copied().collect();
            let hits = found
                .iter()
                .filter(|n| wanted.contains(&(n.id as u32)))
                .count();
            total += hits as f64 / 10.0;
        }
        let elapsed = started.elapsed().as_secs_f64() / query_rows.len() as f64;

        // Fraction of the dataset actually examined, averaged over queries.
        let scanned: f64 = query_rows
            .iter()
            .map(|query| {
                index
                    .search_probe_sizes(query, nprobe)
                    .iter()
                    .sum::<usize>() as f64
            })
            .sum::<f64>()
            / query_rows.len() as f64
            / index.len() as f64;

        let label = if nprobe == clusters {
            format!("{nprobe}*")
        } else {
            nprobe.to_string()
        };
        println!(
            "  {label:>7}  {:>9.4}  {:>10.2}  {:>11.1}%",
            total / query_rows.len() as f64,
            elapsed * 1000.0,
            scanned * 100.0,
        );
    }

    println!(
        "\n  * probes every cluster, so its recall is the QUANTIZER's ceiling with no\n\
         \x20   pruning loss. The gap between it and a smaller nprobe is what pruning\n\
         \x20   cost, and more bits cannot recover that part."
    );

    Ok(())
}
