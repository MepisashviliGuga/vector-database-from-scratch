//! Measures extended RaBitQ's accuracy on a real dataset, against the
//! brute-force oracle.
//!
//! ```text
//! cargo run --release --example rabitq_recall
//! cargo run --release --example rabitq_recall benchmark/datasets/sift/sift 200
//! ```
//!
//! # What is measured
//!
//! For each bit width `B`, every base vector is quantized to `B` bits per
//! dimension, and queries are answered by **ranking on estimated distances
//! alone** — no re-ranking against raw vectors. Recall is then computed against
//! exact k-NN from the same data.
//!
//! That is the setting paper 03 targets: the raw vectors are assumed *not* to be
//! accessible at query time, because the whole point is to not keep them in RAM.
//!
//! # What is not measured
//!
//! Speed. The arithmetic here is scalar, where the paper uses SIMD FastScan, so
//! a timing comparison would be meaningless. Accuracy and memory are directly
//! comparable; throughput is not.

use std::time::Instant;

use vectordb::ann::{fvecs, rabitq::RaBitQ, squared_l2};

fn main() -> std::io::Result<()> {
    let prefix = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "benchmark/datasets/siftsmall/siftsmall".to_string());
    let query_limit: usize = std::env::args()
        .nth(2)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(100);
    // Encoding costs O(2^B · D log D) per vector, so on a million-vector set the
    // wide codes dominate the runtime. Selecting bit widths keeps a full run
    // tractable without silently skipping any.
    let bit_widths: Vec<u32> = match std::env::args().nth(3) {
        Some(list) => list
            .split(',')
            .filter_map(|b| b.trim().parse().ok())
            .collect(),
        None => vec![1, 2, 3, 4, 5, 6, 7, 8],
    };

    let base = fvecs::read_fvecs(format!("{prefix}_base.fvecs"))?;
    let queries = fvecs::read_fvecs(format!("{prefix}_query.fvecs"))?;
    let dimension = base.dimension;
    let count = base.count();

    println!("extended RaBitQ on {prefix}");
    println!("  {count} base vectors x {dimension} dimensions");

    let base_rows = base.rows();
    let query_rows: Vec<Vec<f32>> = queries.rows().into_iter().take(query_limit).collect();
    println!("  {} queries evaluated\n", query_rows.len());

    // A single centroid over the whole set. Paper 03 pairs the quantizer with
    // IVF, where each cluster has its own centroid and residuals are far
    // smaller; one global centroid is the harder setting, so these figures are a
    // LOWER bound on what the method achieves in its intended configuration.
    let mut centroid = vec![0.0f32; dimension];
    for vector in &base_rows {
        for (slot, value) in centroid.iter_mut().zip(vector.iter()) {
            *slot += value;
        }
    }
    for slot in &mut centroid {
        *slot /= count as f32;
    }

    // Exact answers from the same data, so recall is measured against ground
    // truth for *this* base set rather than a published file.
    println!("computing exact neighbours...");
    let exact: Vec<Vec<usize>> = query_rows
        .iter()
        .map(|query| {
            let mut scored: Vec<(f32, usize)> = base_rows
                .iter()
                .enumerate()
                .map(|(index, vector)| (squared_l2(vector, query), index))
                .collect();
            scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            scored.into_iter().take(100).map(|(_, i)| i).collect()
        })
        .collect();

    let raw_bytes = count * dimension * 4;
    println!(
        "\n  raw vectors: {:.1} MiB\n",
        raw_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  {:>3}  {:>9}  {:>7}  {:>10}  {:>10}  {:>10}",
        "B", "MiB", "ratio", "recall@1", "recall@10", "recall@100"
    );

    for bits in bit_widths {
        let quantizer = RaBitQ::new(dimension, bits, 0x5EED);

        let encode_start = Instant::now();
        let codes: Vec<_> = base_rows
            .iter()
            .map(|vector| quantizer.encode(vector, &centroid))
            .collect();
        let encode_seconds = encode_start.elapsed().as_secs_f64();

        let mut recalls = [0.0f64; 3];
        for (query, truth) in query_rows.iter().zip(exact.iter()) {
            let prepared = quantizer.prepare_query(query, &centroid);
            let mut ranked: Vec<(f32, usize)> = codes
                .iter()
                .enumerate()
                .map(|(index, code)| (quantizer.estimate_squared_distance(code, &prepared), index))
                .collect();
            ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

            for (slot, &k) in recalls.iter_mut().zip([1usize, 10, 100].iter()) {
                let wanted: std::collections::HashSet<usize> =
                    truth.iter().take(k).copied().collect();
                let hits = ranked
                    .iter()
                    .take(k)
                    .filter(|(_, index)| wanted.contains(index))
                    .count();
                *slot += hits as f64 / k as f64;
            }
        }

        let queries = query_rows.len() as f64;
        // Bit-packed size, which is the figure comparable to the paper's
        // compression rates; our in-memory codes are one byte per dimension.
        let packed = count * quantizer.packed_code_bytes();
        println!(
            "  {bits:>3}  {:>9.1}  {:>6.1}x  {:>10.4}  {:>10.4}  {:>10.4}   ({:.1}s to encode)",
            packed as f64 / (1024.0 * 1024.0),
            raw_bytes as f64 / packed as f64,
            recalls[0] / queries,
            recalls[1] / queries,
            recalls[2] / queries,
            encode_seconds,
        );
    }

    println!(
        "\nRanking uses estimated distances only -- no re-ranking against raw\n\
         vectors, which is the setting paper 03 targets. A single global centroid\n\
         is used; the paper pairs the quantizer with IVF, where per-cluster\n\
         centroids make residuals much smaller, so these are a LOWER bound on\n\
         what the method achieves in its intended configuration."
    );

    Ok(())
}
