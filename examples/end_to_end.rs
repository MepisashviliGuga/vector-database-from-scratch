//! The full system: durable storage plus approximate search, measured together.
//!
//! ```text
//! cargo run --release --example end_to_end
//! cargo run --release --example end_to_end benchmark/datasets/sift/sift 100
//! ```
//!
//! # The question
//!
//! Does composing the two layers beat either alone? The quantized index is fast
//! and lossy; the storage engine is exact and slow to scan. Re-ranking index
//! candidates against full-precision vectors read back from storage should
//! recover most of the quantization loss — ranking errors *inside* the candidate
//! set vanish, and only candidates the index never proposed are lost.
//!
//! Each row varies `rerank_candidates` at a fixed `nprobe`, so the only thing
//! moving is how many storage reads the query is willing to pay for.

use std::time::Instant;

use vectordb::ann::{fvecs, squared_l2};
use vectordb::bench::BenchDir;
use vectordb::engine::{SearchParams, VectorStore, VectorStoreConfig};
use vectordb::storage::{GrowthKind, LsmConfig, MergeKind, SyncPolicy};

fn main() -> std::io::Result<()> {
    let prefix = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "benchmark/datasets/siftsmall/siftsmall".to_string());
    let query_limit: usize = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(100);

    let base = fvecs::read_fvecs(format!("{prefix}_base.fvecs"))?;
    let queries = fvecs::read_fvecs(format!("{prefix}_query.fvecs"))?;
    let dimension = base.dimension;
    let count = base.count();

    println!("End-to-end vector database on {prefix}");
    println!("  {count} vectors x {dimension} dimensions");

    let mut config = VectorStoreConfig::new(dimension);
    config.storage = LsmConfig {
        memtable_threshold_bytes: 4 * 1024 * 1024,
        sync_policy: SyncPolicy::Manual,
        target_file_size_bytes: 16 * 1024 * 1024,
        growth: GrowthKind::Vertical {
            buffer_bytes: 4 * 1024 * 1024,
            size_ratio: 8,
        },
        merge: MergeKind::Leveling,
        ..Default::default()
    };
    config.index.clusters = (count as f64).sqrt() as usize;
    config.index.bits = 5;
    config.index.training_sample = Some(50_000);
    config.training_threshold = 1_000;

    let dir = BenchDir::new("end-to-end")?;
    let mut store = VectorStore::open(dir.path(), config)?;

    // Ingest through the real write path: WAL, memtable, flush, compaction.
    print!("ingesting... ");
    let ingest_start = Instant::now();
    for id in 0..count {
        let vector = base.get(id).expect("vector");
        store.insert(id as u32, vector, b"")?;
    }
    store.sync()?;
    // Rebuild once so every vector is indexed against final centroids, rather
    // than the ones trained on the first 1,000 to arrive.
    store.rebuild_index()?;
    let ingest_seconds = ingest_start.elapsed().as_secs_f64();

    let (index_bytes, disk_bytes) = store.footprint();
    println!("{ingest_seconds:.1}s");
    println!(
        "  index {:.1} MiB in memory, storage {:.1} MiB on disk\n",
        index_bytes as f64 / (1024.0 * 1024.0),
        disk_bytes as f64 / (1024.0 * 1024.0),
    );

    let query_rows: Vec<Vec<f32>> = queries.rows().into_iter().take(query_limit).collect();

    print!("computing exact neighbours... ");
    let truth_start = Instant::now();
    let truth: Vec<Vec<u32>> = query_rows
        .iter()
        .map(|query| {
            let mut scored: Vec<(f32, u32)> = (0..count)
                .map(|i| (squared_l2(base.get(i).expect("vector"), query), i as u32))
                .collect();
            scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            scored.into_iter().take(10).map(|(_, id)| id).collect()
        })
        .collect();
    println!("{:.1}s\n", truth_start.elapsed().as_secs_f64());

    let nprobe = 32;

    // Warm the page cache before timing anything. Re-ranking reads SSTable
    // blocks, so the first configuration swept would otherwise be charged for
    // faulting in a multi-hundred-megabyte store and report a *higher* cost
    // than the strictly larger budgets that follow it.
    print!("warming... ");
    let warm_start = Instant::now();
    for query in &query_rows {
        store.search(
            query,
            10,
            SearchParams {
                nprobe,
                rerank_candidates: 500,
            },
        )?;
    }
    println!("{:.1}s\n", warm_start.elapsed().as_secs_f64());

    println!(
        "  {:>10}  {:>10}  {:>10}",
        "candidates", "recall@10", "ms/query"
    );

    for rerank in [10usize, 20, 50, 100, 200, 500] {
        let params = SearchParams {
            nprobe,
            rerank_candidates: rerank,
        };

        let started = Instant::now();
        let mut total = 0.0;
        for (query, want) in query_rows.iter().zip(truth.iter()) {
            let found = store.search(query, 10, params)?;
            let wanted: std::collections::HashSet<u32> = want.iter().copied().collect();
            total += found
                .iter()
                .filter(|n| wanted.contains(&(n.id as u32)))
                .count() as f64
                / 10.0;
        }
        let elapsed = started.elapsed().as_secs_f64() / query_rows.len() as f64;

        println!(
            "  {rerank:>10}  {:>10.4}  {:>10.2}",
            total / query_rows.len() as f64,
            elapsed * 1000.0,
        );
    }

    println!(
        "\n  nprobe fixed at {nprobe}; only the re-rank budget varies. The row at\n\
         \x20 candidates = 10 is the index's own ordering with nothing to re-rank,\n\
         \x20 so the gain over it is exactly what reading full-precision vectors\n\
         \x20 back from storage buys.\n\
         \x20 Every returned distance is exact, computed from the stored vector."
    );

    Ok(())
}
