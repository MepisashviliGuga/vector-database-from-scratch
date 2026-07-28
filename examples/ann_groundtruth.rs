//! Validates the brute-force oracle against a real dataset's published ground
//! truth.
//!
//! ```text
//! cargo run --release --example ann_groundtruth                          # siftsmall
//! cargo run --release --example ann_groundtruth <prefix> [max_queries]
//! ```
//!
//! `max_queries` caps how many of the dataset's queries are evaluated. SIFT1M
//! ships 10,000 queries and a brute-force scan costs a million distance
//! computations each, so evaluating all of them takes hours. A capped run is a
//! **sample**, and the output says so — it is a correctness check on the oracle,
//! not a throughput measurement.
//!
//! # Why this run matters more than it looks
//!
//! Everything in Phase 5 and 6 — every recall figure for the quantizer and the
//! graph index — is measured against our brute-force scan. If that scan is
//! wrong, every downstream number is wrong in a way no downstream test would
//! reveal, because they all agree with each other by construction.
//!
//! The SIFT datasets ship ground truth computed independently, decades of use
//! ago, by the people who published the benchmark. Reproducing it exactly is an
//! *external* check on our oracle — the one thing our own test suite cannot
//! provide.
//!
//! A recall below 1.0 here means our scan, our distance function, or our reader
//! is wrong. It does not mean the dataset is.

use std::time::Instant;

use vectordb::ann::{fvecs, recall_at_k, squared_l2, BruteForceIndex};

fn main() -> std::io::Result<()> {
    let prefix = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "benchmark/datasets/siftsmall/siftsmall".to_string());
    let max_queries: Option<usize> = std::env::args().nth(2).and_then(|arg| arg.parse().ok());

    let base_path = format!("{prefix}_base.fvecs");
    let query_path = format!("{prefix}_query.fvecs");
    let truth_path = format!("{prefix}_groundtruth.ivecs");

    println!("Loading {prefix}...");
    let base = fvecs::read_fvecs(&base_path)?;
    let queries = fvecs::read_fvecs(&query_path)?;
    let published = fvecs::read_ivecs(&truth_path)?;

    println!(
        "  base     {} vectors x {} dimensions ({:.1} MiB as f32)",
        base.count(),
        base.dimension,
        (base.data.len() * 4) as f64 / (1024.0 * 1024.0)
    );
    println!("  queries  {}", queries.count());
    println!(
        "  truth    {} queries x {} neighbours",
        published.len(),
        published.first().map_or(0, Vec::len)
    );

    assert_eq!(
        base.dimension, queries.dimension,
        "base and query dimensions disagree; these files do not belong together"
    );
    assert_eq!(
        queries.count(),
        published.len(),
        "one ground-truth row is expected per query"
    );
    assert!(!published.is_empty(), "the ground-truth file is empty");

    let index = BruteForceIndex::from_flat(base.dimension, base.data);
    let mut query_rows = queries.rows();
    let mut published = published;

    if let Some(limit) = max_queries {
        if limit < query_rows.len() {
            println!(
                "\n  SAMPLING the first {limit} of {} queries. Recall below is over that\n\
                 \x20 sample, not the whole query set.",
                query_rows.len()
            );
            query_rows.truncate(limit);
            published.truncate(limit);
        }
    }

    // recall@1 and recall@10 are the figures the ANN literature reports.
    for k in [1usize, 10, 100] {
        let available = published.first().map_or(0, Vec::len);
        if k > available {
            println!("\nrecall@{k}: skipped, the published truth only holds {available}");
            continue;
        }

        let started = Instant::now();
        let mut total_recall = 0.0;
        let mut exact_matches = 0usize;
        let mut tied_queries = 0usize;
        let mut genuinely_wrong = Vec::new();

        for (position, (query, truth)) in query_rows.iter().zip(published.iter()).enumerate() {
            let found = index.search(query, k);
            let recall = recall_at_k(&found, truth, k);
            total_recall += recall;
            if recall == 1.0 {
                exact_matches += 1;
                continue;
            }

            // A different answer is not necessarily a wrong one. If our results
            // sit at exactly the same distances as the published ones, both
            // rankings are correct and the datasets simply broke a tie
            // differently. Deciding this by comparing the *distances* rather
            // than the ids is the only way to tell the two cases apart.
            let ours: Vec<f32> = found.iter().map(|neighbour| neighbour.distance).collect();
            let mut theirs: Vec<f32> = truth
                .iter()
                .take(k)
                .filter_map(|&id| index.vector(id as usize))
                .map(|vector| squared_l2(query, vector))
                .collect();
            theirs.sort_by(f32::total_cmp);

            if ours == theirs {
                tied_queries += 1;
            } else {
                genuinely_wrong.push((position, ours, theirs));
            }
        }

        let elapsed = started.elapsed();
        let mean_recall = total_recall / query_rows.len() as f64;
        let per_query = elapsed.as_secs_f64() / query_rows.len() as f64;

        println!(
            "\nrecall@{k}  {:.6}   ({}/{} queries matched exactly)",
            mean_recall,
            exact_matches,
            query_rows.len()
        );
        println!(
            "  {:.2} ms per query, {:.0} queries/sec, {} distance computations each",
            per_query * 1000.0,
            1.0 / per_query,
            index.len()
        );

        if tied_queries > 0 {
            println!(
                "  {tied_queries} queries returned different ids at identical distances.\n\
                 \x20 Both rankings are correct; the tie was broken differently. SIFT\n\
                 \x20 vectors are integers 0-255, so squared distances are exact in f32\n\
                 \x20 (max ~8.3M, well under 2^24) and this is not a precision artefact."
            );
        }

        if genuinely_wrong.is_empty() {
            println!("  No genuine disagreements: the oracle is correct.");
        } else {
            println!(
                "  {} GENUINE MISMATCHES. Our distance function, reader, or search is\n\
                 \x20 wrong. Nothing downstream of this oracle can be trusted until it\n\
                 \x20 is fixed.",
                genuinely_wrong.len()
            );
            for (position, ours, theirs) in genuinely_wrong.iter().take(3) {
                println!("    query {position}:");
                println!("      ours   {:?}", &ours[..ours.len().min(5)]);
                println!("      theirs {:?}", &theirs[..theirs.len().min(5)]);
            }
        }
    }

    println!(
        "\nBrute force holds {:.1} MiB of f32 vectors. That figure is the baseline\n\
         the quantizer has to beat on memory, and the per-query time above is the\n\
         baseline the graph index has to beat on speed.",
        index.data_bytes() as f64 / (1024.0 * 1024.0)
    );

    Ok(())
}
