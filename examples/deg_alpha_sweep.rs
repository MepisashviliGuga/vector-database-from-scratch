//! Does a fixed-α index really degrade as the query's α moves? — paper 05 Fig 2.
//!
//! ```text
//! cargo run --release --example deg_alpha_sweep
//! cargo run --release --example deg_alpha_sweep benchmark/datasets/sift/sift 20000
//! ```
//!
//! # The claim under test
//!
//! §3.3 measures two existing approaches to the Hybrid Vector Query and finds
//! each strong over part of the α range and weak elsewhere:
//!
//! - **Fusion** builds one graph at a fixed α (0.5). Good near 0.5, degrades
//!   towards the extremes.
//! - **Merging** builds one graph per modality and re-ranks. Good at the
//!   extremes, degrades near 0.5.
//!
//! DEG is meant to hold across the whole range. All three are built here by the
//! same [`DegIndex`] code path, differing only in [`PruningPolicy`], so a
//! difference in the table is a difference in pruning and not in beam search,
//! reverse edges, or candidate acquisition.
//!
//! # The data is synthetic, and deliberately so
//!
//! DEG needs two vectors per object; SIFT has one. The second modality here is a
//! generated 2-D coordinate — which is a shape the paper itself uses, not an
//! invention: Ins-SG and Twitter-US are text embeddings paired with geographic
//! coordinates at `m = 2` (Table 2).
//!
//! Two regimes are swept, because the experiment is vacuous in one of them:
//!
//! - **correlated** — the coordinate is a fixed random projection of the vector.
//!   The modalities largely agree, so α barely changes the answer and every index
//!   looks fine. This is the control.
//! - **independent** — the coordinate is drawn independently. Now the modalities
//!   disagree, α genuinely matters, and a fixed-α index has something to get
//!   wrong. This is where the claim is actually tested.

use std::time::Instant;

use vectordb::ann::deg::{DegConfig, DegIndex, HybridSet, PruningPolicy};
use vectordb::ann::{fvecs, recall_at_k, Neighbor};
use vectordb::workload::Rng;

const K: usize = 10;
const SECONDARY_DIM: usize = 2;
const ALPHAS: [f32; 7] = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0];
const MIN_TIMED_SECONDS: f64 = 0.5;
const MIN_PASSES: usize = 3;

/// How the second modality relates to the first.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Regime {
    Correlated,
    Independent,
}

impl Regime {
    fn label(&self) -> &'static str {
        match self {
            Regime::Correlated => "correlated",
            Regime::Independent => "independent",
        }
    }
}

/// Generate the second modality for every object.
///
/// Correlated coordinates come from a fixed random projection of the primary
/// vector, so nearby vectors get nearby coordinates. Independent ones ignore the
/// vector entirely.
fn secondary_modality(
    primary: &[f32],
    dim_e: usize,
    count: usize,
    regime: Regime,
    seed: u64,
) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    match regime {
        Regime::Independent => (0..count * SECONDARY_DIM)
            .map(|_| rng.next_f64() as f32)
            .collect(),
        Regime::Correlated => {
            // One projection matrix, reused for every object — that reuse is
            // what makes the coordinate a function of the vector rather than
            // noise.
            let projection: Vec<f32> = (0..dim_e * SECONDARY_DIM)
                .map(|_| rng.next_f64() as f32 - 0.5)
                .collect();
            let mut out = Vec::with_capacity(count * SECONDARY_DIM);
            for id in 0..count {
                let vector = &primary[id * dim_e..(id + 1) * dim_e];
                for axis in 0..SECONDARY_DIM {
                    let value: f32 = vector
                        .iter()
                        .zip(&projection[axis * dim_e..(axis + 1) * dim_e])
                        .map(|(v, p)| v * p)
                        .sum();
                    out.push(value);
                }
            }
            out
        }
    }
}

/// Time a search closure and return (recall@K, QPS).
///
/// Reports the *fastest* pass, for the reason recorded in `results/end_to_end.md`:
/// every noise source here can only add time, and averaging let background load
/// invert two configurations in Phase 6.
fn measure<F>(queries: &[(Vec<f32>, Vec<f32>)], truth: &[Vec<u32>], mut search: F) -> (f64, f64)
where
    F: FnMut(&[f32], &[f32]) -> Vec<Neighbor>,
{
    let found: Vec<Vec<Neighbor>> = queries.iter().map(|(e, s)| search(e, s)).collect();
    let mut recall = 0.0;
    for (got, want) in found.iter().zip(truth.iter()) {
        recall += recall_at_k(got, want, K);
    }
    let recall = recall / queries.len() as f64;

    let overall = Instant::now();
    let mut passes = 0usize;
    let mut best = f64::INFINITY;
    while passes < MIN_PASSES || overall.elapsed().as_secs_f64() < MIN_TIMED_SECONDS {
        let pass = Instant::now();
        for (e, s) in queries {
            std::hint::black_box(search(e, s));
        }
        best = best.min(pass.elapsed().as_secs_f64());
        passes += 1;
    }

    (recall, queries.len() as f64 / best)
}

/// The Merging baseline: search both single-modality graphs, then re-rank the
/// union by the true hybrid distance at this α.
fn merging_search(
    primary_graph: &DegIndex,
    secondary_graph: &DegIndex,
    query_e: &[f32],
    query_s: &[f32],
    alpha: f32,
    fetch: usize,
    beam: usize,
) -> Vec<Neighbor> {
    let mut ids: Vec<u32> = primary_graph
        .search(query_e, query_s, 1.0, fetch, beam)
        .iter()
        .map(|n| n.id as u32)
        .collect();
    ids.extend(
        secondary_graph
            .search(query_e, query_s, 0.0, fetch, beam)
            .iter()
            .map(|n| n.id as u32),
    );
    ids.sort_unstable();
    ids.dedup();

    let set = primary_graph.set();
    let mut scored: Vec<Neighbor> = ids
        .iter()
        .map(|&id| Neighbor {
            id: id as u64,
            distance: set.query_distance(query_e, query_s, id as usize).at(alpha),
        })
        .collect();
    scored.sort_unstable();
    scored.truncate(K);
    scored
}

fn main() -> std::io::Result<()> {
    let prefix = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "benchmark/datasets/siftsmall/siftsmall".to_string());
    let limit: usize = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(10_000);
    let query_limit: usize = std::env::args()
        .nth(3)
        .and_then(|a| a.parse().ok())
        .unwrap_or(100);
    // Search beam and build parameters, settable so an operating point can be
    // reported rather than assumed. A graph index that looks bad at a small beam
    // is often just being asked to answer with too little work.
    let beam: usize = std::env::args()
        .nth(4)
        .and_then(|a| a.parse().ok())
        .unwrap_or(48);
    let degree: usize = std::env::args()
        .nth(5)
        .and_then(|a| a.parse().ok())
        .unwrap_or(32);
    let build_pool: usize = std::env::args()
        .nth(6)
        .and_then(|a| a.parse().ok())
        .unwrap_or(64);

    let base = fvecs::read_fvecs(format!("{prefix}_base.fvecs"))?;
    let raw_queries = fvecs::read_fvecs(format!("{prefix}_query.fvecs"))?;
    let dim_e = base.dimension;
    let count = base.count().min(limit);
    let primary: Vec<f32> = base.data[..count * dim_e].to_vec();

    println!("DEG α sweep — paper 05 Figure 2");
    println!("  {count} objects, primary {dim_e}-D (SIFT), secondary {SECONDARY_DIM}-D (synthetic)");
    println!("  {} queries, recall@{K}", query_limit.min(raw_queries.count()));
    println!("  beam {beam}, max_degree {degree}, build_pool {build_pool}\n");

    for regime in [Regime::Independent, Regime::Correlated] {
        let secondary = secondary_modality(&primary, dim_e, count, regime, 0xD_E9);
        let set = HybridSet::new(
            primary.clone(),
            dim_e,
            secondary.clone(),
            SECONDARY_DIM,
            100_000,
            17,
        );
        let (emax, smax) = set.maxima();

        // Queries: real SIFT query vectors, with synthetic coordinates drawn the
        // same way as the base set's so they live in the same space.
        let query_rows: Vec<Vec<f32>> = raw_queries.rows().into_iter().take(query_limit).collect();
        let query_secondary = secondary_modality(
            &query_rows.concat(),
            dim_e,
            query_rows.len(),
            regime,
            // A different seed for the independent regime so queries are not
            // copies of base coordinates; the correlated regime derives them from
            // the query vectors, where the seed must match to reuse the same
            // projection matrix.
            if regime == Regime::Correlated { 0xD_E9 } else { 0xB_EE },
        );
        let queries: Vec<(Vec<f32>, Vec<f32>)> = query_rows
            .iter()
            .enumerate()
            .map(|(i, q)| {
                (
                    q.clone(),
                    query_secondary[i * SECONDARY_DIM..(i + 1) * SECONDARY_DIM].to_vec(),
                )
            })
            .collect();

        println!("== {} modalities ==", regime.label());
        println!("  emax {emax:.4}, smax {smax:.4} (sampled)");

        let config = DegConfig {
            max_degree: degree,
            build_pool,
            ..DegConfig::default()
        };
        print!("  building deg... ");
        let started = Instant::now();
        let deg = DegIndex::build(set.clone(), config);
        let deg_build = started.elapsed().as_secs_f64();
        println!("{deg_build:.1}s");

        print!("  building fusion (α=0.5)... ");
        let started = Instant::now();
        let fusion = DegIndex::build(
            set.clone(),
            DegConfig {
                policy: PruningPolicy::Fixed(0.5),
                ..config
            },
        );
        let fusion_build = started.elapsed().as_secs_f64();
        println!("{fusion_build:.1}s");

        print!("  building merging (two graphs)... ");
        let started = Instant::now();
        let merge_primary = DegIndex::build(
            set.clone(),
            DegConfig {
                policy: PruningPolicy::SingleModality { primary: true },
                ..config
            },
        );
        let merge_secondary = DegIndex::build(
            set,
            DegConfig {
                policy: PruningPolicy::SingleModality { primary: false },
                ..config
            },
        );
        let merging_build = started.elapsed().as_secs_f64();
        println!("{merging_build:.1}s");

        let (vectors, ranges) = deg.memory_bytes();
        println!(
            "  deg mean out-degree {:.1}, active ranges {:.2} MiB against {:.2} MiB of vectors",
            deg.mean_out_degree(),
            ranges as f64 / (1024.0 * 1024.0),
            vectors as f64 / (1024.0 * 1024.0),
        );

        println!(
            "\n  {:>5}  {:>18}  {:>18}  {:>18}",
            "α", "DEG recall/QPS", "Fusion recall/QPS", "Merging recall/QPS"
        );
        println!("  {}", "-".repeat(66));

        for &alpha in &ALPHAS {
            // Ground truth at this α, by exact scan over Eq 1.
            let truth: Vec<Vec<u32>> = queries
                .iter()
                .map(|(e, s)| {
                    deg.exact_search(e, s, alpha, K)
                        .iter()
                        .map(|n| n.id as u32)
                        .collect()
                })
                .collect();

            let (deg_recall, deg_qps) =
                measure(&queries, &truth, |e, s| deg.search(e, s, alpha, K, beam));
            let (fusion_recall, fusion_qps) =
                measure(&queries, &truth, |e, s| fusion.search(e, s, alpha, K, beam));
            let (merge_recall, merge_qps) = measure(&queries, &truth, |e, s| {
                merging_search(&merge_primary, &merge_secondary, e, s, alpha, K, beam)
            });

            println!(
                "  {alpha:>5.1}  {deg_recall:>9.4} {deg_qps:>8.0}  \
                 {fusion_recall:>9.4} {fusion_qps:>8.0}  \
                 {merge_recall:>9.4} {merge_qps:>8.0}"
            );
        }
        println!();
    }

    println!(
        "  All three arms share one build and search path, differing only in how\n\
         \x20 edges are pruned, so a gap between columns is a pruning difference.\n\
         \x20 Ground truth is an exact scan over Eq 1 at each α.\n\
         \x20 QPS is the fastest of several passes; see results/end_to_end.md."
    );

    Ok(())
}
