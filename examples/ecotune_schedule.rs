//! Prints the compaction schedules EcoTune's dynamic program produces.
//!
//! Run with `cargo run --release --example ecotune_schedule`.
//!
//! The point is to see paper 02's central claim directly: a policy should be
//! aggressive early in a compaction round and lazy near the end, because a
//! sorted run created early survives to serve more queries before the
//! round-ending global compaction erases it. A policy built on physical levels
//! cannot express that — its aggressiveness is fixed for the whole round.
//!
//! **Reading the output.** Merge widths are *cumulative*: a width-4 merge at
//! position 4 absorbs an earlier width-2 merge plus two new runs. So the taper
//! to look for is in the **gaps** between merges, not the widths. It is visible
//! in the point-heavy workload; the scan-heavy one saturates at "merge every
//! time", which is just leveling and has no taper to show.

use vectordb::storage::compaction::{EcoTuneConfig, EcoTunePolicy};

fn show(label: &str, config: EcoTuneConfig) {
    let policy = EcoTunePolicy::solve(config);
    let schedule = policy.schedule();

    println!("\n{label}");
    println!(
        "  R = {}, C = {}, r = {}, T_c = {}, beta = {}",
        config.runs_per_round,
        config.last_level_ratio,
        config.long_range_ratio,
        config.rewrite_time,
        config.mlc_query_factor,
    );
    println!("  score {:.4}", policy.score());

    // A timeline: one column per unit run, marking where merges land.
    let mut timeline = vec![String::from("  .."); config.runs_per_round + 1];
    for merge in &schedule {
        if merge.after_unit_runs < timeline.len() {
            timeline[merge.after_unit_runs] = if merge.is_global {
                format!("{:>4}", "G")
            } else {
                format!("{:>4}", merge.width)
            };
        }
    }
    print!("  runs   ");
    for i in 0..=config.runs_per_round {
        print!("{i:>4}");
    }
    println!();
    print!("  merge  ");
    for cell in &timeline {
        print!("{cell}");
    }
    println!();

    let chosen = policy.chosen_merges();
    let mut gaps = Vec::new();
    let mut previous = 0usize;
    for merge in &chosen {
        gaps.push(merge.after_unit_runs - previous);
        previous = merge.after_unit_runs;
    }
    println!("  {} chosen merges, gaps between them {gaps:?}", chosen.len());
}

fn main() {
    println!("EcoTune schedules. Numbers are merge widths in unit runs; G is the");
    println!("round-ending global compaction, which happens regardless of policy.");

    let base = EcoTuneConfig {
        runs_per_round: 12,
        last_level_ratio: 3,
        tm_interval: 1.0,
        rewrite_time: 0.02,
        long_range_ratio: 0.6,
        false_positive_rate: 0.01,
        mlc_query_factor: 0.5,
    };

    show("scan-heavy (r = 0.6), cheap merges", base);

    show(
        "point-heavy (r = 0.02): run count barely matters, so merging earns little",
        EcoTuneConfig {
            long_range_ratio: 0.02,
            ..base
        },
    );

    show(
        "expensive merges (T_c = 0.5): the investment stops paying off",
        EcoTuneConfig {
            rewrite_time: 0.5,
            ..base
        },
    );

    show(
        "queries blocked during merges (beta = 0.05)",
        EcoTuneConfig {
            mlc_query_factor: 0.05,
            ..base
        },
    );
}
