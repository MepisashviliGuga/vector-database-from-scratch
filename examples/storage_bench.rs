//! The Phase 3 storage benchmark sweep.
//!
//! Run with `cargo run --release --example storage_bench`. Writes
//! `results/storage_benchmarks.csv` and prints a summary.
//!
//! # Four independent axes
//!
//! Growth scheme, merge policy, key distribution, and value size. Each sweep
//! below varies **one** of them and holds the rest fixed, because two earlier
//! comparisons in this project had to be thrown out for confounding two axes at
//! once. A full cross product is a stretch goal; controlled slices are the
//! deliverable.
//!
//! # What is deliberately *not* claimed
//!
//! - `SyncPolicy::Manual` throughout, so writes are not `fsync`ed per operation.
//!   The figures are therefore about compaction and I/O structure, not about
//!   durability cost. The policy travels in the CSV so a row cannot be quoted
//!   without it.
//! - Single-threaded, with compaction running synchronously inside `put`. Both
//!   papers assume background compaction on multi-core hardware. Absolute
//!   throughput here is not comparable to theirs; the *relative* ordering of
//!   configurations is what these runs are for.
//! - Data sizes are small enough to sit in the OS page cache. Read
//!   amplification is counted in blocks read by the engine, which is unaffected,
//!   but latency figures understate real disk cost.

use std::io::Write;

use vectordb::bench::{calibrate, run_benchmark, BenchDir, BenchResult, Calibration};
use vectordb::storage::compaction::EcoTuneConfig;
use vectordb::storage::{GrowthKind, HorizontalPolicy, LsmConfig, MergeKind, SyncPolicy};
use vectordb::workload::{KeyDistribution, WorkloadSpec};

const KEY_COUNT: u64 = 20_000;
const RUN_OPS: usize = 20_000;

/// Engine settings held fixed across every run, so only the swept axis moves.
fn base_config(growth: GrowthKind, merge: MergeKind) -> LsmConfig {
    LsmConfig {
        memtable_threshold_bytes: 64 * 1024,
        sync_policy: SyncPolicy::Manual,
        block_target_bytes: 4096,
        target_file_size_bytes: 256 * 1024,
        growth,
        merge,
        ..Default::default()
    }
}

fn vertical() -> GrowthKind {
    GrowthKind::Vertical {
        buffer_bytes: 64 * 1024,
        size_ratio: 4,
    }
}

fn horizontal() -> GrowthKind {
    GrowthKind::HorizontalLeveling { levels: 4 }
}

fn vertiorizon() -> GrowthKind {
    GrowthKind::Vertiorizon {
        horizontal_levels: 2,
        size_ratio: 4,
        buffer_bytes: 64 * 1024,
        initial_n: 8,
        policy: HorizontalPolicy::Leveling,
    }
}

fn ecotune(calibration: Calibration) -> GrowthKind {
    GrowthKind::EcoTune {
        config: EcoTuneConfig {
            runs_per_round: 8,
            last_level_ratio: 3,
            // Measured on this machine, not invented. See the calibration
            // caveat printed at the top of the run.
            tm_interval: calibration.flush_interval_seconds.max(1e-9),
            rewrite_time: calibration.rewrite_seconds.max(1e-9),
            long_range_ratio: 0.3,
            false_positive_rate: 0.01,
            mlc_query_factor: calibration.mlc_query_factor,
        },
        top_capacity_bytes: 256 * 1024,
    }
}

/// A mixed read/write workload, varying only the distribution and value size.
fn mixed(distribution: KeyDistribution, value_bytes: usize, keys: u64) -> WorkloadSpec {
    WorkloadSpec {
        key_count: keys,
        value_bytes,
        distribution,
        get_ratio: 0.5,
        put_ratio: 0.5,
        delete_ratio: 0.0,
        scan_ratio: 0.0,
        scan_length: 100,
        seed: 20_260_727,
    }
}

/// Scan-heavy: the workload the compaction policies actually differ on, since a
/// point lookup is answered from a bloom filter almost regardless of run count.
fn scan_heavy(distribution: KeyDistribution, keys: u64) -> WorkloadSpec {
    WorkloadSpec {
        get_ratio: 0.2,
        put_ratio: 0.2,
        scan_ratio: 0.6,
        scan_length: 100,
        ..mixed(distribution, 100, keys)
    }
}

struct Runner {
    rows: Vec<BenchResult>,
}

impl Runner {
    fn run(
        &mut self,
        label: &str,
        growth: GrowthKind,
        merge: MergeKind,
        spec: WorkloadSpec,
        run_ops: usize,
    ) {
        let dir = match BenchDir::new(label) {
            Ok(dir) => dir,
            Err(error) => {
                eprintln!("  {label}: could not create a scratch directory: {error}");
                return;
            }
        };
        match run_benchmark(dir.path(), base_config(growth, merge), spec, run_ops) {
            Ok(result) => {
                print_row(&result);
                self.rows.push(result);
            }
            // Report and continue: one failing configuration must not silently
            // shrink the result set.
            Err(error) => eprintln!("  {label}: FAILED: {error}"),
        }
    }
}

fn print_header() {
    println!(
        "  {:<20} {:<9} {:<11} {:>5} {:>8} {:>7} {:>7} {:>7} {:>8} {:>7}",
        "growth", "merge", "keys", "val", "kops/s", "p50us", "p99us", "WA", "spaceA", "runs"
    );
}

fn print_row(result: &BenchResult) {
    fn show(value: Option<f64>, precision: usize) -> String {
        value.map_or("-".into(), |v| format!("{v:.precision$}"))
    }
    println!(
        "  {:<20} {:<9} {:<11} {:>5} {:>8.1} {:>7} {:>7} {:>7} {:>8} {:>7}",
        result.growth,
        result.merge,
        result.distribution,
        result.value_bytes,
        result.throughput_ops_per_sec / 1000.0,
        show(result.p50_micros, 1),
        show(result.p99_micros, 1),
        show(result.write_amplification, 2),
        show(result.space_amplification, 2),
        result.run_count,
    );
}

fn main() -> std::io::Result<()> {
    println!("Phase 3 storage benchmarks");
    println!("  single-threaded, compaction synchronous inside put, SyncPolicy::Manual");
    println!("  relative ordering is the point; absolute throughput is not comparable");
    println!("  to the papers, which assume background compaction on many cores\n");

    // ---- Calibrate EcoTune's hardware parameters. ----
    let calibration = {
        let dir = BenchDir::new("calibration")?;
        calibrate(
            dir.path(),
            base_config(vertical(), MergeKind::Leveling),
            mixed(KeyDistribution::Sequential, 100, 10_000),
        )?
    };
    println!("EcoTune calibration on this machine:");
    println!("  T_w (flush interval)   {:.6} s", calibration.flush_interval_seconds);
    println!("  T_c (rewrite one unit) {:.6} s", calibration.rewrite_seconds);
    println!("  beta (query during ML) {:.3}", calibration.mlc_query_factor);
    println!(
        "  NOTE: beta here is the share of wall-clock time compaction occupies, not\n\
         \x20       the paper's contended-thread measurement. This engine compacts\n\
         \x20       synchronously, so there is no concurrency to observe. Approximate."
    );

    let mut runner = Runner { rows: Vec::new() };

    // ---- Sweep 1: growth scheme. Everything else fixed. ----
    println!("\n[1] Growth scheme (merge=leveling, uniform keys, 100 B values)");
    print_header();
    for growth in [vertical(), horizontal(), vertiorizon()] {
        runner.run(
            "growth",
            growth,
            MergeKind::Leveling,
            mixed(KeyDistribution::Uniform, 100, KEY_COUNT),
            RUN_OPS,
        );
    }

    // ---- Sweep 2: merge policy. ----
    println!("\n[2] Merge policy (growth=vertical, uniform keys, 100 B values)");
    print_header();
    for merge in [MergeKind::Leveling, MergeKind::Tiering { runs_per_level: 4 }] {
        runner.run(
            "merge",
            vertical(),
            merge,
            mixed(KeyDistribution::Uniform, 100, KEY_COUNT),
            RUN_OPS,
        );
    }

    // ---- Sweep 3: key distribution. The axis that hides policy differences. ----
    println!("\n[3] Key distribution (100 B values) -- sequential should flatter leveling");
    print_header();
    for distribution in [
        KeyDistribution::Sequential,
        KeyDistribution::Uniform,
        KeyDistribution::Zipfian { theta: 0.99 },
    ] {
        for merge in [MergeKind::Leveling, MergeKind::Tiering { runs_per_level: 4 }] {
            runner.run(
                "dist",
                vertical(),
                merge,
                mixed(distribution, 100, KEY_COUNT),
                RUN_OPS,
            );
        }
    }

    // ---- Sweep 4: value size. 100 B is the papers' regime; 512 is SIFT, 3840 GIST. ----
    println!("\n[4] Value size (growth=vertical, merge=leveling, uniform keys)");
    print_header();
    for (value_bytes, keys) in [(100usize, KEY_COUNT), (512, KEY_COUNT / 2), (3840, 4_000)] {
        runner.run(
            "value",
            vertical(),
            MergeKind::Leveling,
            mixed(KeyDistribution::Uniform, value_bytes, keys),
            RUN_OPS / 2,
        );
    }

    // ---- Sweep 5: scan-heavy, where run count actually drives I/O. ----
    println!("\n[5] Scan-heavy, 60% scans of 100 entries (uniform keys, 100 B values)");
    print_header();
    for (growth, merge) in [
        (vertical(), MergeKind::Leveling),
        (vertical(), MergeKind::Tiering { runs_per_level: 4 }),
        (vertiorizon(), MergeKind::Leveling),
        (ecotune(calibration), MergeKind::Tiering { runs_per_level: 8 }),
    ] {
        runner.run(
            "scan",
            growth,
            merge,
            scan_heavy(KeyDistribution::Uniform, KEY_COUNT / 4),
            RUN_OPS / 10,
        );
    }

    // ---- Write the CSV. ----
    std::fs::create_dir_all("results")?;
    let path = "results/storage_benchmarks.csv";
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "{}", BenchResult::csv_header())?;
    for row in &runner.rows {
        writeln!(file, "{}", row.to_csv())?;
    }
    println!("\nWrote {} rows to {path}", runner.rows.len());

    // ---- The question the two papers disagree about. ----
    println!("\nWrite amplification versus throughput, across every row above:");
    let mut pairs: Vec<(f64, f64)> = runner
        .rows
        .iter()
        .filter_map(|row| {
            row.write_amplification
                .map(|wa| (wa, row.throughput_ops_per_sec))
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    if pairs.len() >= 4 {
        println!("  correlation {:+.3}", correlation(&pairs));
        println!(
            "  (Vertiorizon minimises write amplification; EcoTune argues it is free.\n\
             \x20  A correlation near zero supports EcoTune's premise on this hardware.\n\
             \x20  These rows mix workloads, so this is a smell test, not a controlled\n\
             \x20  experiment -- the controlled version needs one config swept by\n\
             \x20  memtable size at a fixed workload.)"
        );
    }

    Ok(())
}

/// Pearson correlation.
fn correlation(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len() as f64;
    let mean_x = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = pairs.iter().map(|p| p.1).sum::<f64>() / n;

    let mut covariance = 0.0;
    let mut variance_x = 0.0;
    let mut variance_y = 0.0;
    for (x, y) in pairs {
        let dx = x - mean_x;
        let dy = y - mean_y;
        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }
    if variance_x <= 0.0 || variance_y <= 0.0 {
        return 0.0;
    }
    covariance / (variance_x.sqrt() * variance_y.sqrt())
}
