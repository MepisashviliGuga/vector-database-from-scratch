//! Measurement harness for the storage benchmarks.
//!
//! # What gets measured, and why each one
//!
//! - **Write amplification** — bytes written to disk per byte of user data. The
//!   quantity Vertiorizon minimises and EcoTune argues is free. Measuring it is
//!   how those two papers get adjudicated on real hardware.
//! - **Read amplification** — data blocks read per read operation. Taken from
//!   the per-SSTable counters rather than inferred, and reported separately for
//!   point lookups and scans, because bloom filters make those two diverge
//!   sharply.
//! - **Space amplification** — bytes on disk per byte of live data. Live bytes
//!   are counted by iterating the tree, not estimated from the workload, so
//!   obsolete versions and un-dropped tombstones are included where they belong.
//! - **Throughput and latency** — p50/p99/p999 from the run phase.
//!
//! # Two phases, YCSB-style
//!
//! A **load** phase inserts the whole key space once, then a **run** phase
//! executes the operation mix. Latency and read amplification come from the run
//! phase only; write amplification covers both, since the load phase is where
//! most of the writing happens.
//!
//! The load phase inserts sequentially, as YCSB does. That matters: sequential
//! inserts are the case where leveling and tiering do *identical* work, so a
//! run phase that also writes is what exposes the difference. Both are reported.
//!
//! # Honesty constraints
//!
//! Everything here is deterministic given a seed, and every field of
//! [`BenchResult`] comes from a counter or a clock — nothing is modelled or
//! extrapolated. Where a number cannot be measured it is absent rather than
//! estimated.

use std::io;
use std::path::Path;
use std::time::Instant;

use crate::storage::{LsmConfig, LsmTree};
use crate::workload::{KeyDistribution, Operation, Workload, WorkloadSpec};

/// Collected operation latencies, in nanoseconds.
///
/// Stores every sample rather than bucketing them. At benchmark scales that is
/// a few megabytes, and it makes percentiles exact — a histogram's bucket width
/// is exactly the kind of quiet approximation that should not sit underneath a
/// reported p99.
#[derive(Debug, Default, Clone)]
pub struct Latencies {
    samples: Vec<u64>,
}

impl Latencies {
    pub fn record(&mut self, nanos: u64) {
        self.samples.push(nanos);
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The `p`th percentile in microseconds, with `p` in `[0, 100]`.
    ///
    /// Nearest-rank: `index = ceil(p/100 · n) − 1`. That is the convention where
    /// "p99" means "the smallest value at least 99% of samples fall at or below",
    /// which is what a latency SLO means by it. Interpolating between
    /// neighbouring samples would report a latency that never occurred.
    ///
    /// Sorts in place. Returns `None` for an empty sample set rather than 0, so
    /// "no data" cannot be mistaken for "instant".
    pub fn percentile_micros(&mut self, p: f64) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        self.samples.sort_unstable();
        let count = self.samples.len();
        let rank = (p.clamp(0.0, 100.0) / 100.0 * count as f64).ceil() as usize;
        let index = rank.max(1) - 1;
        Some(self.samples[index.min(count - 1)] as f64 / 1000.0)
    }

    pub fn mean_micros(&self) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        let total: u128 = self.samples.iter().map(|&n| n as u128).sum();
        Some(total as f64 / self.samples.len() as f64 / 1000.0)
    }
}

/// One row of results.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub growth: String,
    pub merge: String,
    pub distribution: String,
    pub value_bytes: usize,
    pub workload: String,
    /// Whether every write was `fsync`ed. A throughput number is meaningless
    /// without it, so it travels with the row rather than living in a comment.
    pub sync_policy: String,

    pub load_ops: usize,
    pub load_seconds: f64,
    pub run_ops: usize,
    pub run_seconds: f64,

    /// Run-phase operations per second.
    pub throughput_ops_per_sec: f64,
    pub p50_micros: Option<f64>,
    pub p99_micros: Option<f64>,
    pub p999_micros: Option<f64>,

    /// Bytes written to disk per byte of user data, over both phases.
    pub write_amplification: Option<f64>,
    /// Same, over the load phase alone — the sequential-insert case.
    pub load_write_amplification: Option<f64>,
    /// Data blocks read per point lookup during the run phase.
    pub blocks_per_point_read: Option<f64>,
    /// Data blocks read per scan during the run phase.
    pub blocks_per_scan: Option<f64>,
    /// Bytes on disk per byte of live data.
    pub space_amplification: Option<f64>,

    pub run_count: usize,
    pub file_count: usize,
    pub runs_per_level: Vec<usize>,
    pub disk_bytes: u64,
    pub live_bytes: u64,
    pub flush_count: u64,
    pub compaction_count: u64,
    pub bloom_rejections: u64,
}

impl BenchResult {
    pub fn csv_header() -> &'static str {
        "growth,merge,distribution,value_bytes,workload,sync_policy,\
         load_ops,load_seconds,run_ops,run_seconds,throughput_ops_per_sec,\
         p50_micros,p99_micros,p999_micros,\
         write_amplification,load_write_amplification,\
         blocks_per_point_read,blocks_per_scan,space_amplification,\
         run_count,file_count,levels,disk_bytes,live_bytes,\
         flush_count,compaction_count,bloom_rejections"
    }

    pub fn to_csv(&self) -> String {
        /// Empty rather than a stand-in number: a missing measurement must not
        /// look like a measured zero in a spreadsheet.
        fn optional(value: Option<f64>) -> String {
            value.map_or(String::new(), |v| format!("{v:.4}"))
        }

        format!(
            "{},{},{},{},{},{},{},{:.4},{},{:.4},{:.1},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.growth,
            self.merge,
            self.distribution,
            self.value_bytes,
            self.workload,
            self.sync_policy,
            self.load_ops,
            self.load_seconds,
            self.run_ops,
            self.run_seconds,
            self.throughput_ops_per_sec,
            optional(self.p50_micros),
            optional(self.p99_micros),
            optional(self.p999_micros),
            optional(self.write_amplification),
            optional(self.load_write_amplification),
            optional(self.blocks_per_point_read),
            optional(self.blocks_per_scan),
            optional(self.space_amplification),
            self.run_count,
            self.file_count,
            self.runs_per_level.len(),
            self.disk_bytes,
            self.live_bytes,
            self.flush_count,
            self.compaction_count,
            self.bloom_rejections,
        )
    }
}

/// Run one configuration against one workload.
///
/// `dir` must be empty; the caller owns it and is responsible for cleanup, so a
/// benchmark can be inspected after the fact.
pub fn run_benchmark(
    dir: &Path,
    config: LsmConfig,
    spec: WorkloadSpec,
    run_ops: usize,
) -> io::Result<BenchResult> {
    let growth = config.growth.name().to_string();
    let merge = config.merge.name().to_string();
    let sync_policy = format!("{:?}", config.sync_policy);
    let mut tree = LsmTree::open(dir, config)?;

    // ---- Load phase: insert the key space once, sequentially. ----
    let value = vec![0xAB; spec.value_bytes];
    let load_start = Instant::now();
    for id in 0..spec.key_count {
        tree.put(crate::workload::encode_key(id), value.clone())?;
    }
    tree.flush()?;
    let load_seconds = load_start.elapsed().as_secs_f64();

    let after_load = tree.stats();
    let load_write_amplification = after_load.write_amplification();

    // ---- Run phase: the operation mix. ----
    tree.reset_io_counters();
    let mut workload = Workload::new(spec);
    let mut latencies = Latencies::default();
    let mut point_reads = 0usize;
    let mut scans = 0usize;
    let mut blocks_in_scans = 0u64;

    let run_start = Instant::now();
    for _ in 0..run_ops {
        let operation = workload.generate();
        let started = Instant::now();

        match &operation {
            Operation::Put { key, value } => {
                tree.put(key.clone(), value.clone())?;
            }
            Operation::Get { key } => {
                point_reads += 1;
                tree.get(key)?;
            }
            Operation::Delete { key } => {
                tree.delete(key.clone())?;
            }
            Operation::Scan { key, length } => {
                scans += 1;
                let blocks_before_scans = tree.stats().blocks_read;
                // Consumed eagerly: a lazy iterator would move the I/O outside
                // the timed region and report a scan as free.
                let mut seen = 0usize;
                for entry in tree.range_from(key).take(*length) {
                    entry?;
                    seen += 1;
                }
                std::hint::black_box(seen);
                blocks_in_scans += tree.stats().blocks_read - blocks_before_scans;
            }
        }
        latencies.record(started.elapsed().as_nanos() as u64);
    }
    let run_seconds = run_start.elapsed().as_secs_f64();

    // ---- Final state. ----
    let stats = tree.stats();
    let live_bytes: u64 = tree
        .iter()
        .map(|entry| entry.map(|(key, value)| (key.len() + value.len()) as u64))
        .collect::<io::Result<Vec<u64>>>()?
        .into_iter()
        .sum();

    let blocks_in_point_reads = stats.blocks_read.saturating_sub(blocks_in_scans);

    Ok(BenchResult {
        growth,
        merge,
        distribution: spec.distribution.name().to_string(),
        value_bytes: spec.value_bytes,
        workload: spec.label(),
        sync_policy,

        load_ops: spec.key_count as usize,
        load_seconds,
        run_ops,
        run_seconds,

        throughput_ops_per_sec: if run_seconds > 0.0 {
            run_ops as f64 / run_seconds
        } else {
            0.0
        },
        p50_micros: latencies.percentile_micros(50.0),
        p99_micros: latencies.percentile_micros(99.0),
        p999_micros: latencies.percentile_micros(99.9),

        write_amplification: stats.write_amplification(),
        load_write_amplification,
        blocks_per_point_read: (point_reads > 0)
            .then(|| blocks_in_point_reads as f64 / point_reads as f64),
        blocks_per_scan: (scans > 0).then(|| blocks_in_scans as f64 / scans as f64),
        space_amplification: (live_bytes > 0).then(|| stats.disk_bytes as f64 / live_bytes as f64),

        run_count: stats.run_count,
        file_count: stats.file_count,
        runs_per_level: stats.runs_per_level,
        disk_bytes: stats.disk_bytes,
        live_bytes,
        flush_count: stats.flush_count,
        compaction_count: stats.compaction_count,
        bloom_rejections: stats.bloom_rejections,
    })
}

/// Hardware parameters EcoTune's cost model needs, measured rather than assumed.
///
/// Paper 02 measures both on the machine it runs on. Leaving them as invented
/// constants would make every EcoTune result a statement about those constants
/// rather than about the algorithm.
#[derive(Debug, Clone, Copy)]
pub struct Calibration {
    /// `T_c`: seconds to rewrite one unit run's worth of data.
    pub rewrite_seconds: f64,
    /// `T_w`: seconds between two flushes at the observed write rate.
    pub flush_interval_seconds: f64,
    /// `β`: query speed while a merge is running, as a fraction of the
    /// undisturbed speed.
    pub mlc_query_factor: f64,
}

/// Measure [`Calibration`] by running a small write-then-read workload.
///
/// **Single-threaded caveat.** Paper 02's `β` compares query speed with merge
/// threads busy against query speed with them idle, on a system where
/// compaction runs in the background. This engine compacts synchronously inside
/// `put`, so there is no concurrent contention to observe. What is measured here
/// is the *share of wall-clock time* compaction occupies, which bounds β from
/// the same resource-competition argument but is not the same quantity. It is
/// reported as an approximation and labelled in the writeup.
pub fn calibrate(dir: &Path, config: LsmConfig, spec: WorkloadSpec) -> io::Result<Calibration> {
    let mut tree = LsmTree::open(dir, config.clone())?;
    let value = vec![0xAB; spec.value_bytes];

    let start = Instant::now();
    for id in 0..spec.key_count {
        tree.put(crate::workload::encode_key(id), value.clone())?;
    }
    tree.flush()?;
    let elapsed = start.elapsed().as_secs_f64();

    let stats = tree.stats();
    let flushes = stats.flush_count.max(1) as f64;
    let flush_interval_seconds = elapsed / flushes;

    // Seconds per byte of compaction output, scaled to one buffer's worth.
    let rewrite_seconds = if stats.compaction_bytes_written > 0 {
        let compaction_share =
            stats.compaction_bytes_written as f64 / stats.sstable_bytes_written.max(1) as f64;
        let compaction_seconds = elapsed * compaction_share;
        let buffers =
            stats.compaction_bytes_written as f64 / config.memtable_threshold_bytes.max(1) as f64;
        compaction_seconds / buffers.max(1.0)
    } else {
        0.0
    };

    // Reads with the tree settled, against reads while compaction is in flight.
    tree.reset_io_counters();
    let quiet_start = Instant::now();
    for id in 0..spec.key_count.min(2000) {
        tree.get(&crate::workload::encode_key(id))?;
    }
    let quiet = quiet_start.elapsed().as_secs_f64();

    let busy_start = Instant::now();
    for id in 0..spec.key_count.min(2000) {
        tree.put(crate::workload::encode_key(id), value.clone())?;
        tree.get(&crate::workload::encode_key(id))?;
    }
    let busy = busy_start.elapsed().as_secs_f64();

    let mlc_query_factor = if busy > 0.0 && quiet > 0.0 {
        (quiet / busy).clamp(0.01, 1.0)
    } else {
        1.0
    };

    Ok(Calibration {
        rewrite_seconds,
        flush_interval_seconds,
        mlc_query_factor,
    })
}

/// A scratch directory that removes itself.
#[derive(Debug)]
pub struct BenchDir {
    path: std::path::PathBuf,
}

impl BenchDir {
    pub fn new(label: &str) -> io::Result<Self> {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);

        let path = std::env::temp_dir().join(format!(
            "vectordb-bench-{label}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path)?;
        Ok(Self { path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for BenchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// A small, fast configuration for smoke-testing the harness itself.
pub fn smoke_spec(distribution: KeyDistribution) -> WorkloadSpec {
    WorkloadSpec {
        key_count: 2_000,
        value_bytes: 100,
        distribution,
        get_ratio: 0.5,
        put_ratio: 0.5,
        delete_ratio: 0.0,
        scan_ratio: 0.0,
        scan_length: 50,
        seed: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{GrowthKind, MergeKind, SyncPolicy};

    fn test_config() -> LsmConfig {
        LsmConfig {
            memtable_threshold_bytes: 16 * 1024,
            sync_policy: SyncPolicy::Manual,
            block_target_bytes: 1024,
            target_file_size_bytes: 64 * 1024,
            growth: GrowthKind::Vertical {
                buffer_bytes: 16 * 1024,
                size_ratio: 4,
            },
            merge: MergeKind::Leveling,
            ..Default::default()
        }
    }

    // -----------------------------------------------------------------
    // Percentiles
    // -----------------------------------------------------------------

    #[test]
    fn percentiles_are_exact_on_a_known_sample() {
        let mut latencies = Latencies::default();
        for nanos in 1..=100u64 {
            latencies.record(nanos * 1000);
        }
        assert_eq!(latencies.percentile_micros(0.0), Some(1.0));
        assert_eq!(latencies.percentile_micros(50.0), Some(50.0));
        assert_eq!(latencies.percentile_micros(99.0), Some(99.0));
        assert_eq!(latencies.percentile_micros(100.0), Some(100.0));
    }

    #[test]
    fn percentiles_do_not_depend_on_insertion_order() {
        let mut ascending = Latencies::default();
        let mut descending = Latencies::default();
        for n in 1..=1000u64 {
            ascending.record(n);
            descending.record(1001 - n);
        }
        assert_eq!(
            ascending.percentile_micros(95.0),
            descending.percentile_micros(95.0)
        );
    }

    /// An empty sample set has no percentile. Returning 0 would make "we
    /// measured nothing" indistinguishable from "it was instant".
    #[test]
    fn an_empty_sample_set_has_no_percentile() {
        let mut latencies = Latencies::default();
        assert_eq!(latencies.percentile_micros(50.0), None);
        assert_eq!(latencies.mean_micros(), None);
        assert!(latencies.is_empty());
    }

    // -----------------------------------------------------------------
    // The harness
    // -----------------------------------------------------------------

    #[test]
    fn a_benchmark_produces_plausible_measurements() {
        let dir = BenchDir::new("smoke").expect("dir");
        let result = run_benchmark(
            dir.path(),
            test_config(),
            smoke_spec(KeyDistribution::Uniform),
            2_000,
        )
        .expect("benchmark");

        assert_eq!(result.load_ops, 2_000);
        assert_eq!(result.run_ops, 2_000);
        assert!(result.throughput_ops_per_sec > 0.0);
        assert!(result.p50_micros.is_some());
        assert!(
            result.p99_micros >= result.p50_micros,
            "p99 {:?} cannot be below p50 {:?}",
            result.p99_micros,
            result.p50_micros
        );

        let amplification = result.write_amplification.expect("write amplification");
        assert!(
            amplification >= 1.0,
            "writing less than the user supplied is impossible: {amplification}"
        );
        assert!(result.live_bytes > 0);
        assert!(result.disk_bytes > 0);
        assert!(result.compaction_count > 0, "compaction never ran");
    }

    /// Space amplification is disk over *live* bytes, and live bytes are counted
    /// by iterating rather than assumed from the workload.
    #[test]
    fn space_amplification_is_at_least_one() {
        let dir = BenchDir::new("space").expect("dir");
        let result = run_benchmark(
            dir.path(),
            test_config(),
            smoke_spec(KeyDistribution::Sequential),
            500,
        )
        .expect("benchmark");

        let space = result.space_amplification.expect("space amplification");
        assert!(
            space >= 1.0,
            "disk cannot hold less than the live data: {space}"
        );
        assert!(
            space < 20.0,
            "space amplification of {space} is implausible"
        );
    }

    /// Read amplification must come from the block counters, and a workload with
    /// no reads must report none rather than zero.
    #[test]
    fn read_amplification_is_absent_when_nothing_was_read() {
        let dir = BenchDir::new("no-reads").expect("dir");
        let spec = WorkloadSpec {
            get_ratio: 0.0,
            put_ratio: 1.0,
            ..smoke_spec(KeyDistribution::Uniform)
        };
        let result = run_benchmark(dir.path(), test_config(), spec, 500).expect("benchmark");

        assert_eq!(result.blocks_per_point_read, None);
        assert_eq!(result.blocks_per_scan, None);
    }

    #[test]
    fn scans_are_measured_separately_from_point_reads() {
        let dir = BenchDir::new("scans").expect("dir");
        let spec = WorkloadSpec {
            get_ratio: 0.5,
            put_ratio: 0.0,
            delete_ratio: 0.0,
            scan_ratio: 0.5,
            scan_length: 20,
            ..smoke_spec(KeyDistribution::Uniform)
        };
        let result = run_benchmark(dir.path(), test_config(), spec, 400).expect("benchmark");

        let per_scan = result.blocks_per_scan.expect("scan measurement");
        let per_read = result.blocks_per_point_read.expect("point measurement");
        assert!(
            per_scan > per_read,
            "a 20-entry scan ({per_scan} blocks) should cost more than a point \
             lookup ({per_read} blocks)"
        );
    }

    /// The same seed and configuration must give the same *work*, even though
    /// wall-clock timings vary run to run.
    #[test]
    fn benchmarks_are_reproducible_in_everything_but_timing() {
        let run = || {
            let dir = BenchDir::new("repeat").expect("dir");
            run_benchmark(
                dir.path(),
                test_config(),
                smoke_spec(KeyDistribution::Zipfian { theta: 0.99 }),
                1_000,
            )
            .expect("benchmark")
        };

        let first = run();
        let second = run();

        assert_eq!(first.disk_bytes, second.disk_bytes);
        assert_eq!(first.live_bytes, second.live_bytes);
        assert_eq!(first.compaction_count, second.compaction_count);
        assert_eq!(first.run_count, second.run_count);
        assert_eq!(first.write_amplification, second.write_amplification);
    }

    #[test]
    fn the_csv_row_has_one_field_per_header_column() {
        let dir = BenchDir::new("csv").expect("dir");
        let result = run_benchmark(
            dir.path(),
            test_config(),
            smoke_spec(KeyDistribution::Uniform),
            200,
        )
        .expect("benchmark");

        let headers = BenchResult::csv_header().split(',').count();
        let fields = result.to_csv().split(',').count();
        assert_eq!(headers, fields, "header and row disagree on column count");
    }

    /// A missing measurement must be an empty cell, not a zero that a
    /// spreadsheet would happily average.
    #[test]
    fn absent_measurements_render_as_empty_cells() {
        let dir = BenchDir::new("empty-cells").expect("dir");
        let spec = WorkloadSpec {
            get_ratio: 0.0,
            put_ratio: 1.0,
            ..smoke_spec(KeyDistribution::Uniform)
        };
        let result = run_benchmark(dir.path(), test_config(), spec, 200).expect("benchmark");

        let row = result.to_csv();
        let fields: Vec<&str> = row.split(',').collect();
        let headers: Vec<&str> = BenchResult::csv_header().split(',').collect();
        let column = headers
            .iter()
            .position(|h| h.trim() == "blocks_per_point_read")
            .expect("column");
        assert_eq!(fields[column], "", "an unmeasured value must be blank");
    }

    #[test]
    fn calibration_reports_finite_parameters() {
        let dir = BenchDir::new("calibrate").expect("dir");
        let calibration = calibrate(
            dir.path(),
            test_config(),
            smoke_spec(KeyDistribution::Sequential),
        )
        .expect("calibration");

        assert!(calibration.rewrite_seconds >= 0.0);
        assert!(calibration.rewrite_seconds.is_finite());
        assert!(calibration.flush_interval_seconds > 0.0);
        assert!(
            (0.01..=1.0).contains(&calibration.mlc_query_factor),
            "beta of {} is outside its definition",
            calibration.mlc_query_factor
        );
    }
}
