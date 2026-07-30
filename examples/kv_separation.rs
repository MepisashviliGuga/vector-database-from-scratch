//! Does key–value separation actually fix the 22.65× write amplification?
//!
//! ```text
//! cargo run --release --example kv_separation
//! ```
//!
//! # The finding this answers
//!
//! Phase 3 swept value size and found write amplification rising from **3.50× at
//! 100 B to 22.65× at 3,840 B** — the size of a GIST vector — with p99 write
//! latency up 360× (`results/README.md`). The cause is structural: compaction
//! rewrites the whole entry, so a byte of value pays the full rewrite cost at
//! every level its key descends.
//!
//! No source paper in this project addresses it. They compare growth schemes and
//! merge policies on ~100 B values, where the key dominates and value size is
//! noise. This sweeps the same value sizes with separation off and on, changing
//! nothing else.
//!
//! # Reading the columns
//!
//! `total` counts **everything written** — SSTables plus the value log — because
//! separation moves bytes rather than removing them, and a figure that ignored
//! where they went would flatter itself by construction. `tree` counts only what
//! the LSM rewrote, which is what compaction policy actually controls. The gap
//! between them is the point.

use std::time::Instant;

use vectordb::bench::BenchDir;
use vectordb::storage::{GrowthKind, LsmConfig, LsmTree, MergeKind, SyncPolicy};
use vectordb::workload::Rng;

/// The Phase 3 sweep: a small key-value record, a SIFT vector, a GIST vector,
/// each with the key count that sweep used at that size.
const VALUE_SIZES: [(usize, u32); 3] = [(100, 20_000), (512, 10_000), (3840, 4_000)];

fn config(separated: bool) -> LsmConfig {
    // A small buffer on purpose. Write amplification is driven by how many
    // levels a key descends, so a large memtable hides the very effect this
    // measures: at a 256 KiB buffer the unseparated baseline came out at 3.72×
    // rather than the 22.65× Phase 3 reported, simply because the tree was too
    // shallow to compact much. 64 KiB puts the sweep back in a regime where
    // compaction dominates, which is where the finding lives.
    let buffer = 64 * 1024;
    LsmConfig {
        memtable_threshold_bytes: buffer,
        sync_policy: SyncPolicy::Manual,
        target_file_size_bytes: 256 * 1024,
        growth: GrowthKind::Vertical {
            buffer_bytes: buffer as u64,
            size_ratio: 4,
        },
        merge: MergeKind::Leveling,
        // 64 B: comfortably below the smallest value swept, so at every size
        // tested the values are diverted and the comparison is like for like.
        value_log_threshold: separated.then_some(64),
        ..Default::default()
    }
}

struct Run {
    total_amplification: f64,
    tree_amplification: f64,
    disk_bytes: u64,
    value_log_bytes: u64,
    seconds: f64,
    read_micros: f64,
}

fn measure(value_bytes: usize, keys: u32, separated: bool) -> std::io::Result<Run> {
    let dir = BenchDir::new(if separated { "kv-on" } else { "kv-off" })?;
    let mut tree = LsmTree::open(dir.path(), config(separated))?;

    // Uniformly distributed keys, matching Phase 3. This is not a detail:
    // sequential keys produce runs with disjoint key ranges, so leveling has
    // almost nothing to rewrite and write amplification collapses to near 1
    // whatever the value size. Overlap is what makes compaction expensive, and
    // overlap is what a uniform distribution creates.
    let mut rng = Rng::new(0x5EED_5EED);
    let value = vec![0xAB; value_bytes];
    let started = Instant::now();
    for _ in 0..keys {
        let key = rng.below(u64::from(keys));
        tree.put(format!("key{key:08}").into_bytes(), value.clone())?;
    }
    tree.flush()?;
    tree.compact_until_quiet()?;
    let seconds = started.elapsed().as_secs_f64();

    // Point lookups, which is where separation costs an extra read.
    let probes = 2_000u32;
    let started = Instant::now();
    let mut hits = 0u32;
    for i in 0..probes {
        let key = format!("key{:08}", u64::from((i * 7) % keys));
        if let Some(found) = tree.get(key.as_bytes())? {
            assert_eq!(
                found.len(),
                value_bytes,
                "lookup lost data at {value_bytes} B, separated={separated}"
            );
            hits += 1;
        }
    }
    let read_micros = started.elapsed().as_secs_f64() * 1e6 / f64::from(probes);
    assert!(hits > probes / 4, "too few hits ({hits}) to time reads");

    let stats = tree.stats();
    Ok(Run {
        total_amplification: stats.write_amplification().unwrap_or(0.0),
        tree_amplification: stats.tree_write_amplification().unwrap_or(0.0),
        disk_bytes: stats.disk_bytes,
        value_log_bytes: stats.value_log_bytes_written,
        seconds,
        read_micros,
    })
}

fn main() -> std::io::Result<()> {
    println!("Key–value separation against the Phase 3 value-size sweep");
    println!("  uniform keys, 64 KiB buffer, vertical growth at ratio 4, leveling");
    println!("  — the same configuration Phase 3 measured 22.65× amplification in\n");

    println!(
        "  {:>7}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}",
        "value B", "WA off", "WA on", "tree on", "write s", "read µs"
    );
    println!("  {}", "-".repeat(64));

    let mut rows = Vec::new();
    for (size, keys) in VALUE_SIZES {
        let off = measure(size, keys, false)?;
        let on = measure(size, keys, true)?;
        println!(
            "  {size:>7}  {:>9.2} {:>9.2}  {:>9.2}  {:>9.1}  {:>9.1}",
            off.total_amplification,
            on.total_amplification,
            on.tree_amplification,
            on.seconds,
            on.read_micros,
        );
        rows.push((size, off, on));
    }

    println!("\n  Space, and what the separation costs to read:\n");
    println!(
        "  {:>7}  {:>12}  {:>12}  {:>12}  {:>12}",
        "value B", "disk off", "disk on", "read off µs", "read on µs"
    );
    println!("  {}", "-".repeat(64));
    for (size, off, on) in &rows {
        println!(
            "  {size:>7}  {:>10.1} MiB  {:>10.1} MiB  {:>12.1}  {:>12.1}",
            off.disk_bytes as f64 / (1024.0 * 1024.0),
            (on.disk_bytes + on.value_log_bytes) as f64 / (1024.0 * 1024.0),
            off.read_micros,
            on.read_micros,
        );
    }

    println!(
        "\n  WA counts every byte written, value log included — separation moves\n\
         \x20 bytes rather than removing them. `tree on` is the LSM's share alone,\n\
         \x20 which is what compaction policy controls and what separation frees.\n\
         \x20 `disk on` includes the value log. No garbage collection runs, so an\n\
         \x20 update-heavy workload would grow it further; this one is insert-only."
    );

    Ok(())
}
