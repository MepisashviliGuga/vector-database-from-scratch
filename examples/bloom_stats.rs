//! Measures what the SSTable bloom filter actually buys, on a 100k-entry table.
//!
//! Run with `cargo run --release --example bloom_stats`. Reports bits per key,
//! block reads for present keys, and the observed false-positive rate against
//! the 1% target — the numbers that justify the filter's resident memory cost.
//!
//! The absent keys probed here are *interleaved between* present keys. Probing
//! keys outside the table's range instead would measure nothing: the sparse
//! index rejects those before the filter is ever consulted.

use vectordb::storage::{SSTable, SSTableWriter, Value};

fn main() -> std::io::Result<()> {
    let dir = std::env::temp_dir().join("vectordb-bloom-stats");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("stats.sst");

    // Even keys only, so the odd keys below are absent but interleaved inside
    // the table's range — the sparse index cannot rule them out, so what we
    // measure is genuinely the filter's work.
    let n = 100_000usize;
    let mut writer = SSTableWriter::create(&path)?;
    for i in 0..n {
        let key = format!("key{:08}", i * 2);
        writer.append(key.as_bytes(), &Value::Put(vec![b'v'; 100]))?;
    }
    let meta = writer.finish()?;

    let table = SSTable::open(&path)?;
    println!("entries        {}", meta.entry_count);
    println!("blocks         {}", meta.block_count);
    println!(
        "file size      {:.2} MiB",
        meta.file_size_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "bloom bytes    {} ({:.2} bits/key)",
        table.bloom_bytes(),
        table.bloom_bytes() as f64 * 8.0 / n as f64
    );

    // Present keys: every lookup must succeed and cost exactly one block read.
    table.reset_counters();
    for i in 0..n {
        let key = format!("key{:08}", i * 2);
        assert!(table.get(key.as_bytes())?.is_some());
    }
    println!(
        "\npresent keys   {n} lookups -> {} block reads, {} bloom rejections",
        table.blocks_read(),
        table.bloom_rejections()
    );

    // Absent keys interleaved between present ones: only the filter can save
    // these reads.
    table.reset_counters();
    let probes = 100_000usize;
    for i in 0..probes {
        let key = format!("key{:08}", i * 2 + 1);
        assert!(table.get(key.as_bytes())?.is_none());
    }
    let reads = table.blocks_read();
    println!(
        "absent keys    {probes} lookups -> {reads} block reads, {} bloom rejections",
        table.bloom_rejections()
    );
    println!(
        "measured FPR   {:.4}% (target 1%)",
        reads as f64 / probes as f64 * 100.0
    );

    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
