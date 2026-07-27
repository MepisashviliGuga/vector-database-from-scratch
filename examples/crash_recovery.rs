//! Kills a real process mid-workload and checks what survives.
//!
//! Run with `cargo run --release --example crash_recovery`.
//!
//! The unit tests simulate a crash by dropping the tree, which only exercises
//! replay. This spawns a child process that writes with `SyncPolicy::EveryWrite`
//! and then calls `std::process::abort()` — no unwinding, no destructors, no
//! flush, no clean close — and then reopens the database in the parent.
//!
//! **What this does and does not prove.** It proves that a write which returned
//! `Ok` is still there after abrupt process death. It does *not* exercise a torn
//! log record: `abort()` leaves the data in the OS page cache, which the kernel
//! still writes out, so the log ends on a clean record boundary. Genuine torn
//! tails need power loss, and are covered instead by the `wal` unit tests that
//! truncate the file at every byte offset.

use std::path::Path;
use std::process::Command;

use vectordb::storage::{LsmConfig, LsmTree, SyncPolicy};

/// Writes acknowledged before the child kills itself.
const ACKNOWLEDGED_WRITES: usize = 5_000;

fn config() -> LsmConfig {
    LsmConfig {
        // Small enough that the run crosses the flush threshold several times,
        // so recovery has to reconcile SSTables *and* a live log.
        memtable_threshold_bytes: 64 * 1024,
        sync_policy: SyncPolicy::EveryWrite,
        ..Default::default()
    }
}

fn key(i: usize) -> Vec<u8> {
    format!("key{i:08}").into_bytes()
}

fn value(i: usize) -> Vec<u8> {
    format!("value-{i}").into_bytes()
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--child") {
        return child(Path::new(&args[2]));
    }
    parent()
}

/// Write, acknowledge, then die without any chance to clean up.
fn child(dir: &Path) -> std::io::Result<()> {
    let mut tree = LsmTree::open(dir, config())?;
    for i in 0..ACKNOWLEDGED_WRITES {
        tree.put(key(i), value(i))?;
    }
    // Delete a slice of them, so recovery has to preserve tombstones too.
    for i in (0..ACKNOWLEDGED_WRITES).step_by(10) {
        tree.delete(key(i))?;
    }

    eprintln!("child: {ACKNOWLEDGED_WRITES} writes acknowledged, aborting now");
    // No flush, no drop, no unwinding.
    std::process::abort();
}

fn parent() -> std::io::Result<()> {
    let dir = std::env::temp_dir().join(format!("vectordb-crash-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;

    let exe = std::env::current_exe()?;
    println!("spawning child to write {ACKNOWLEDGED_WRITES} records and abort...");
    let status = Command::new(exe)
        .arg("--child")
        .arg(&dir)
        .status()
        .expect("spawn child");

    assert!(
        !status.success(),
        "the child was supposed to abort, but exited cleanly with {status}"
    );
    println!("child died as expected: {status}\n");

    println!("reopening the database...");
    let tree = LsmTree::open(&dir, config())?;
    let stats = tree.stats();
    println!("  runs on disk    {}", stats.sstable_count);
    println!("  memtable entries {} (replayed from the log)", stats.memtable_entries);

    let mut live = 0usize;
    let mut deleted = 0usize;
    let mut lost = Vec::new();
    let mut wrong = Vec::new();

    for i in 0..ACKNOWLEDGED_WRITES {
        let expected_deleted = i % 10 == 0;
        match tree.get(&key(i))? {
            None if expected_deleted => deleted += 1,
            None => lost.push(i),
            Some(bytes) if expected_deleted => wrong.push((i, format!("resurrected: {bytes:?}"))),
            Some(bytes) if bytes == value(i) => live += 1,
            Some(bytes) => wrong.push((i, format!("wrong value: {bytes:?}"))),
        }
    }

    println!("\n  live keys recovered    {live}");
    println!("  deletions honoured     {deleted}");
    println!("  LOST                   {}", lost.len());
    println!("  WRONG                  {}", wrong.len());

    if !lost.is_empty() || !wrong.is_empty() {
        println!("\nFAILED — acknowledged writes did not survive.");
        if let Some(first) = lost.first() {
            println!("  first lost key: {first}");
        }
        if let Some((index, reason)) = wrong.first() {
            println!("  first wrong key: {index} ({reason})");
        }
        std::process::exit(1);
    }

    // The iteration path must agree with the point lookups after recovery too.
    let iterated = tree.iter().count();
    let expected_live = ACKNOWLEDGED_WRITES - ACKNOWLEDGED_WRITES.div_ceil(10);
    println!("\n  iteration yields       {iterated} (expected {expected_live})");
    assert_eq!(iterated, expected_live, "iteration disagrees with lookups after recovery");

    println!("\nPASSED — every acknowledged write survived an abrupt kill.");
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
