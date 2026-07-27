//! The LSM-tree: memtable, write-ahead log, and levels of immutable runs.
//!
//! This module owns the pieces that the growth-scheme and compaction-policy work
//! plugs into. It deliberately does **no compaction yet** — every flush lands in
//! level 0 — so that flush, recovery, and the read path can be proven correct
//! before any policy is layered on. Phase 1 (Vertiorizon) decides how levels are
//! shaped; Phase 2 (EcoTune) decides when they are merged.
//!
//! # Write path
//!
//! ```text
//! put/delete ──► WAL (append, optionally fsync) ──► memtable
//!                                                       │ threshold reached
//!                                                       ▼
//!                                          SSTable in level 0, then WAL reset
//! ```
//!
//! The log is appended *before* the memtable is touched, so an acknowledged
//! write is recoverable. On flush the SSTable is written and `fsync`ed before
//! the log is discarded. A crash in that window replays log records that are
//! also already in the new SSTable — harmless, because the replayed memtable is
//! newer in the read order and holds identical values.
//!
//! # Read path
//!
//! Newest source first: memtable, then each level, and within a level each run
//! from newest to oldest. The first entry found for the key wins, **including a
//! tombstone**, which terminates the search and reports a miss. Falling through
//! a tombstone to an older run would resurrect deleted keys.
//!
//! # Which files exist: derived from names, not a manifest
//!
//! Runs are named `L{level}-{sequence}.sst`, and recovery rebuilds the tree by
//! listing the directory. Sequence numbers ascend globally, so within a level a
//! higher sequence is newer. Tables are written to a `.tmp` name and renamed
//! into place, so a half-written file can never be mistaken for a run; stray
//! `.tmp` files are swept at open.
//!
//! **Labelled simplification:** RocksDB uses a MANIFEST — a log of version edits
//! — which makes a multi-file change (a compaction adding four files and
//! deleting five) atomic as a group. Directory listing cannot express that, so a
//! crash mid-compaction here can leave both inputs and outputs on disk. The
//! compaction work in Phase 2 has to handle that explicitly, and the trade-off
//! belongs in the writeup.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use super::merge::{memtable_source, sstable_source, MergeIterator, Source};
use super::sstable::DEFAULT_BLOOM_FALSE_POSITIVE_RATE;
use super::{Key, MemTable, SSTable, SSTableWriter, SyncPolicy, UserValue, Value, Wal};

/// Name of the active write-ahead log inside the database directory.
const WAL_FILENAME: &str = "current.wal";

/// Tuning knobs. Every field here is something the Phase 3 benchmarks sweep.
#[derive(Debug, Clone)]
pub struct LsmConfig {
    /// Memtable size that triggers a flush. Larger buffers mean fewer, bigger
    /// runs: less write amplification, more memory, and more data at risk of
    /// replay after a crash.
    pub memtable_threshold_bytes: usize,
    /// Whether each write is `fsync`ed. [`SyncPolicy::Manual`] is much faster
    /// and much less durable; any throughput number must state which was used.
    pub sync_policy: SyncPolicy,
    pub block_target_bytes: usize,
    /// Target false-positive rate for per-run bloom filters; 0 disables them.
    pub bloom_false_positive_rate: f64,
}

impl Default for LsmConfig {
    fn default() -> Self {
        Self {
            memtable_threshold_bytes: 4 * 1024 * 1024,
            sync_policy: SyncPolicy::EveryWrite,
            block_target_bytes: super::sstable::DEFAULT_BLOCK_TARGET_BYTES,
            bloom_false_positive_rate: DEFAULT_BLOOM_FALSE_POSITIVE_RATE,
        }
    }
}

/// Counters for the benchmark suite.
///
/// Write amplification is `bytes_written_to_disk / user_bytes_written`: with no
/// compaction it sits near 1, and it is exactly the quantity the compaction
/// policies trade against read cost.
#[derive(Debug, Clone, Default)]
pub struct LsmStats {
    pub memtable_bytes: usize,
    pub memtable_entries: usize,
    pub sstable_count: usize,
    pub disk_bytes: u64,
    /// Runs per level, level 0 first.
    pub tables_per_level: Vec<usize>,
    pub flush_count: u64,
    /// Key + value bytes handed to `put`/`delete` by the caller.
    pub user_bytes_written: u64,
    /// Bytes of SSTable actually written, including index, filter and footer.
    pub sstable_bytes_written: u64,
    pub blocks_read: u64,
    pub bloom_rejections: u64,
}

impl LsmStats {
    /// Bytes written to disk per byte of user data. Undefined before any write.
    pub fn write_amplification(&self) -> Option<f64> {
        (self.user_bytes_written > 0)
            .then(|| self.sstable_bytes_written as f64 / self.user_bytes_written as f64)
    }
}

/// A persistent log-structured merge-tree over byte keys and values.
#[derive(Debug)]
pub struct LsmTree {
    dir: PathBuf,
    config: LsmConfig,
    memtable: MemTable,
    wal: Wal,
    /// `levels[0]` is the newest level. Within a level, runs are ordered newest
    /// first, which is the order the read path must consult them in.
    levels: Vec<Vec<SSTable>>,
    next_sequence: u64,
    flush_count: u64,
    user_bytes_written: u64,
    sstable_bytes_written: u64,
}

impl LsmTree {
    /// Open (or create) a database in `dir`, recovering any existing state.
    pub fn open(dir: impl AsRef<Path>, config: LsmConfig) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;

        // Sweep half-written tables from a previous crash. They were never
        // renamed into place, so nothing references them.
        let mut discovered: BTreeMap<usize, Vec<(u64, PathBuf)>> = BTreeMap::new();
        let mut highest_sequence = 0u64;
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.ends_with(".tmp") {
                std::fs::remove_file(&path)?;
                continue;
            }
            if let Some((level, sequence)) = parse_table_filename(name) {
                highest_sequence = highest_sequence.max(sequence);
                discovered.entry(level).or_default().push((sequence, path));
            }
        }

        // Newest first within each level, and no gaps between levels.
        let mut levels: Vec<Vec<SSTable>> = Vec::new();
        if let Some(&deepest) = discovered.keys().max() {
            levels.resize_with(deepest + 1, Vec::new);
        }
        for (level, mut files) in discovered {
            files.sort_by_key(|(sequence, _)| std::cmp::Reverse(*sequence));
            for (_, path) in files {
                levels[level].push(SSTable::open(&path)?);
            }
        }

        // Replay the log into a fresh memtable. Records are applied in append
        // order, so a later write to a key overwrites an earlier one exactly as
        // it did before the crash.
        let (wal, replay) = Wal::recover(dir.join(WAL_FILENAME), config.sync_policy)?;
        let mut memtable = MemTable::new();
        for (key, value) in replay.records {
            match value {
                Value::Put(bytes) => memtable.put(key, bytes),
                Value::Tombstone => memtable.delete(key),
            };
        }

        Ok(Self {
            dir,
            config,
            memtable,
            wal,
            levels,
            next_sequence: highest_sequence + 1,
            flush_count: 0,
            user_bytes_written: 0,
            sstable_bytes_written: 0,
        })
    }

    /// Insert or overwrite `key`.
    pub fn put(&mut self, key: Key, value: UserValue) -> io::Result<()> {
        self.user_bytes_written += (key.len() + value.len()) as u64;
        self.write(key, Value::Put(value))
    }

    /// Delete `key`, writing a tombstone.
    pub fn delete(&mut self, key: Key) -> io::Result<()> {
        self.user_bytes_written += key.len() as u64;
        self.write(key, Value::Tombstone)
    }

    fn write(&mut self, key: Key, value: Value) -> io::Result<()> {
        // Log first. If this returns and the process dies, the write is
        // recoverable; if it fails, the memtable was never touched and the
        // caller's error is truthful.
        self.wal.append(&key, &value)?;

        match value {
            Value::Put(bytes) => self.memtable.put(key, bytes),
            Value::Tombstone => self.memtable.delete(key),
        };

        if self
            .memtable
            .should_flush(self.config.memtable_threshold_bytes)
        {
            self.flush()?;
        }
        Ok(())
    }

    /// Look up `key`, newest source first.
    ///
    /// Returns `None` both for a key that was never written and for one that was
    /// deleted; the tombstone stops the search either way.
    pub fn get(&self, key: &[u8]) -> io::Result<Option<UserValue>> {
        if let Some(value) = self.memtable.get(key) {
            return Ok(value.as_bytes().map(<[u8]>::to_vec));
        }

        for level in &self.levels {
            for table in level {
                match table.get(key)? {
                    Some(Value::Put(bytes)) => return Ok(Some(bytes)),
                    // A tombstone is an answer: stop, and do not look deeper.
                    Some(Value::Tombstone) => return Ok(None),
                    None => continue,
                }
            }
        }
        Ok(None)
    }

    /// Whether `key` currently has a live value.
    pub fn contains_key(&self, key: &[u8]) -> io::Result<bool> {
        Ok(self.get(key)?.is_some())
    }

    /// Every live key in ascending order, with deleted keys removed.
    ///
    /// Builds a merge over the memtable and every run, newest first — the same
    /// ordering the point-lookup path uses, which is why both agree.
    pub fn iter(&self) -> impl Iterator<Item = io::Result<(Key, UserValue)>> + '_ {
        let mut sources: Vec<Source<'_>> = vec![memtable_source(&self.memtable)];
        for level in &self.levels {
            for table in level {
                sources.push(sstable_source(table));
            }
        }

        // Tombstones are dropped here because this view spans *every* run: there
        // is nothing older left for them to shadow. That reasoning does not
        // transfer to a partial compaction.
        MergeIterator::dropping_tombstones(sources).map(|entry| {
            entry.map(|(key, value)| match value {
                Value::Put(bytes) => (key, bytes),
                Value::Tombstone => unreachable!("tombstones were dropped by the merge"),
            })
        })
    }

    /// Write the memtable out as a new run in level 0.
    ///
    /// Returns `None` when there is nothing buffered. Ordering is deliberate:
    /// the table is durable before the log protecting it is discarded.
    pub fn flush(&mut self) -> io::Result<Option<PathBuf>> {
        if self.memtable.is_empty() {
            return Ok(None);
        }

        let sequence = self.next_sequence;
        self.next_sequence += 1;
        let final_path = self.dir.join(table_filename(0, sequence));
        let temp_path = final_path.with_extension("sst.tmp");

        let mut writer =
            SSTableWriter::create_with_block_size(&temp_path, self.config.block_target_bytes)?
                .with_bloom_false_positive_rate(self.config.bloom_false_positive_rate);

        // The memtable is already sorted and already free of duplicate keys,
        // which is exactly the SSTable writer's input contract.
        let flushed = std::mem::take(&mut self.memtable);
        for (key, value) in flushed.into_entries() {
            writer.append(&key, &value)?;
        }
        let meta = writer.finish()?;

        // The rename is what publishes the run. Until it happens the file is a
        // `.tmp` that recovery will delete.
        std::fs::rename(&temp_path, &final_path)?;

        if self.levels.is_empty() {
            self.levels.push(Vec::new());
        }
        // Newest first.
        self.levels[0].insert(0, SSTable::open(&final_path)?);

        // Only now is the log redundant.
        self.reset_wal()?;

        self.flush_count += 1;
        self.sstable_bytes_written += meta.file_size_bytes;
        Ok(Some(final_path))
    }

    /// Replace the write-ahead log with an empty one.
    ///
    /// Truncates rather than deleting so the file handle stays valid and no
    /// window exists in which the database has no log at all.
    fn reset_wal(&mut self) -> io::Result<()> {
        let path = self.dir.join(WAL_FILENAME);
        std::fs::remove_file(&path).or_else(|error| match error.kind() {
            io::ErrorKind::NotFound => Ok(()),
            _ => Err(error),
        })?;
        self.wal = Wal::open(&path, self.config.sync_policy)?;
        Ok(())
    }

    /// Force buffered log bytes to the device. A no-op under
    /// [`SyncPolicy::EveryWrite`], which has already synced.
    pub fn sync(&mut self) -> io::Result<()> {
        self.wal.sync()
    }

    pub fn stats(&self) -> LsmStats {
        let mut stats = LsmStats {
            memtable_bytes: self.memtable.approx_size_bytes(),
            memtable_entries: self.memtable.entry_count(),
            flush_count: self.flush_count,
            user_bytes_written: self.user_bytes_written,
            sstable_bytes_written: self.sstable_bytes_written,
            ..Default::default()
        };
        for level in &self.levels {
            stats.tables_per_level.push(level.len());
            for table in level {
                stats.sstable_count += 1;
                stats.disk_bytes += table.file_size_bytes();
                stats.blocks_read += table.blocks_read();
                stats.bloom_rejections += table.bloom_rejections();
            }
        }
        stats
    }

    /// Zero the per-run I/O counters so a benchmark can measure one phase alone.
    pub fn reset_io_counters(&self) {
        for level in &self.levels {
            for table in level {
                table.reset_counters();
            }
        }
    }

    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn config(&self) -> &LsmConfig {
        &self.config
    }
}

fn table_filename(level: usize, sequence: u64) -> String {
    format!("L{level:02}-{sequence:010}.sst")
}

/// Parse `L{level}-{sequence}.sst`, returning `None` for anything else so
/// unrelated files in the directory are ignored rather than misread.
fn parse_table_filename(name: &str) -> Option<(usize, u64)> {
    let stem = name.strip_suffix(".sst")?;
    let rest = stem.strip_prefix('L')?;
    let (level, sequence) = rest.split_once('-')?;
    Some((level.parse().ok()?, sequence.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(label: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static COUNTER: AtomicU32 = AtomicU32::new(0);

            let unique = format!(
                "vectordb-lsm-{label}-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            );
            let path = std::env::temp_dir().join(unique);
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn k(s: &str) -> Key {
        s.as_bytes().to_vec()
    }

    fn v(s: &str) -> UserValue {
        s.as_bytes().to_vec()
    }

    /// A small memtable so tests flush often without writing much data.
    fn test_config() -> LsmConfig {
        LsmConfig {
            memtable_threshold_bytes: 1024,
            sync_policy: SyncPolicy::Manual,
            block_target_bytes: 256,
            ..Default::default()
        }
    }

    #[test]
    fn put_and_get_round_trip_in_memory() {
        let dir = TempDir::new("basic");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");

        tree.put(k("alpha"), v("one")).expect("put");
        assert_eq!(tree.get(b"alpha").expect("get"), Some(v("one")));
        assert_eq!(tree.get(b"missing").expect("get"), None);
    }

    #[test]
    fn overwrites_and_deletes_take_effect() {
        let dir = TempDir::new("overwrite");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");

        tree.put(k("key"), v("first")).expect("put");
        tree.put(k("key"), v("second")).expect("put");
        assert_eq!(tree.get(b"key").expect("get"), Some(v("second")));

        tree.delete(k("key")).expect("delete");
        assert_eq!(tree.get(b"key").expect("get"), None);

        tree.put(k("key"), v("third")).expect("put");
        assert_eq!(tree.get(b"key").expect("get"), Some(v("third")));
    }

    #[test]
    fn a_delete_survives_a_flush_and_still_shadows_older_runs() {
        let dir = TempDir::new("delete-across-runs");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");

        tree.put(k("doomed"), v("value")).expect("put");
        tree.flush().expect("flush");

        tree.delete(k("doomed")).expect("delete");
        tree.flush().expect("flush");

        assert_eq!(tree.stats().sstable_count, 2, "expected two runs on disk");
        assert_eq!(
            tree.get(b"doomed").expect("get"),
            None,
            "the tombstone in the newer run must shadow the value in the older one"
        );
    }

    /// The read path must consult runs newest-first. Written as its own test
    /// because getting this backwards returns stale data on every key that was
    /// ever overwritten.
    #[test]
    fn newer_runs_shadow_older_ones() {
        let dir = TempDir::new("shadowing");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");

        for version in 0..5 {
            tree.put(k("key"), v(&format!("version{version}")))
                .expect("put");
            tree.flush().expect("flush");
        }

        assert_eq!(tree.stats().sstable_count, 5);
        assert_eq!(tree.get(b"key").expect("get"), Some(v("version4")));
    }

    #[test]
    fn flushing_an_empty_memtable_is_a_no_op() {
        let dir = TempDir::new("empty-flush");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");

        assert!(tree.flush().expect("flush").is_none());
        assert_eq!(tree.stats().sstable_count, 0);
    }

    #[test]
    fn crossing_the_threshold_flushes_automatically() {
        let dir = TempDir::new("auto-flush");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");

        for i in 0..200 {
            tree.put(format!("key{i:04}").into_bytes(), vec![b'x'; 64])
                .expect("put");
        }

        let stats = tree.stats();
        assert!(
            stats.flush_count > 0,
            "200 x ~70 bytes should have crossed the 1 KiB threshold"
        );
        assert!(stats.memtable_bytes <= test_config().memtable_threshold_bytes);

        // Everything is still readable, wherever it ended up.
        for i in 0..200 {
            assert_eq!(
                tree.get(format!("key{i:04}").as_bytes()).expect("get"),
                Some(vec![b'x'; 64]),
                "key{i:04} went missing across a flush"
            );
        }
    }

    /// The core durability test: kill the process without flushing, reopen, and
    /// every acknowledged write must still be there.
    #[test]
    fn unflushed_writes_survive_a_reopen_via_the_log() {
        let dir = TempDir::new("recovery");

        {
            let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");
            tree.put(k("alpha"), v("1")).expect("put");
            tree.put(k("bravo"), v("2")).expect("put");
            tree.delete(k("bravo")).expect("delete");
            tree.put(k("charlie"), v("3")).expect("put");
            tree.sync().expect("sync");
            // Dropped without flushing: nothing reached an SSTable.
        }

        let tree = LsmTree::open(&dir.path, test_config()).expect("reopen");
        assert_eq!(tree.stats().sstable_count, 0, "nothing should have flushed");
        assert_eq!(tree.get(b"alpha").expect("get"), Some(v("1")));
        assert_eq!(
            tree.get(b"bravo").expect("get"),
            None,
            "the delete must be replayed too, not just the puts"
        );
        assert_eq!(tree.get(b"charlie").expect("get"), Some(v("3")));
    }

    #[test]
    fn flushed_runs_are_rediscovered_on_reopen() {
        let dir = TempDir::new("rediscover");

        // Recorded before closing rather than hardcoded: the write loop may
        // cross the flush threshold on its own, and how many runs that produces
        // is not what this test is about.
        let runs_before_close;
        {
            let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");
            for i in 0..50 {
                tree.put(format!("key{i:03}").into_bytes(), v("value"))
                    .expect("put");
            }
            tree.flush().expect("flush");
            // A few more that stay in the log.
            tree.put(k("late"), v("in the log")).expect("put");
            tree.sync().expect("sync");
            runs_before_close = tree.stats().sstable_count;
        }
        assert!(runs_before_close >= 1, "the explicit flush must have written a run");

        let tree = LsmTree::open(&dir.path, test_config()).expect("reopen");
        assert_eq!(
            tree.stats().sstable_count,
            runs_before_close,
            "every run on disk must be rediscovered, and none invented"
        );
        for i in 0..50 {
            assert_eq!(
                tree.get(format!("key{i:03}").as_bytes()).expect("get"),
                Some(v("value"))
            );
        }
        assert_eq!(tree.get(b"late").expect("get"), Some(v("in the log")));
    }

    /// Recovery must survive many open/close cycles, mixing flushed and logged
    /// state, without losing or duplicating anything.
    #[test]
    fn state_is_stable_across_repeated_reopens() {
        let dir = TempDir::new("repeated");
        let expected_rounds = 8;

        for round in 0..expected_rounds {
            let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");
            for i in 0..20 {
                tree.put(format!("r{round}-k{i:02}").into_bytes(), v("value"))
                    .expect("put");
            }
            if round % 2 == 0 {
                tree.flush().expect("flush");
            } else {
                tree.sync().expect("sync");
            }
        }

        let tree = LsmTree::open(&dir.path, test_config()).expect("final open");
        for round in 0..expected_rounds {
            for i in 0..20 {
                let key = format!("r{round}-k{i:02}");
                assert_eq!(
                    tree.get(key.as_bytes()).expect("get"),
                    Some(v("value")),
                    "{key} was lost across reopens"
                );
            }
        }

        let live: Vec<_> = tree.iter().map(|entry| entry.expect("iter")).collect();
        assert_eq!(live.len(), expected_rounds * 20, "iteration lost or duplicated keys");
    }

    #[test]
    fn a_half_written_table_is_swept_at_open() {
        let dir = TempDir::new("sweep-tmp");

        {
            let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");
            tree.put(k("real"), v("data")).expect("put");
            tree.flush().expect("flush");
        }

        // Stand in for a crash during a flush: a `.tmp` file that was never
        // renamed into place.
        let orphan = dir.path.join("L00-0000000099.sst.tmp");
        std::fs::write(&orphan, b"garbage that is not an SSTable").expect("write orphan");

        let tree = LsmTree::open(&dir.path, test_config()).expect("reopen");
        assert!(!orphan.exists(), "the orphaned .tmp file must be removed");
        assert_eq!(tree.get(b"real").expect("get"), Some(v("data")));
    }

    #[test]
    fn iteration_returns_live_keys_in_order_without_tombstones() {
        let dir = TempDir::new("iterate");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");

        tree.put(k("charlie"), v("3")).expect("put");
        tree.put(k("alpha"), v("1")).expect("put");
        tree.flush().expect("flush");

        tree.put(k("bravo"), v("2")).expect("put");
        tree.delete(k("charlie")).expect("delete");
        tree.put(k("alpha"), v("overwritten")).expect("put");

        let live: Vec<_> = tree.iter().map(|entry| entry.expect("iter")).collect();
        assert_eq!(
            live,
            vec![
                (k("alpha"), v("overwritten")),
                (k("bravo"), v("2")),
                // charlie was deleted and must not appear.
            ]
        );
    }

    /// `iter` and `get` are two different code paths over the same data; they
    /// must never disagree.
    #[test]
    fn iteration_agrees_with_point_lookups() {
        let dir = TempDir::new("agreement");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");

        // Deterministic pseudo-random churn: overwrites, deletes, resurrections,
        // spread across several runs.
        let mut state = 12345u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };

        for round in 0..12 {
            for _ in 0..60 {
                let key = format!("key{:03}", next() % 150).into_bytes();
                match next() % 4 {
                    0 => tree.delete(key).expect("delete"),
                    _ => tree.put(key, format!("round{round}").into_bytes()).expect("put"),
                }
            }
            if round % 3 == 0 {
                tree.flush().expect("flush");
            }
        }

        let live: Vec<_> = tree.iter().map(|entry| entry.expect("iter")).collect();
        for (key, value) in &live {
            assert_eq!(
                tree.get(key).expect("get").as_ref(),
                Some(value),
                "iter and get disagree on {:?}",
                String::from_utf8_lossy(key)
            );
        }

        // And nothing absent from the iteration is reachable by lookup.
        for i in 0..150 {
            let key = format!("key{i:03}").into_bytes();
            let in_iteration = live.iter().any(|(k, _)| k == &key);
            let found = tree.get(&key).expect("get").is_some();
            assert_eq!(
                in_iteration,
                found,
                "disagreement on {:?}: iter={in_iteration} get={found}",
                String::from_utf8_lossy(&key)
            );
        }
    }

    #[test]
    fn stats_track_flushes_and_write_amplification() {
        let dir = TempDir::new("stats");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");

        assert!(tree.stats().write_amplification().is_none());

        for i in 0..300 {
            tree.put(format!("key{i:04}").into_bytes(), vec![b'x'; 100])
                .expect("put");
        }
        tree.flush().expect("flush");

        let stats = tree.stats();
        assert!(stats.flush_count >= 1);
        assert!(stats.user_bytes_written > 0);
        assert!(stats.sstable_bytes_written > 0);

        let amplification = stats.write_amplification().expect("amplification");
        // With no compaction each byte is written once, plus index, bloom filter
        // and per-entry framing overhead. Well above 3x would mean something is
        // writing data repeatedly.
        assert!(
            (1.0..3.0).contains(&amplification),
            "write amplification of {amplification:.2} is implausible with no compaction"
        );
    }

    #[test]
    fn sequence_numbers_keep_ascending_across_reopens() {
        let dir = TempDir::new("sequences");

        for _ in 0..3 {
            let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");
            tree.put(k("key"), v("value")).expect("put");
            tree.flush().expect("flush");
        }

        let mut names: Vec<String> = std::fs::read_dir(&dir.path)
            .expect("read dir")
            .filter_map(|entry| {
                let name = entry.ok()?.file_name().to_str()?.to_string();
                parse_table_filename(&name).map(|_| name)
            })
            .collect();
        names.sort();

        assert_eq!(names.len(), 3, "each flush must produce a distinct file");
        let sequences: Vec<u64> = names
            .iter()
            .map(|name| parse_table_filename(name).expect("parse").1)
            .collect();
        assert!(
            sequences.windows(2).all(|pair| pair[0] < pair[1]),
            "sequence numbers must not be reused after a reopen: {sequences:?}"
        );
    }

    #[test]
    fn filenames_round_trip_and_reject_foreign_names() {
        assert_eq!(parse_table_filename(&table_filename(0, 1)), Some((0, 1)));
        assert_eq!(parse_table_filename(&table_filename(7, 999_999)), Some((7, 999_999)));

        assert_eq!(parse_table_filename("current.wal"), None);
        assert_eq!(parse_table_filename("L00-0000000001.sst.tmp"), None);
        assert_eq!(parse_table_filename("notes.txt"), None);
        assert_eq!(parse_table_filename("Lxx-0000000001.sst"), None);
    }

    #[test]
    fn empty_keys_and_values_are_handled() {
        let dir = TempDir::new("edges");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");

        tree.put(Vec::new(), v("empty key")).expect("put");
        tree.put(k("empty value"), Vec::new()).expect("put");
        tree.flush().expect("flush");

        assert_eq!(tree.get(b"").expect("get"), Some(v("empty key")));
        assert_eq!(
            tree.get(b"empty value").expect("get"),
            Some(Vec::new()),
            "an empty value is a live value, not a miss"
        );
    }

    /// Vector-sized values are the point of this project, so the storage layer
    /// is exercised at that size rather than only on tiny test strings.
    #[test]
    fn handles_vector_sized_values() {
        let dir = TempDir::new("vectors");
        let mut tree = LsmTree::open(
            &dir.path,
            LsmConfig {
                memtable_threshold_bytes: 64 * 1024,
                ..test_config()
            },
        )
        .expect("open");

        // 128 dimensions of f32, as in SIFT.
        let vector: Vec<u8> = (0..128 * 4).map(|i| (i % 251) as u8).collect();
        for i in 0..200 {
            tree.put(format!("vec{i:05}").into_bytes(), vector.clone())
                .expect("put");
        }
        tree.flush().expect("flush");

        for i in 0..200 {
            assert_eq!(
                tree.get(format!("vec{i:05}").as_bytes()).expect("get"),
                Some(vector.clone())
            );
        }
    }
}
