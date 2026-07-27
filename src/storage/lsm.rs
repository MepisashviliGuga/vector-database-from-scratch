//! The LSM-tree: memtable, write-ahead log, levels of runs, and compaction.
//!
//! # Write path
//!
//! ```text
//! put/delete ──► WAL (append, optionally fsync) ──► memtable
//!                                                       │ threshold reached
//!                                                       ▼
//!                                    run written to level 0, manifest committed,
//!                                    WAL reset, then compaction runs to quiescence
//! ```
//!
//! The log is appended *before* the memtable is touched, so an acknowledged
//! write is recoverable. On flush the run is written and `fsync`ed, the manifest
//! is committed, and only then is the log discarded. A crash in that window
//! replays records that are also already in the new run — harmless, because the
//! replayed memtable is newer in the read order and holds identical values.
//!
//! # Read path
//!
//! Newest source first: memtable, then each level in turn, and within a level
//! each run from newest to oldest. The first entry found wins, **including a
//! tombstone**, which terminates the search and reports a miss.
//!
//! # Runs
//!
//! A [`Run`] is a sorted sequence with no duplicate keys, split across one or
//! more SSTables with disjoint key ranges. Searching a run means binary-searching
//! its files by range and probing exactly one — so a run costs one lookup no
//! matter how many files it holds. Read cost tracks the number of *runs*; file
//! count only bounds how much a compaction has to rewrite.
//!
//! # Commit point
//!
//! Which runs are live is decided by the manifest, not by what is on disk. New
//! SSTables are written under their final names and are simply invisible until
//! the manifest is atomically replaced; unreferenced files are swept at startup.
//! See [`super::manifest`] for why a directory listing is not sufficient.
//!
//! **Labelled simplification:** compaction operates at whole-run granularity —
//! a job merges entire runs, where LevelDB and RocksDB pick a bounded slice of
//! the key range. Write *amplification* is the same either way (both rewrite
//! roughly `T+1` bytes per byte moved down a level), but a single compaction
//! here is larger, so tail write latency is worse than a production engine's.
//! That distinction matters for the p99 numbers in Phase 3 and belongs in the
//! writeup.

use std::collections::HashSet;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::compaction::{
    CompactionJob, CompactionPolicy, LevelShape, Leveling, RunShape, TreeShape,
};
use super::growth::{GrowthScheme, Vertical};
use super::manifest::{table_filename, Manifest, RunEntry};
use super::merge::{memtable_source, MergeIterator, Source};
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
    /// Size at which a run is split into another file. Bounds how much a single
    /// SSTable holds without affecting how many runs exist.
    pub target_file_size_bytes: u64,
    /// Axis 1: how the tree is shaped as it grows.
    pub growth: Arc<dyn GrowthScheme>,
    /// Axis 2: when and what to merge.
    pub compaction: Arc<dyn CompactionPolicy>,
    /// Ceiling on compactions run after a single flush.
    ///
    /// Purely a safety net against a policy that never converges: a buggy policy
    /// would otherwise spin forever inside a `put`. Reaching this limit is not
    /// an error — the next flush simply continues the work.
    pub max_compactions_per_flush: usize,
}

impl Default for LsmConfig {
    fn default() -> Self {
        let memtable_threshold_bytes = 4 * 1024 * 1024;
        Self {
            memtable_threshold_bytes,
            sync_policy: SyncPolicy::EveryWrite,
            block_target_bytes: super::sstable::DEFAULT_BLOCK_TARGET_BYTES,
            bloom_false_positive_rate: DEFAULT_BLOOM_FALSE_POSITIVE_RATE,
            target_file_size_bytes: 8 * 1024 * 1024,
            // Level 0 holds a few flushes' worth, and each level below is ten
            // times the last — the conventional starting point.
            growth: Arc::new(Vertical::new(memtable_threshold_bytes as u64 * 4, 10)),
            compaction: Arc::new(Leveling::default()),
            max_compactions_per_flush: 64,
        }
    }
}

/// Counters for the benchmark suite.
#[derive(Debug, Clone, Default)]
pub struct LsmStats {
    pub memtable_bytes: usize,
    pub memtable_entries: usize,
    /// Runs across all levels. This is what read cost scales with.
    pub run_count: usize,
    /// SSTable files across all levels.
    pub file_count: usize,
    pub disk_bytes: u64,
    /// Runs per level, level 0 first.
    pub runs_per_level: Vec<usize>,
    pub level_bytes: Vec<u64>,
    pub flush_count: u64,
    pub compaction_count: u64,
    /// Key + value bytes handed to `put`/`delete` by the caller.
    pub user_bytes_written: u64,
    /// Bytes of SSTable written, by flushes and compactions together.
    pub sstable_bytes_written: u64,
    /// Bytes written by compaction alone — the cost the policies trade against
    /// read performance.
    pub compaction_bytes_written: u64,
    pub blocks_read: u64,
    pub bloom_rejections: u64,
}

impl LsmStats {
    /// Bytes written to disk per byte of user data.
    ///
    /// This is the headline write-amplification figure. With no compaction it
    /// sits just above 1; leveling drives it up in exchange for fewer runs to
    /// probe on a read.
    pub fn write_amplification(&self) -> Option<f64> {
        (self.user_bytes_written > 0)
            .then(|| self.sstable_bytes_written as f64 / self.user_bytes_written as f64)
    }
}

/// A sorted run: SSTables with disjoint key ranges, ascending, no duplicate
/// keys between them.
#[derive(Debug)]
pub struct Run {
    /// Creation order. Within a level, higher is newer.
    sequence: u64,
    /// Ascending by key range, non-overlapping.
    tables: Vec<SSTable>,
}

impl Run {
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn file_count(&self) -> usize {
        self.tables.len()
    }

    pub fn bytes(&self) -> u64 {
        self.tables.iter().map(SSTable::file_size_bytes).sum()
    }

    pub fn entry_count(&self) -> u64 {
        self.tables.iter().map(SSTable::entry_count).sum()
    }

    pub fn min_key(&self) -> Option<&Key> {
        self.tables.first().and_then(SSTable::min_key)
    }

    pub fn max_key(&self) -> Option<&Key> {
        self.tables.last().and_then(SSTable::max_key)
    }

    /// Look up `key` in this run.
    ///
    /// The files partition the key space, so at most one can contain the key:
    /// binary-search for the last file whose range starts at or before it, then
    /// probe that file alone.
    pub fn get(&self, key: &[u8]) -> io::Result<Option<Value>> {
        let past = self
            .tables
            .partition_point(|table| match table.min_key() {
                Some(min) => min.as_slice() <= key,
                None => true,
            });
        match past.checked_sub(1) {
            Some(index) => self.tables[index].get(key),
            // The key sorts before every file in this run.
            None => Ok(None),
        }
    }

    /// Every entry in ascending key order, tombstones included.
    fn source(&self) -> Source<'_> {
        Box::new(self.tables.iter().flat_map(SSTable::iter))
    }

    fn paths(&self) -> Vec<PathBuf> {
        self.tables
            .iter()
            .map(|table| table.path().to_path_buf())
            .collect()
    }

    fn reset_counters(&self) {
        for table in &self.tables {
            table.reset_counters();
        }
    }
}

/// A persistent log-structured merge-tree over byte keys and values.
#[derive(Debug)]
pub struct LsmTree {
    dir: PathBuf,
    config: LsmConfig,
    memtable: MemTable,
    wal: Wal,
    /// `levels[0]` is the newest level. Within a level, runs are newest first —
    /// the order the read path must consult them in.
    levels: Vec<Vec<Run>>,
    next_sequence: u64,
    flush_count: u64,
    compaction_count: u64,
    user_bytes_written: u64,
    sstable_bytes_written: u64,
    compaction_bytes_written: u64,
}

impl LsmTree {
    /// Open (or create) a database in `dir`, recovering any existing state.
    pub fn open(dir: impl AsRef<Path>, config: LsmConfig) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;

        let manifest = Manifest::load(&dir)?.unwrap_or_default();
        Self::sweep_unreferenced_files(&dir, &manifest)?;

        // Rebuild levels. Runs are grouped by level and ordered newest first.
        let deepest = manifest.runs.iter().map(|run| run.level).max();
        let mut levels: Vec<Vec<Run>> = Vec::new();
        if let Some(deepest) = deepest {
            levels.resize_with(deepest + 1, Vec::new);
        }

        let mut entries = manifest.runs.clone();
        entries.sort_by_key(|run| (run.level, std::cmp::Reverse(run.sequence)));
        for entry in entries {
            let mut tables = Vec::with_capacity(entry.files.len());
            for file in &entry.files {
                tables.push(SSTable::open(dir.join(file))?);
            }
            levels[entry.level].push(Run {
                sequence: entry.sequence,
                tables,
            });
        }

        // Replay the log into a fresh memtable, in append order, so a later
        // write to a key overwrites an earlier one exactly as it did before.
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
            next_sequence: manifest.next_sequence.max(1),
            flush_count: 0,
            compaction_count: 0,
            user_bytes_written: 0,
            sstable_bytes_written: 0,
            compaction_bytes_written: 0,
        })
    }

    /// Delete SSTables the manifest does not name.
    ///
    /// These are outputs of a flush or compaction that crashed before its
    /// manifest commit. They were never live, so nothing can reference them.
    fn sweep_unreferenced_files(dir: &Path, manifest: &Manifest) -> io::Result<()> {
        let referenced: HashSet<PathBuf> = manifest.referenced_files().into_iter().collect();

        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            let Some(name) = path.file_name() else {
                continue;
            };
            let is_table = name.to_str().is_some_and(|name| name.ends_with(".sst"));
            let is_scratch = name.to_str().is_some_and(|name| name.ends_with(".tmp"));

            if (is_table && !referenced.contains(Path::new(name))) || is_scratch {
                std::fs::remove_file(&path)?;
            }
        }
        Ok(())
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
            for run in level {
                match run.get(key)? {
                    Some(Value::Put(bytes)) => return Ok(Some(bytes)),
                    // A tombstone is an answer: stop, and do not look deeper.
                    Some(Value::Tombstone) => return Ok(None),
                    None => continue,
                }
            }
        }
        Ok(None)
    }

    pub fn contains_key(&self, key: &[u8]) -> io::Result<bool> {
        Ok(self.get(key)?.is_some())
    }

    /// Every live key in ascending order, with deleted keys removed.
    ///
    /// Sources are ordered exactly as the point-lookup path consults them, which
    /// is why the two always agree.
    pub fn iter(&self) -> impl Iterator<Item = io::Result<(Key, UserValue)>> + '_ {
        let mut sources: Vec<Source<'_>> = vec![memtable_source(&self.memtable)];
        for level in &self.levels {
            for run in level {
                sources.push(run.source());
            }
        }

        // Safe here because this view spans *every* run: there is nothing older
        // left for a tombstone to shadow. The same reasoning does not transfer
        // to a partial compaction.
        MergeIterator::dropping_tombstones(sources).map(|entry| {
            entry.map(|(key, value)| match value {
                Value::Put(bytes) => (key, bytes),
                Value::Tombstone => unreachable!("tombstones were dropped by the merge"),
            })
        })
    }

    /// Write the memtable out as a new run in level 0, then compact.
    ///
    /// Returns whether anything was written.
    pub fn flush(&mut self) -> io::Result<bool> {
        if self.memtable.is_empty() {
            return Ok(false);
        }

        let sequence = self.take_sequence();
        let flushed = std::mem::take(&mut self.memtable);

        // Tombstones are kept: level 0 may sit above older runs holding the keys
        // they delete, and only compaction knows when that is no longer true.
        let entries = flushed.into_entries().map(Ok);
        let run = write_run(&self.dir, &self.config, 0, sequence, entries)?;

        if let Some(run) = run {
            self.sstable_bytes_written += run.bytes();
            if self.levels.is_empty() {
                self.levels.push(Vec::new());
            }
            self.levels[0].insert(0, run);
        }

        // The commit point. Before this the new files are invisible.
        self.store_manifest()?;
        // Only now is the log redundant.
        self.reset_wal()?;
        self.flush_count += 1;

        self.compact_until_quiet()?;
        Ok(true)
    }

    /// Run compactions until the policy is satisfied.
    pub fn compact_until_quiet(&mut self) -> io::Result<()> {
        for _ in 0..self.config.max_compactions_per_flush {
            let shape = self.shape();
            let policy = Arc::clone(&self.config.compaction);
            let Some(job) = policy.pick(&shape, self.config.growth.as_ref()) else {
                return Ok(());
            };
            self.execute(job)?;
        }
        // Hitting the ceiling is not an error; the next flush carries on.
        Ok(())
    }

    /// Merge the runs a job names and install the result.
    fn execute(&mut self, job: CompactionJob) -> io::Result<()> {
        // Take ownership of the inputs up front. Nothing else may reference them
        // while the merge streams through.
        let source_runs = take_runs(&mut self.levels[job.source_level], &job.source_runs);
        let target_runs = match self.levels.get_mut(job.target_level) {
            Some(level) => take_runs(level, &job.target_runs),
            None => Vec::new(),
        };

        let sequence = self.take_sequence();
        let new_run = {
            // Newest first: sources sit at a shallower level, so they hold newer
            // data than anything already at the target.
            let mut sources: Vec<Source<'_>> = Vec::new();
            for run in source_runs.iter().chain(target_runs.iter()) {
                sources.push(run.source());
            }

            let merged: Box<dyn Iterator<Item = io::Result<(Key, Value)>>> = if job.drop_tombstones
            {
                Box::new(MergeIterator::dropping_tombstones(sources))
            } else {
                Box::new(MergeIterator::new(sources))
            };

            write_run(&self.dir, &self.config, job.target_level, sequence, merged)?
        };

        while self.levels.len() <= job.target_level {
            self.levels.push(Vec::new());
        }
        if let Some(run) = new_run {
            let bytes = run.bytes();
            self.sstable_bytes_written += bytes;
            self.compaction_bytes_written += bytes;
            // Newest at the target level: it carries newer data than the runs
            // that arrived there earlier.
            self.levels[job.target_level].insert(0, run);
        }

        // A policy declaring one run per level relies on the executor to keep
        // that true. Compacting a contiguous key range produces an output that
        // is disjoint from the runs already there rather than overlapping them,
        // so they can be folded together without rewriting anything.
        if self.config.compaction.runs_per_level() == 1 {
            coalesce_disjoint_runs(&mut self.levels[job.target_level]);
        }

        // Commit before deleting: after this the old files are unreferenced, so
        // a crash before the unlink merely leaves garbage for the next startup.
        self.store_manifest()?;

        for run in source_runs.iter().chain(target_runs.iter()) {
            for path in run.paths() {
                if let Err(error) = std::fs::remove_file(&path) {
                    if error.kind() != io::ErrorKind::NotFound {
                        return Err(error);
                    }
                }
            }
        }

        self.compaction_count += 1;
        Ok(())
    }

    /// The tree as the compaction policy sees it.
    fn shape(&self) -> TreeShape {
        TreeShape {
            levels: self
                .levels
                .iter()
                .map(|runs| LevelShape {
                    runs: runs
                        .iter()
                        .enumerate()
                        .map(|(index, run)| RunShape {
                            index,
                            bytes: run.bytes(),
                            min_key: run.min_key().cloned(),
                            max_key: run.max_key().cloned(),
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    fn manifest(&self) -> Manifest {
        let mut runs = Vec::new();
        for (level, level_runs) in self.levels.iter().enumerate() {
            for run in level_runs {
                runs.push(RunEntry {
                    level,
                    sequence: run.sequence,
                    files: run
                        .tables
                        .iter()
                        .map(|table| {
                            table
                                .path()
                                .file_name()
                                .expect("an SSTable always has a filename")
                                .to_string_lossy()
                                .into_owned()
                        })
                        .collect(),
                });
            }
        }
        Manifest {
            runs,
            next_sequence: self.next_sequence,
        }
    }

    fn store_manifest(&self) -> io::Result<()> {
        self.manifest().store(&self.dir)
    }

    fn take_sequence(&mut self) -> u64 {
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        sequence
    }

    /// Replace the write-ahead log with an empty one.
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
            compaction_count: self.compaction_count,
            user_bytes_written: self.user_bytes_written,
            sstable_bytes_written: self.sstable_bytes_written,
            compaction_bytes_written: self.compaction_bytes_written,
            ..Default::default()
        };

        for level in &self.levels {
            stats.runs_per_level.push(level.len());
            let mut level_bytes = 0;
            for run in level {
                stats.run_count += 1;
                stats.file_count += run.file_count();
                level_bytes += run.bytes();
                for table in &run.tables {
                    stats.blocks_read += table.blocks_read();
                    stats.bloom_rejections += table.bloom_rejections();
                }
            }
            stats.level_bytes.push(level_bytes);
            stats.disk_bytes += level_bytes;
        }
        stats
    }

    /// Zero the per-table I/O counters so a benchmark can measure one phase.
    pub fn reset_io_counters(&self) {
        for level in &self.levels {
            for run in level {
                run.reset_counters();
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

/// Remove the runs at `indices`, returning them in the order given.
///
/// Removal proceeds from the highest index down so earlier indices stay valid,
/// then the result is reordered to match the caller's list — a policy names runs
/// newest-first and the merge depends on that order.
fn take_runs(level: &mut Vec<Run>, indices: &[usize]) -> Vec<Run> {
    let mut sorted: Vec<usize> = indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    let mut taken: Vec<(usize, Run)> = Vec::with_capacity(sorted.len());
    for &index in sorted.iter().rev() {
        if index < level.len() {
            taken.push((index, level.remove(index)));
        }
    }

    indices
        .iter()
        .filter_map(|wanted| {
            taken
                .iter()
                .position(|(index, _)| index == wanted)
                .map(|position| taken.remove(position).1)
        })
        .collect()
}

/// Fold mutually disjoint runs in a level into a single run.
///
/// A run is just a list of files with disjoint ranges, so merging runs that do
/// not overlap each other is a metadata operation — no data is read or
/// rewritten. This is what keeps leveling's "one run per level" invariant true
/// when compaction of a contiguous key range produces output beside, rather than
/// on top of, what is already there.
///
/// Leaves the level untouched if any two runs overlap, since folding those would
/// put two versions of a key in one run with no way to tell which is newer. That
/// should not happen under leveling; the check is here so a policy bug degrades
/// performance instead of corrupting data.
fn coalesce_disjoint_runs(level: &mut Vec<Run>) {
    if level.len() <= 1 {
        return;
    }

    // Check before touching anything: if the runs overlap, the level must be
    // left exactly as it is, recency order intact.
    let mut ranges: Vec<(&Key, &Key)> = Vec::with_capacity(level.len());
    for run in level.iter() {
        match (run.min_key(), run.max_key()) {
            (Some(min), Some(max)) => ranges.push((min, max)),
            // A run with no key range cannot be reasoned about; leave the level.
            _ => return,
        }
    }
    ranges.sort_by(|a, b| a.0.cmp(b.0));
    let disjoint = ranges
        .windows(2)
        .all(|pair| pair[0].1 < pair[1].0);
    if !disjoint {
        return;
    }

    let newest_sequence = level.iter().map(Run::sequence).max().unwrap_or(0);
    let mut tables: Vec<SSTable> = level
        .drain(..)
        .flat_map(|run| run.tables.into_iter())
        .collect();
    tables.sort_by(|a, b| a.min_key().cmp(&b.min_key()));

    level.push(Run {
        sequence: newest_sequence,
        tables,
    });
}

/// Stream `entries` into a new run at `level`, splitting by target file size.
///
/// Files are written under their final names. They are inert until a manifest
/// commit names them, so no scratch-and-rename dance is needed here — the
/// manifest is the commit point.
///
/// Returns `None` when the merge produced nothing, which happens when a
/// bottom-level compaction drops every entry as a tombstone.
fn write_run(
    dir: &Path,
    config: &LsmConfig,
    level: usize,
    sequence: u64,
    entries: impl Iterator<Item = io::Result<(Key, Value)>>,
) -> io::Result<Option<Run>> {
    let mut tables: Vec<SSTable> = Vec::new();
    let mut writer: Option<(SSTableWriter, PathBuf)> = None;
    let mut bytes_in_part: u64 = 0;

    for entry in entries {
        let (key, value) = entry?;

        if writer.is_none() {
            let path = dir.join(table_filename(level, sequence, tables.len()));
            let new_writer =
                SSTableWriter::create_with_block_size(&path, config.block_target_bytes)?
                    .with_bloom_false_positive_rate(config.bloom_false_positive_rate);
            writer = Some((new_writer, path));
            bytes_in_part = 0;
        }

        let (active, _) = writer.as_mut().expect("just created");
        active.append(&key, &value)?;
        bytes_in_part += (key.len() + value.byte_len()) as u64;

        if bytes_in_part >= config.target_file_size_bytes {
            let (active, path) = writer.take().expect("active writer");
            active.finish()?;
            tables.push(SSTable::open(&path)?);
        }
    }

    if let Some((active, path)) = writer.take() {
        active.finish()?;
        tables.push(SSTable::open(&path)?);
    }

    if tables.is_empty() {
        return Ok(None);
    }
    Ok(Some(Run { sequence, tables }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::compaction::Tiering;
    use crate::storage::growth::Horizontal;
    use crate::storage::manifest::MANIFEST_FILENAME;

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

    /// Small everything, so tests flush and compact often without writing much.
    fn test_config() -> LsmConfig {
        LsmConfig {
            memtable_threshold_bytes: 1024,
            sync_policy: SyncPolicy::Manual,
            block_target_bytes: 256,
            target_file_size_bytes: 4096,
            growth: Arc::new(Vertical::new(4096, 4)),
            compaction: Arc::new(Leveling::new(4)),
            ..Default::default()
        }
    }

    /// Compaction disabled, for tests about flush and recovery alone.
    fn no_compaction_config() -> LsmConfig {
        LsmConfig {
            compaction: Arc::new(Leveling::new(usize::MAX)),
            ..test_config()
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
        let mut tree = LsmTree::open(&dir.path, no_compaction_config()).expect("open");

        tree.put(k("doomed"), v("value")).expect("put");
        tree.flush().expect("flush");
        tree.delete(k("doomed")).expect("delete");
        tree.flush().expect("flush");

        assert_eq!(tree.stats().run_count, 2);
        assert_eq!(
            tree.get(b"doomed").expect("get"),
            None,
            "the tombstone in the newer run must shadow the value in the older one"
        );
    }

    #[test]
    fn newer_runs_shadow_older_ones() {
        let dir = TempDir::new("shadowing");
        let mut tree = LsmTree::open(&dir.path, no_compaction_config()).expect("open");

        for version in 0..5 {
            tree.put(k("key"), v(&format!("version{version}")))
                .expect("put");
            tree.flush().expect("flush");
        }

        assert_eq!(tree.stats().run_count, 5);
        assert_eq!(tree.get(b"key").expect("get"), Some(v("version4")));
    }

    #[test]
    fn flushing_an_empty_memtable_is_a_no_op() {
        let dir = TempDir::new("empty-flush");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");

        assert!(!tree.flush().expect("flush"));
        assert_eq!(tree.stats().run_count, 0);
    }

    // ---------------------------------------------------------------
    // Compaction
    // ---------------------------------------------------------------

    /// Leveling must not let level 0 grow past its run trigger.
    #[test]
    fn leveling_keeps_level_zero_bounded() {
        let dir = TempDir::new("leveling-l0");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");

        for i in 0..40 {
            tree.put(format!("key{i:04}").into_bytes(), vec![b'x'; 100])
                .expect("put");
            tree.flush().expect("flush");
        }

        let stats = tree.stats();
        assert!(stats.compaction_count > 0, "compaction never ran");
        assert!(
            stats.runs_per_level[0] < 4,
            "level 0 holds {} runs, above the trigger of 4",
            stats.runs_per_level[0]
        );
    }

    /// The invariant that defines leveling: one run per level below level 0.
    #[test]
    fn leveling_maintains_one_run_per_level() {
        let dir = TempDir::new("leveling-invariant");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");

        for i in 0..500 {
            tree.put(format!("key{i:05}").into_bytes(), vec![b'v'; 60])
                .expect("put");
        }
        tree.flush().expect("flush");

        let stats = tree.stats();
        for (level, &runs) in stats.runs_per_level.iter().enumerate().skip(1) {
            assert!(
                runs <= 1,
                "level {level} holds {runs} runs; leveling permits at most one"
            );
        }
    }

    /// Tiering's invariant: at most `T` runs per level.
    #[test]
    fn tiering_keeps_runs_per_level_bounded() {
        let dir = TempDir::new("tiering");
        let mut tree = LsmTree::open(
            &dir.path,
            LsmConfig {
                compaction: Arc::new(Tiering::new(4)),
                ..test_config()
            },
        )
        .expect("open");

        for i in 0..500 {
            tree.put(format!("key{i:05}").into_bytes(), vec![b'v'; 60])
                .expect("put");
        }
        tree.flush().expect("flush");

        let stats = tree.stats();
        assert!(stats.compaction_count > 0);
        for (level, &runs) in stats.runs_per_level.iter().enumerate() {
            assert!(
                runs < 4,
                "level {level} holds {runs} runs, at or above the limit of 4"
            );
        }
    }

    /// Compaction must never change what the database contains. This is the
    /// single most important test in the module.
    #[test]
    fn compaction_preserves_contents() {
        for (label, config) in [
            ("leveling", test_config()),
            (
                "tiering",
                LsmConfig {
                    compaction: Arc::new(Tiering::new(4)),
                    ..test_config()
                },
            ),
            (
                "horizontal-leveling",
                LsmConfig {
                    growth: Arc::new(Horizontal::new(3, 4, 4096)),
                    ..test_config()
                },
            ),
        ] {
            let dir = TempDir::new(label);
            let mut tree = LsmTree::open(&dir.path, config).expect("open");

            // Deterministic churn: overwrites, deletes and resurrections.
            let mut expected = std::collections::BTreeMap::new();
            let mut state = 987_654_321u64;
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as usize
            };

            for round in 0..2000 {
                let key = format!("key{:04}", next() % 800).into_bytes();
                if next() % 5 == 0 {
                    tree.delete(key.clone()).expect("delete");
                    expected.remove(&key);
                } else {
                    let value = format!("round{round}").into_bytes();
                    tree.put(key.clone(), value.clone()).expect("put");
                    expected.insert(key, value);
                }
            }
            tree.flush().expect("flush");

            let stats = tree.stats();
            assert!(
                stats.compaction_count > 0,
                "{label}: compaction never ran, so this proves nothing"
            );

            for (key, value) in &expected {
                assert_eq!(
                    tree.get(key).expect("get").as_ref(),
                    Some(value),
                    "{label}: wrong value for {:?} after compaction",
                    String::from_utf8_lossy(key)
                );
            }

            let live: Vec<_> = tree.iter().map(|entry| entry.expect("iter")).collect();
            let expected_live: Vec<_> = expected.into_iter().collect();
            assert_eq!(
                live, expected_live,
                "{label}: iteration disagrees with the expected contents"
            );
        }
    }

    /// A deleted key must stay deleted through a bottom-level compaction that
    /// discards its tombstone.
    #[test]
    fn deletions_survive_tombstone_dropping() {
        let dir = TempDir::new("tombstone-drop");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");

        for i in 0..600 {
            tree.put(format!("key{i:04}").into_bytes(), vec![b'v'; 50])
                .expect("put");
        }
        tree.flush().expect("flush");

        for i in (0..600).step_by(3) {
            tree.delete(format!("key{i:04}").into_bytes())
                .expect("delete");
        }
        tree.flush().expect("flush");
        tree.compact_until_quiet().expect("compact");

        for i in 0..600 {
            let key = format!("key{i:04}");
            let expected = if i % 3 == 0 { None } else { Some(vec![b'v'; 50]) };
            assert_eq!(
                tree.get(key.as_bytes()).expect("get"),
                expected,
                "{key} is wrong after compaction"
            );
        }
    }

    #[test]
    fn compaction_reduces_the_run_count() {
        let dir = TempDir::new("run-count");
        let mut tree = LsmTree::open(&dir.path, no_compaction_config()).expect("open");

        for i in 0..12 {
            tree.put(format!("key{i:04}").into_bytes(), vec![b'x'; 200])
                .expect("put");
            tree.flush().expect("flush");
        }
        let before = tree.stats().run_count;
        assert_eq!(before, 12);

        // Same data, now with a policy that will act on it.
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("reopen");
        tree.compact_until_quiet().expect("compact");
        let after = tree.stats().run_count;

        assert!(
            after < before,
            "compaction should have reduced {before} runs, but left {after}"
        );
        for i in 0..12 {
            assert_eq!(
                tree.get(format!("key{i:04}").as_bytes()).expect("get"),
                Some(vec![b'x'; 200])
            );
        }
    }

    /// Write amplification under each policy, for `key_order` inserted keys.
    fn amplification(
        policy: Arc<dyn CompactionPolicy>,
        label: &str,
        key_order: impl Iterator<Item = usize>,
    ) -> f64 {
        let dir = TempDir::new(label);
        let mut tree = LsmTree::open(
            &dir.path,
            LsmConfig {
                compaction: policy,
                ..test_config()
            },
        )
        .expect("open");

        for i in key_order {
            tree.put(format!("key{i:05}").into_bytes(), vec![b'v'; 80])
                .expect("put");
        }
        tree.flush().expect("flush");
        tree.stats().write_amplification().expect("amplification")
    }

    /// Leveling rewrites data more than tiering. This is the trade-off the whole
    /// second axis exists to expose.
    ///
    /// The keys are **scattered**, so each flush spans the whole key space and
    /// its run overlaps what is already below. That overlap is what leveling
    /// pays to remove — see the sequential case below, where it pays nothing.
    #[test]
    fn leveling_writes_more_than_tiering_when_runs_overlap() {
        // 7919 is coprime with 3000, so this is a permutation: the same 3000
        // keys as a sequential run, in scrambled order.
        let scattered = || (0..3000).map(|i| (i * 7919) % 3000);

        let leveling = amplification(Arc::new(Leveling::new(4)), "amp-leveling", scattered());
        let tiering = amplification(Arc::new(Tiering::new(4)), "amp-tiering", scattered());

        assert!(
            leveling > tiering,
            "leveling ({leveling:.2}x) should write more than tiering ({tiering:.2}x) \
             when runs overlap"
        );
    }

    /// A finding worth pinning: with strictly ascending keys, every flush covers
    /// a key range disjoint from every earlier one, so leveling finds nothing to
    /// rewrite and costs exactly what tiering costs.
    ///
    /// The read-write trade-off between the two policies is therefore a property
    /// of the *workload*, not of the policies alone. Any Phase 3 benchmark that
    /// only inserts sequential keys would show the two as identical and conclude,
    /// wrongly, that the choice does not matter.
    #[test]
    fn leveling_costs_nothing_extra_on_sequential_keys() {
        let sequential = || 0..3000;

        let leveling = amplification(Arc::new(Leveling::new(4)), "seq-leveling", sequential());
        let tiering = amplification(Arc::new(Tiering::new(4)), "seq-tiering", sequential());

        assert!(
            (leveling - tiering).abs() < 0.01,
            "with disjoint runs the two policies should do identical work, but \
             leveling was {leveling:.2}x and tiering {tiering:.2}x"
        );
    }

    // ---------------------------------------------------------------
    // Durability and recovery
    // ---------------------------------------------------------------

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
        }

        let tree = LsmTree::open(&dir.path, test_config()).expect("reopen");
        assert_eq!(tree.stats().run_count, 0, "nothing should have flushed");
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

        let runs_before_close;
        {
            let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");
            for i in 0..50 {
                tree.put(format!("key{i:03}").into_bytes(), v("value"))
                    .expect("put");
            }
            tree.flush().expect("flush");
            tree.put(k("late"), v("in the log")).expect("put");
            tree.sync().expect("sync");
            runs_before_close = tree.stats().run_count;
        }
        assert!(runs_before_close >= 1);

        let tree = LsmTree::open(&dir.path, test_config()).expect("reopen");
        assert_eq!(
            tree.stats().run_count,
            runs_before_close,
            "every run must be rediscovered, and none invented"
        );
        for i in 0..50 {
            assert_eq!(
                tree.get(format!("key{i:03}").as_bytes()).expect("get"),
                Some(v("value"))
            );
        }
        assert_eq!(tree.get(b"late").expect("get"), Some(v("in the log")));
    }

    #[test]
    fn state_is_stable_across_repeated_reopens() {
        let dir = TempDir::new("repeated");
        let rounds = 8;

        for round in 0..rounds {
            let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");
            for i in 0..40 {
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
        for round in 0..rounds {
            for i in 0..40 {
                let key = format!("r{round}-k{i:02}");
                assert_eq!(
                    tree.get(key.as_bytes()).expect("get"),
                    Some(v("value")),
                    "{key} was lost across reopens"
                );
            }
        }
        assert_eq!(tree.iter().count(), rounds * 40);
    }

    /// Files written by a flush or compaction that never committed must be
    /// deleted, not adopted.
    #[test]
    fn uncommitted_files_are_swept_at_open() {
        let dir = TempDir::new("sweep");

        {
            let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");
            tree.put(k("real"), v("data")).expect("put");
            tree.flush().expect("flush");
        }

        // Stand in for a crash between writing a run and committing it.
        let orphan = dir.path.join(table_filename(0, 9999, 0));
        std::fs::write(&orphan, b"not a real SSTable").expect("write orphan");
        let scratch = dir.path.join("something.tmp");
        std::fs::write(&scratch, b"scratch").expect("write scratch");

        let tree = LsmTree::open(&dir.path, test_config()).expect("reopen");
        assert!(!orphan.exists(), "an uncommitted run must be swept");
        assert!(!scratch.exists(), "scratch files must be swept");
        assert_eq!(tree.get(b"real").expect("get"), Some(v("data")));
    }

    /// The manifest is what makes a run live, so a database that has compacted
    /// must still reopen to exactly the same contents.
    #[test]
    fn contents_survive_a_reopen_after_compaction() {
        let dir = TempDir::new("reopen-after-compaction");

        let expected: Vec<(Key, UserValue)> = (0..400)
            .map(|i| (format!("key{i:04}").into_bytes(), vec![b'v'; 70]))
            .collect();

        {
            let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");
            for (key, value) in &expected {
                tree.put(key.clone(), value.clone()).expect("put");
            }
            tree.flush().expect("flush");
            assert!(tree.stats().compaction_count > 0, "compaction never ran");
        }

        let tree = LsmTree::open(&dir.path, test_config()).expect("reopen");
        let live: Vec<_> = tree.iter().map(|entry| entry.expect("iter")).collect();
        assert_eq!(live, expected);
    }

    #[test]
    fn sequence_numbers_keep_ascending_across_reopens() {
        let dir = TempDir::new("sequences");
        let mut highest = 0;

        for _ in 0..4 {
            let mut tree = LsmTree::open(&dir.path, no_compaction_config()).expect("open");
            tree.put(k("key"), v("value")).expect("put");
            tree.flush().expect("flush");

            let lowest_new = tree
                .levels
                .iter()
                .flatten()
                .map(|run| run.sequence())
                .max()
                .expect("a run");
            assert!(
                lowest_new > highest,
                "sequence {lowest_new} was reused after a reopen (previous high {highest})"
            );
            highest = lowest_new;
        }
    }

    #[test]
    fn a_corrupt_manifest_refuses_to_open() {
        let dir = TempDir::new("bad-manifest");

        {
            let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");
            tree.put(k("key"), v("value")).expect("put");
            tree.flush().expect("flush");
        }

        let path = dir.path.join(MANIFEST_FILENAME);
        let contents = std::fs::read_to_string(&path).expect("read");
        std::fs::write(&path, contents.replace("next-sequence", "next-sequenceX")).expect("write");

        assert!(
            LsmTree::open(&dir.path, test_config()).is_err(),
            "a corrupt manifest must be an error, not a silently empty database"
        );
    }

    // ---------------------------------------------------------------
    // Edges
    // ---------------------------------------------------------------

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
        for i in 0..400 {
            tree.put(format!("vec{i:05}").into_bytes(), vector.clone())
                .expect("put");
        }
        tree.flush().expect("flush");

        assert!(tree.stats().compaction_count > 0);
        for i in 0..400 {
            assert_eq!(
                tree.get(format!("vec{i:05}").as_bytes()).expect("get"),
                Some(vector.clone()),
                "vector {i} did not survive compaction intact"
            );
        }
    }

    #[test]
    fn a_run_is_split_into_files_at_the_target_size() {
        let dir = TempDir::new("splitting");
        let mut tree = LsmTree::open(
            &dir.path,
            LsmConfig {
                // High enough that the write loop never auto-flushes, so the
                // single explicit flush below is the only run created.
                memtable_threshold_bytes: 4 * 1024 * 1024,
                target_file_size_bytes: 4096,
                compaction: Arc::new(Leveling::new(usize::MAX)),
                ..test_config()
            },
        )
        .expect("open");

        for i in 0..1000 {
            tree.put(format!("key{i:05}").into_bytes(), vec![b'v'; 100])
                .expect("put");
        }
        tree.flush().expect("flush");

        let stats = tree.stats();
        assert_eq!(stats.run_count, 1, "one flush is one run");
        assert!(
            stats.file_count > 5,
            "~100 KiB of data at a 4 KiB target should span many files, got {}",
            stats.file_count
        );

        // And the run still reads correctly across its file boundaries.
        for i in 0..1000 {
            assert_eq!(
                tree.get(format!("key{i:05}").as_bytes()).expect("get"),
                Some(vec![b'v'; 100]),
                "key {i} was lost at a file boundary"
            );
        }
    }

    #[test]
    fn taking_runs_preserves_the_requested_order() {
        fn dummy(sequence: u64) -> Run {
            Run {
                sequence,
                tables: Vec::new(),
            }
        }

        let mut level = vec![dummy(10), dummy(9), dummy(8), dummy(7)];
        let taken = take_runs(&mut level, &[0, 2]);

        assert_eq!(
            taken.iter().map(Run::sequence).collect::<Vec<_>>(),
            vec![10, 8],
            "runs must come back in the order the policy asked for"
        );
        assert_eq!(
            level.iter().map(Run::sequence).collect::<Vec<_>>(),
            vec![9, 7],
            "the untouched runs must remain, in order"
        );
    }
}
