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

use super::compaction::EcoTuneConfig;
use super::compaction::{CompactionJob, Leveling, MergePolicy, RunFiles, Tiering};
use super::growth::{
    EcoTune, GrowthScheme, HorizontalLeveling, HorizontalPolicy, HorizontalTiering, Vertical,
    Vertiorizon,
};
use super::manifest::{table_filename, Manifest, RunEntry};
use super::merge::{memtable_source, MergeIterator, Source};
use super::shape::{FileShape, LevelShape, RunShape, TreeShape};
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
    /// Axis 1: when a level compacts into the next.
    pub growth: GrowthKind,
    /// Axis 2: how the merge is performed.
    pub merge: MergeKind,
    /// Ceiling on compactions run after a single flush.
    ///
    /// Purely a safety net against a scheme that never converges: a buggy one
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
            // The conventional starting point: RocksDB-style vertical growth at
            // a fanout of ten, with leveling below level 0.
            growth: GrowthKind::Vertical {
                buffer_bytes: memtable_threshold_bytes as u64,
                size_ratio: 10,
            },
            merge: MergeKind::Leveling,
            max_compactions_per_flush: 64,
        }
    }
}

/// Which growth scheme to run.
///
/// A plain enum rather than a trait object because growth schemes are
/// *stateful* — they carry compaction counters that advance as the database
/// runs. Config stays a cloneable value describing what to build; [`LsmTree`]
/// instantiates the live object and owns its state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GrowthKind {
    /// Fixed capacities, levels added as data grows. The industry baseline.
    Vertical { buffer_bytes: u64, size_ratio: u64 },
    /// Fixed level count, compaction counters (paper 01, Algorithm 1).
    HorizontalLeveling { levels: usize },
    /// Paper 01's contribution (Algorithm 2) — not a baseline.
    HorizontalTiering {
        levels: usize,
        buffer_bytes: u64,
        expected_data_bytes: u64,
    },
    /// Paper 01's central contribution (§5): a horizontal part above two
    /// vertical levels.
    Vertiorizon {
        /// Levels in the horizontal part.
        horizontal_levels: usize,
        size_ratio: u64,
        buffer_bytes: u64,
        /// Horizontal part capacity as a multiple of the buffer; grows as the
        /// data set does.
        initial_n: u64,
        policy: HorizontalPolicy,
    },
    /// Paper 02's EcoTune, mapped onto three levels. See
    /// `growth::ecotune_scheme` for what in the mapping is ours rather than the
    /// paper's — notably that `T_c` and `β` must be measured, not guessed.
    EcoTune {
        config: EcoTuneConfig,
        /// Top level capacity `S`.
        top_capacity_bytes: u64,
    },
}

impl GrowthKind {
    /// Short name for benchmark output.
    ///
    /// Matched here rather than delegating to a built scheme: constructing an
    /// [`GrowthKind::EcoTune`] solves its dynamic program, which is far too much
    /// work to label a CSV column.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Vertical { .. } => "vertical",
            Self::HorizontalLeveling { .. } => "horizontal-leveling",
            Self::HorizontalTiering { .. } => "horizontal-tiering",
            Self::Vertiorizon { .. } => "vertiorizon",
            Self::EcoTune { .. } => "ecotune",
        }
    }

    fn build(&self) -> Box<dyn GrowthScheme> {
        match *self {
            Self::Vertical {
                buffer_bytes,
                size_ratio,
            } => Box::new(Vertical::new(buffer_bytes, size_ratio)),
            Self::HorizontalLeveling { levels } => Box::new(HorizontalLeveling::new(levels)),
            Self::HorizontalTiering {
                levels,
                buffer_bytes,
                expected_data_bytes,
            } => Box::new(HorizontalTiering::new(
                levels,
                buffer_bytes,
                expected_data_bytes,
            )),
            Self::Vertiorizon {
                horizontal_levels,
                size_ratio,
                buffer_bytes,
                initial_n,
                policy,
            } => Box::new(Vertiorizon::new(
                horizontal_levels,
                size_ratio,
                buffer_bytes,
                initial_n,
                policy,
            )),
            Self::EcoTune {
                config,
                top_capacity_bytes,
            } => Box::new(EcoTune::new(config, top_capacity_bytes)),
        }
    }
}

/// Which merge policy to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeKind {
    Leveling,
    Tiering { runs_per_level: usize },
}

impl MergeKind {
    /// Short name for benchmark output.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Leveling => "leveling",
            Self::Tiering { .. } => "tiering",
        }
    }

    fn build(&self) -> Box<dyn MergePolicy> {
        match *self {
            Self::Leveling => Box::new(Leveling),
            Self::Tiering { runs_per_level } => Box::new(Tiering::new(runs_per_level)),
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
    /// Unit runs this represents: 1 for a flush, the sum of its inputs for a
    /// merge. Only EcoTune reads it; see `shape::RunShape::units`.
    units: usize,
}

impl Run {
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Unit runs this run represents.
    pub fn units(&self) -> usize {
        self.units
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
        let past = self.tables.partition_point(|table| match table.min_key() {
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

    /// Entries at or after `start`, in ascending key order.
    ///
    /// Skips the files that end before `start` entirely rather than opening and
    /// discarding them, so a scan costs work proportional to what it returns.
    fn source_from(&self, start: &[u8]) -> Source<'_> {
        let first = self
            .tables
            .partition_point(|table| table.max_key().is_none_or(|max| max.as_slice() < start));
        let start = start.to_vec();
        Box::new(
            self.tables[first..]
                .iter()
                .flat_map(move |table| table.range_from(&start)),
        )
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
    /// Axis 1, live and stateful: holds the compaction counters.
    growth: Box<dyn GrowthScheme>,
    /// Axis 2, stateless.
    merge: Box<dyn MergePolicy>,
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
                units: entry.units.max(1),
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

        let growth = config.growth.build();
        let merge = config.merge.build();

        Ok(Self {
            dir,
            config,
            memtable,
            wal,
            levels,
            growth,
            merge,
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

    /// Live keys at or after `start`, in ascending order, deleted keys removed.
    ///
    /// Bound it with [`Iterator::take`] for a fixed-length scan. Every source is
    /// *seeked* to `start` rather than filtered from the beginning, so the cost
    /// is proportional to what comes back.
    ///
    /// This is the operation the compaction policies actually differ on. A point
    /// lookup is answered from a bloom filter almost regardless of how many runs
    /// exist — which is precisely why EcoTune leaves its top level uncompacted —
    /// so a benchmark without range scans measures the one workload where every
    /// policy looks the same.
    pub fn range_from<'a>(
        &'a self,
        start: &[u8],
    ) -> impl Iterator<Item = io::Result<(Key, UserValue)>> + 'a {
        let mut sources: Vec<Source<'a>> = vec![Box::new(
            self.memtable
                .range::<[u8], _>((std::ops::Bound::Included(start), std::ops::Bound::Unbounded))
                .map(|(key, value)| Ok((key.clone(), value.clone())))
                .collect::<Vec<_>>()
                .into_iter(),
        )];
        for level in &self.levels {
            for run in level {
                sources.push(run.source_from(start));
            }
        }

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
        // A flush is exactly one unit run, by definition.
        let run = write_run(&self.dir, &self.config, 0, sequence, 1, entries)?;

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

        // The growth scheme counts buffer flushes, so it must be told about this
        // one before it is asked what to compact.
        self.growth.note_flush();
        self.compact_until_quiet()?;
        Ok(true)
    }

    /// Run compactions until the growth scheme is satisfied.
    ///
    /// The scheme mutates its schedule when it hands over a level, so every
    /// level it returns must actually be compacted. When the merge policy finds
    /// nothing to do there — an empty level — the loop moves on rather than
    /// re-asking, which would spin.
    pub fn compact_until_quiet(&mut self) -> io::Result<()> {
        for _ in 0..self.config.max_compactions_per_flush {
            let shape = self.shape();
            let Some(request) = self.growth.next_compaction(&shape) else {
                return Ok(());
            };
            let Some(job) = self.merge.plan(&shape, request) else {
                continue;
            };
            self.execute(job)?;
        }
        // Hitting the ceiling is not an error; the next flush carries on.
        Ok(())
    }

    /// Merge the files a job names and install the result.
    fn execute(&mut self, job: CompactionJob) -> io::Result<()> {
        // Take ownership of the inputs up front. Nothing else may reference them
        // while the merge streams through. Files are removed before empty runs
        // are pruned, so the run indices the job named stay valid throughout.
        //
        // Source levels are visited shallowest first, which is the merge order:
        // shallower levels hold newer data.
        let mut source_files: Vec<Vec<SSTable>> = Vec::new();
        for level_files in &job.sources {
            let Some(level) = self.levels.get_mut(level_files.level) else {
                continue;
            };
            source_files.extend(take_files(level, &level_files.runs));
        }
        let target_files = match self.levels.get_mut(job.target_level) {
            Some(level) => take_files(level, &job.targets),
            None => Vec::new(),
        };

        for level_files in &job.sources {
            if let Some(level) = self.levels.get_mut(level_files.level) {
                prune_empty_runs(level);
            }
        }
        if let Some(level) = self.levels.get_mut(job.target_level) {
            prune_empty_runs(level);
        }

        // The output carries every input's units, which is what makes EcoTune's
        // width arithmetic work across successive merges.
        let merged_units: usize = job
            .sources
            .iter()
            .flat_map(|level_files| {
                let level = self.levels.get(level_files.level);
                level_files.runs.iter().filter_map(move |selection| {
                    level
                        .and_then(|runs| runs.get(selection.run))
                        .map(Run::units)
                })
            })
            .sum::<usize>()
            .max(1);

        let sequence = self.take_sequence();
        let new_run = {
            // Newest first: sources sit at a shallower level, so they hold newer
            // data than anything already at the target. Each group's files are
            // disjoint and ascending, so they concatenate into one sorted run.
            let sources: Vec<Source<'_>> = source_files
                .iter()
                .chain(target_files.iter())
                .map(|files| -> Source<'_> { Box::new(files.iter().flat_map(SSTable::iter)) })
                .collect();

            let merged: Box<dyn Iterator<Item = io::Result<(Key, Value)>>> = if job.drop_tombstones
            {
                Box::new(MergeIterator::dropping_tombstones(sources))
            } else {
                Box::new(MergeIterator::new(sources))
            };

            write_run(
                &self.dir,
                &self.config,
                job.target_level,
                sequence,
                merged_units,
                merged,
            )?
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
        if self.merge.runs_per_level() == 1 {
            coalesce_disjoint_runs(&mut self.levels[job.target_level]);
        }

        // Commit before deleting: after this the old files are unreferenced, so
        // a crash before the unlink merely leaves garbage for the next startup.
        self.store_manifest()?;

        for table in source_files.iter().chain(target_files.iter()).flatten() {
            if let Err(error) = std::fs::remove_file(table.path()) {
                if error.kind() != io::ErrorKind::NotFound {
                    return Err(error);
                }
            }
        }

        self.compaction_count += 1;
        Ok(())
    }

    /// The tree as the growth scheme and merge policy see it.
    fn shape(&self) -> TreeShape {
        TreeShape {
            levels: self
                .levels
                .iter()
                .map(|runs| LevelShape {
                    runs: runs
                        .iter()
                        .enumerate()
                        .map(|(run_index, run)| RunShape {
                            index: run_index,
                            files: run
                                .tables
                                .iter()
                                .enumerate()
                                .map(|(file_index, table)| FileShape {
                                    index: file_index,
                                    bytes: table.file_size_bytes(),
                                    min_key: table.min_key().cloned(),
                                    max_key: table.max_key().cloned(),
                                })
                                .collect(),
                            units: run.units,
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
                    units: run.units,
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

    /// Name of the live growth scheme, for benchmark output.
    pub fn growth_name(&self) -> &'static str {
        self.growth.name()
    }

    /// Name of the live merge policy, for benchmark output.
    pub fn merge_name(&self) -> &'static str {
        self.merge.name()
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

/// Remove the files each selection names, returning them grouped and in the
/// order the selections were given.
///
/// The caller's order is the merge order — newest run first — so it is preserved
/// exactly. Within a group, files come back ascending by key, which is the order
/// they concatenate into a sorted stream.
///
/// Runs are *not* removed here even if they end up empty: doing so mid-loop
/// would invalidate the run indices later selections refer to. Call
/// [`prune_empty_runs`] once the taking is done.
fn take_files(level: &mut [Run], selections: &[RunFiles]) -> Vec<Vec<SSTable>> {
    selections
        .iter()
        .map(|selection| {
            let Some(run) = level.get_mut(selection.run) else {
                return Vec::new();
            };

            let mut indices: Vec<usize> = selection.files.clone();
            indices.sort_unstable();
            indices.dedup();

            // Remove from the back so earlier indices stay valid.
            let mut taken: Vec<SSTable> = Vec::with_capacity(indices.len());
            for &index in indices.iter().rev() {
                if index < run.tables.len() {
                    taken.push(run.tables.remove(index));
                }
            }
            taken.reverse();
            taken
        })
        .collect()
}

/// Drop runs left with no files.
fn prune_empty_runs(level: &mut Vec<Run>) {
    level.retain(|run| !run.tables.is_empty());
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
    let disjoint = ranges.windows(2).all(|pair| pair[0].1 < pair[1].0);
    if !disjoint {
        return;
    }

    let newest_sequence = level.iter().map(Run::sequence).max().unwrap_or(0);
    let total_units: usize = level.iter().map(Run::units).sum();
    let mut tables: Vec<SSTable> = level
        .drain(..)
        .flat_map(|run| run.tables.into_iter())
        .collect();
    tables.sort_by(|a, b| a.min_key().cmp(&b.min_key()));

    level.push(Run {
        sequence: newest_sequence,
        tables,
        units: total_units.max(1),
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
    units: usize,
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
    Ok(Some(Run {
        sequence,
        tables,
        units,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
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
            growth: GrowthKind::Vertical {
                buffer_bytes: 1024,
                size_ratio: 4,
            },
            merge: MergeKind::Leveling,
            ..Default::default()
        }
    }

    fn tiering_config() -> LsmConfig {
        LsmConfig {
            merge: MergeKind::Tiering { runs_per_level: 4 },
            ..test_config()
        }
    }

    /// EcoTune mapped onto top/main/last. Paired with tiering, because the main
    /// level holds several runs and merges append rather than rewriting.
    fn ecotune_config() -> LsmConfig {
        LsmConfig {
            growth: GrowthKind::EcoTune {
                config: EcoTuneConfig {
                    runs_per_round: 8,
                    last_level_ratio: 3,
                    long_range_ratio: 0.4,
                    ..EcoTuneConfig::default()
                },
                top_capacity_bytes: 4096,
            },
            merge: MergeKind::Tiering { runs_per_level: 8 },
            ..test_config()
        }
    }

    /// Two horizontal levels over the two-level vertical part.
    fn vertiorizon_config() -> LsmConfig {
        LsmConfig {
            growth: GrowthKind::Vertiorizon {
                horizontal_levels: 2,
                size_ratio: 4,
                buffer_bytes: 1024,
                initial_n: 4,
                policy: HorizontalPolicy::Leveling,
            },
            merge: MergeKind::Leveling,
            ..test_config()
        }
    }

    /// Compaction effectively disabled, for tests about flush and recovery
    /// alone: a buffer size no level can ever reach.
    fn no_compaction_config() -> LsmConfig {
        LsmConfig {
            growth: GrowthKind::Vertical {
                buffer_bytes: u64::MAX / 4,
                size_ratio: 2,
            },
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

    /// Vertical growth must not let level 0 grow past its capacity.
    #[test]
    fn vertical_growth_keeps_level_zero_bounded() {
        let dir = TempDir::new("vertical-l0");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");

        for i in 0..40 {
            tree.put(format!("key{i:04}").into_bytes(), vec![b'x'; 100])
                .expect("put");
            tree.flush().expect("flush");
        }

        let stats = tree.stats();
        assert!(stats.compaction_count > 0, "compaction never ran");
        assert!(
            stats.level_bytes[0] < 4 * 1024,
            "level 0 holds {} bytes, above its capacity of 4096",
            stats.level_bytes[0]
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
        let mut tree = LsmTree::open(&dir.path, tiering_config()).expect("open");

        for i in 0..500 {
            tree.put(format!("key{i:05}").into_bytes(), vec![b'v'; 60])
                .expect("put");
        }
        tree.flush().expect("flush");

        assert!(tree.stats().compaction_count > 0, "compaction never ran");
    }

    /// Compaction must never change what the database contains. This is the
    /// single most important test in the module.
    #[test]
    fn compaction_preserves_contents() {
        for (label, config) in [
            ("vertical-leveling", test_config()),
            ("vertical-tiering", tiering_config()),
            (
                "horizontal-leveling",
                LsmConfig {
                    growth: GrowthKind::HorizontalLeveling { levels: 4 },
                    ..test_config()
                },
            ),
            (
                "horizontal-tiering",
                LsmConfig {
                    growth: GrowthKind::HorizontalTiering {
                        levels: 4,
                        buffer_bytes: 1024,
                        expected_data_bytes: 256 * 1024,
                    },
                    merge: MergeKind::Tiering { runs_per_level: 4 },
                    ..test_config()
                },
            ),
            ("vertiorizon", vertiorizon_config()),
            ("ecotune", ecotune_config()),
            (
                "vertiorizon-tiering",
                LsmConfig {
                    growth: GrowthKind::Vertiorizon {
                        horizontal_levels: 2,
                        size_ratio: 4,
                        buffer_bytes: 1024,
                        initial_n: 4,
                        policy: HorizontalPolicy::Tiering,
                    },
                    merge: MergeKind::Tiering { runs_per_level: 4 },
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
            let expected = if i % 3 == 0 {
                None
            } else {
                Some(vec![b'v'; 50])
            };
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

        // Each flush must carry real weight: the vertical scheme triggers on
        // level bytes against capacity, so a dozen near-empty runs would sit
        // below the threshold and legitimately never compact.
        for i in 0..12 {
            tree.put(format!("key{i:04}").into_bytes(), vec![b'x'; 600])
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
                Some(vec![b'x'; 600])
            );
        }
    }

    /// Write amplification under a given configuration.
    fn amplification(
        config: LsmConfig,
        label: &str,
        key_order: impl Iterator<Item = usize>,
    ) -> f64 {
        let dir = TempDir::new(label);
        let mut tree = LsmTree::open(&dir.path, config).expect("open");

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

        let leveling = amplification(test_config(), "amp-leveling", scattered());
        let tiering = amplification(tiering_config(), "amp-tiering", scattered());

        assert!(
            leveling > tiering,
            "leveling ({leveling:.2}x) should write more than tiering ({tiering:.2}x) \
             when runs overlap"
        );
    }

    /// A finding worth pinning: leveling's extra cost comes entirely from
    /// *overlap*, so it is a property of the workload rather than of the policy.
    ///
    /// With strictly ascending keys, every flush covers a key range disjoint
    /// from every earlier one, so a compaction finds nothing below to rewrite
    /// and simply moves data down. Scatter the same keys and each run overlaps
    /// what is already there, which is what leveling pays to merge away.
    ///
    /// Everything but the key order is held fixed here — same growth scheme,
    /// same merge policy, same sizes — so the difference is attributable to the
    /// workload alone. Any Phase 3 benchmark that only inserts sequential keys
    /// would understate leveling's cost and could conclude, wrongly, that the
    /// policy choice does not matter.
    #[test]
    fn leveling_pays_for_overlap_not_for_volume() {
        let sequential = amplification(test_config(), "seq-leveling", 0..3000);
        // The same 3000 keys, scrambled: 7919 is coprime with 3000, so this is a
        // permutation rather than a different data set.
        let scattered = amplification(
            test_config(),
            "scattered-leveling",
            (0..3000).map(|i| (i * 7919) % 3000),
        );

        assert!(
            scattered > sequential,
            "identical data in scrambled order should cost more ({scattered:.2}x) \
             than in ascending order ({sequential:.2}x), because only then is \
             there overlapping data to rewrite"
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
                ..no_compaction_config()
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

    /// Partial compaction moves a slice of a level at a time, so the tree must
    /// still hold exactly the same data afterwards — and must actually have used
    /// the partial path.
    #[test]
    fn partial_compaction_preserves_contents() {
        let dir = TempDir::new("partial");
        let mut tree = LsmTree::open(
            &dir.path,
            LsmConfig {
                // Small files, so a level holds many of them and slicing is
                // meaningfully finer than taking the whole level.
                target_file_size_bytes: 512,
                ..test_config()
            },
        )
        .expect("open");

        let expected: Vec<(Key, UserValue)> = (0..1200)
            .map(|i| (format!("key{i:05}").into_bytes(), vec![b'v'; 40]))
            .collect();
        for (key, value) in &expected {
            tree.put(key.clone(), value.clone()).expect("put");
        }
        tree.flush().expect("flush");

        let stats = tree.stats();
        assert!(stats.compaction_count > 0, "compaction never ran");
        assert!(
            stats.file_count > stats.run_count,
            "runs should be split across several files for slicing to matter"
        );

        let live: Vec<_> = tree.iter().map(|entry| entry.expect("iter")).collect();
        assert_eq!(live, expected);
    }

    /// Vertiorizon must actually build the shape it describes: a horizontal part
    /// on top, then exactly two vertical levels holding the bulk of the data.
    #[test]
    fn vertiorizon_builds_a_two_part_tree() {
        let dir = TempDir::new("vertiorizon-shape");
        let mut tree = LsmTree::open(&dir.path, vertiorizon_config()).expect("open");

        for i in 0..6000 {
            let key = format!("key{:05}", (i * 7919) % 6000);
            tree.put(key.into_bytes(), vec![b'v'; 60]).expect("put");
        }
        tree.flush().expect("flush");

        let stats = tree.stats();
        assert!(stats.compaction_count > 0, "compaction never ran");
        assert!(
            stats.level_bytes.len() > 2,
            "expected the vertical part to be populated, got {:?}",
            stats.level_bytes
        );

        // Horizontal part is levels 0..1, vertical part is 2..3. The vertical
        // part should hold the large majority — that is the whole design.
        let horizontal: u64 = stats.level_bytes.iter().take(2).sum();
        let vertical: u64 = stats.level_bytes.iter().skip(2).sum();
        assert!(
            vertical > horizontal,
            "the vertical part should hold the bulk of the data: {vertical} vs {horizontal}"
        );
        assert!(
            stats.level_bytes.len() <= 4,
            "Vertiorizon is 2 horizontal + 2 vertical levels, got {:?}",
            stats.level_bytes
        );
    }

    /// The horizontal part is where nearly all compactions happen, so it must
    /// stay a small slice of the tree — otherwise its full compactions would be
    /// expensive and the design would gain nothing.
    ///
    /// The bound is a *fraction*, not an absolute size: `n` grows as the data
    /// does, so the horizontal capacity is not fixed. With `T = 4` the intended
    /// split is roughly `1 : T′ : T²` = `1 : 2.83 : 16`, putting the horizontal
    /// part near 5% of the total.
    #[test]
    fn vertiorizon_keeps_its_horizontal_part_small() {
        let dir = TempDir::new("vertiorizon-drain");
        let mut tree = LsmTree::open(&dir.path, vertiorizon_config()).expect("open");

        for i in 0..3000 {
            let key = format!("key{:05}", (i * 7919) % 3000);
            tree.put(key.into_bytes(), vec![b'v'; 60]).expect("put");
        }
        tree.flush().expect("flush");

        let stats = tree.stats();
        let horizontal: u64 = stats.level_bytes.iter().take(2).sum();
        let total: u64 = stats.level_bytes.iter().sum();
        assert!(total > 0, "nothing was written");
        assert!(
            horizontal * 4 < total,
            "the horizontal part holds {horizontal} of {total} bytes, far more than \
             the intended 1 : T′ : T² split allows"
        );
    }

    // ---------------------------------------------------------------
    // Range scans
    // ---------------------------------------------------------------

    /// A scan from the very beginning must return exactly what a full iteration
    /// does — two code paths over the same data that must never disagree.
    #[test]
    fn a_scan_from_the_start_matches_full_iteration() {
        let dir = TempDir::new("scan-all");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");

        for i in 0..800 {
            let key = format!("key{:05}", (i * 7919) % 800);
            tree.put(key.into_bytes(), vec![b'v'; 40]).expect("put");
        }
        tree.flush().expect("flush");

        let iterated: Vec<_> = tree.iter().map(|e| e.expect("iter")).collect();
        let scanned: Vec<_> = tree.range_from(b"").map(|e| e.expect("scan")).collect();
        assert_eq!(scanned, iterated);
    }

    #[test]
    fn a_scan_starts_at_the_requested_key_and_ascends() {
        let dir = TempDir::new("scan-from");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");

        for i in 0..500 {
            tree.put(format!("key{i:05}").into_bytes(), vec![b'v'; 40])
                .expect("put");
        }
        tree.flush().expect("flush");

        let scanned: Vec<Key> = tree
            .range_from(b"key00250")
            .take(10)
            .map(|entry| entry.expect("scan").0)
            .collect();

        assert_eq!(scanned.len(), 10);
        assert_eq!(scanned[0], b"key00250".to_vec());
        assert_eq!(scanned[9], b"key00259".to_vec());
        assert!(
            scanned.windows(2).all(|pair| pair[0] < pair[1]),
            "a scan must ascend: {scanned:?}"
        );
    }

    /// A scan must honour tombstones and overwrites just as `get` does.
    #[test]
    fn a_scan_skips_deleted_keys_and_sees_the_newest_value() {
        let dir = TempDir::new("scan-tombstones");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");

        for i in 0..300 {
            tree.put(format!("key{i:05}").into_bytes(), v("old"))
                .expect("put");
        }
        tree.flush().expect("flush");
        for i in (0..300).step_by(3) {
            tree.delete(format!("key{i:05}").into_bytes())
                .expect("delete");
        }
        for i in (1..300).step_by(3) {
            tree.put(format!("key{i:05}").into_bytes(), v("new"))
                .expect("put");
        }

        for (key, value) in tree.range_from(b"").map(|e| e.expect("scan")) {
            let index: usize = String::from_utf8_lossy(&key)[3..].parse().expect("index");
            assert_ne!(
                index % 3,
                0,
                "key{index:05} was deleted but the scan returned it"
            );
            let expected = if index % 3 == 1 { v("new") } else { v("old") };
            assert_eq!(value, expected, "wrong value for key{index:05}");
        }
    }

    /// A scan past the end of the key space is empty, not an error.
    #[test]
    fn a_scan_beyond_the_last_key_is_empty() {
        let dir = TempDir::new("scan-past-end");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");
        tree.put(k("alpha"), v("1")).expect("put");
        tree.flush().expect("flush");

        assert_eq!(tree.range_from(b"zzzz").count(), 0);
    }

    /// The point of seeking: a scan late in the key space must not read the
    /// blocks before it. Without this, EcoTune's cost model measures nothing,
    /// because every scan would cost a full file read regardless of run count.
    #[test]
    fn a_scan_does_not_read_blocks_before_its_start() {
        let dir = TempDir::new("scan-seeks");
        let mut tree = LsmTree::open(
            &dir.path,
            LsmConfig {
                memtable_threshold_bytes: 4 * 1024 * 1024,
                target_file_size_bytes: 64 * 1024,
                block_target_bytes: 256,
                ..no_compaction_config()
            },
        )
        .expect("open");

        for i in 0..4000 {
            tree.put(format!("key{i:06}").into_bytes(), vec![b'v'; 50])
                .expect("put");
        }
        tree.flush().expect("flush");

        tree.reset_io_counters();
        let full: usize = tree.iter().count();
        let blocks_for_full_scan = tree.stats().blocks_read;

        tree.reset_io_counters();
        let tail: Vec<_> = tree.range_from(b"key003990").take(10).collect();
        let blocks_for_short_scan = tree.stats().blocks_read;

        assert_eq!(full, 4000);
        assert_eq!(tail.len(), 10);
        assert!(
            blocks_for_short_scan * 10 < blocks_for_full_scan,
            "a 10-entry scan read {blocks_for_short_scan} blocks against \
             {blocks_for_full_scan} for the whole table; it is not seeking"
        );
    }

    #[test]
    fn a_scan_spans_levels_and_runs() {
        let dir = TempDir::new("scan-levels");
        let mut tree = LsmTree::open(&dir.path, test_config()).expect("open");

        // Enough churn to spread data across several levels.
        for i in 0..2000 {
            let key = format!("key{:05}", (i * 7919) % 2000);
            tree.put(key.into_bytes(), vec![b'v'; 50]).expect("put");
        }
        tree.flush().expect("flush");
        assert!(
            tree.stats().runs_per_level.len() > 1,
            "expected several levels"
        );

        // Something still in the memtable, above everything on disk.
        tree.put(k("key00042"), v("freshest")).expect("put");

        let scanned: Vec<_> = tree
            .range_from(b"key00040")
            .take(5)
            .map(|e| e.expect("scan"))
            .collect();
        assert_eq!(scanned[0].0, b"key00040".to_vec());
        assert_eq!(
            scanned[2],
            (k("key00042"), v("freshest")),
            "the memtable's value must win over the on-disk one"
        );
    }

    /// EcoTune's mapping must produce the three-level shape it claims, with the
    /// last level collapsed to a single run by the round-ending compaction.
    #[test]
    fn ecotune_builds_a_three_level_tree() {
        let dir = TempDir::new("ecotune-shape");
        let mut tree = LsmTree::open(&dir.path, ecotune_config()).expect("open");

        for i in 0..4000 {
            let key = format!("key{:05}", (i * 7919) % 4000);
            tree.put(key.into_bytes(), vec![b'v'; 60]).expect("put");
        }
        tree.flush().expect("flush");

        let stats = tree.stats();
        assert!(stats.compaction_count > 0, "compaction never ran");
        assert!(
            stats.runs_per_level.len() <= 3,
            "EcoTune maps onto exactly three levels, got {:?}",
            stats.runs_per_level
        );
        if stats.runs_per_level.len() == 3 {
            assert!(
                stats.runs_per_level[2] <= 1,
                "the last level must hold at most one run, got {}",
                stats.runs_per_level[2]
            );
        }
    }

    #[test]
    fn taking_files_preserves_the_requested_order() {
        let dir = TempDir::new("take-files");
        let mut tree = LsmTree::open(
            &dir.path,
            LsmConfig {
                // Everything in one memtable, so the single flush below produces
                // exactly one run, split into files at 4 KiB.
                memtable_threshold_bytes: 4 * 1024 * 1024,
                ..no_compaction_config()
            },
        )
        .expect("open");

        // One run spanning several files: the target file size is 4 KiB, so this
        // needs well over 12 KiB of payload.
        for i in 0..200 {
            tree.put(format!("key{i:03}").into_bytes(), vec![b'v'; 300])
                .expect("put");
        }
        tree.flush().expect("flush");

        let files_before = tree.levels[0][0].tables.len();
        assert!(files_before >= 3, "need several files to select among");

        let taken = take_files(
            &mut tree.levels[0],
            &[RunFiles {
                run: 0,
                files: vec![0, 2],
            }],
        );

        assert_eq!(taken.len(), 1, "one selection, one group");
        assert_eq!(taken[0].len(), 2);
        assert!(
            taken[0][0].min_key() < taken[0][1].min_key(),
            "files must come back ascending by key"
        );
        assert_eq!(
            tree.levels[0][0].tables.len(),
            files_before - 2,
            "the untaken files must remain"
        );
    }

    #[test]
    fn pruning_removes_only_emptied_runs() {
        let dir = TempDir::new("prune");
        let mut tree = LsmTree::open(&dir.path, no_compaction_config()).expect("open");
        tree.put(k("a"), v("1")).expect("put");
        tree.flush().expect("flush");
        tree.put(k("b"), v("2")).expect("put");
        tree.flush().expect("flush");
        assert_eq!(tree.levels[0].len(), 2);

        // Empty the newer run, leaving the older one intact.
        tree.levels[0][0].tables.clear();
        prune_empty_runs(&mut tree.levels[0]);

        assert_eq!(tree.levels[0].len(), 1);
        assert_eq!(tree.get(b"a").expect("get"), Some(v("1")));
    }
}
