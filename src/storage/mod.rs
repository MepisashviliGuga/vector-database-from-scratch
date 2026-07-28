//! LSM-tree storage engine.
//!
//! Write path: `put`/`delete` land in the [`memtable`]; when it exceeds a size
//! threshold it is frozen and flushed to an immutable, sorted on-disk run.
//! Read path: newest source first (memtable, then progressively older on-disk
//! runs), stopping at the first entry found for a key — including a tombstone.

pub mod bloom;
pub mod compaction;
pub mod crc32;
pub mod growth;
pub mod lsm;
pub mod manifest;
pub mod memtable;
pub mod merge;
pub mod shape;
pub mod sstable;
pub mod wal;

pub use bloom::BloomFilter;
pub use compaction::{CompactionJob, Leveling, MergePolicy, Tiering};
pub use growth::{
    EcoTune, GrowthScheme, HorizontalLeveling, HorizontalPolicy, HorizontalTiering, Vertical,
    Vertiorizon,
};
pub use lsm::{GrowthKind, LsmConfig, LsmStats, LsmTree, MergeKind, Run};
pub use shape::{LevelShape, RunShape, TreeShape};
pub use manifest::Manifest;
pub use memtable::{MemTable, Value};
pub use merge::MergeIterator;
pub use sstable::{SSTable, SSTableMeta, SSTableWriter};
pub use wal::{Replay, SyncPolicy, Wal};

/// Keys are opaque byte strings, ordered lexicographically. The engine never
/// interprets them; higher layers encode vector IDs (and later, secondary index
/// keys) into this space.
pub type Key = Vec<u8>;

/// Values are opaque byte strings. The vector layer serialises `(vector,
/// metadata)` into one of these.
pub type UserValue = Vec<u8>;
