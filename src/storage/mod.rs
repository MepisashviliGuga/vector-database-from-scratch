//! LSM-tree storage engine.
//!
//! Write path: `put`/`delete` land in the [`memtable`]; when it exceeds a size
//! threshold it is frozen and flushed to an immutable, sorted on-disk run.
//! Read path: newest source first (memtable, then progressively older on-disk
//! runs), stopping at the first entry found for a key — including a tombstone.

pub mod crc32;
pub mod memtable;
pub mod wal;

pub use memtable::{MemTable, Value};
pub use wal::{Replay, SyncPolicy, Wal};

/// Keys are opaque byte strings, ordered lexicographically. The engine never
/// interprets them; higher layers encode vector IDs (and later, secondary index
/// keys) into this space.
pub type Key = Vec<u8>;

/// Values are opaque byte strings. The vector layer serialises `(vector,
/// metadata)` into one of these.
pub type UserValue = Vec<u8>;
