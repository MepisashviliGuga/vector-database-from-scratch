//! In-memory write buffer.
//!
//! The memtable is the only mutable structure in the engine. Every write goes
//! here (after being appended to the WAL, once that exists); everything below it
//! is immutable. It must therefore support:
//!
//! - ordered iteration, so a flush produces a *sorted* run in one pass;
//! - in-place overwrite, so a hot key does not consume unbounded memory;
//! - tombstones, so a delete of a key that only exists in an older on-disk run
//!   can shadow it;
//! - size accounting, so the engine knows when to flush.
//!
//! Backed by a `BTreeMap`. Production engines (RocksDB, LevelDB) use a skip
//! list, because a lock-free skip list allows concurrent readers during writes.
//! `BTreeMap` gives the same ordered semantics with better cache behaviour and
//! no `unsafe`; the concurrency difference does not matter yet because the engine
//! is single-threaded. This is engineering choice, not a paper reproduction.

use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::ops::RangeBounds;

use super::{Key, UserValue};

/// Flat per-entry cost charged to the size accounting, on top of key and value
/// bytes.
///
/// This is a *budgeting* constant for deciding when to flush, not a measurement
/// of actual heap usage: it stands in for the `BTreeMap`'s internal node share
/// plus the length prefixes each entry will occupy once serialised into an
/// SSTable. Being consistently wrong by a constant factor is fine here; being
/// inconsistent between `put` and `delete` is not, because the accounting is
/// maintained incrementally.
const ENTRY_OVERHEAD_BYTES: usize = 24;

/// What the memtable holds for a key.
///
/// The distinction between "no entry for this key" and "an entry saying this key
/// is deleted" is load-bearing for the whole engine. A [`Value::Tombstone`] is a
/// positive statement that the key is gone, and it must stop the read path from
/// descending into older runs where a stale `Put` may still live. Collapsing the
/// two into `Option` would resurrect deleted keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    /// The key is live and holds these bytes.
    Put(UserValue),
    /// The key was deleted. Shadows anything older.
    Tombstone,
}

impl Value {
    /// Bytes of user payload; a tombstone carries none.
    pub fn byte_len(&self) -> usize {
        match self {
            Value::Put(bytes) => bytes.len(),
            Value::Tombstone => 0,
        }
    }

    pub fn is_tombstone(&self) -> bool {
        matches!(self, Value::Tombstone)
    }

    /// The payload if live, `None` if this is a tombstone.
    ///
    /// Use this only where a tombstone and a miss are genuinely interchangeable
    /// (e.g. a user-facing `get` on the top-level engine). Inside the read path,
    /// match on the variant instead.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Value::Put(bytes) => Some(bytes),
            Value::Tombstone => None,
        }
    }
}

/// Ordered in-memory write buffer with tombstone support and incremental size
/// accounting.
#[derive(Debug, Default)]
pub struct MemTable {
    entries: BTreeMap<Key, Value>,
    approx_size_bytes: usize,
}

impl MemTable {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            approx_size_bytes: 0,
        }
    }

    /// Insert or overwrite `key`.
    ///
    /// Returns the entry this replaced, if any — including a tombstone, which
    /// means the key is being resurrected.
    pub fn put(&mut self, key: Key, value: UserValue) -> Option<Value> {
        self.insert(key, Value::Put(value))
    }

    /// Record that `key` is deleted.
    ///
    /// Always writes a tombstone, even if the key is absent from this memtable:
    /// absence here says nothing about the on-disk runs below, so the tombstone
    /// is the only way to shadow them. Returns the entry it replaced, if any.
    pub fn delete(&mut self, key: Key) -> Option<Value> {
        self.insert(key, Value::Tombstone)
    }

    fn insert(&mut self, key: Key, value: Value) -> Option<Value> {
        let key_len = key.len();
        let added = Self::entry_charge(key_len, value.byte_len());

        let previous = self.entries.insert(key, value);

        match &previous {
            Some(old) => {
                // Overwrite: swap the old charge for the new one rather than
                // accumulating both, or a repeatedly-updated key would flush the
                // memtable far too early.
                let removed = Self::entry_charge(key_len, old.byte_len());
                debug_assert!(
                    self.approx_size_bytes >= removed,
                    "size accounting underflow: tracked {} < charge {removed} being removed",
                    self.approx_size_bytes,
                );
                self.approx_size_bytes = self.approx_size_bytes.saturating_sub(removed) + added;
            }
            None => self.approx_size_bytes += added,
        }

        previous
    }

    /// Look up `key` in this memtable only.
    ///
    /// Three outcomes the caller must distinguish:
    /// - `Some(Value::Put(_))` — found, stop searching.
    /// - `Some(Value::Tombstone)` — deleted, stop searching, report a miss.
    /// - `None` — this memtable knows nothing; continue to older runs.
    pub fn get(&self, key: &[u8]) -> Option<&Value> {
        self.entries.get(key)
    }

    /// Entries in ascending key order, tombstones included.
    ///
    /// Tombstones are deliberately visible: a flush must persist them, and a
    /// merging iterator must see them to shadow older runs.
    pub fn iter(&self) -> impl Iterator<Item = (&Key, &Value)> {
        self.entries.iter()
    }

    /// Entries whose keys fall in `range`, in ascending key order.
    ///
    /// The signature mirrors [`BTreeMap::range`] so callers can scan by owned
    /// keys (`k("b")..k("d")`) or, without allocating, by borrowed slices via a
    /// `(Bound, Bound)` pair. The plain `a..b` form cannot be used with `[u8]`
    /// directly: `Range<&T>` only implements `RangeBounds<T>` for sized `T`.
    pub fn range<Q, R>(&self, range: R) -> impl Iterator<Item = (&Key, &Value)>
    where
        Q: Ord + ?Sized,
        Key: Borrow<Q>,
        R: RangeBounds<Q>,
    {
        self.entries.range(range)
    }

    /// Consume the memtable, yielding entries in ascending key order.
    ///
    /// This is the flush path: one sequential pass, already sorted, straight into
    /// an SSTable writer.
    pub fn into_entries(self) -> impl Iterator<Item = (Key, Value)> {
        self.entries.into_iter()
    }

    /// Number of entries, counting tombstones.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Estimated memory footprint, in bytes, used to trigger flushes.
    ///
    /// See [`ENTRY_OVERHEAD_BYTES`]: this is a consistent budget, not a true
    /// heap measurement.
    pub fn approx_size_bytes(&self) -> usize {
        self.approx_size_bytes
    }

    /// Whether the memtable has reached its flush threshold.
    pub fn should_flush(&self, threshold_bytes: usize) -> bool {
        self.approx_size_bytes >= threshold_bytes
    }

    fn entry_charge(key_len: usize, value_len: usize) -> usize {
        ENTRY_OVERHEAD_BYTES + key_len + value_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Bound;

    fn k(s: &str) -> Key {
        s.as_bytes().to_vec()
    }

    fn v(s: &str) -> UserValue {
        s.as_bytes().to_vec()
    }

    #[test]
    fn empty_memtable_has_no_entries() {
        let table = MemTable::new();
        assert!(table.is_empty());
        assert_eq!(table.entry_count(), 0);
        assert_eq!(table.approx_size_bytes(), 0);
        assert_eq!(table.get(b"absent"), None);
    }

    #[test]
    fn put_then_get_returns_the_value() {
        let mut table = MemTable::new();
        assert_eq!(table.put(k("alpha"), v("one")), None);
        assert_eq!(table.get(b"alpha"), Some(&Value::Put(v("one"))));
        assert_eq!(table.entry_count(), 1);
    }

    #[test]
    fn overwrite_returns_newest_value_and_keeps_one_entry() {
        let mut table = MemTable::new();
        table.put(k("alpha"), v("one"));
        let replaced = table.put(k("alpha"), v("two"));

        assert_eq!(replaced, Some(Value::Put(v("one"))));
        assert_eq!(table.get(b"alpha"), Some(&Value::Put(v("two"))));
        assert_eq!(table.entry_count(), 1, "overwrite must not add an entry");
    }

    #[test]
    fn delete_writes_a_tombstone_not_a_removal() {
        let mut table = MemTable::new();
        table.put(k("alpha"), v("one"));
        let replaced = table.delete(k("alpha"));

        assert_eq!(replaced, Some(Value::Put(v("one"))));
        // The crucial assertion: `get` is Some(Tombstone), not None. A None here
        // would let the read path fall through to an older run and resurrect the
        // key.
        assert_eq!(table.get(b"alpha"), Some(&Value::Tombstone));
        assert_eq!(table.entry_count(), 1);
    }

    #[test]
    fn delete_of_unknown_key_still_writes_a_tombstone() {
        let mut table = MemTable::new();
        assert_eq!(table.delete(k("never-seen")), None);
        assert_eq!(table.get(b"never-seen"), Some(&Value::Tombstone));
        assert_eq!(
            table.entry_count(),
            1,
            "the tombstone must exist to shadow older on-disk runs"
        );
    }

    #[test]
    fn put_after_delete_resurrects_the_key() {
        let mut table = MemTable::new();
        table.put(k("alpha"), v("one"));
        table.delete(k("alpha"));
        let replaced = table.put(k("alpha"), v("three"));

        assert_eq!(replaced, Some(Value::Tombstone));
        assert_eq!(table.get(b"alpha"), Some(&Value::Put(v("three"))));
    }

    #[test]
    fn iteration_is_in_sorted_key_order_including_tombstones() {
        let mut table = MemTable::new();
        // Inserted out of order on purpose.
        table.put(k("delta"), v("4"));
        table.put(k("alpha"), v("1"));
        table.put(k("charlie"), v("3"));
        table.delete(k("bravo"));

        let keys: Vec<Key> = table.iter().map(|(key, _)| key.clone()).collect();
        assert_eq!(keys, vec![k("alpha"), k("bravo"), k("charlie"), k("delta")]);

        let values: Vec<&Value> = table.iter().map(|(_, value)| value).collect();
        assert!(
            values[1].is_tombstone(),
            "tombstones must survive iteration"
        );
    }

    #[test]
    fn range_scan_respects_bounds() {
        let mut table = MemTable::new();
        for key in ["a", "b", "c", "d", "e"] {
            table.put(k(key), v(key));
        }

        // Owned-key form.
        let scanned: Vec<Key> = table
            .range(k("b")..k("d"))
            .map(|(key, _)| key.clone())
            .collect();
        assert_eq!(scanned, vec![k("b"), k("c")], "end bound is exclusive");

        let inclusive: Vec<Key> = table
            .range(k("b")..=k("d"))
            .map(|(key, _)| key.clone())
            .collect();
        assert_eq!(inclusive, vec![k("b"), k("c"), k("d")]);

        // Borrowed-slice form, which the read path will use to avoid allocating
        // a key just to bound a scan.
        let borrowed: Vec<Key> = table
            .range::<[u8], _>((Bound::Included(b"c".as_slice()), Bound::Unbounded))
            .map(|(key, _)| key.clone())
            .collect();
        assert_eq!(borrowed, vec![k("c"), k("d"), k("e")]);
    }

    #[test]
    fn size_accounting_charges_key_value_and_overhead() {
        let mut table = MemTable::new();
        table.put(k("alpha"), v("one")); // 5 + 3 + overhead
        assert_eq!(table.approx_size_bytes(), ENTRY_OVERHEAD_BYTES + 5 + 3);

        table.put(k("bb"), v("cccc")); // 2 + 4 + overhead
        assert_eq!(
            table.approx_size_bytes(),
            2 * ENTRY_OVERHEAD_BYTES + 5 + 3 + 2 + 4
        );
    }

    #[test]
    fn size_accounting_swaps_charges_on_overwrite() {
        let mut table = MemTable::new();
        table.put(k("alpha"), v("one"));
        table.put(k("alpha"), v("a much longer value"));

        assert_eq!(
            table.approx_size_bytes(),
            ENTRY_OVERHEAD_BYTES + 5 + "a much longer value".len(),
            "an overwrite must replace the old charge, not add to it"
        );
    }

    #[test]
    fn size_accounting_shrinks_when_a_put_becomes_a_tombstone() {
        let mut table = MemTable::new();
        table.put(k("alpha"), v("a long-ish value"));
        let before = table.approx_size_bytes();

        table.delete(k("alpha"));

        assert!(table.approx_size_bytes() < before);
        assert_eq!(
            table.approx_size_bytes(),
            ENTRY_OVERHEAD_BYTES + 5,
            "a tombstone costs key bytes plus overhead only"
        );
    }

    #[test]
    fn size_accounting_never_underflows_under_churn() {
        let mut table = MemTable::new();
        for round in 0..50 {
            for key in 0..20u32 {
                let key = key.to_be_bytes().to_vec();
                if round % 3 == 0 {
                    table.delete(key);
                } else {
                    table.put(key, vec![b'x'; round]);
                }
            }
        }
        // Reachable only if no intermediate step wrapped around.
        assert!(table.approx_size_bytes() >= 20 * ENTRY_OVERHEAD_BYTES);
        assert_eq!(table.entry_count(), 20);
    }

    #[test]
    fn should_flush_tracks_the_threshold() {
        let mut table = MemTable::new();
        assert!(!table.should_flush(100));
        table.put(k("alpha"), vec![b'x'; 200]);
        assert!(table.should_flush(100));
    }

    #[test]
    fn into_entries_yields_sorted_pairs_for_flush() {
        let mut table = MemTable::new();
        table.put(k("zulu"), v("z"));
        table.put(k("alpha"), v("a"));
        table.delete(k("mike"));

        let flushed: Vec<(Key, Value)> = table.into_entries().collect();
        assert_eq!(
            flushed,
            vec![
                (k("alpha"), Value::Put(v("a"))),
                (k("mike"), Value::Tombstone),
                (k("zulu"), Value::Put(v("z"))),
            ]
        );
    }
}
