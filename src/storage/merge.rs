//! K-way merge over sorted runs, resolving key collisions by recency.
//!
//! Every part of the engine that has to look at more than one run at a time goes
//! through here: range scans, memtable flushes that span runs, and both
//! compaction policies. Getting the shadowing rule right in one place means the
//! compaction strategies in Phase 2 can be judged on scheduling alone, without
//! each of them reimplementing correctness.
//!
//! # The shadowing rule
//!
//! Sources are supplied **newest first**. When several runs hold the same key,
//! only the newest one's entry is emitted and the rest are discarded — that is
//! what makes an overwrite or a delete take effect, since an LSM-tree never
//! modifies an older run in place.
//!
//! A tombstone is a real entry here, not an absence. It is emitted like any
//! other value so that a caller merging into a *non-bottom* level keeps it: the
//! tombstone must survive to shadow whatever older runs still sit below.
//!
//! # Dropping tombstones
//!
//! [`MergeIterator::dropping_tombstones`] discards them instead. This is only
//! sound when merging into the **bottom** level, where no older run can exist
//! for the tombstone to shadow. Dropping one anywhere else resurrects every
//! deleted key that still has a live entry further down — a silent, unbounded
//! correctness bug, which is why the choice is an explicit constructor rather
//! than a default.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io;

use super::{Key, Value};

/// An entry sitting at the head of one source, waiting to be merged.
struct HeapEntry {
    key: Key,
    value: Value,
    /// Index into the source list. Lower means newer, so it wins a collision.
    source: usize,
}

impl Ord for HeapEntry {
    /// Reversed on both fields, because [`BinaryHeap`] is a max-heap and this
    /// merge wants the *smallest* key first, and among equal keys the *lowest*
    /// source index (the newest run).
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .key
            .cmp(&self.key)
            .then_with(|| other.source.cmp(&self.source))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.source == other.source
    }
}

impl Eq for HeapEntry {}

/// A boxed source of sorted entries. Each source must yield keys in strictly
/// ascending order; SSTables and memtables both do.
pub type Source<'a> = Box<dyn Iterator<Item = io::Result<(Key, Value)>> + 'a>;

/// Merges several sorted runs into one ascending stream, newest entry winning.
pub struct MergeIterator<'a> {
    sources: Vec<Source<'a>>,
    heap: BinaryHeap<HeapEntry>,
    drop_tombstones: bool,
    /// An error hit while refilling a source, surfaced on the next `next` call.
    /// Errors cannot be swallowed: a compaction that silently skipped an
    /// unreadable block would drop live data on the floor.
    pending_error: Option<io::Error>,
}

impl<'a> MergeIterator<'a> {
    /// Merge `sources`, **ordered newest first**.
    ///
    /// The ordering is the entire contract. Passing them oldest-first does not
    /// fail loudly — it quietly returns stale values and undoes deletions.
    pub fn new(sources: Vec<Source<'a>>) -> Self {
        let mut merger = Self {
            sources,
            heap: BinaryHeap::new(),
            drop_tombstones: false,
            pending_error: None,
        };
        for index in 0..merger.sources.len() {
            merger.refill(index);
        }
        merger
    }

    /// Merge, discarding tombstones.
    ///
    /// Only valid when merging into the bottom level — see the module docs.
    pub fn dropping_tombstones(sources: Vec<Source<'a>>) -> Self {
        let mut merger = Self::new(sources);
        merger.drop_tombstones = true;
        merger
    }

    /// Pull the next entry from `index` into the heap, if it has one.
    fn refill(&mut self, index: usize) {
        match self.sources[index].next() {
            Some(Ok((key, value))) => self.heap.push(HeapEntry {
                key,
                value,
                source: index,
            }),
            Some(Err(error)) => {
                // Keep the first error; later ones are usually cascade noise
                // from the same bad file.
                self.pending_error.get_or_insert(error);
            }
            None => {}
        }
    }
}

impl Iterator for MergeIterator<'_> {
    type Item = io::Result<(Key, Value)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(error) = self.pending_error.take() {
                return Some(Err(error));
            }

            let winner = self.heap.pop()?;
            self.refill(winner.source);

            // Every other source holding this key is older by construction, so
            // discard their entries without looking at them.
            while let Some(shadowed) = self.heap.peek() {
                if shadowed.key != winner.key {
                    break;
                }
                let shadowed = self.heap.pop().expect("just peeked");
                self.refill(shadowed.source);
            }

            if self.drop_tombstones && winner.value.is_tombstone() {
                continue;
            }
            return Some(Ok((winner.key, winner.value)));
        }
    }
}

/// Adapt a memtable into a [`Source`].
///
/// Clones each entry: the memtable outlives the merge and cannot hand out
/// ownership of its contents. Flushes are bounded by the memtable size, so the
/// copy is proportional to work already being done.
pub fn memtable_source(memtable: &super::MemTable) -> Source<'_> {
    Box::new(
        memtable
            .iter()
            .map(|(key, value)| Ok((key.clone(), value.clone()))),
    )
}

/// Adapt an SSTable into a [`Source`].
pub fn sstable_source(table: &super::SSTable) -> Source<'_> {
    Box::new(table.iter())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(s: &str) -> Key {
        s.as_bytes().to_vec()
    }

    fn put(s: &str) -> Value {
        Value::Put(s.as_bytes().to_vec())
    }

    /// Build a source from a literal list, as an in-memory stand-in for a run.
    fn source(entries: Vec<(Key, Value)>) -> Source<'static> {
        Box::new(entries.into_iter().map(Ok))
    }

    fn collect(merger: MergeIterator<'_>) -> Vec<(Key, Value)> {
        merger.map(|entry| entry.expect("merge entry")).collect()
    }

    #[test]
    fn merges_disjoint_runs_into_sorted_order() {
        let newest = source(vec![(k("a"), put("1")), (k("d"), put("4"))]);
        let oldest = source(vec![(k("b"), put("2")), (k("c"), put("3"))]);

        let merged = collect(MergeIterator::new(vec![newest, oldest]));
        assert_eq!(
            merged,
            vec![
                (k("a"), put("1")),
                (k("b"), put("2")),
                (k("c"), put("3")),
                (k("d"), put("4")),
            ]
        );
    }

    #[test]
    fn the_newest_run_wins_a_collision() {
        let newest = source(vec![(k("shared"), put("new"))]);
        let middle = source(vec![(k("shared"), put("middle"))]);
        let oldest = source(vec![(k("shared"), put("old"))]);

        let merged = collect(MergeIterator::new(vec![newest, middle, oldest]));
        assert_eq!(
            merged,
            vec![(k("shared"), put("new"))],
            "a key present in three runs must be emitted once, from the newest"
        );
    }

    /// Source order is the whole contract, so pin what reversing it does.
    #[test]
    fn source_order_determines_the_winner() {
        let build = || {
            (
                source(vec![(k("shared"), put("new"))]),
                source(vec![(k("shared"), put("old"))]),
            )
        };

        let (newest, oldest) = build();
        assert_eq!(
            collect(MergeIterator::new(vec![newest, oldest])),
            vec![(k("shared"), put("new"))]
        );

        let (newest, oldest) = build();
        assert_eq!(
            collect(MergeIterator::new(vec![oldest, newest])),
            vec![(k("shared"), put("old"))],
            "passing sources oldest-first returns stale data; it is the caller's \
             job to order them"
        );
    }

    #[test]
    fn a_newer_tombstone_shadows_an_older_value() {
        let newest = source(vec![(k("gone"), Value::Tombstone)]);
        let oldest = source(vec![(k("gone"), put("still here"))]);

        let merged = collect(MergeIterator::new(vec![newest, oldest]));
        assert_eq!(
            merged,
            vec![(k("gone"), Value::Tombstone)],
            "the tombstone must be emitted so it keeps shadowing lower levels"
        );
    }

    #[test]
    fn a_newer_value_resurrects_over_an_older_tombstone() {
        let newest = source(vec![(k("key"), put("rewritten"))]);
        let oldest = source(vec![(k("key"), Value::Tombstone)]);

        let merged = collect(MergeIterator::new(vec![newest, oldest]));
        assert_eq!(merged, vec![(k("key"), put("rewritten"))]);
    }

    #[test]
    fn dropping_tombstones_removes_them_and_what_they_shadow() {
        let newest = source(vec![(k("a"), put("1")), (k("b"), Value::Tombstone)]);
        let oldest = source(vec![(k("b"), put("doomed")), (k("c"), put("3"))]);

        let merged = collect(MergeIterator::dropping_tombstones(vec![newest, oldest]));
        assert_eq!(
            merged,
            vec![(k("a"), put("1")), (k("c"), put("3"))],
            "at the bottom level both the tombstone and the value it shadows go away"
        );
    }

    #[test]
    fn keeping_tombstones_is_the_default() {
        let newest = source(vec![(k("b"), Value::Tombstone)]);
        let oldest = source(vec![(k("b"), put("doomed"))]);

        let merged = collect(MergeIterator::new(vec![newest, oldest]));
        assert_eq!(merged, vec![(k("b"), Value::Tombstone)]);
    }

    #[test]
    fn handles_empty_and_absent_sources() {
        assert_eq!(collect(MergeIterator::new(vec![])), vec![]);
        assert_eq!(collect(MergeIterator::new(vec![source(vec![])])), vec![]);

        let merged = collect(MergeIterator::new(vec![
            source(vec![]),
            source(vec![(k("a"), put("1"))]),
            source(vec![]),
        ]));
        assert_eq!(merged, vec![(k("a"), put("1"))]);
    }

    #[test]
    fn runs_of_very_different_lengths_merge_correctly() {
        let long: Vec<(Key, Value)> = (0..1000)
            .map(|i| (format!("key{i:05}").into_bytes(), put("old")))
            .collect();
        // Overwrites every hundredth key.
        let short: Vec<(Key, Value)> = (0..1000)
            .step_by(100)
            .map(|i| (format!("key{i:05}").into_bytes(), put("new")))
            .collect();

        let merged = collect(MergeIterator::new(vec![source(short), source(long)]));
        assert_eq!(merged.len(), 1000, "collisions must not duplicate keys");

        for (index, (key, value)) in merged.iter().enumerate() {
            assert_eq!(key, &format!("key{index:05}").into_bytes());
            let expected = if index % 100 == 0 { "new" } else { "old" };
            assert_eq!(value, &put(expected), "wrong winner at index {index}");
        }
    }

    #[test]
    fn output_is_strictly_ascending_across_many_overlapping_runs() {
        // Ten runs, each holding every tenth key with an offset, so every key
        // appears in exactly one run but the runs interleave heavily.
        let sources: Vec<Source<'static>> = (0..10)
            .map(|offset| {
                let entries: Vec<(Key, Value)> = (0..100)
                    .map(|i| {
                        let n = i * 10 + offset;
                        (format!("key{n:05}").into_bytes(), put("v"))
                    })
                    .collect();
                source(entries)
            })
            .collect();

        let merged = collect(MergeIterator::new(sources));
        assert_eq!(merged.len(), 1000);
        for pair in merged.windows(2) {
            assert!(
                pair[0].0 < pair[1].0,
                "merge output must strictly ascend: {:?} then {:?}",
                String::from_utf8_lossy(&pair[0].0),
                String::from_utf8_lossy(&pair[1].0)
            );
        }
    }

    /// Every run holding the same key is the pathological collision case.
    #[test]
    fn a_key_in_every_run_is_emitted_once() {
        let sources: Vec<Source<'static>> = (0..20)
            .map(|i| source(vec![(k("hot"), put(&format!("version{i}")))]))
            .collect();

        let merged = collect(MergeIterator::new(sources));
        assert_eq!(merged, vec![(k("hot"), put("version0"))]);
    }

    /// An I/O error must reach the caller. A compaction that silently skipped an
    /// unreadable block would drop live data.
    #[test]
    fn errors_are_surfaced_not_swallowed() {
        let healthy = source(vec![(k("a"), put("1")), (k("z"), put("26"))]);
        let broken: Source<'static> = Box::new(
            vec![
                Ok((k("b"), put("2"))),
                Err(io::Error::new(io::ErrorKind::InvalidData, "bad block")),
            ]
            .into_iter(),
        );

        let results: Vec<_> = MergeIterator::new(vec![healthy, broken]).collect();
        assert!(
            results.iter().any(|entry| entry.is_err()),
            "the merge must report the failing source"
        );
    }

    #[test]
    fn merging_real_sstables_and_a_memtable() {
        use crate::storage::{MemTable, SSTable, SSTableWriter};

        let dir = std::env::temp_dir().join(format!("vectordb-merge-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");

        // Oldest run on disk.
        let old_path = dir.join("old.sst");
        let mut writer = SSTableWriter::create(&old_path).expect("create");
        for key in ["alpha", "bravo", "charlie"] {
            writer.append(key.as_bytes(), &put("old")).expect("append");
        }
        writer.finish().expect("finish");

        // Newer run on disk: overwrites bravo, deletes charlie.
        let new_path = dir.join("new.sst");
        let mut writer = SSTableWriter::create(&new_path).expect("create");
        writer.append(b"bravo", &put("newer")).expect("append");
        writer
            .append(b"charlie", &Value::Tombstone)
            .expect("append");
        writer.finish().expect("finish");

        // Newest of all: the in-memory buffer.
        let mut memtable = MemTable::new();
        memtable.put(k("delta"), b"fresh".to_vec());
        memtable.put(k("alpha"), b"newest".to_vec());

        let old_table = SSTable::open(&old_path).expect("open");
        let new_table = SSTable::open(&new_path).expect("open");

        let merged = collect(MergeIterator::new(vec![
            memtable_source(&memtable),
            sstable_source(&new_table),
            sstable_source(&old_table),
        ]));

        assert_eq!(
            merged,
            vec![
                (k("alpha"), put("newest")),
                (k("bravo"), put("newer")),
                (k("charlie"), Value::Tombstone),
                (k("delta"), put("fresh")),
            ]
        );

        // The same merge at the bottom level drops the tombstone.
        let compacted = collect(MergeIterator::dropping_tombstones(vec![
            memtable_source(&memtable),
            sstable_source(&new_table),
            sstable_source(&old_table),
        ]));
        assert_eq!(
            compacted,
            vec![
                (k("alpha"), put("newest")),
                (k("bravo"), put("newer")),
                (k("delta"), put("fresh")),
            ]
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
