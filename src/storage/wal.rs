//! Write-ahead log.
//!
//! Every mutation is appended here *before* it is applied to the memtable, so a
//! crash loses nothing that was acknowledged. On restart the log is replayed to
//! rebuild the memtable, and is deleted only once that memtable has been flushed
//! to a durable on-disk run.
//!
//! # Record format
//!
//! ```text
//! ┌───────────────┬───────────────┬─────────────────────────────────┐
//! │ payload_len   │ crc32         │ payload                         │
//! │ u32 LE        │ u32 LE        │ payload_len bytes               │
//! └───────────────┴───────────────┴─────────────────────────────────┘
//!
//! payload:
//! ┌────────┬───────────────┬───────────┬──────────────────────────┐
//! │ tag u8 │ key_len u32LE │ key bytes │ value bytes (to the end) │
//! └────────┴───────────────┴───────────┴──────────────────────────┘
//! ```
//!
//! The value's length is implied by `payload_len`, so it is not stored twice.
//! A tombstone carries zero value bytes. The CRC covers the payload only.
//!
//! # Torn tails are normal, not corruption
//!
//! A process killed mid-`write` leaves a partial record at the end of the file.
//! That is the *expected* state after a crash, so replay stops cleanly at the
//! first record it cannot verify and reports how many bytes it discarded, rather
//! than refusing to open the database. Reopening then physically truncates those
//! bytes — see [`Wal::recover`] for why that step is not optional.
//!
//! **Labelled simplification:** LevelDB and RocksDB write their logs in
//! fixed-size 32 KiB blocks with per-fragment record types, which lets recovery
//! resynchronise *past* a corrupt region in the middle of a file. This log is a
//! flat record stream, so it can only truncate at the first bad record — a
//! mid-file bit flip costs every record after it. That is an acceptable trade
//! here (the log is short-lived and bounded by the memtable flush threshold) but
//! it is a real difference from production engines, not an equivalent design.

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use super::crc32::Crc32;
use super::{Key, Value};

/// Bytes of fixed header before each payload: `payload_len` + `crc32`.
const HEADER_BYTES: usize = 8;

/// Tag byte marking a live write.
const TAG_PUT: u8 = 0;
/// Tag byte marking a deletion.
const TAG_TOMBSTONE: u8 = 1;

/// Largest payload replay will attempt to allocate.
///
/// Without this, a corrupted length field of `0xFFFFFFFF` would ask for a 4 GiB
/// buffer before the CRC ever gets a chance to reject the record. Any record
/// claiming to be larger than this is treated as a torn tail.
const MAX_PAYLOAD_BYTES: u32 = 64 * 1024 * 1024;

/// When to force log bytes down to the physical device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPolicy {
    /// `fsync` after every append. Durable across a machine crash, and slow —
    /// throughput becomes a function of device sync latency.
    EveryWrite,
    /// Buffer in userspace and only sync when [`Wal::sync`] is called. Survives
    /// a *process* crash once the bytes reach the OS, but a power loss can lose
    /// the tail. This is the honest setting to benchmark "no-sync" throughput
    /// with, and it must be labelled as such in any reported number.
    Manual,
}

/// An appendable write-ahead log.
#[derive(Debug)]
pub struct Wal {
    writer: BufWriter<File>,
    path: PathBuf,
    sync_policy: SyncPolicy,
    bytes_written: u64,
}

/// Outcome of replaying a log from disk.
#[derive(Debug, Default)]
pub struct Replay {
    /// Records in the order they were appended. Replaying them in this order
    /// into a fresh memtable reproduces the pre-crash state, because later
    /// writes to a key overwrite earlier ones.
    pub records: Vec<(Key, Value)>,
    /// Bytes at the end of the file that could not be verified and were
    /// discarded. Non-zero after a crash mid-write; zero after a clean shutdown.
    pub discarded_tail_bytes: u64,
}

impl Wal {
    /// Open `path` for appending, recovering its contents and repairing a torn
    /// tail. Creates the file if absent.
    ///
    /// This is the entry point the engine uses at startup: it returns both the
    /// recovered records (to rebuild the memtable) and a log ready to accept new
    /// writes.
    ///
    /// **Truncating the torn tail is mandatory, not tidiness.** Appending after
    /// an unverifiable fragment would strand every subsequent record behind
    /// bytes that replay stops at — the log would keep accepting writes and
    /// silently fail to recover a single one of them. So the file is physically
    /// shortened to the last verified record boundary before the append handle
    /// is opened. The discarded bytes were, by definition, never acknowledged to
    /// a caller.
    pub fn recover(
        path: impl AsRef<Path>,
        sync_policy: SyncPolicy,
    ) -> io::Result<(Self, Replay)> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let replay = Self::replay(&path)?;
        if replay.discarded_tail_bytes > 0 {
            let file = OpenOptions::new().write(true).open(&path)?;
            let keep = file.metadata()?.len() - replay.discarded_tail_bytes;
            file.set_len(keep)?;
            // Make the shortened length durable before writing past it, so a
            // second crash during recovery cannot resurrect the fragment.
            file.sync_all()?;
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;
        let bytes_written = file.metadata()?.len();

        let wal = Self {
            writer: BufWriter::new(file),
            path,
            sync_policy,
            bytes_written,
        };
        Ok((wal, replay))
    }

    /// Open `path` for appending, discarding the recovered records.
    ///
    /// Convenience for callers that only want to write. Still repairs a torn
    /// tail — see [`Wal::recover`].
    pub fn open(path: impl AsRef<Path>, sync_policy: SyncPolicy) -> io::Result<Self> {
        Ok(Self::recover(path, sync_policy)?.0)
    }

    /// Append one mutation.
    ///
    /// Under [`SyncPolicy::EveryWrite`] this returns only once the bytes are on
    /// the device; the caller may treat a successful return as durable.
    pub fn append(&mut self, key: &[u8], value: &Value) -> io::Result<()> {
        let (tag, value_bytes): (u8, &[u8]) = match value {
            Value::Put(bytes) => (TAG_PUT, bytes),
            Value::Tombstone => (TAG_TOMBSTONE, &[]),
        };

        let key_len: u32 = key
            .len()
            .try_into()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "key too large for the log"))?;
        let payload_len: u32 = (1 + 4 + key.len() + value_bytes.len())
            .try_into()
            .map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "record too large for the log")
            })?;
        if payload_len > MAX_PAYLOAD_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("record of {payload_len} bytes exceeds the {MAX_PAYLOAD_BYTES} byte limit"),
            ));
        }

        // Checksum the payload incrementally so it never has to be assembled
        // into a scratch buffer just to be hashed.
        let key_len_bytes = key_len.to_le_bytes();
        let mut checksum = Crc32::new();
        checksum.update(&[tag]);
        checksum.update(&key_len_bytes);
        checksum.update(key);
        checksum.update(value_bytes);

        self.writer.write_all(&payload_len.to_le_bytes())?;
        self.writer.write_all(&checksum.finish().to_le_bytes())?;
        self.writer.write_all(&[tag])?;
        self.writer.write_all(&key_len_bytes)?;
        self.writer.write_all(key)?;
        self.writer.write_all(value_bytes)?;

        self.bytes_written += (HEADER_BYTES + payload_len as usize) as u64;

        if self.sync_policy == SyncPolicy::EveryWrite {
            self.sync()?;
        }
        Ok(())
    }

    /// Flush userspace buffers and force the file's data to the device.
    ///
    /// Uses `sync_data` rather than `sync_all`: the log's metadata (timestamps)
    /// does not need to be durable, only its contents, and skipping the metadata
    /// flush is measurably cheaper on most filesystems.
    pub fn sync(&mut self) -> io::Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_data()
    }

    /// Total bytes appended, including headers. Used to decide when to roll the
    /// log over after a memtable flush.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read every intact record from the log at `path`, in append order.
    ///
    /// A missing file replays as empty — that is a database that has never been
    /// written to, not an error. Stops at the first record that fails to verify;
    /// see the module docs on torn tails.
    pub fn replay(path: impl AsRef<Path>) -> io::Result<Replay> {
        let file = match File::open(path.as_ref()) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Replay::default()),
            Err(error) => return Err(error),
        };

        let total_bytes = file.metadata()?.len();
        let mut reader = BufReader::new(file);
        let mut records = Vec::new();
        let mut verified_bytes = 0u64;

        // `CleanEnd` and `Unverifiable` both stop replay; they differ only in
        // whether bytes are left unaccounted for, which the caller reads off
        // `discarded_tail_bytes` below.
        while let RecordRead::Record { key, value, bytes } = read_record(&mut reader)? {
            records.push((key, value));
            verified_bytes += bytes;
        }

        Ok(Replay {
            records,
            discarded_tail_bytes: total_bytes - verified_bytes,
        })
    }

    /// Delete the log file. Called once the memtable it protects has been
    /// durably flushed, at which point replaying it would be redundant work.
    pub fn remove(self) -> io::Result<()> {
        let path = self.path.clone();
        drop(self);
        match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }
}

enum RecordRead {
    Record {
        key: Key,
        value: Value,
        /// Header plus payload, for the caller's byte accounting.
        bytes: u64,
    },
    /// The file ended exactly on a record boundary.
    CleanEnd,
    /// A truncated or checksum-failing record: everything from here on is
    /// discarded.
    Unverifiable,
}

fn read_record(reader: &mut impl Read) -> io::Result<RecordRead> {
    let mut header = [0u8; HEADER_BYTES];
    match fill(reader, &mut header)? {
        0 => return Ok(RecordRead::CleanEnd),
        n if n < HEADER_BYTES => return Ok(RecordRead::Unverifiable),
        _ => {}
    }

    let payload_len = u32::from_le_bytes(header[0..4].try_into().expect("4 bytes"));
    let expected_crc = u32::from_le_bytes(header[4..8].try_into().expect("4 bytes"));

    // Reject an implausible length before allocating anything for it.
    if payload_len > MAX_PAYLOAD_BYTES || (payload_len as usize) < 1 + 4 {
        return Ok(RecordRead::Unverifiable);
    }

    let mut payload = vec![0u8; payload_len as usize];
    if fill(reader, &mut payload)? < payload.len() {
        return Ok(RecordRead::Unverifiable);
    }

    let mut checksum = Crc32::new();
    checksum.update(&payload);
    if checksum.finish() != expected_crc {
        return Ok(RecordRead::Unverifiable);
    }

    // The CRC has passed, so the field widths below are trustworthy; a
    // length that overruns the payload would have to be a CRC collision.
    let tag = payload[0];
    let key_len = u32::from_le_bytes(payload[1..5].try_into().expect("4 bytes")) as usize;
    let key_end = match 5usize.checked_add(key_len) {
        Some(end) if end <= payload.len() => end,
        _ => return Ok(RecordRead::Unverifiable),
    };

    let key = payload[5..key_end].to_vec();
    let value = match tag {
        TAG_PUT => Value::Put(payload[key_end..].to_vec()),
        TAG_TOMBSTONE if key_end == payload.len() => Value::Tombstone,
        _ => return Ok(RecordRead::Unverifiable),
    };

    Ok(RecordRead::Record {
        key,
        value,
        bytes: (HEADER_BYTES + payload.len()) as u64,
    })
}

/// Read until `buf` is full or the source is exhausted, returning bytes read.
///
/// `Read::read_exact` cannot be used here because it collapses "the file ended
/// on a boundary" and "the file ended mid-record" into the same `UnexpectedEof`,
/// and replay must tell those apart.
fn fill(reader: &mut impl Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(filled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::MemTable;
    use std::fs::OpenOptions;

    /// A scratch directory that deletes itself, so tests leave no residue and
    /// can run concurrently without colliding.
    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(label: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static COUNTER: AtomicU32 = AtomicU32::new(0);

            let unique = format!(
                "vectordb-{label}-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            );
            let path = std::env::temp_dir().join(unique);
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }

        fn file(&self, name: &str) -> PathBuf {
            self.path.join(name)
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

    fn put(s: &str) -> Value {
        Value::Put(s.as_bytes().to_vec())
    }

    #[test]
    fn replaying_a_missing_log_yields_nothing() {
        let dir = TempDir::new("missing");
        let replay = Wal::replay(dir.file("absent.wal")).expect("replay");
        assert!(replay.records.is_empty());
        assert_eq!(replay.discarded_tail_bytes, 0);
    }

    #[test]
    fn records_round_trip_in_append_order() {
        let dir = TempDir::new("roundtrip");
        let path = dir.file("test.wal");

        {
            let mut wal = Wal::open(&path, SyncPolicy::EveryWrite).expect("open");
            wal.append(b"alpha", &put("one")).expect("append");
            wal.append(b"bravo", &Value::Tombstone).expect("append");
            wal.append(b"alpha", &put("two")).expect("append");
            wal.sync().expect("sync");
        }

        let replay = Wal::replay(&path).expect("replay");
        assert_eq!(
            replay.records,
            vec![
                (k("alpha"), put("one")),
                (k("bravo"), Value::Tombstone),
                (k("alpha"), put("two")),
            ],
            "order matters: replaying in sequence is what makes the last write win"
        );
        assert_eq!(replay.discarded_tail_bytes, 0, "a clean log discards nothing");
    }

    #[test]
    fn handles_empty_keys_values_and_binary_data() {
        let dir = TempDir::new("edges");
        let path = dir.file("test.wal");

        let binary: Vec<u8> = (0..=255u8).collect();
        {
            let mut wal = Wal::open(&path, SyncPolicy::Manual).expect("open");
            wal.append(b"", &put("empty key")).expect("append");
            wal.append(b"empty value", &Value::Put(Vec::new()))
                .expect("append");
            wal.append(&binary, &Value::Put(binary.clone()))
                .expect("append");
            wal.sync().expect("sync");
        }

        let replay = Wal::replay(&path).expect("replay");
        assert_eq!(replay.records.len(), 3);
        assert_eq!(replay.records[0], (Vec::new(), put("empty key")));
        assert_eq!(
            replay.records[1],
            (k("empty value"), Value::Put(Vec::new())),
            "an empty value must stay a Put, not decay into a tombstone"
        );
        assert_eq!(replay.records[2], (binary.clone(), Value::Put(binary)));
        assert_eq!(replay.discarded_tail_bytes, 0);
    }

    #[test]
    fn reopening_appends_rather_than_truncating() {
        let dir = TempDir::new("reopen");
        let path = dir.file("test.wal");

        {
            let mut wal = Wal::open(&path, SyncPolicy::EveryWrite).expect("open");
            wal.append(b"first", &put("1")).expect("append");
        }
        {
            let mut wal = Wal::open(&path, SyncPolicy::EveryWrite).expect("reopen");
            assert!(wal.bytes_written() > 0, "reopen must see the existing bytes");
            wal.append(b"second", &put("2")).expect("append");
        }

        let replay = Wal::replay(&path).expect("replay");
        assert_eq!(replay.records.len(), 2);
    }

    /// The core crash test: a process killed mid-`write` leaves a partial
    /// record. Every complete record before it must survive, and the engine must
    /// still be usable afterwards.
    #[test]
    fn a_torn_tail_costs_only_the_partial_record() {
        let dir = TempDir::new("torn");
        let path = dir.file("test.wal");

        {
            let mut wal = Wal::open(&path, SyncPolicy::EveryWrite).expect("open");
            for i in 0..10 {
                wal.append(format!("key{i:02}").as_bytes(), &put(&format!("value{i}")))
                    .expect("append");
            }
        }

        let full_len = std::fs::metadata(&path).expect("metadata").len();
        let intact = Wal::replay(&path).expect("replay").records.len();
        assert_eq!(intact, 10);

        // Simulate the crash: lop off 5 bytes, landing mid-record.
        let truncated_to = full_len - 5;
        OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open for truncate")
            .set_len(truncated_to)
            .expect("truncate");

        let replay = Wal::replay(&path).expect("replay after truncation");
        assert_eq!(
            replay.records.len(),
            9,
            "the nine complete records must survive a torn tenth"
        );
        assert!(
            replay.discarded_tail_bytes > 0,
            "the partial record must be reported as discarded"
        );

        // And the log is still writable: recovery is not a dead end.
        let mut wal = Wal::open(&path, SyncPolicy::EveryWrite).expect("reopen after crash");
        wal.append(b"after-crash", &put("ok")).expect("append");
        wal.sync().expect("sync");

        let reopened = Wal::replay(&path).expect("replay");
        assert_eq!(
            reopened.records.last().expect("a last record").0,
            k("after-crash")
        );
    }

    /// Truncating at *every* possible offset must never panic, never loop, and
    /// never return a record that was not fully written.
    #[test]
    fn truncation_at_any_offset_is_survivable() {
        let dir = TempDir::new("truncate-sweep");
        let path = dir.file("test.wal");

        {
            let mut wal = Wal::open(&path, SyncPolicy::Manual).expect("open");
            for i in 0..6 {
                wal.append(format!("key{i}").as_bytes(), &put(&format!("value-{i}")))
                    .expect("append");
            }
            wal.sync().expect("sync");
        }

        let full = std::fs::read(&path).expect("read log");
        let complete = Wal::replay(&path).expect("replay").records;

        for cut in 0..=full.len() {
            let partial_path = dir.file(&format!("cut-{cut}.wal"));
            std::fs::write(&partial_path, &full[..cut]).expect("write partial");

            let replay = Wal::replay(&partial_path).expect("replay partial");
            assert!(
                replay.records.len() <= complete.len(),
                "truncation invented records at cut {cut}"
            );
            assert_eq!(
                replay.records[..],
                complete[..replay.records.len()],
                "surviving records must be a prefix of the original at cut {cut}"
            );
        }
    }

    /// A bit flip inside a record body must be caught by the CRC rather than
    /// silently returning wrong data.
    #[test]
    fn corruption_is_detected_not_returned() {
        let dir = TempDir::new("corrupt");
        let path = dir.file("test.wal");

        {
            let mut wal = Wal::open(&path, SyncPolicy::Manual).expect("open");
            wal.append(b"alpha", &put("the original value"))
                .expect("append");
            wal.append(b"bravo", &put("second")).expect("append");
            wal.sync().expect("sync");
        }

        let mut bytes = std::fs::read(&path).expect("read");
        // Flip a bit in the first record's value, well past its header.
        let victim = HEADER_BYTES + 12;
        bytes[victim] ^= 0b0000_1000;
        std::fs::write(&path, &bytes).expect("write corrupted");

        let replay = Wal::replay(&path).expect("replay corrupted");
        assert!(
            replay.records.is_empty(),
            "a corrupt first record must stop replay, not be handed back as data"
        );
        assert_eq!(replay.discarded_tail_bytes, bytes.len() as u64);
    }

    #[test]
    fn a_garbage_length_field_does_not_allocate_wildly() {
        let dir = TempDir::new("garbage-len");
        let path = dir.file("test.wal");

        // A header claiming a 4 GiB payload, with a CRC that will never be
        // reached. Replay must reject it on the length check alone.
        let mut bytes = u32::MAX.to_le_bytes().to_vec();
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(b"not really four gigabytes");
        std::fs::write(&path, &bytes).expect("write");

        let replay = Wal::replay(&path).expect("replay");
        assert!(replay.records.is_empty());
        assert_eq!(replay.discarded_tail_bytes, bytes.len() as u64);
    }

    /// The end-to-end recovery story: writes go to the log, the process dies,
    /// and a fresh memtable rebuilt from the log matches what was there before —
    /// including overwrites and deletions.
    #[test]
    fn replay_rebuilds_an_equivalent_memtable() {
        let dir = TempDir::new("rebuild");
        let path = dir.file("test.wal");

        let mut before = MemTable::new();
        {
            let mut wal = Wal::open(&path, SyncPolicy::EveryWrite).expect("open");
            let mutations: Vec<(&[u8], Value)> = vec![
                (b"alpha", put("1")),
                (b"bravo", put("2")),
                (b"alpha", put("overwritten")),
                (b"charlie", put("3")),
                (b"bravo", Value::Tombstone),
                (b"delta", Value::Tombstone),
            ];
            for (key, value) in mutations {
                wal.append(key, &value).expect("append");
                match value {
                    Value::Put(bytes) => {
                        before.put(key.to_vec(), bytes);
                    }
                    Value::Tombstone => {
                        before.delete(key.to_vec());
                    }
                }
            }
        }

        let mut after = MemTable::new();
        for (key, value) in Wal::replay(&path).expect("replay").records {
            match value {
                Value::Put(bytes) => after.put(key, bytes),
                Value::Tombstone => after.delete(key),
            };
        }

        let expected: Vec<_> = before.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let recovered: Vec<_> = after.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        assert_eq!(expected, recovered);
        assert_eq!(
            after.get(b"alpha").and_then(Value::as_bytes),
            Some(b"overwritten".as_slice()),
            "the later write must win after replay"
        );
        assert_eq!(
            after.get(b"bravo"),
            Some(&Value::Tombstone),
            "a delete must not be undone by replaying the earlier put"
        );
    }

    /// Regression test. Recovery must physically shorten the file, not just
    /// refuse to read the fragment: appending past a torn tail would strand
    /// every later write behind bytes replay stops at, so the log would accept
    /// writes and recover none of them.
    #[test]
    fn recover_physically_truncates_the_torn_tail() {
        let dir = TempDir::new("truncate-on-open");
        let path = dir.file("test.wal");

        {
            let mut wal = Wal::open(&path, SyncPolicy::Manual).expect("open");
            wal.append(b"alpha", &put("one")).expect("append");
            wal.append(b"bravo", &put("two")).expect("append");
            wal.sync().expect("sync");
        }

        // Append 7 bytes of junk, standing in for a record that was half-written
        // when the process died.
        {
            let mut file = OpenOptions::new().append(true).open(&path).expect("open");
            file.write_all(b"\x40\x00\x00\x00\x99\x99\x99").expect("write junk");
        }
        let torn_len = std::fs::metadata(&path).expect("metadata").len();

        let (wal, replay) = Wal::recover(&path, SyncPolicy::Manual).expect("recover");
        let repaired_len = std::fs::metadata(&path).expect("metadata").len();

        assert_eq!(replay.records.len(), 2);
        assert_eq!(replay.discarded_tail_bytes, 7);
        assert_eq!(
            repaired_len,
            torn_len - 7,
            "the fragment must be gone from the file, not merely skipped"
        );

        drop(wal);
        let mut wal = Wal::open(&path, SyncPolicy::Manual).expect("reopen");
        wal.append(b"charlie", &put("three")).expect("append");
        wal.sync().expect("sync");

        let after = Wal::replay(&path).expect("replay");
        assert_eq!(after.records.len(), 3, "the post-crash write must be recoverable");
        assert_eq!(after.discarded_tail_bytes, 0);
    }

    #[test]
    fn remove_deletes_the_file() {
        let dir = TempDir::new("remove");
        let path = dir.file("test.wal");

        let mut wal = Wal::open(&path, SyncPolicy::Manual).expect("open");
        wal.append(b"key", &put("value")).expect("append");
        wal.sync().expect("sync");
        assert!(path.exists());

        wal.remove().expect("remove");
        assert!(!path.exists());
    }
}
