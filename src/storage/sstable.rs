//! Sorted String Table: an immutable, sorted, on-disk run.
//!
//! A memtable flush produces exactly one of these. Once written it is never
//! modified — compaction creates new SSTables and deletes old ones. Immutability
//! is what lets reads proceed without locking and lets the same file be shared
//! between levels during a merge.
//!
//! # File layout
//!
//! ```text
//! ┌──────────────────┐ offset 0
//! │ data block 0     │  entries, sorted, ~4 KiB each
//! │ data block 1     │
//! │ ...              │
//! ├──────────────────┤
//! │ index block      │  one entry per data block (sparse)
//! ├──────────────────┤
//! │ bloom filter     │  optional; absent = zero length
//! ├──────────────────┤
//! │ footer (44 B)    │  fixed size, located by seeking to end - 44
//! └──────────────────┘
//! ```
//!
//! Every block carries a trailing CRC-32 over its own bytes, so a corrupt block
//! is caught when it is read rather than silently returning wrong data.
//!
//! # Why the index is sparse
//!
//! The index stores the *first key* of each block, not every key. For 4 KiB
//! blocks that is roughly one index entry per few dozen records, so the index of
//! a large SSTable stays small enough to hold in memory permanently. A lookup
//! binary-searches the in-memory index to find the one block that could contain
//! the key, then reads that single block and scans it. Cost: one disk read per
//! lookup, and memory proportional to the number of blocks rather than the
//! number of keys. This is the standard LevelDB/RocksDB arrangement.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::bloom::{hash_key, BloomFilter};
use super::crc32::crc32;
use super::{Key, Value};

/// Target size of a data block before it is closed and a new one started.
///
/// Blocks are the unit of I/O for a point lookup, so this trades read
/// amplification (larger blocks read more bytes than needed) against index size
/// (smaller blocks mean more index entries in memory). 4 KiB matches the typical
/// page size and LevelDB's default.
pub const DEFAULT_BLOCK_TARGET_BYTES: usize = 4 * 1024;

/// Default target false-positive rate for the per-table bloom filter.
///
/// 1% costs ~9.6 bits per key. Lowering it shrinks wasted block reads but grows
/// memory that every open table holds permanently; this is one of the knobs the
/// Phase 3 read-amplification benchmarks should sweep.
pub const DEFAULT_BLOOM_FALSE_POSITIVE_RATE: f64 = 0.01;

/// Bytes of CRC appended to every block.
const BLOCK_CRC_BYTES: usize = 4;

/// Fixed footer size: see [`Footer`].
const FOOTER_BYTES: usize = 44;

/// Identifies the file as one of ours and pins the format version. Bumping this
/// invalidates old files rather than misparsing them.
const MAGIC: u64 = 0x5353_5442_4456_0001;

const TAG_PUT: u8 = 0;
const TAG_TOMBSTONE: u8 = 1;

/// Largest single encoded entry accepted when reading, guarding against a
/// corrupt length field demanding an enormous allocation.
const MAX_ENTRY_BYTES: u32 = 64 * 1024 * 1024;

/// One entry in the sparse index: the first key of a data block, and where that
/// block lives in the file.
#[derive(Debug, Clone, PartialEq, Eq)]
struct IndexEntry {
    first_key: Key,
    offset: u64,
    /// Block length including its CRC trailer.
    length: u32,
}

/// Summary of a written SSTable, returned by [`SSTableWriter::finish`].
#[derive(Debug, Clone)]
pub struct SSTableMeta {
    pub path: PathBuf,
    pub entry_count: u64,
    pub file_size_bytes: u64,
    pub block_count: usize,
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Builds an SSTable by appending entries in ascending key order.
#[derive(Debug)]
pub struct SSTableWriter {
    writer: BufWriter<File>,
    path: PathBuf,
    /// Offset of the next byte to be written.
    offset: u64,
    /// Buffer for the block currently being filled.
    block: Vec<u8>,
    block_first_key: Option<Key>,
    index: Vec<IndexEntry>,
    entry_count: u64,
    last_key: Option<Key>,
    block_target_bytes: usize,
    /// Hash of every key appended, kept so the bloom filter can be sized against
    /// the true key count at `finish` rather than an estimate supplied up front.
    /// Costs 8 bytes per key transiently — small next to the memtable being
    /// flushed.
    key_hashes: Vec<u64>,
    bloom_false_positive_rate: f64,
}

impl SSTableWriter {
    /// Create a new SSTable at `path`, truncating anything already there.
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::create_with_block_size(path, DEFAULT_BLOCK_TARGET_BYTES)
    }

    /// As [`SSTableWriter::create`], with an explicit block size. Tests use a
    /// tiny block size to exercise multi-block behaviour without writing
    /// megabytes.
    pub fn create_with_block_size(
        path: impl AsRef<Path>,
        block_target_bytes: usize,
    ) -> io::Result<Self> {
        assert!(block_target_bytes > 0, "block size must be positive");

        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;

        Ok(Self {
            writer: BufWriter::new(file),
            path,
            offset: 0,
            block: Vec::with_capacity(block_target_bytes + 128),
            block_first_key: None,
            index: Vec::new(),
            entry_count: 0,
            last_key: None,
            block_target_bytes,
            key_hashes: Vec::new(),
            bloom_false_positive_rate: DEFAULT_BLOOM_FALSE_POSITIVE_RATE,
        })
    }

    /// Override the bloom filter's target false-positive rate. A rate of 0
    /// disables the filter entirely, which is useful for measuring what it buys.
    pub fn with_bloom_false_positive_rate(mut self, rate: f64) -> Self {
        self.bloom_false_positive_rate = rate;
        self
    }

    /// Append one entry. Keys must arrive in strictly ascending order.
    ///
    /// The ordering requirement is not a convenience — the whole file format
    /// depends on it, and a memtable flush or a merge iterator naturally
    /// provides it. Violating it is a programming error, so it is rejected
    /// loudly rather than producing a subtly unsearchable file.
    pub fn append(&mut self, key: &[u8], value: &Value) -> io::Result<()> {
        if let Some(last) = &self.last_key {
            if key <= last.as_slice() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "SSTable keys must strictly ascend: {:?} followed {:?}",
                        String::from_utf8_lossy(key),
                        String::from_utf8_lossy(last)
                    ),
                ));
            }
        }

        let (tag, value_bytes): (u8, &[u8]) = match value {
            Value::Put(bytes) => (TAG_PUT, bytes),
            Value::Tombstone => (TAG_TOMBSTONE, &[]),
        };

        let key_len: u32 = key.len().try_into().map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "key too large for an SSTable")
        })?;
        let value_len: u32 = value_bytes.len().try_into().map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "value too large for an SSTable")
        })?;

        if self.block_first_key.is_none() {
            self.block_first_key = Some(key.to_vec());
        }

        self.block.push(tag);
        self.block.extend_from_slice(&key_len.to_le_bytes());
        self.block.extend_from_slice(&value_len.to_le_bytes());
        self.block.extend_from_slice(key);
        self.block.extend_from_slice(value_bytes);

        self.entry_count += 1;
        self.last_key = Some(key.to_vec());
        if self.bloom_false_positive_rate > 0.0 {
            self.key_hashes.push(hash_key(key));
        }

        if self.block.len() >= self.block_target_bytes {
            self.flush_block()?;
        }
        Ok(())
    }

    /// Write the pending block out and record it in the index.
    ///
    /// An entry larger than the block target gets a block to itself; blocks are
    /// a target, not a hard cap, because an entry may never be split across two.
    fn flush_block(&mut self) -> io::Result<()> {
        if self.block.is_empty() {
            return Ok(());
        }

        let first_key = self
            .block_first_key
            .take()
            .expect("a non-empty block always recorded its first key");
        let checksum = crc32(&self.block);

        self.writer.write_all(&self.block)?;
        self.writer.write_all(&checksum.to_le_bytes())?;

        let length = (self.block.len() + BLOCK_CRC_BYTES) as u32;
        self.index.push(IndexEntry {
            first_key,
            offset: self.offset,
            length,
        });
        self.offset += length as u64;
        self.block.clear();
        Ok(())
    }

    /// Finish the file: flush the last block, write the index and footer, and
    /// `fsync`.
    ///
    /// The footer is written last and contains the offsets of everything else,
    /// so a file whose footer is present and checksums correctly is a file that
    /// was fully written. A crash mid-write leaves a footerless file that
    /// [`SSTable::open`] rejects — which is the desired behaviour, since a
    /// partial SSTable should never be adopted into the tree.
    pub fn finish(mut self) -> io::Result<SSTableMeta> {
        self.flush_block()?;

        let index_offset = self.offset;
        let mut index_bytes = Vec::new();
        for entry in &self.index {
            let key_len = entry.first_key.len() as u32;
            index_bytes.extend_from_slice(&key_len.to_le_bytes());
            index_bytes.extend_from_slice(&entry.first_key);
            index_bytes.extend_from_slice(&entry.offset.to_le_bytes());
            index_bytes.extend_from_slice(&entry.length.to_le_bytes());
        }
        let index_checksum = crc32(&index_bytes);
        self.writer.write_all(&index_bytes)?;
        self.writer.write_all(&index_checksum.to_le_bytes())?;
        let index_length = (index_bytes.len() + BLOCK_CRC_BYTES) as u32;
        self.offset += index_length as u64;

        // Bloom filter, sized against the actual key count. Tombstone keys go in
        // too: a lookup has to reach the tombstone to learn the key is deleted,
        // and a filter that omitted them would let the search fall through to an
        // older run and resurrect the key.
        let (bloom_offset, bloom_length) = if self.bloom_false_positive_rate > 0.0 {
            let mut filter = BloomFilter::with_capacity(
                self.key_hashes.len(),
                self.bloom_false_positive_rate,
            );
            for &hash in &self.key_hashes {
                filter.insert_hash(hash);
            }
            let bloom_bytes = filter.encode();
            let bloom_checksum = crc32(&bloom_bytes);
            let offset = self.offset;
            self.writer.write_all(&bloom_bytes)?;
            self.writer.write_all(&bloom_checksum.to_le_bytes())?;
            let length = (bloom_bytes.len() + BLOCK_CRC_BYTES) as u32;
            self.offset += length as u64;
            (offset, length)
        } else {
            // Zero length marks the filter absent; readers then check every
            // candidate block.
            (0, 0)
        };

        let footer = Footer {
            index_offset,
            index_length,
            bloom_offset,
            bloom_length,
            entry_count: self.entry_count,
        };
        self.writer.write_all(&footer.encode())?;
        self.offset += FOOTER_BYTES as u64;

        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;

        Ok(SSTableMeta {
            path: self.path,
            entry_count: self.entry_count,
            file_size_bytes: self.offset,
            block_count: self.index.len(),
        })
    }
}

// ---------------------------------------------------------------------------
// Footer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Footer {
    index_offset: u64,
    index_length: u32,
    bloom_offset: u64,
    bloom_length: u32,
    entry_count: u64,
}

impl Footer {
    /// 44 bytes: 32 of payload, a CRC over exactly those 32, then the magic.
    fn encode(&self) -> [u8; FOOTER_BYTES] {
        let mut buffer = [0u8; FOOTER_BYTES];
        buffer[0..8].copy_from_slice(&self.index_offset.to_le_bytes());
        buffer[8..12].copy_from_slice(&self.index_length.to_le_bytes());
        buffer[12..20].copy_from_slice(&self.bloom_offset.to_le_bytes());
        buffer[20..24].copy_from_slice(&self.bloom_length.to_le_bytes());
        buffer[24..32].copy_from_slice(&self.entry_count.to_le_bytes());
        let checksum = crc32(&buffer[0..32]);
        buffer[32..36].copy_from_slice(&checksum.to_le_bytes());
        buffer[36..44].copy_from_slice(&MAGIC.to_le_bytes());
        buffer
    }

    fn decode(buffer: &[u8; FOOTER_BYTES]) -> io::Result<Self> {
        let magic = u64::from_le_bytes(buffer[36..44].try_into().expect("8 bytes"));
        if magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not an SSTable, or written by an incompatible format version",
            ));
        }
        let expected = u32::from_le_bytes(buffer[32..36].try_into().expect("4 bytes"));
        if crc32(&buffer[0..32]) != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SSTable footer failed its checksum",
            ));
        }
        Ok(Self {
            index_offset: u64::from_le_bytes(buffer[0..8].try_into().expect("8 bytes")),
            index_length: u32::from_le_bytes(buffer[8..12].try_into().expect("4 bytes")),
            bloom_offset: u64::from_le_bytes(buffer[12..20].try_into().expect("8 bytes")),
            bloom_length: u32::from_le_bytes(buffer[20..24].try_into().expect("4 bytes")),
            entry_count: u64::from_le_bytes(buffer[24..32].try_into().expect("8 bytes")),
        })
    }
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// A read handle on an immutable SSTable.
///
/// Holds the sparse index in memory; data blocks are read on demand. Reads take
/// `&self` (via positioned reads that do not disturb a shared file cursor), so a
/// single open table can serve concurrent lookups later without interior
/// mutability.
#[derive(Debug)]
pub struct SSTable {
    file: File,
    path: PathBuf,
    index: Vec<IndexEntry>,
    entry_count: u64,
    file_size_bytes: u64,
    /// Largest key in the file. Cached at open by reading the final block, which
    /// avoids a variable-length field in the fixed-size footer.
    max_key: Option<Key>,
    /// Absent for tables written with the filter disabled.
    bloom: Option<BloomFilter>,
    /// Data blocks actually read from disk. This is the raw material for the
    /// read-amplification numbers in Phase 3, so it is instrumentation rather
    /// than debug scaffolding.
    blocks_read: AtomicU64,
    /// Lookups the bloom filter answered without any disk read.
    bloom_rejections: AtomicU64,
}

impl SSTable {
    /// Open an existing SSTable and load its index.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let file_size_bytes = file.metadata()?.len();

        if file_size_bytes < FOOTER_BYTES as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file is too short to contain an SSTable footer",
            ));
        }

        let mut footer_bytes = [0u8; FOOTER_BYTES];
        read_exact_at(
            &file,
            &mut footer_bytes,
            file_size_bytes - FOOTER_BYTES as u64,
        )?;
        let footer = Footer::decode(&footer_bytes)?;

        let index_bytes = read_checked_block(&file, footer.index_offset, footer.index_length)?;
        let index = decode_index(&index_bytes)?;

        // The filter is loaded once and held for the table's lifetime — that
        // resident cost is the whole point, since it buys disk reads back.
        let bloom = if footer.bloom_length > 0 {
            let bloom_bytes =
                read_checked_block(&file, footer.bloom_offset, footer.bloom_length)?;
            Some(BloomFilter::decode(&bloom_bytes)?)
        } else {
            None
        };

        let mut table = Self {
            file,
            path,
            index,
            entry_count: footer.entry_count,
            file_size_bytes,
            max_key: None,
            bloom,
            blocks_read: AtomicU64::new(0),
            bloom_rejections: AtomicU64::new(0),
        };
        table.max_key = match table.index.len() {
            0 => None,
            n => table.read_block(n - 1)?.pop().map(|(key, _)| key),
        };
        // That last read was setup, not a query. Zero the counters so they only
        // ever reflect lookup work.
        table.blocks_read.store(0, Ordering::Relaxed);
        Ok(table)
    }

    /// Look up `key` in this table only.
    ///
    /// As with the memtable, three outcomes matter: a live value, a tombstone
    /// (stop searching, report a miss), and absence (keep looking in older
    /// tables).
    pub fn get(&self, key: &[u8]) -> io::Result<Option<Value>> {
        // The filter's negative answer is exact, so this returns without
        // touching the disk. On a read-heavy workload over many runs, this is
        // the difference between one block read per lookup and one per run.
        if let Some(bloom) = &self.bloom {
            if !bloom.may_contain(key) {
                self.bloom_rejections.fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
        }

        let Some(block_index) = self.block_for_key(key) else {
            return Ok(None);
        };
        // One block read, then a linear scan of a few KiB — cheaper than the
        // extra index memory that a per-key index would cost.
        for (candidate, value) in self.read_block(block_index)? {
            match candidate.as_slice().cmp(key) {
                std::cmp::Ordering::Less => continue,
                std::cmp::Ordering::Equal => return Ok(Some(value)),
                // Entries ascend, so passing the key means it is absent.
                std::cmp::Ordering::Greater => return Ok(None),
            }
        }
        Ok(None)
    }

    /// Index of the only block that could contain `key`, if any.
    ///
    /// Blocks are identified by their first key, so the candidate is the last
    /// block whose first key is `<= key`. A key smaller than every block's first
    /// key cannot be in the file at all.
    fn block_for_key(&self, key: &[u8]) -> Option<usize> {
        let past = self
            .index
            .partition_point(|entry| entry.first_key.as_slice() <= key);
        past.checked_sub(1)
    }

    /// Read and verify data block `index`, decoding its entries.
    fn read_block(&self, index: usize) -> io::Result<Vec<(Key, Value)>> {
        let entry = &self.index[index];
        self.blocks_read.fetch_add(1, Ordering::Relaxed);
        let bytes = read_checked_block(&self.file, entry.offset, entry.length)?;
        decode_block(&bytes)
    }

    /// Every entry in ascending key order, tombstones included.
    pub fn iter(&self) -> SSTableIter<'_> {
        SSTableIter {
            table: self,
            next_block: 0,
            buffered: Vec::new().into_iter(),
        }
    }

    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }

    pub fn block_count(&self) -> usize {
        self.index.len()
    }

    pub fn file_size_bytes(&self) -> u64 {
        self.file_size_bytes
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Whether this table carries a bloom filter.
    pub fn has_bloom_filter(&self) -> bool {
        self.bloom.is_some()
    }

    /// Resident bytes of bloom filter, for the memory side of the Phase 3
    /// filter-sizing trade-off.
    pub fn bloom_bytes(&self) -> usize {
        self.bloom
            .as_ref()
            .map_or(0, |filter| (filter.num_bits() / 8) as usize)
    }

    /// Data blocks read from disk since the table was opened.
    pub fn blocks_read(&self) -> u64 {
        self.blocks_read.load(Ordering::Relaxed)
    }

    /// Lookups the bloom filter resolved with no disk read at all.
    pub fn bloom_rejections(&self) -> u64 {
        self.bloom_rejections.load(Ordering::Relaxed)
    }

    /// Zero the I/O counters, so a benchmark can measure one phase in isolation.
    pub fn reset_counters(&self) {
        self.blocks_read.store(0, Ordering::Relaxed);
        self.bloom_rejections.store(0, Ordering::Relaxed);
    }

    /// Smallest key in the file. With [`SSTable::max_key`] this gives the key
    /// range compaction uses to decide which tables overlap.
    pub fn min_key(&self) -> Option<&Key> {
        self.index.first().map(|entry| &entry.first_key)
    }

    pub fn max_key(&self) -> Option<&Key> {
        self.max_key.as_ref()
    }

    /// Whether this table's key range overlaps `other`'s. Non-overlapping
    /// tables can be moved between levels without a merge.
    pub fn overlaps(&self, other: &SSTable) -> bool {
        match (
            self.min_key(),
            self.max_key(),
            other.min_key(),
            other.max_key(),
        ) {
            (Some(self_min), Some(self_max), Some(other_min), Some(other_max)) => {
                self_min <= other_max && other_min <= self_max
            }
            _ => false,
        }
    }
}

/// Iterator over an SSTable's entries.
///
/// Yields `io::Result` because a block read can fail partway through; silently
/// ending iteration on an I/O error would let a compaction quietly drop data.
pub struct SSTableIter<'a> {
    table: &'a SSTable,
    next_block: usize,
    buffered: std::vec::IntoIter<(Key, Value)>,
}

impl Iterator for SSTableIter<'_> {
    type Item = io::Result<(Key, Value)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(entry) = self.buffered.next() {
                return Some(Ok(entry));
            }
            if self.next_block >= self.table.index.len() {
                return None;
            }
            match self.table.read_block(self.next_block) {
                Ok(entries) => {
                    self.next_block += 1;
                    self.buffered = entries.into_iter();
                }
                Err(error) => {
                    // Skip past the bad block so a caller that keeps polling
                    // does not spin on the same failure forever.
                    self.next_block += 1;
                    return Some(Err(error));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Decoding helpers
// ---------------------------------------------------------------------------

/// Read `length` bytes at `offset` and verify the CRC trailer, returning the
/// payload without it.
fn read_checked_block(file: &File, offset: u64, length: u32) -> io::Result<Vec<u8>> {
    if (length as usize) < BLOCK_CRC_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "block length is too small to hold a checksum",
        ));
    }
    let mut bytes = vec![0u8; length as usize];
    read_exact_at(file, &mut bytes, offset)?;

    let split = bytes.len() - BLOCK_CRC_BYTES;
    let expected = u32::from_le_bytes(bytes[split..].try_into().expect("4 bytes"));
    if crc32(&bytes[..split]) != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("SSTable block at offset {offset} failed its checksum"),
        ));
    }
    bytes.truncate(split);
    Ok(bytes)
}

fn decode_block(bytes: &[u8]) -> io::Result<Vec<(Key, Value)>> {
    let mut entries = Vec::new();
    let mut cursor = 0usize;

    while cursor < bytes.len() {
        if cursor + 9 > bytes.len() {
            return Err(malformed("truncated entry header"));
        }
        let tag = bytes[cursor];
        let key_len = u32::from_le_bytes(bytes[cursor + 1..cursor + 5].try_into().expect("4 bytes"));
        let value_len =
            u32::from_le_bytes(bytes[cursor + 5..cursor + 9].try_into().expect("4 bytes"));
        cursor += 9;

        if key_len > MAX_ENTRY_BYTES || value_len > MAX_ENTRY_BYTES {
            return Err(malformed("entry length exceeds the maximum"));
        }
        let key_end = cursor + key_len as usize;
        let value_end = key_end + value_len as usize;
        if value_end > bytes.len() {
            return Err(malformed("entry overruns its block"));
        }

        let key = bytes[cursor..key_end].to_vec();
        let value = match tag {
            TAG_PUT => Value::Put(bytes[key_end..value_end].to_vec()),
            TAG_TOMBSTONE if value_len == 0 => Value::Tombstone,
            _ => return Err(malformed("unrecognised entry tag")),
        };
        entries.push((key, value));
        cursor = value_end;
    }
    Ok(entries)
}

fn decode_index(bytes: &[u8]) -> io::Result<Vec<IndexEntry>> {
    let mut entries = Vec::new();
    let mut cursor = 0usize;

    while cursor < bytes.len() {
        if cursor + 4 > bytes.len() {
            return Err(malformed("truncated index entry"));
        }
        let key_len =
            u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().expect("4 bytes")) as usize;
        cursor += 4;

        let key_end = cursor + key_len;
        if key_end + 12 > bytes.len() {
            return Err(malformed("index entry overruns the index block"));
        }
        let first_key = bytes[cursor..key_end].to_vec();
        let offset = u64::from_le_bytes(bytes[key_end..key_end + 8].try_into().expect("8 bytes"));
        let length =
            u32::from_le_bytes(bytes[key_end + 8..key_end + 12].try_into().expect("4 bytes"));
        cursor = key_end + 12;

        entries.push(IndexEntry {
            first_key,
            offset,
            length,
        });
    }
    Ok(entries)
}

fn malformed(reason: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("malformed SSTable: {reason}"),
    )
}

/// Read exactly `buf.len()` bytes starting at `offset`, without using or
/// disturbing the file's cursor.
///
/// Positioned reads keep [`SSTable::get`] on `&self`. Unix and Windows spell
/// this differently and neither guarantees a full read in one call, hence the
/// loop.
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    use std::os::unix::fs::FileExt;
    #[cfg(windows)]
    use std::os::windows::fs::FileExt;

    let mut buf = buf;
    let mut offset = offset;
    while !buf.is_empty() {
        #[cfg(unix)]
        let read = file.read_at(buf, offset)?;
        #[cfg(windows)]
        let read = file.seek_read(buf, offset)?;

        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "SSTable ended before the requested bytes",
            ));
        }
        let rest = std::mem::take(&mut buf);
        buf = &mut rest[read..];
        offset += read as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::MemTable;

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(label: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static COUNTER: AtomicU32 = AtomicU32::new(0);

            let unique = format!(
                "vectordb-sst-{label}-{}-{}",
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

    /// Keys that sort lexicographically in the same order as their index, so
    /// tests can reason about ordering without surprises.
    fn numbered_key(i: usize) -> Key {
        format!("key{i:06}").into_bytes()
    }

    fn write_table(path: &Path, entries: &[(Key, Value)], block_size: usize) -> SSTableMeta {
        let mut writer =
            SSTableWriter::create_with_block_size(path, block_size).expect("create writer");
        for (key, value) in entries {
            writer.append(key, value).expect("append");
        }
        writer.finish().expect("finish")
    }

    #[test]
    fn round_trips_a_single_block() {
        let dir = TempDir::new("single");
        let path = dir.file("test.sst");
        let entries = vec![
            (k("alpha"), put("1")),
            (k("bravo"), Value::Tombstone),
            (k("charlie"), put("3")),
        ];

        let meta = write_table(&path, &entries, DEFAULT_BLOCK_TARGET_BYTES);
        assert_eq!(meta.entry_count, 3);
        assert_eq!(meta.block_count, 1);

        let table = SSTable::open(&path).expect("open");
        let read: Vec<_> = table.iter().map(|r| r.expect("read entry")).collect();
        assert_eq!(read, entries);
    }

    #[test]
    fn tombstones_survive_the_round_trip() {
        let dir = TempDir::new("tombstone");
        let path = dir.file("test.sst");
        write_table(
            &path,
            &[
                (k("deleted"), Value::Tombstone),
                (k("empty-value"), Value::Put(Vec::new())),
            ],
            DEFAULT_BLOCK_TARGET_BYTES,
        );

        let table = SSTable::open(&path).expect("open");
        assert_eq!(table.get(b"deleted").expect("get"), Some(Value::Tombstone));
        assert_eq!(
            table.get(b"empty-value").expect("get"),
            Some(Value::Put(Vec::new())),
            "an empty value must not be confused with a tombstone on disk"
        );
    }

    #[test]
    fn point_lookups_work_across_many_blocks() {
        let dir = TempDir::new("multiblock");
        let path = dir.file("test.sst");

        let entries: Vec<(Key, Value)> = (0..500)
            .map(|i| (numbered_key(i), Value::Put(format!("value-{i}").into_bytes())))
            .collect();
        // A small block size forces many blocks, exercising the index search.
        let meta = write_table(&path, &entries, 256);
        assert!(
            meta.block_count > 20,
            "expected many blocks, got {}",
            meta.block_count
        );

        let table = SSTable::open(&path).expect("open");
        assert_eq!(table.entry_count(), 500);

        for (key, expected) in &entries {
            assert_eq!(
                table.get(key).expect("get").as_ref(),
                Some(expected),
                "lookup failed for {:?}",
                String::from_utf8_lossy(key)
            );
        }
    }

    #[test]
    fn absent_keys_return_none_everywhere_in_the_range() {
        let dir = TempDir::new("absent");
        let path = dir.file("test.sst");

        // Even keys only, so every odd key is a miss.
        let entries: Vec<(Key, Value)> = (0..200)
            .filter(|i| i % 2 == 0)
            .map(|i| (numbered_key(i), put("v")))
            .collect();
        write_table(&path, &entries, 128);

        let table = SSTable::open(&path).expect("open");
        for i in (1..200).step_by(2) {
            assert_eq!(
                table.get(&numbered_key(i)).expect("get"),
                None,
                "odd key {i} should be absent"
            );
        }
        // Before the first key and after the last: the two boundary misses.
        assert_eq!(table.get(b"aaaa").expect("get"), None);
        assert_eq!(table.get(b"zzzz").expect("get"), None);
    }

    #[test]
    fn iteration_returns_every_entry_in_order() {
        let dir = TempDir::new("iterate");
        let path = dir.file("test.sst");

        let entries: Vec<(Key, Value)> = (0..300)
            .map(|i| {
                let value = if i % 7 == 0 {
                    Value::Tombstone
                } else {
                    Value::Put(vec![b'x'; i % 50])
                };
                (numbered_key(i), value)
            })
            .collect();
        write_table(&path, &entries, 200);

        let table = SSTable::open(&path).expect("open");
        let read: Vec<_> = table.iter().map(|r| r.expect("read entry")).collect();
        assert_eq!(read, entries);
    }

    #[test]
    fn key_range_is_reported_for_overlap_checks() {
        let dir = TempDir::new("range");

        let left = dir.file("left.sst");
        write_table(
            &left,
            &[(k("a"), put("1")), (k("m"), put("2"))],
            DEFAULT_BLOCK_TARGET_BYTES,
        );
        let right = dir.file("right.sst");
        write_table(
            &right,
            &[(k("n"), put("3")), (k("z"), put("4"))],
            DEFAULT_BLOCK_TARGET_BYTES,
        );
        let straddling = dir.file("straddling.sst");
        write_table(
            &straddling,
            &[(k("f"), put("5")), (k("t"), put("6"))],
            DEFAULT_BLOCK_TARGET_BYTES,
        );

        let left = SSTable::open(&left).expect("open");
        let right = SSTable::open(&right).expect("open");
        let straddling = SSTable::open(&straddling).expect("open");

        assert_eq!(left.min_key(), Some(&k("a")));
        assert_eq!(left.max_key(), Some(&k("m")));

        assert!(!left.overlaps(&right), "a..m and n..z must not overlap");
        assert!(left.overlaps(&straddling));
        assert!(straddling.overlaps(&right));
        assert!(left.overlaps(&left));
    }

    /// The max key lives at the end of the final block, so it must survive a
    /// multi-block file rather than reporting the last block's *first* key.
    #[test]
    fn max_key_comes_from_the_end_of_the_last_block() {
        let dir = TempDir::new("maxkey");
        let path = dir.file("test.sst");

        let entries: Vec<(Key, Value)> = (0..100).map(|i| (numbered_key(i), put("v"))).collect();
        let meta = write_table(&path, &entries, 64);
        assert!(meta.block_count > 1);

        let table = SSTable::open(&path).expect("open");
        assert_eq!(table.min_key(), Some(&numbered_key(0)));
        assert_eq!(table.max_key(), Some(&numbered_key(99)));
    }

    #[test]
    fn out_of_order_appends_are_rejected() {
        let dir = TempDir::new("ordering");
        let path = dir.file("test.sst");

        let mut writer = SSTableWriter::create(&path).expect("create");
        writer.append(b"bravo", &put("1")).expect("append");

        let backwards = writer.append(b"alpha", &put("2"));
        assert!(backwards.is_err(), "a descending key must be rejected");

        let duplicate = writer.append(b"bravo", &put("3"));
        assert!(
            duplicate.is_err(),
            "a duplicate key must be rejected: the flush path is responsible for \
             collapsing duplicates before they reach the writer"
        );
    }

    #[test]
    fn an_empty_sstable_is_readable() {
        let dir = TempDir::new("empty");
        let path = dir.file("test.sst");

        let meta = write_table(&path, &[], DEFAULT_BLOCK_TARGET_BYTES);
        assert_eq!(meta.entry_count, 0);
        assert_eq!(meta.block_count, 0);

        let table = SSTable::open(&path).expect("open");
        assert_eq!(table.get(b"anything").expect("get"), None);
        assert_eq!(table.iter().count(), 0);
        assert_eq!(table.min_key(), None);
        assert_eq!(table.max_key(), None);
    }

    #[test]
    fn large_values_get_their_own_blocks() {
        let dir = TempDir::new("large");
        let path = dir.file("test.sst");

        // A GIST vector is 3840 bytes: comfortably larger than one 256-byte
        // block, so entries must never be split across blocks.
        let big = vec![0xABu8; 4000];
        let entries = vec![
            (k("a"), Value::Put(big.clone())),
            (k("b"), put("small")),
            (k("c"), Value::Put(big.clone())),
        ];
        write_table(&path, &entries, 256);

        let table = SSTable::open(&path).expect("open");
        assert_eq!(table.get(b"a").expect("get"), Some(Value::Put(big.clone())));
        assert_eq!(table.get(b"c").expect("get"), Some(Value::Put(big)));
        assert_eq!(table.get(b"b").expect("get"), Some(put("small")));
    }

    #[test]
    fn a_memtable_flushes_into_a_readable_sstable() {
        let dir = TempDir::new("flush");
        let path = dir.file("test.sst");

        let mut memtable = MemTable::new();
        memtable.put(k("charlie"), b"3".to_vec());
        memtable.put(k("alpha"), b"1".to_vec());
        memtable.delete(k("bravo"));
        memtable.put(k("alpha"), b"overwritten".to_vec());

        let mut writer = SSTableWriter::create(&path).expect("create");
        for (key, value) in memtable.into_entries() {
            writer.append(&key, &value).expect("append");
        }
        writer.finish().expect("finish");

        let table = SSTable::open(&path).expect("open");
        assert_eq!(
            table.get(b"alpha").expect("get"),
            Some(Value::Put(b"overwritten".to_vec()))
        );
        assert_eq!(table.get(b"bravo").expect("get"), Some(Value::Tombstone));
        assert_eq!(table.get(b"charlie").expect("get"), Some(put("3")));
    }

    #[test]
    fn a_truncated_file_is_rejected_rather_than_half_read() {
        let dir = TempDir::new("truncated");
        let path = dir.file("test.sst");

        let entries: Vec<(Key, Value)> = (0..100).map(|i| (numbered_key(i), put("v"))).collect();
        write_table(&path, &entries, 128);

        // Simulate a crash during the flush: the footer never made it to disk.
        let full = std::fs::metadata(&path).expect("metadata").len();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open")
            .set_len(full - 10)
            .expect("truncate");

        assert!(
            SSTable::open(&path).is_err(),
            "a footerless file must be rejected: adopting a partial SSTable into \
             the tree would silently lose the missing entries"
        );
    }

    #[test]
    fn block_corruption_is_caught_by_the_checksum() {
        let dir = TempDir::new("corrupt");
        let path = dir.file("test.sst");

        let entries: Vec<(Key, Value)> = (0..50)
            .map(|i| (numbered_key(i), Value::Put(vec![b'v'; 20])))
            .collect();
        write_table(&path, &entries, 256);

        let mut bytes = std::fs::read(&path).expect("read");
        bytes[30] ^= 0b0001_0000;
        std::fs::write(&path, &bytes).expect("write");

        let table = SSTable::open(&path).expect("open: the footer and index are intact");
        let first_block_read = table.get(&numbered_key(0));
        assert!(
            first_block_read.is_err(),
            "a corrupted block must surface as an error, not as wrong data"
        );
    }

    #[test]
    fn a_corrupt_footer_is_rejected() {
        let dir = TempDir::new("bad-footer");
        let path = dir.file("test.sst");
        write_table(&path, &[(k("a"), put("1"))], DEFAULT_BLOCK_TARGET_BYTES);

        let mut bytes = std::fs::read(&path).expect("read");
        let len = bytes.len();
        // Corrupt the index offset, inside the footer's checksummed region.
        bytes[len - FOOTER_BYTES] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("write");

        assert!(SSTable::open(&path).is_err());
    }

    /// The payoff: a lookup for a key that is not in the table should almost
    /// never touch the disk.
    #[test]
    fn the_bloom_filter_eliminates_most_reads_for_absent_keys() {
        let dir = TempDir::new("bloom-saves-io");
        let path = dir.file("test.sst");

        // Even keys only. The probes below are the odd keys, which are absent
        // but sit *between* present keys — so the sparse index cannot rule them
        // out and only the filter can prevent a block read. Probing keys outside
        // the table's range instead would make this test pass with no filter at
        // all.
        let entries: Vec<(Key, Value)> = (0..2000).map(|i| (numbered_key(i * 2), put("v"))).collect();
        write_table(&path, &entries, 512);

        let table = SSTable::open(&path).expect("open");
        assert!(table.has_bloom_filter());
        assert_eq!(table.blocks_read(), 0, "open must not count as query I/O");

        let probes = 2000usize;
        for i in 0..probes {
            let absent = numbered_key(i * 2 + 1);
            assert_eq!(table.get(&absent).expect("get"), None);
        }

        // At a 1% target rate, ~20 of 2000 probes should slip through and cost a
        // block read. Allow generous slack for statistical noise while still
        // failing loudly if the filter is not being consulted at all.
        assert!(
            table.blocks_read() < (probes / 10) as u64,
            "expected the filter to absorb nearly all {probes} probes, but {} block \
             reads happened",
            table.blocks_read()
        );
        assert!(table.bloom_rejections() > (probes * 9 / 10) as u64);
    }

    /// Correctness must not depend on the filter. With it disabled, every
    /// lookup still returns exactly the same answers — just with more I/O.
    #[test]
    fn results_are_identical_with_and_without_the_filter() {
        let dir = TempDir::new("bloom-parity");
        let entries: Vec<(Key, Value)> = (0..400)
            .map(|i| {
                let value = if i % 5 == 0 {
                    Value::Tombstone
                } else {
                    Value::Put(format!("v{i}").into_bytes())
                };
                (numbered_key(i * 2), value)
            })
            .collect();

        let with_filter = dir.file("with.sst");
        write_table(&with_filter, &entries, 256);

        let without_filter = dir.file("without.sst");
        let mut writer = SSTableWriter::create_with_block_size(&without_filter, 256)
            .expect("create")
            .with_bloom_false_positive_rate(0.0);
        for (key, value) in &entries {
            writer.append(key, value).expect("append");
        }
        writer.finish().expect("finish");

        let with = SSTable::open(&with_filter).expect("open");
        let without = SSTable::open(&without_filter).expect("open");
        assert!(with.has_bloom_filter());
        assert!(!without.has_bloom_filter());

        // Probe present keys, absent keys, and tombstoned keys alike.
        for i in 0..900 {
            let key = numbered_key(i);
            assert_eq!(
                with.get(&key).expect("get"),
                without.get(&key).expect("get"),
                "filter changed the answer for {:?}",
                String::from_utf8_lossy(&key)
            );
        }
        assert!(
            with.blocks_read() < without.blocks_read(),
            "the filter should have saved reads: {} vs {}",
            with.blocks_read(),
            without.blocks_read()
        );
    }

    /// A tombstone must be findable through the filter. If deleted keys were
    /// left out, a lookup would skip the tombstone, fall through to an older
    /// run, and resurrect the key.
    #[test]
    fn tombstoned_keys_are_present_in_the_filter() {
        let dir = TempDir::new("bloom-tombstones");
        let path = dir.file("test.sst");

        let entries: Vec<(Key, Value)> = (0..300)
            .map(|i| (numbered_key(i), Value::Tombstone))
            .collect();
        write_table(&path, &entries, 256);

        let table = SSTable::open(&path).expect("open");
        for i in 0..300 {
            assert_eq!(
                table.get(&numbered_key(i)).expect("get"),
                Some(Value::Tombstone),
                "tombstone for key {i} was filtered out"
            );
        }
        assert_eq!(
            table.bloom_rejections(),
            0,
            "no inserted key may be rejected by the filter"
        );
    }

    #[test]
    fn a_foreign_file_is_not_mistaken_for_an_sstable() {
        let dir = TempDir::new("foreign");
        let path = dir.file("not-an-sstable.bin");
        std::fs::write(&path, vec![0x42u8; 500]).expect("write");

        assert!(SSTable::open(&path).is_err());
    }
}
