//! Key–value separation: an append-only log holding values the tree only points at.
//!
//! # Why this exists
//!
//! Phase 3 measured write amplification rising from **3.50× at 100 B values to
//! 22.65× at 3,840 B** — the size of a GIST vector — with p99 latency up 360×
//! (`results/README.md`). The cause is structural: every compaction rewrites the
//! *whole* entry, so a byte of value pays the full `T+1` rewrite cost at every
//! level it descends, however many times.
//!
//! None of this project's source papers address it. They compare growth schemes
//! and merge policies on ~100 B values, where the key and its bookkeeping dominate
//! and value size is noise. At vector sizes the ordering reverses, and no choice
//! of policy recovers the difference — which is the finding that motivates this
//! module.
//!
//! The fix is WiscKey's: write the value once to an append-only log and store only
//! a pointer in the tree. Compaction then rewrites pointers, not payloads, so a
//! value is written once no matter how far its key travels.
//!
//! # What it costs
//!
//! Three things, all real:
//!
//! - **An extra read.** A lookup that used to end at the SSTable block now
//!   follows a pointer into the log. Bloom filters and the sparse index still
//!   prune, so this is one extra positioned read on the hits only.
//! - **Space amplification.** Overwriting or deleting a key orphans its log
//!   record, and nothing here reclaims it — see the note on garbage collection
//!   below. Write amplification improves at the cost of space.
//! - **Scan locality.** Values are laid out in *insertion* order, not key order,
//!   so a range scan that reads values walks the log randomly. Point lookups are
//!   unaffected. This is WiscKey's known weakness and it is not fixed here.
//!
//! # Garbage collection — deliberately absent, and labelled
//!
//! WiscKey reclaims space by scanning the log tail, checking each record's key
//! against the tree, and re-appending the ones still live. That is not implemented
//! here. The consequence is stated rather than hidden: a workload that overwrites
//! keys grows the log without bound, so **space amplification is unbounded under
//! updates** and the benchmark below reports it. Adding GC would not change the
//! write-amplification result, which is what this module exists to measure.
//!
//! # Record format
//!
//! ```text
//! [crc32: u32][len: u32][payload: len bytes]
//! ```
//!
//! The CRC covers the length and the payload, so a torn or corrupted record is
//! detectable rather than silently returning wrong bytes. Recovery truncates a
//! torn tail before appending, the same discipline [`super::wal`] uses — without
//! it, a crash mid-append would strand every later write behind an unreadable
//! record.

use std::fs::{File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use super::crc32::crc32;

/// Bytes of framing per record: the CRC and the length.
const HEADER_BYTES: u64 = 8;

/// Where a value lives in the log.
///
/// Stored in the tree in place of the payload. Twelve bytes on the wire against a
/// 3,840-byte GIST vector, which is the whole point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValuePointer {
    /// Offset of the record header, not of the payload.
    pub offset: u64,
    /// Payload length in bytes.
    pub len: u32,
}

impl ValuePointer {
    /// Bytes of a serialised pointer.
    pub const ENCODED_BYTES: usize = 12;

    pub fn encode(&self) -> [u8; Self::ENCODED_BYTES] {
        let mut out = [0u8; Self::ENCODED_BYTES];
        out[..8].copy_from_slice(&self.offset.to_le_bytes());
        out[8..].copy_from_slice(&self.len.to_le_bytes());
        out
    }

    /// Returns `None` if the slice is the wrong length.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != Self::ENCODED_BYTES {
            return None;
        }
        let mut offset = [0u8; 8];
        offset.copy_from_slice(&bytes[..8]);
        let mut len = [0u8; 4];
        len.copy_from_slice(&bytes[8..]);
        Some(Self {
            offset: u64::from_le_bytes(offset),
            len: u32::from_le_bytes(len),
        })
    }
}

/// An append-only log of values.
#[derive(Debug)]
pub struct ValueLog {
    file: File,
    path: PathBuf,
    /// Where the next record will be written; also the live size of the log.
    tail: u64,
    /// Payload bytes appended over this handle's lifetime, for amplification
    /// accounting. Excludes framing so it is comparable with the user's byte
    /// count.
    payload_written: u64,
}

impl ValueLog {
    /// Open or create the log, truncating any torn tail.
    ///
    /// Scans forward validating each record. The first one that fails to verify
    /// marks the end of the intact prefix, and the file is physically truncated
    /// there before any append — so a later crash cannot leave a valid record
    /// stranded behind a corrupt one.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        let size = file.metadata()?.len();
        let intact = Self::scan_intact_prefix(&file, size)?;
        if intact < size {
            file.set_len(intact)?;
            file.sync_all()?;
        }
        file.seek(SeekFrom::Start(intact))?;

        Ok(Self {
            file,
            path,
            tail: intact,
            payload_written: 0,
        })
    }

    /// The offset just past the last fully valid record.
    fn scan_intact_prefix(file: &File, size: u64) -> io::Result<u64> {
        let mut offset = 0u64;
        loop {
            if offset + HEADER_BYTES > size {
                return Ok(offset);
            }
            let mut header = [0u8; HEADER_BYTES as usize];
            if super::sstable::read_exact_at(file, &mut header, offset).is_err() {
                return Ok(offset);
            }
            let expected = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
            let len = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);

            let end = offset + HEADER_BYTES + u64::from(len);
            if end > size {
                // The payload was never fully written.
                return Ok(offset);
            }

            let mut payload = vec![0u8; len as usize];
            if super::sstable::read_exact_at(file, &mut payload, offset + HEADER_BYTES).is_err() {
                return Ok(offset);
            }
            let mut check = Vec::with_capacity(4 + payload.len());
            check.extend_from_slice(&header[4..]);
            check.extend_from_slice(&payload);
            if crc32(&check) != expected {
                return Ok(offset);
            }
            offset = end;
        }
    }

    /// Append a value, returning where it landed.
    pub fn append(&mut self, value: &[u8]) -> io::Result<ValuePointer> {
        let len = u32::try_from(value.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("value of {} bytes exceeds the 4 GiB record limit", value.len()),
            )
        })?;

        let mut check = Vec::with_capacity(4 + value.len());
        check.extend_from_slice(&len.to_le_bytes());
        check.extend_from_slice(value);
        let crc = crc32(&check);

        let offset = self.tail;
        let mut record = Vec::with_capacity(HEADER_BYTES as usize + value.len());
        record.extend_from_slice(&crc.to_le_bytes());
        record.extend_from_slice(&len.to_le_bytes());
        record.extend_from_slice(value);
        self.file.write_all(&record)?;

        self.tail += HEADER_BYTES + u64::from(len);
        self.payload_written += u64::from(len);
        Ok(ValuePointer { offset, len })
    }

    /// Read the value a pointer names.
    ///
    /// Verifies the CRC, so a pointer into a corrupted or overwritten region
    /// fails loudly rather than returning plausible-looking bytes.
    pub fn read(&self, pointer: ValuePointer) -> io::Result<Vec<u8>> {
        let mut header = [0u8; HEADER_BYTES as usize];
        super::sstable::read_exact_at(&self.file, &mut header, pointer.offset)?;
        let expected = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        let len = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);

        if len != pointer.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "value log record at {} is {len} bytes, pointer says {}",
                    pointer.offset, pointer.len
                ),
            ));
        }

        let mut payload = vec![0u8; len as usize];
        super::sstable::read_exact_at(&self.file, &mut payload, pointer.offset + HEADER_BYTES)?;

        let mut check = Vec::with_capacity(4 + payload.len());
        check.extend_from_slice(&header[4..]);
        check.extend_from_slice(&payload);
        if crc32(&check) != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("value log record at {} failed its checksum", pointer.offset),
            ));
        }
        Ok(payload)
    }

    /// Flush buffered writes to the operating system, then to the device.
    pub fn sync(&mut self) -> io::Result<()> {
        self.file.flush()?;
        self.file.sync_all()
    }

    /// Total bytes on disk, framing included.
    pub fn disk_bytes(&self) -> u64 {
        self.tail
    }

    /// Payload bytes appended over this handle's lifetime, framing excluded.
    ///
    /// This is the numerator that makes key–value separation measurable: with
    /// separation on, a value contributes to this exactly once no matter how many
    /// compactions its key survives.
    pub fn payload_written(&self) -> u64 {
        self.payload_written
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::BenchDir;

    fn log(dir: &BenchDir) -> ValueLog {
        ValueLog::open(dir.path().join("values.log")).expect("open")
    }

    #[test]
    fn a_value_round_trips() {
        let dir = BenchDir::new("vlog-roundtrip").expect("dir");
        let mut vlog = log(&dir);
        let pointer = vlog.append(b"hello world").expect("append");
        assert_eq!(vlog.read(pointer).expect("read"), b"hello world");
    }

    #[test]
    fn many_values_round_trip_in_any_order() {
        let dir = BenchDir::new("vlog-many").expect("dir");
        let mut vlog = log(&dir);
        let values: Vec<Vec<u8>> = (0..200u32)
            .map(|i| vec![(i % 251) as u8; (i as usize % 97) + 1])
            .collect();
        let pointers: Vec<ValuePointer> = values
            .iter()
            .map(|v| vlog.append(v).expect("append"))
            .collect();

        // Read back out of order, since that is how the tree will use it.
        for index in (0..values.len()).rev() {
            assert_eq!(
                vlog.read(pointers[index]).expect("read"),
                values[index],
                "value {index}"
            );
        }
    }

    #[test]
    fn an_empty_value_round_trips() {
        let dir = BenchDir::new("vlog-empty").expect("dir");
        let mut vlog = log(&dir);
        let pointer = vlog.append(b"").expect("append");
        assert_eq!(pointer.len, 0);
        assert!(vlog.read(pointer).expect("read").is_empty());
    }

    #[test]
    fn pointers_encode_and_decode() {
        let pointer = ValuePointer {
            offset: 0xDEAD_BEEF_1234,
            len: 3840,
        };
        let encoded = pointer.encode();
        assert_eq!(encoded.len(), ValuePointer::ENCODED_BYTES);
        assert_eq!(ValuePointer::decode(&encoded), Some(pointer));
        assert_eq!(ValuePointer::decode(&encoded[..11]), None);
    }

    #[test]
    fn a_pointer_is_far_smaller_than_the_value_it_replaces() {
        // The whole premise: this ratio is what compaction stops rewriting.
        let dir = BenchDir::new("vlog-ratio").expect("dir");
        let mut vlog = log(&dir);
        let gist = vec![0u8; 3840];
        let pointer = vlog.append(&gist).expect("append");
        let shrink = gist.len() as f64 / pointer.encode().len() as f64;
        assert!(
            shrink > 100.0,
            "a pointer should be orders of magnitude smaller, got {shrink:.0}×"
        );
    }

    #[test]
    fn values_survive_a_reopen() {
        let dir = BenchDir::new("vlog-reopen").expect("dir");
        let path = dir.path().join("values.log");
        let pointers: Vec<ValuePointer> = {
            let mut vlog = ValueLog::open(&path).expect("open");
            let pointers = (0..50u32)
                .map(|i| vlog.append(&[i as u8; 64]).expect("append"))
                .collect();
            vlog.sync().expect("sync");
            pointers
        };

        let vlog = ValueLog::open(&path).expect("reopen");
        for (index, pointer) in pointers.iter().enumerate() {
            assert_eq!(vlog.read(*pointer).expect("read"), vec![index as u8; 64]);
        }
    }

    #[test]
    fn appending_after_a_reopen_continues_the_log() {
        let dir = BenchDir::new("vlog-append-after").expect("dir");
        let path = dir.path().join("values.log");
        let first = {
            let mut vlog = ValueLog::open(&path).expect("open");
            let p = vlog.append(b"before").expect("append");
            vlog.sync().expect("sync");
            p
        };

        let mut vlog = ValueLog::open(&path).expect("reopen");
        let second = vlog.append(b"after").expect("append");
        assert!(second.offset > first.offset, "the tail should advance");
        assert_eq!(vlog.read(first).expect("read"), b"before");
        assert_eq!(vlog.read(second).expect("read"), b"after");
    }

    #[test]
    fn a_torn_tail_is_truncated_and_earlier_records_survive() {
        // The WAL discipline: a half-written record must not strand what follows.
        let dir = BenchDir::new("vlog-torn").expect("dir");
        let path = dir.path().join("values.log");
        let good = {
            let mut vlog = ValueLog::open(&path).expect("open");
            let p = vlog.append(b"intact record").expect("append");
            vlog.sync().expect("sync");
            p
        };

        // Append a header claiming more payload than actually follows.
        {
            let mut file = OpenOptions::new().append(true).open(&path).expect("open");
            file.write_all(&0u32.to_le_bytes()).expect("crc");
            file.write_all(&9999u32.to_le_bytes()).expect("len");
            file.write_all(b"short").expect("payload");
            file.sync_all().expect("sync");
        }

        let mut vlog = ValueLog::open(&path).expect("reopen");
        assert_eq!(vlog.read(good).expect("read"), b"intact record");
        // And the log is usable again: the next append lands on the truncated tail.
        let next = vlog.append(b"after recovery").expect("append");
        assert_eq!(vlog.read(next).expect("read"), b"after recovery");
    }

    #[test]
    fn a_corrupted_payload_is_detected_rather_than_returned() {
        let dir = BenchDir::new("vlog-corrupt").expect("dir");
        let path = dir.path().join("values.log");
        let pointer = {
            let mut vlog = ValueLog::open(&path).expect("open");
            let p = vlog.append(b"original contents").expect("append");
            vlog.sync().expect("sync");
            p
        };

        // Flip a byte inside the payload, leaving the header intact.
        {
            use std::io::Read;
            let mut bytes = Vec::new();
            File::open(&path)
                .expect("open")
                .read_to_end(&mut bytes)
                .expect("read");
            let target = HEADER_BYTES as usize + 3;
            bytes[target] ^= 0xFF;
            std::fs::write(&path, &bytes).expect("write");
        }

        // `open` scans and truncates the now-invalid record, so the pointer no
        // longer addresses anything — the corruption cannot be read as data.
        let vlog = ValueLog::open(&path).expect("reopen");
        assert!(
            vlog.read(pointer).is_err(),
            "a corrupted record must not be returned as a value"
        );
    }

    #[test]
    fn a_pointer_with_the_wrong_length_is_rejected() {
        let dir = BenchDir::new("vlog-badptr").expect("dir");
        let mut vlog = log(&dir);
        let mut pointer = vlog.append(b"twelve bytes").expect("append");
        pointer.len += 1;
        assert!(vlog.read(pointer).is_err());
    }

    #[test]
    fn payload_written_counts_each_value_once() {
        // The property the whole module exists for: a value is written once,
        // regardless of what the tree above does with its key afterwards.
        let dir = BenchDir::new("vlog-accounting").expect("dir");
        let mut vlog = log(&dir);
        for _ in 0..10 {
            vlog.append(&vec![7u8; 3840]).expect("append");
        }
        assert_eq!(vlog.payload_written(), 10 * 3840);
        // Framing is excluded from the payload count but present on disk.
        assert_eq!(
            vlog.disk_bytes(),
            10 * (3840 + HEADER_BYTES),
            "disk includes framing"
        );
    }

    #[test]
    fn framing_overhead_is_negligible_at_vector_sizes() {
        let dir = BenchDir::new("vlog-framing").expect("dir");
        let mut vlog = log(&dir);
        vlog.append(&vec![0u8; 3840]).expect("append");
        let overhead = vlog.disk_bytes() as f64 / 3840.0 - 1.0;
        assert!(
            overhead < 0.01,
            "framing is {overhead:.4} of a GIST vector, expected under 1%"
        );
    }
}
