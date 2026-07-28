//! Readers for the `.fvecs` / `.ivecs` / `.bvecs` format used by the standard
//! ANN benchmark datasets (SIFT, GIST, from <http://corpus-texmex.irisa.fr/>).
//!
//! # Format
//!
//! A flat sequence of records, each of which is its own header:
//!
//! ```text
//!   [i32 dimension][dimension × element]
//! ```
//!
//! Little-endian, elements being `f32` for `.fvecs`, `i32` for `.ivecs`, and
//! `u8` for `.bvecs`. The dimension is repeated in *every* record, which is
//! redundant but makes the file self-describing and gives a cheap consistency
//! check: a file whose records disagree on dimension is corrupt or is not what
//! it claims to be.
//!
//! # Why read these rather than convert
//!
//! The ground-truth files (`*_groundtruth.ivecs`) are published alongside the
//! vectors, so recall can be measured against the community's answers as well as
//! against our own brute-force scan. Two independent oracles is meaningfully
//! better than one, and it costs a few dozen lines.

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

/// A collection of equal-length vectors, stored flat.
#[derive(Debug, Clone, Default)]
pub struct VectorSet {
    pub dimension: usize,
    /// `count · dimension` values; vector `i` occupies `[i·d, (i+1)·d)`.
    pub data: Vec<f32>,
}

impl VectorSet {
    pub fn count(&self) -> usize {
        self.data.len().checked_div(self.dimension).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    pub fn get(&self, index: usize) -> Option<&[f32]> {
        let start = index.checked_mul(self.dimension)?;
        self.data.get(start..start + self.dimension)
    }

    /// Copy out each vector, for callers that want owned rows.
    pub fn rows(&self) -> Vec<Vec<f32>> {
        (0..self.count())
            .filter_map(|index| self.get(index).map(<[f32]>::to_vec))
            .collect()
    }
}

/// Guard against a corrupt or misidentified file demanding an enormous
/// allocation before anything has been validated.
const MAX_DIMENSION: i32 = 1 << 20;

/// Read a `.fvecs` file of 32-bit float vectors.
pub fn read_fvecs(path: impl AsRef<Path>) -> io::Result<VectorSet> {
    read_records(path, |bytes| {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    })
}

/// Read an `.ivecs` file, as used for ground-truth neighbour ids.
///
/// Returns the ids per query rather than a [`VectorSet`], since that is what
/// callers do with them.
pub fn read_ivecs(path: impl AsRef<Path>) -> io::Result<Vec<Vec<u32>>> {
    let set = read_records(path, |bytes| {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32
    })?;
    Ok((0..set.count())
        .map(|index| {
            set.get(index)
                .unwrap_or(&[])
                .iter()
                .map(|&value| value as u32)
                .collect()
        })
        .collect())
}

/// Shared record loop. `decode` turns four little-endian bytes into a value.
fn read_records(path: impl AsRef<Path>, decode: fn(&[u8]) -> f32) -> io::Result<VectorSet> {
    let path = path.as_ref();
    let mut reader = BufReader::new(File::open(path)?);

    let mut dimension: Option<usize> = None;
    let mut data = Vec::new();
    let mut header = [0u8; 4];

    loop {
        match fill(&mut reader, &mut header)? {
            0 => break,
            4 => {}
            partial => {
                return Err(malformed(&format!(
                    "{} ends with {partial} stray bytes, so it is truncated",
                    path.display()
                )))
            }
        }

        let declared = i32::from_le_bytes(header);
        if declared <= 0 || declared > MAX_DIMENSION {
            return Err(malformed(&format!(
                "{} declares a dimension of {declared}, which is not a vector file",
                path.display()
            )));
        }
        let declared = declared as usize;

        match dimension {
            None => dimension = Some(declared),
            // Every record repeats the dimension, so a disagreement means the
            // file is corrupt or is not the format it claims.
            Some(expected) if expected != declared => {
                return Err(malformed(&format!(
                    "{} mixes {expected}- and {declared}-dimensional records",
                    path.display()
                )))
            }
            Some(_) => {}
        }

        let mut payload = vec![0u8; declared * 4];
        if fill(&mut reader, &mut payload)? < payload.len() {
            return Err(malformed(&format!(
                "{} ends mid-vector",
                path.display()
            )));
        }
        data.extend(payload.chunks_exact(4).map(decode));
    }

    Ok(VectorSet {
        dimension: dimension.unwrap_or(0),
        data,
    })
}

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

fn malformed(reason: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, reason.to_string())
}

/// Write a `.fvecs` file. Used by the tests, and for exporting a subset.
pub fn write_fvecs(path: impl AsRef<Path>, dimension: usize, data: &[f32]) -> io::Result<()> {
    use std::io::Write;

    assert!(dimension > 0, "vectors need at least one dimension");
    assert_eq!(
        data.len() % dimension,
        0,
        "{} values do not divide into {dimension}-dimensional vectors",
        data.len()
    );

    let mut file = std::io::BufWriter::new(File::create(path)?);
    for vector in data.chunks_exact(dimension) {
        file.write_all(&(dimension as i32).to_le_bytes())?;
        for value in vector {
            file.write_all(&value.to_le_bytes())?;
        }
    }
    file.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir {
        path: std::path::PathBuf,
    }

    impl TempDir {
        fn new(label: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static COUNTER: AtomicU32 = AtomicU32::new(0);

            let path = std::env::temp_dir().join(format!(
                "vectordb-fvecs-{label}-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&path).expect("temp dir");
            Self { path }
        }

        fn file(&self, name: &str) -> std::path::PathBuf {
            self.path.join(name)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn fvecs_round_trip() {
        let dir = TempDir::new("round-trip");
        let path = dir.file("test.fvecs");
        let data: Vec<f32> = (0..30).map(|i| i as f32 * 0.5).collect();

        write_fvecs(&path, 5, &data).expect("write");
        let set = read_fvecs(&path).expect("read");

        assert_eq!(set.dimension, 5);
        assert_eq!(set.count(), 6);
        assert_eq!(set.data, data);
        assert_eq!(set.get(0), Some([0.0, 0.5, 1.0, 1.5, 2.0].as_slice()));
        assert_eq!(set.get(6), None);
    }

    #[test]
    fn rows_match_the_flat_layout() {
        let dir = TempDir::new("rows");
        let path = dir.file("test.fvecs");
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        write_fvecs(&path, 3, &data).expect("write");

        let rows = read_fvecs(&path).expect("read").rows();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0], vec![0.0, 1.0, 2.0]);
        assert_eq!(rows[3], vec![9.0, 10.0, 11.0]);
    }

    /// The ground-truth files are `.ivecs`, and their values are neighbour ids.
    #[test]
    fn ivecs_reads_neighbour_ids() {
        let dir = TempDir::new("ivecs");
        let path = dir.file("truth.ivecs");

        // Two queries, three neighbours each, written by hand.
        let mut bytes = Vec::new();
        for ids in [[7i32, 3, 9], [1, 4, 1]] {
            bytes.extend_from_slice(&3i32.to_le_bytes());
            for id in ids {
                bytes.extend_from_slice(&id.to_le_bytes());
            }
        }
        std::fs::write(&path, &bytes).expect("write");

        let truth = read_ivecs(&path).expect("read");
        assert_eq!(truth, vec![vec![7, 3, 9], vec![1, 4, 1]]);
    }

    #[test]
    fn an_empty_file_reads_as_empty() {
        let dir = TempDir::new("empty");
        let path = dir.file("empty.fvecs");
        std::fs::write(&path, []).expect("write");

        let set = read_fvecs(&path).expect("read");
        assert!(set.is_empty());
        assert_eq!(set.count(), 0);
    }

    /// A record repeats the dimension, so disagreement means the file is not
    /// what it claims. Reading on regardless would produce plausible garbage.
    #[test]
    fn records_disagreeing_on_dimension_are_rejected() {
        let dir = TempDir::new("ragged");
        let path = dir.file("ragged.fvecs");

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        bytes.extend_from_slice(&3i32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        bytes.extend_from_slice(&3.0f32.to_le_bytes());
        std::fs::write(&path, &bytes).expect("write");

        let error = read_fvecs(&path).expect_err("must reject");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn a_truncated_file_is_rejected() {
        let dir = TempDir::new("truncated");
        let path = dir.file("short.fvecs");

        let mut bytes = 4i32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        std::fs::write(&path, &bytes).expect("write");

        assert!(read_fvecs(&path).is_err());
    }

    /// A wrong file would otherwise be read as a dimension of billions and
    /// demand an enormous allocation before anything was validated.
    #[test]
    fn an_absurd_dimension_is_rejected_before_allocating() {
        let dir = TempDir::new("absurd");
        let path = dir.file("not-vectors.fvecs");
        std::fs::write(&path, i32::MAX.to_le_bytes()).expect("write");

        let error = read_fvecs(&path).expect_err("must reject");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);

        let negative = dir.file("negative.fvecs");
        std::fs::write(&negative, (-5i32).to_le_bytes()).expect("write");
        assert!(read_fvecs(&negative).is_err());
    }

    #[test]
    fn a_missing_file_is_a_not_found_error() {
        let error = read_fvecs("definitely-not-here.fvecs").expect_err("must fail");
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }

    /// The reader has to feed the brute-force index directly, since that is the
    /// only thing it exists for.
    #[test]
    fn a_read_set_builds_a_searchable_index() {
        use crate::ann::BruteForceIndex;

        let dir = TempDir::new("into-index");
        let path = dir.file("vectors.fvecs");
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        write_fvecs(&path, 4, &data).expect("write");

        let set = read_fvecs(&path).expect("read");
        let index = BruteForceIndex::from_flat(set.dimension, set.data);

        assert_eq!(index.len(), 25);
        let nearest = index.search(&[0.0, 1.0, 2.0, 3.0], 1);
        assert_eq!(nearest[0].id, 0);
    }
}
