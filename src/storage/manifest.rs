//! The manifest: which runs are live, updated atomically.
//!
//! # Why a directory listing is not enough
//!
//! An earlier version of this engine derived the tree from filenames alone.
//! That is fine for flushes, but it breaks on compaction, and the failure is
//! subtle enough to be worth recording.
//!
//! A compaction reads runs from level `i`, writes the result to level `i+1`, and
//! deletes the inputs. Those three steps cannot be made atomic against a crash.
//! If the process dies after the output is renamed into place but before the
//! inputs are deleted, both exist — and the read path consults levels
//! shallowest-first, so it finds the **stale input at level `i`** before ever
//! reaching the fresh output at level `i+1`. Silent stale reads.
//!
//! Ordering the steps differently only moves the problem: deleting inputs first
//! risks losing them outright.
//!
//! Nor can recency be recovered from sequence numbers alone. A compaction deep
//! in the tree produces a run with a *high* sequence number holding *old* data,
//! while a shallower run with a lower sequence holds newer data — so sorting all
//! runs by sequence is not a valid recency order either.
//!
//! # The fix
//!
//! One small file naming every live run, replaced by an atomic rename. The
//! rename is the commit point: before it, the old set of runs is live; after it,
//! the new set is. Files on disk that the manifest does not name are garbage
//! from an interrupted operation and are deleted at startup.
//!
//! This is a scaled-down version of what RocksDB's MANIFEST does. RocksDB
//! appends *version edits* to a log, so a long-running database does not rewrite
//! the full file list on every change; this rewrites the whole manifest each
//! time. At the run counts this project reaches that costs microseconds, and the
//! atomicity property — the part correctness depends on — is identical.

use std::io::{self, Write};
use std::path::{Path, PathBuf};

use super::crc32::crc32;

/// Current manifest filename. Replaced wholesale on every change.
pub const MANIFEST_FILENAME: &str = "MANIFEST";
/// Scratch name the next manifest is written to before being renamed over
/// [`MANIFEST_FILENAME`].
const MANIFEST_TEMP_FILENAME: &str = "MANIFEST.tmp";

const FORMAT_HEADER: &str = "vectordb-manifest 1";

/// One live run: which level it sits in, its recency within that level, and the
/// files it is made of.
///
/// Filenames are stored rather than derived. A run's files are created together
/// and named after it, but runs can later be *folded* — two runs with disjoint
/// key ranges become one without any data being rewritten — and the resulting
/// run holds files originally named for different sequences. Deriving names
/// from `(level, sequence, part)` would then reconstruct names that do not
/// exist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunEntry {
    pub level: usize,
    /// Creation order. Within a level, higher is newer.
    pub sequence: u64,
    /// Files, in ascending key order, holding disjoint slices of the range.
    pub files: Vec<String>,
    /// How many unit runs (memtable flushes' worth) this run represents.
    ///
    /// Only EcoTune reads it, but it must be durable: its schedule specifies
    /// merge widths in units, so a restart that reset every run to 1 would make
    /// the scheduler merge the wrong sets.
    pub units: usize,
}

/// The set of runs that make up the database.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Manifest {
    pub runs: Vec<RunEntry>,
    /// Next unused sequence number. Persisted so sequences never repeat across a
    /// restart, which would make two different runs share a filename.
    pub next_sequence: u64,
}

impl Manifest {
    /// Read the manifest from `dir`, or `None` if the database is new.
    ///
    /// A manifest that fails its checksum is an error rather than a fresh start:
    /// silently treating a corrupt manifest as an empty database would present
    /// a full database as empty, and the next flush would begin overwriting it.
    pub fn load(dir: &Path) -> io::Result<Option<Self>> {
        let path = dir.join(MANIFEST_FILENAME);
        let contents = match std::fs::read_to_string(&path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };

        let (body, checksum_line) = contents
            .rsplit_once("crc32 ")
            .ok_or_else(|| malformed("manifest has no checksum line; it was probably truncated"))?;

        let expected = u32::from_str_radix(checksum_line.trim(), 16)
            .map_err(|_| malformed("manifest checksum is not a hex number"))?;
        if crc32(body.as_bytes()) != expected {
            return Err(malformed("manifest failed its checksum"));
        }

        let mut lines = body.lines();
        if lines.next().map(str::trim) != Some(FORMAT_HEADER) {
            return Err(malformed(
                "not a manifest, or written by an incompatible format version",
            ));
        }

        let mut manifest = Self::default();
        for line in lines {
            let mut fields = line.split_whitespace();
            match fields.next() {
                Some("next-sequence") => {
                    manifest.next_sequence = parse_field(fields.next(), "next-sequence")?;
                }
                Some("run") => {
                    let level = parse_field(fields.next(), "run level")?;
                    let sequence = parse_field(fields.next(), "run sequence")?;
                    let units: usize = parse_field(fields.next(), "run unit count")?;
                    let files: Vec<String> = fields.map(str::to_string).collect();
                    if files.is_empty() {
                        return Err(malformed("a run entry names no files"));
                    }
                    manifest.runs.push(RunEntry {
                        level,
                        sequence,
                        files,
                        units,
                    });
                }
                // Blank lines and anything unrecognised are skipped so a future
                // format can add fields without breaking this reader.
                _ => continue,
            }
        }
        Ok(Some(manifest))
    }

    /// Write the manifest to `dir`, atomically.
    ///
    /// The rename is the commit point. Everything before it is preparation that
    /// a crash discards; everything after it is durable.
    pub fn store(&self, dir: &Path) -> io::Result<()> {
        let mut body = String::new();
        body.push_str(FORMAT_HEADER);
        body.push('\n');
        body.push_str(&format!("next-sequence {}\n", self.next_sequence));
        for run in &self.runs {
            body.push_str(&format!("run {} {} {}", run.level, run.sequence, run.units));
            for file in &run.files {
                body.push(' ');
                body.push_str(file);
            }
            body.push('\n');
        }

        let mut contents = body.clone();
        contents.push_str(&format!("crc32 {:08x}\n", crc32(body.as_bytes())));

        let temp_path = dir.join(MANIFEST_TEMP_FILENAME);
        {
            let mut file = std::fs::File::create(&temp_path)?;
            file.write_all(contents.as_bytes())?;
            // The new manifest must be on disk *before* the rename publishes it,
            // or a crash could leave the name pointing at unwritten bytes.
            file.sync_all()?;
        }
        std::fs::rename(&temp_path, dir.join(MANIFEST_FILENAME))?;
        Ok(())
    }

    /// Every filename this manifest references.
    pub fn referenced_files(&self) -> Vec<PathBuf> {
        self.runs
            .iter()
            .flat_map(|run| run.files.iter().map(PathBuf::from))
            .collect()
    }
}

/// `L{level}-R{sequence}-P{part}.sst`.
///
/// The name carries enough to identify a file's role while debugging, but the
/// manifest, not the name, is what makes a file live.
pub fn table_filename(level: usize, sequence: u64, part: usize) -> String {
    format!("L{level:02}-R{sequence:010}-P{part:04}.sst")
}

fn parse_field<T: std::str::FromStr>(field: Option<&str>, what: &str) -> io::Result<T> {
    field
        .ok_or_else(|| malformed(&format!("manifest is missing a {what}")))?
        .parse()
        .map_err(|_| malformed(&format!("manifest has an unparseable {what}")))
}

fn malformed(reason: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("manifest: {reason}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(label: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static COUNTER: AtomicU32 = AtomicU32::new(0);

            let unique = format!(
                "vectordb-manifest-{label}-{}-{}",
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

    fn entry(level: usize, sequence: u64, parts: usize) -> RunEntry {
        RunEntry {
            level,
            sequence,
            files: (0..parts)
                .map(|part| table_filename(level, sequence, part))
                .collect(),
            units: 1,
        }
    }

    fn sample() -> Manifest {
        Manifest {
            runs: vec![entry(0, 7, 1), entry(0, 6, 1), entry(1, 5, 4)],
            next_sequence: 8,
        }
    }

    #[test]
    fn a_missing_manifest_means_a_new_database() {
        let dir = TempDir::new("missing");
        assert_eq!(Manifest::load(&dir.path).expect("load"), None);
    }

    #[test]
    fn round_trips_through_disk() {
        let dir = TempDir::new("roundtrip");
        let manifest = sample();
        manifest.store(&dir.path).expect("store");

        let loaded = Manifest::load(&dir.path).expect("load").expect("present");
        assert_eq!(loaded, manifest);
    }

    #[test]
    fn an_empty_manifest_round_trips() {
        let dir = TempDir::new("empty");
        let manifest = Manifest {
            runs: Vec::new(),
            next_sequence: 1,
        };
        manifest.store(&dir.path).expect("store");

        let loaded = Manifest::load(&dir.path).expect("load").expect("present");
        assert_eq!(loaded, manifest);
    }

    #[test]
    fn storing_twice_replaces_rather_than_appends() {
        let dir = TempDir::new("replace");
        sample().store(&dir.path).expect("store");

        let replacement = Manifest {
            runs: vec![entry(2, 9, 2)],
            next_sequence: 10,
        };
        replacement.store(&dir.path).expect("store again");

        let loaded = Manifest::load(&dir.path).expect("load").expect("present");
        assert_eq!(loaded, replacement);
        assert!(
            !dir.path.join(MANIFEST_TEMP_FILENAME).exists(),
            "the scratch file must be renamed away, not left behind"
        );
    }

    /// A corrupt manifest must be an error. Treating it as a new database would
    /// present a full database as empty, and the next flush would start
    /// overwriting live data.
    #[test]
    fn a_corrupt_manifest_is_an_error_not_an_empty_database() {
        let dir = TempDir::new("corrupt");
        sample().store(&dir.path).expect("store");

        let path = dir.path.join(MANIFEST_FILENAME);
        let contents = std::fs::read_to_string(&path).expect("read");
        // Change a run's level without fixing the checksum.
        let tampered = contents.replace("run 1 5 ", "run 3 5 ");
        assert_ne!(
            tampered, contents,
            "the tamper must actually change something"
        );
        std::fs::write(&path, tampered).expect("write");

        let error = Manifest::load(&dir.path).expect_err("must reject");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn a_truncated_manifest_is_rejected() {
        let dir = TempDir::new("truncated");
        sample().store(&dir.path).expect("store");

        let path = dir.path.join(MANIFEST_FILENAME);
        let contents = std::fs::read_to_string(&path).expect("read");
        // Lose the checksum line, as a crash mid-write would.
        let truncated = &contents[..contents.len() / 2];
        std::fs::write(&path, truncated).expect("write");

        assert!(Manifest::load(&dir.path).is_err());
    }

    #[test]
    fn a_foreign_file_is_not_read_as_a_manifest() {
        let dir = TempDir::new("foreign");
        let body = "something else entirely\n";
        let contents = format!("{body}crc32 {:08x}\n", crc32(body.as_bytes()));
        std::fs::write(dir.path.join(MANIFEST_FILENAME), contents).expect("write");

        assert!(Manifest::load(&dir.path).is_err());
    }

    #[test]
    fn filenames_are_derived_from_run_identity() {
        assert_eq!(table_filename(3, 42, 0), "L03-R0000000042-P0000.sst");
        assert_eq!(table_filename(3, 42, 1), "L03-R0000000042-P0001.sst");
    }

    #[test]
    fn referenced_files_covers_every_part_of_every_run() {
        let files = sample().referenced_files();
        assert_eq!(files.len(), 1 + 1 + 4);
        assert!(files.contains(&PathBuf::from("L01-R0000000005-P0003.sst")));
        assert!(!files.contains(&PathBuf::from("L01-R0000000005-P0004.sst")));
    }

    /// A folded run holds files originally named for different sequences, which
    /// is exactly why names are stored rather than derived.
    #[test]
    fn a_run_may_hold_files_named_for_other_sequences() {
        let dir = TempDir::new("folded");
        let folded = Manifest {
            runs: vec![RunEntry {
                level: 1,
                sequence: 12,
                files: vec![
                    table_filename(1, 4, 0),
                    table_filename(1, 9, 0),
                    table_filename(1, 12, 0),
                ],
                units: 3,
            }],
            next_sequence: 13,
        };
        folded.store(&dir.path).expect("store");

        let loaded = Manifest::load(&dir.path).expect("load").expect("present");
        assert_eq!(loaded, folded);
        assert_eq!(loaded.referenced_files().len(), 3);
    }

    #[test]
    fn a_run_naming_no_files_is_rejected() {
        let dir = TempDir::new("no-files");
        let body = format!("{FORMAT_HEADER}\nnext-sequence 3\nrun 0 1\n");
        let contents = format!("{body}crc32 {:08x}\n", crc32(body.as_bytes()));
        std::fs::write(dir.path.join(MANIFEST_FILENAME), contents).expect("write");

        assert!(Manifest::load(&dir.path).is_err());
    }
}
