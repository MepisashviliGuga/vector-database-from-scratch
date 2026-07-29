//! A vector database built from scratch.
//!
//! Two subsystems, each usable on its own, plus the layer that joins them:
//!
//! - [`storage`]: a durable LSM-tree engine — write-ahead log, SSTables with
//!   bloom filters and sparse indexes, an atomic manifest, and a pluggable
//!   growth scheme and merge policy.
//! - [`ann`]: approximate nearest-neighbour search — extended RaBitQ
//!   quantization, IVF clustering, a navigable proximity graph, and SymphonyQG.
//! - [`engine`]: a [`VectorStore`](engine::VectorStore) that writes
//!   full-precision vectors to the storage engine and quantized codes to the
//!   index, then re-ranks index candidates against storage so every returned
//!   distance is exact.
//!
//! [`bench`] and [`workload`] support measurement: a seeded workload generator
//! and nearest-rank latency percentiles. Nothing in the query or compaction path
//! depends on them.

pub mod ann;
pub mod bench;
pub mod engine;
pub mod storage;
pub mod workload;
