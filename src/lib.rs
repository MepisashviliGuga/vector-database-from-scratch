//! A vector database built from scratch.
//!
//! Two independently testable subsystems:
//!
//! - [`storage`]: a durable LSM-tree engine (write-ahead log, SSTables, pluggable
//!   growth scheme and compaction policy).
//! - `ann_index` (not yet started): quantization + graph-based approximate
//!   nearest-neighbour search.

pub mod ann;
pub mod bench;
pub mod storage;
pub mod workload;
