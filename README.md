# A vector database, built from scratch

A durable LSM-tree storage engine and a modern approximate-nearest-neighbour
index, both written from the papers in Rust, with no embedded database and no
ANN library underneath.

The goal is not a usable product. It is to implement the two hardest layers of a
real vector database faithfully enough that the results can be compared against
the papers that describe them, and honestly enough that the places they diverge
are reported rather than hidden.

## Status

Phase 0 is in progress. **Nothing below is claimed until it is implemented,
tested, and measured** — see [Reported numbers](#reported-numbers).

| Phase | Component | State |
|---|---|---|
| 0 | Memtable, WAL, SSTable, bloom filter, merge iterator, LSM tree | done |
| 0 | Growth schemes: vertical, horizontal | not started |
| 0 | Compaction: leveling, tiering | not started |
| 1 | Vertiorizon growth scheme (paper 01) | not started |
| 2 | EcoTune compaction policy (paper 02) | not started |
| 3 | Storage benchmarks across both axes | not started |
| 4 | Brute-force exact k-NN baseline | not started |
| 5 | Extended RaBitQ quantizer (paper 03) + SymphonyQG graph (paper 04) | not started |
| 6 | Integration, recall@k vs. QPS curves | not started |
| 7 | Stretch: filtered search (paper 05), RusKey RL compaction (paper 06) | not started |

87 unit tests, clippy clean at `-D warnings`.

## Architecture

```
              ┌──────────────────────┐
  insert ────►│     Query layer      │◄──── search(vector, k, filters)
              └──────────┬───────────┘
                 ┌───────┴────────┐
                 ▼                ▼
      ┌────────────────────┐  ┌──────────────────────┐
      │  Storage layer     │  │  ANN index           │
      │  (LSM-tree)        │◄─┤  quantized codes     │
      │                    │─►│  navigable graph     │
      │  full-precision    │  │  candidate search    │
      │  vectors + metadata│  │  re-rank via storage │
      └────────────────────┘  └──────────────────────┘
```

The ANN index holds compressed vectors and graph edges for fast candidate
generation; the storage layer holds the durable, full-precision vectors. A query
searches the graph approximately, then re-ranks the top candidates against exact
vectors fetched from storage.

The storage layer varies along **two independent axes**, which the papers treat
separately and this project keeps separately testable:

- **Tree shape** — how the tree accommodates growth (vertical / horizontal /
  Vertiorizon).
- **Compaction policy** — when and how aggressively to merge (leveling / tiering
  / EcoTune).

### Storage layer, as built

```
put/delete ──► WAL (append, fsync) ──► memtable ──┐ threshold
                                                   ▼
                                       SSTable in level 0, then WAL reset
```

Reads consult sources newest-first — memtable, then each level, newest run
first — and stop at the first entry found for the key, **including a tombstone**.

| File | What it is |
|---|---|
| `src/storage/memtable.rs` | Ordered write buffer with tombstones and incremental size accounting |
| `src/storage/wal.rs` | CRC-checked record log; replay stops at the first unverifiable record and recovery truncates the torn tail |
| `src/storage/sstable.rs` | Immutable sorted run: CRC'd blocks, sparse index, bloom filter, footer written last |
| `src/storage/bloom.rs` | Bloom filter; k hashes from two via Kirsch-Mitzenmacher |
| `src/storage/crc32.rs` | Table-driven CRC-32, matching zlib |
| `src/storage/merge.rs` | K-way merge resolving key collisions by recency |
| `src/storage/lsm.rs` | The tree: flush pipeline, multi-run read path, crash recovery |

## Running it

```bash
cargo test                                        # unit tests
cargo run --release --example bloom_stats         # what the bloom filter buys
cargo run --release --example crash_recovery      # kill a process, verify durability
```

`crash_recovery` spawns a child that writes 5,000 records with per-write `fsync`,
calls `abort()`, and then reopens the database in the parent to check that every
acknowledged write survived.

## Reported numbers

Every figure in this repo comes from running the code on this machine. Nothing is
estimated, extrapolated, or copied from a paper.

Where an implementation departs from its source paper, it is labelled in the
module documentation as one of:

- **faithful reproduction** — the paper's algorithm as described;
- **engineering glue** — necessary code the paper does not specify;
- **labelled simplification** — a deliberate deviation, with its cost stated.

Two simplifications exist so far, both documented at the top of their modules:

1. **WAL recovery** is a flat record stream, so a mid-file bit flip costs every
   record after it. LevelDB and RocksDB use fixed-size blocks with fragment
   types, which can resynchronise past corruption.
2. **File tracking** is by filename (`L{level}-{sequence}.sst`) plus directory
   listing, not a MANIFEST. RocksDB's version-edit log makes a multi-file
   compaction atomic as a group; directory listing cannot express that.

## Source papers

Six papers, indexed in [`Papers/README.md`](Papers/README.md) with citations,
links, and the phase each belongs to. The PDFs themselves are not committed.

The authors of the two ANN papers publish a reference C++ implementation,
[RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library). This project
uses it **only as a correctness oracle** — comparing quantized codes and recall
on identical inputs, so a gap can be attributed to a bug here rather than a
misreading of the paper. No reported result comes from running it.

## Licence

Not yet chosen.
