# A vector database, built from scratch

A durable LSM-tree storage engine and a modern approximate-nearest-neighbour
index, both written from the papers in Rust, with no embedded database and no ANN
library underneath.

The goal is not a usable product. It is to implement the two hardest layers of a
real vector database faithfully enough that the results can be compared against
the papers that describe them, and honestly enough that the places they diverge
are reported rather than hidden.

**439 tests, clippy clean at `-D warnings`, five SIGMOD papers reproduced.**

## Headline results

Every number here comes from running the code on one laptop, single-threaded and
scalar. Nothing is estimated or copied from a paper.

| | measured |
|---|---|
| **SIFT1M, end to end** | **recall@10 0.9870** at 13.4 ms/query — the index alone reaches 0.9420 |
| **Quantization** | 95.8% recall@10 at **6.4× compression**, 5.84 ms/query against brute force's 62 ms |
| **Graph index, 100k** | recall@10 **0.9990** at 938 QPS — **14× brute force**, and the margin widens with scale |
| **Storage** | vector-sized values cost **22.65× write amplification** against 3.50× for the ~100 B values compaction research benchmarks on |
| **Durability** | 5,000 acknowledged writes survive `abort()` mid-stream with zero loss or corruption |

Three results worth more than the headline figures:

- **Composition beats either layer.** The quantized index plateaus at 0.9420 on
  SIFT1M and cannot be pushed higher by probing harder — 5-bit codes have lost the
  information. Re-ranking twenty candidates against full-precision vectors read
  back from the LSM-tree reaches **0.9870 at no measurable added latency**, because
  re-ranking removes *ordering* errors entirely and leaves only the failure to
  *propose*. This is why the project needs a storage engine and not a `Vec<Vec<f32>>`.
- **Value size dominates every compaction policy.** Write amplification runs
  3.50× → 7.86× → **22.65×** as values grow 100 B → 512 B → 3,840 B, and p99
  latency rises 360×. The source papers benchmark ~100 B values; at vector sizes,
  key–value separation matters more than any growth scheme or merge policy they
  compare.
- **A reproduction that failed, then didn't.** Paper 05's claim did not reproduce
  at first. Isolating why — by swapping one component at a time against a known-good
  baseline — showed my own configuration had starved the candidate search, not that
  the paper was wrong. The verdict was retracted in place. See
  [`results/deg.md`](results/deg.md).

Full write-ups: [storage](results/README.md) · [ANN](results/ann_recall.md) ·
[end to end](results/end_to_end.md) · [hybrid search](results/deg.md).

## Status

| Phase | Component | State |
|---|---|---|
| 0 | Memtable, WAL, SSTable, bloom filter, merge iterator, manifest, LSM tree | done |
| 0 | Growth schemes: vertical, horizontal (Alg 1); merge policies: leveling, tiering | done |
| 1 | Horizontal-tiering (paper 01, Alg 2 — the paper's contribution) | done |
| 1 | **Vertiorizon** (paper 01 §5): two-part layout, `T′ = T/√2`, dynamic `n` | done |
| 1 | Vertiorizon self-tuning (§5.2), skew adaptation (§5.3) | **not reproduced** — see below |
| 2 | **EcoTune** cost model + DP scheduler (paper 02, Alg 1) | done |
| 2 | EcoTune §4.3.3 pending-runs refinement | **not reproduced** — the paper omits its derivation |
| 3 | Storage benchmarks, workload generator, range scans | done — [results](results/README.md) |
| 4 | Brute-force exact k-NN oracle | done — validated against published SIFT ground truth |
| 5 | **Extended RaBitQ** quantizer (paper 03) | done — [results](results/ann_recall.md) |
| 5 | IVF over quantized residuals; navigable proximity graph | done |
| 5 | **SymphonyQG** (paper 04) | done — reproduced, and measurably *not* worth it without SIMD |
| 6 | Integration: `VectorStore`, recall@k vs QPS across all four indexes | done — [results](results/end_to_end.md) |
| 7 | **DEG** hybrid vector search (paper 05) | done — [results](results/deg.md) |
| 7 | RusKey RL compaction (paper 06), Zombie Hashing | not started |

## Architecture

```
              ┌──────────────────────┐
  insert ────►│   VectorStore        │◄──── search(vector, k)
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

Insert writes the full-precision vector to the LSM-tree and a quantized code to
the index. Search uses the index to *propose* candidates and storage to *score*
them exactly, so every returned distance is exact.

Consistency is documented rather than glossed: **storage is the source of truth,
the index is a hint.** An overwrite leaves a stale code, so candidate *selection*
may use old data while the returned *distance* never does; a delete is filtered
until rebuild. Neither is corruption — what degrades is recall, and
`rebuild_index` restores it.

The storage layer varies along **two independent axes**, which the papers treat
separately and this project keeps separately testable:

- **Tree shape** — how the tree accommodates growth (vertical / horizontal /
  Vertiorizon).
- **Compaction policy** — when and how aggressively to merge (leveling / tiering /
  EcoTune).

### Storage layer

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
| `src/storage/manifest.rs` | Which runs are live; replaced by atomic rename, the commit point |
| `src/storage/shape.rs` | Read-only view of the tree, shared by both axes |
| `src/storage/growth/` | Axis 1 — *when* to compact: `vertical`, `horizontal`, `horizontal_tiering`, `vertiorizon`, `ecotune_scheme` |
| `src/storage/compaction/` | Axis 2 — *how* to merge: `leveling`, `tiering`, `ecotune` (the DP) |
| `src/storage/lsm.rs` | The tree: flush pipeline, multi-run read path, compaction, recovery |

### ANN layer

| File | What it is |
|---|---|
| `src/ann/brute_force.rs` | Exact k-NN by full scan — the oracle every recall figure is measured against |
| `src/ann/fvecs.rs` | Readers for the SIFT/GIST `.fvecs` and `.ivecs` formats |
| `src/ann/rotation.rs` | Random orthogonal matrix; moves the randomness out of the data and into `P` |
| `src/ann/rabitq.rs` | Extended RaBitQ: normalized-grid codebook, Algorithm 1 encoding, unbiased estimator |
| `src/ann/kmeans.rs` | Lloyd's k-means with k-means++ seeding |
| `src/ann/ivf.rs` | Inverted-file index over quantized residuals |
| `src/ann/graph.rs` | Navigable proximity graph with angle-based diversity pruning |
| `src/ann/symphony.rs` | SymphonyQG: §3.1.1 LUT decomposition, implicit re-ranking, degree alignment |
| `src/ann/deg/` | DEG hybrid search: α-interval algebra, Pareto frontiers, GPS, dynamic edge pruning |
| `src/engine.rs` | `VectorStore` — the layer that joins storage and index |
| `src/workload.rs` | Deterministic YCSB-style workload generator |
| `src/bench.rs` | Measurement harness: amplification, throughput, nearest-rank percentiles |

## Running it

```bash
cargo test                                        # 439 unit tests

# Storage
cargo run --release --example bloom_stats         # what the bloom filter buys
cargo run --release --example crash_recovery      # kill a process, verify durability
cargo run --release --example storage_bench       # the Phase 3 sweep
cargo run --release --example ecotune_schedule    # what EcoTune's DP decides

# ANN. Fetch a dataset first: benchmark/datasets/fetch.sh siftsmall
cargo run --release --example ann_groundtruth     # validate the oracle
cargo run --release --example rabitq_recall       # recall vs bits per dimension
cargo run --release --example ivf_recall          # recall vs nprobe
cargo run --release --example index_comparison    # all four indexes on one axis

# End to end, and hybrid search
cargo run --release --example end_to_end          # what re-ranking against storage buys
cargo run --release --example deg_alpha_sweep     # paper 05, Figure 2
cargo run --release --example deg_diagnosis       # isolating a defect by controlled swap
```

`crash_recovery` spawns a child that writes 5,000 records with per-write `fsync`,
calls `abort()`, and then reopens the database in the parent to check that every
acknowledged write survived.

## What the papers got wrong, or left out

Reproducing an algorithm closely enough to test it surfaces things reading it does
not. Each of these is pinned by a test.

**Paper 05 (DEG), §4.3 Case 4 is wrong at a boundary — and it fires on the
paper's own worked example.** The pruning range for `B < 0, A < 0` is given as
`[min(1, B/A), 1]`. When `B/A ≥ 1` no α satisfies the inequality, so the answer is
empty, but the clamp returns the point `[1,1]` — and a non-empty pruning range
*prunes an edge that should be kept*. Table 1's first example has `A = 0` in exact
arithmetic and −4×10⁻⁸ in f32, routing it into Case 4 with `B/A = 1.25×10⁷`, so the
published formula prunes the very edge the paper says is never pruned.

**Paper 05, `B = 0` is uncovered** by all four cases, which each require `B`
strictly signed. It is reachable whenever two second-modality distances coincide —
common at low dimension, and two of the paper's five datasets use `m = 2` — and
`min(1, B/A)` divides by zero when `A = 0` too.

**Paper 05's "active range" is not a range.** Algorithm 2 unions the pruning
ranges of every selected neighbour and complements them, which is generally
several disjoint intervals.

**Paper 05 §4.5 calls the active-range storage negligible against the vectors.**
Measured: 45% of the vector bytes at degree 19.

**Paper 04 (SymphonyQG) uses more memory than the vectors it compresses** — codes
are replicated once per in-edge, ~5× the raw vectors at degree 32. And its recall
is *non-monotonic* in code width, peaking at 3–4 bits: as codes widen, search
converges to exact greedy beam search, which explores less.

**Paper 01 §5.1 is ambiguous:** `n` is incremented "by a factor of `1/T`", which
admits both `n ← n(1 + 1/T)` and `n ← n/T`. The second would shrink the horizontal
part as data grows, contradicting the stated purpose, so the first is implemented.
Flagged rather than guessed.

## Honest reporting

Every figure comes from running the code. Where an implementation departs from its
source paper, the module documentation labels it **faithful reproduction**,
**engineering glue**, or **labelled simplification** — with its cost stated.

**Claims that do not reproduce here, and why:**

- **Throughput claims in papers 03 and 04.** Both rest on a SIMD `FastScan`
  kernel that is not implemented. Measurement shows why it matters: scalar code
  comparison is *slower* than exact f32 distance, so SymphonyQG pays the accuracy
  cost of quantization and collects none of the speed. Accuracy and memory are
  comparable to the papers; **query throughput is not, and no timing claim is made.**
- **Growth schemes showed no measurable difference** (1–5% against the paper's
  3.2×/6× claims). The configuration is too small to test the claim, and that is
  recorded as such rather than presented as a refutation.
- **EcoTune is structurally unmeasurable here.** The engine compacts synchronously
  inside `put`, so there is no background contention for its scheduler to arbitrate.

**Open, unexplained:** brute-force search scales *sublinearly* — 149 ns/vector at
100k against 63 ns/vector at 1M, doing identical work with no early exit. Three
hypotheses were tested and rejected. Documented in
[`results/end_to_end.md`](results/end_to_end.md) rather than smoothed over.

**Labelled simplifications:**

1. **WAL recovery** is a flat record stream, so a mid-file bit flip costs every
   record after it. LevelDB and RocksDB use fixed-size blocks with fragment types,
   which can resynchronise past corruption.
2. **Compaction is whole-run.** Write *amplification* is the same either way, but
   each individual compaction is larger, so tail write latency is worse than a
   production engine's.
3. **DEG's `emax`/`smax`** normalisation constants are estimated from a seeded
   sample rather than computed over all pairs (O(N²)). They only rescale the
   modalities, and index and queries share them.

An earlier fourth simplification — tracking live files by directory listing rather
than a manifest — turned out **not to be safe** and was removed rather than kept:
it allowed stale reads after a crash mid-compaction.
`src/storage/manifest.rs` explains the failure in full.

## Source papers

Six papers, indexed in [`Papers/README.md`](Papers/README.md) with citations,
links, and the phase each belongs to. The PDFs themselves are not committed.

The authors of the two ANN papers publish a reference C++ implementation,
[RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library). This project
uses it **only as a correctness oracle** — comparing quantized codes and recall on
identical inputs, so a gap can be attributed to a bug here rather than a
misreading of the paper. No reported result comes from running it.

## Licence

Not yet chosen.
