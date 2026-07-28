# Phase 3 storage benchmark results

Produced by `cargo run --release --example storage_bench`, first run 2026-07-27.
Raw data in [`storage_benchmarks.csv`](storage_benchmarks.csv).

Every number below came from running the code on one machine. Nothing is
estimated, and where a claim from a paper did **not** appear, that is recorded
rather than omitted.

## Setup and its limits

- Single-threaded. **Compaction runs synchronously inside `put`.**
- `SyncPolicy::Manual` — writes are not `fsync`ed per operation, so these figures
  are about compaction and I/O structure, not durability cost.
- 20,000 keys at 100 B unless stated; 64 KiB memtable, 4 KiB blocks, 256 KiB
  target file size.
- Data fits in the OS page cache. Block-read counts are unaffected (they are
  counted by the engine), but latencies understate real disk cost.

Absolute throughput here is **not** comparable to either paper, both of which
assume background compaction on many cores. Relative ordering is the point.

## Finding 1: value size dominates everything — and it is our workload

| value size | write amplification | throughput | p99 |
|---|---|---|---|
| 100 B (the papers' regime) | **3.50×** | 7.0 kops/s | 79 µs |
| 512 B (SIFT vector) | **7.86×** | 0.7 kops/s | 169 µs |
| 3840 B (GIST vector) | **22.65×** | 0.2 kops/s | **28.6 ms** |

Growth scheme, merge policy and key distribution held fixed; only value size
moves. Write amplification rises **6.5×** and p99 rises **360×** between the size
LSM compaction research is measured at and the size a GIST vector actually is.

This was predicted at the start of this project and is now measured: compaction
research is conducted on ~100-byte values, and at vector sizes write
amplification is dominated by recopying value bytes. That is the problem
key-value separation (WiscKey, RocksDB BlobDB) exists to solve, and this engine
does not do it.

**It is the most consequential result here.** A vector database built on a
textbook LSM-tree pays a 6.5× write-amplification penalty precisely because its
values are large — so the interesting engineering question for this project is
key-value separation, not which compaction policy to pick.

## Finding 2: the leveling/tiering trade-off reproduces cleanly

| merge policy | write amplification | space amplification | throughput |
|---|---|---|---|
| leveling | 4.20× | 1.24× | 8.4 kops/s |
| tiering | **2.81×** | **1.60×** | 16.4 kops/s |

Textbook, and reassuring: tiering writes less and uses more space; leveling the
reverse. This is the axis the engine measures most convincingly.

## Finding 3: leveling pays for key overlap, tiering does not

| distribution | leveling WA | tiering WA |
|---|---|---|
| sequential | 3.65× | 2.83× |
| uniform | **4.20×** | 2.81× |
| zipfian | 3.26× | 2.54× |

Leveling's write amplification moves with the key distribution; tiering's is flat
to within 0.02×. That is exactly the mechanism — leveling pays to merge away
*overlapping* data, and sequential inserts produce runs with no overlap to merge.

It also confirms why key distribution must be a swept axis: a suite that only
inserted sequential keys would have understated leveling's cost.

## Finding 4: the growth-scheme differences did NOT appear

| growth scheme | write amplification | space amplification |
|---|---|---|
| vertical | 4.20× | 1.24× |
| horizontal-leveling | 4.15× | **1.52×** |
| vertiorizon | **4.01×** | 1.24× |

Directionally consistent with paper 01 — horizontal is worse on space, Vertiorizon
gets the lower write cost while keeping vertical's space cost — but the margins
are **1–5%**, far short of the paper's reported 3.2× throughput and 6× space
gains.

Honest reading: **this configuration does not test the claim.** 20,000 keys at
100 B is 2 MB, roughly 32 flushes and a handful of compactions. The horizontal
scheme's advantage is asymptotic in the number of compactions per round, and the
schedules have barely begun to diverge at this scale. Testing it properly needs a
data set orders of magnitude larger.

An earlier ad-hoc measurement in this project showed horizontal at 11.1× against
vertical at 20.7×, and **that gap does not reproduce here.** The earlier figure
came from a tiny test configuration with no controls and should not be quoted.

## Finding 5: EcoTune cannot be evaluated in this engine

| configuration | throughput | p50 | write amplification |
|---|---|---|---|
| vertical + leveling | 9.1 kops/s | 138 µs | 2.03× |
| vertical + tiering | 8.3 kops/s | 149 µs | 2.03× |
| vertiorizon + leveling | **9.8 kops/s** | **92 µs** | 2.64× |
| ecotune + tiering | 8.3 kops/s | 146 µs | 2.03× |

EcoTune shows no advantage, and the reason is structural rather than a tuning
failure. Its entire thesis is that **compaction and queries compete for the same
CPU and I/O**, so a policy should decline compactions whose cost exceeds their
return. This engine compacts *synchronously inside `put`*. There is no
concurrency, so there is no contention to arbitrate — compaction time is simply
serialised cost, and no scheduling decision can recover it.

Evaluating EcoTune faithfully requires **background compaction threads**. Until
then its DP is verified against the paper's own recurrence (see
`src/storage/compaction/ecotune.rs`) but its benefit is unmeasurable here, and no
throughput claim should be made for it either way.

Its `β` parameter has the same problem: calibration measured 0.165, but that is
the share of wall-clock time compaction occupies, not the paper's
contended-thread ratio.

## The papers' disagreement: not settled

Paper 01 minimises write amplification; paper 02 argues it is free on modern
NVMe. Across all 18 rows, write amplification and throughput correlate at
**−0.649** — but that number is **confounded and should not be quoted.** It is
driven almost entirely by the value-size sweep, where high write amplification
and low throughput share a common cause (large values), not a causal link.

The controlled experiment is: one configuration, one workload, sweeping *memtable
size* to vary write amplification while holding everything else fixed. That is
not yet run.

## What to do next

1. **Key-value separation.** Finding 1 says this matters more for a vector
   database than any compaction policy.
2. **Background compaction.** Without it, Findings 4 and 5 are untestable.
3. **A larger data set**, to give the growth schemes room to diverge.
4. **The controlled write-amplification experiment** described above.
