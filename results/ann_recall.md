# Extended RaBitQ recall (Phase 5a)

Produced by `cargo run --release --example rabitq_recall`, 2026-07-27.

Ranking uses **estimated distances only** — no re-ranking against raw vectors,
which is the setting paper 03 targets, since the whole point is not to keep the
raw vectors in RAM.

## SIFT1M — 1,000,000 vectors × 128 dimensions, 100 queries

| B | packed size | ratio | recall@1 | recall@10 | recall@100 | encode time |
|---|---|---|---|---|---|---|
| 1 | 15.3 MiB | 32.0× | 0.3000 | 0.2730 | 0.3589 | 11 s |
| 4 | 61.0 MiB | 8.0× | 0.7600 | 0.8490 | 0.8907 | 125 s |
| 5 | 76.3 MiB | 6.4× | 0.9000 | 0.9210 | 0.9430 | 257 s |
| 7 | 106.8 MiB | 4.6× | 0.9800 | 0.9810 | 0.9859 | 990 s |

Raw vectors: **488.3 MiB**.

## siftsmall — 10,000 vectors × 128 dimensions, 100 queries

| B | ratio | recall@1 | recall@10 | recall@100 |
|---|---|---|---|---|
| 1 | 32.0× | 0.4400 | 0.5370 | 0.6449 |
| 2 | 16.0× | 0.6600 | 0.7590 | 0.8140 |
| 3 | 10.7× | 0.8100 | 0.8580 | 0.8982 |
| 4 | 8.0× | 0.8900 | 0.9210 | 0.9445 |
| 5 | 6.4× | 0.9400 | 0.9470 | 0.9712 |
| 6 | 5.3× | 0.9600 | 0.9680 | 0.9839 |
| 7 | 4.6× | 1.0000 | 0.9790 | 0.9910 |
| 8 | 4.0× | 1.0000 | 0.9940 | 0.9957 |

## Against the paper's claims

Paper 03 reports **">95% and 99% recall at about 6.4× and 4.5× compression,
without accessing raw vectors for re-ranking"**.

| compression | paper | ours (siftsmall) | ours (SIFT1M) |
|---|---|---|---|
| 6.4× (B=5) | >95% | 94.7% | 92.1% |
| ~4.5× (B=7) | >99% | 97.9% | 98.1% |

We land **1–3 points under** at recall@10, and the gap widens with dataset size.
That is a real shortfall, and there is a concrete reason for it rather than an
appeal to noise.

### Why: one centroid, where the paper uses IVF

Paper 03 pairs the quantizer with an **IVF index**, where the data is clustered
and each vector is encoded as a residual from *its own cluster's* centroid. This
implementation uses a **single global centroid** for the entire dataset.

The consequence is direct: with one centroid over a million vectors, the residual
`o_r − c` is large and points in essentially arbitrary directions, so the unit
vector being quantized carries the full spread of the data. Per-cluster centroids
make residuals far smaller and far more concentrated, and a quantizer of fixed
precision does correspondingly better on them.

**So our configuration is strictly harder than the paper's**, and these figures
are a lower bound on the method rather than a refutation of its claim. The
scaling supports that reading: recall falls from 10k to 1M vectors at every bit
width, which is what a single global centroid predicts and what IVF exists to
prevent.

Closing the gap means implementing IVF, which is Phase 5b anyway.

### B = 1 behaves as the paper describes

At 32× compression recall@1 is 0.30 on SIFT1M — matching paper 03's own remark
that one-bit RaBitQ "can hardly produce reasonable recall" without re-ranking.
The extension to `B > 1` exists precisely because of this.

## Encoding cost: our gap is parallelism, not the algorithm

990 s to encode a million vectors at `B = 7`. Paper 03 reports a million-scale
3072-dimensional dataset finishing "in a few minutes" at the same bit width —
24× the dimensions, and faster.

The complexity is the same `O(2^B · D log D)` per vector. The difference is
implementation: encoding is **embarrassingly parallel** across vectors and this
is single-threaded scalar Rust. That is a known, fixable gap and it is not
evidence about the algorithm.

Encoding time also rises steeply in `B` (11 → 125 → 257 → 990 s for B = 1, 4, 5,
7), exactly as `2^B` predicts.

## What is not measured

**Query speed.** The arithmetic here is scalar, where paper 03 uses SIMD
FastScan and a two-stage most-significant-bit pruning path. Accuracy and memory
are directly comparable to the paper; throughput is not, and no timing claim is
made.
