# ANN recall (Phase 5)

## Headline: IVF closes the gap to the paper

Paper 03 claims **>95% recall at about 6.4× compression, without re-ranking**.

SIFT1M, 1M vectors × 128 dims, B = 5 (6.4× compression), 1024 clusters, 100
queries. Recall is against exact k-NN over the same base set.

| nprobe | recall@10 | ms/query | data scanned |
|---|---|---|---|
| 1 | 0.4080 | 0.28 | 0.1% |
| 8 | 0.8600 | 1.62 | 1.0% |
| 16 | 0.9230 | 3.07 | 1.9% |
| **32** | **0.9580** | **5.84** | **3.6%** |
| 64 | 0.9630 | 11.09 | 7.0% |
| 1024 (all) | 0.9650 | 160.63 | 100% |

Against the single global centroid of Phase 5a, at the same bit width:

| configuration | recall@10 |
|---|---|
| single global centroid, full scan | 0.9210 |
| IVF, probing all 1024 clusters | **0.9650** |

Two independent effects, both real:

1. **Better codes.** At an identical bit width, clustering lifts the quantizer's
   ceiling from **0.921 to 0.965**. Residuals from a nearby centroid are small
   and locally concentrated; residuals from one global centroid carry the entire
   spread of the dataset. This is what closes the gap to the published claim.
2. **Less work.** `nprobe = 32` reaches **0.958 — above the paper's >95%** —
   while touching 3.6% of the data.

Against the exact brute-force baseline of **62 ms/query**, IVF at `nprobe = 32`
answers in **5.84 ms — 10.6× faster — at 95.8% recall and 6.4× less memory**.
Both are scalar and single-threaded, so the comparison is like for like.

### All the speed comes from pruning, none from the codes

Worth stating plainly, because it is easy to assume otherwise: probing *every*
cluster costs **160 ms/query against brute force's 62 ms**. Comparing a
quantized code is, in this implementation, *slower* than computing an exact
`f32` distance — the estimator adds an integer-to-float conversion, an offset
correction and two divisions per candidate.

So IVF's speedup here is entirely the pruning, and the compression buys memory
rather than time. Making the *codes* faster to compare than raw vectors is
exactly what paper 03's SIMD FastScan does, and it is the part this project
does not reproduce. That is the single largest gap between these numbers and the
paper's.

### Caveats

Cluster balance is imperfect: 1024/1024 lists used, mean 977 members, largest
3718. A skewed list means `nprobe` buys less than its fraction suggests, which is
why "% scanned" is measured per query rather than assumed to be
`nprobe / clusters`.

Index build took 207 s for a million vectors (k-means on a 100k sample, then
assignment and encoding of the full set).

An earlier version of `IvfIndex::search` collected every candidate and sorted
them rather than keeping a bounded heap. Recall was unaffected, but the
exhaustive row cost 203 ms instead of 161 ms — the sort rivalled the distance
estimation. The figures above are all post-fix.

---

# Extended RaBitQ alone, without IVF (Phase 5a)

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

### Why: one centroid, where the paper uses IVF — now confirmed

**This section's hypothesis was subsequently tested and held.** Adding IVF lifted
recall@10 at B=5 from 0.921 to 0.965 on SIFT1M, with everything else unchanged.
The reasoning below was written before that measurement and is left as it stood.

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
