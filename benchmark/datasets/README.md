# ANN benchmark datasets

The vector files are **not committed** — SIFT1M alone is 168 MB compressed. Fetch
them with [`fetch.sh`](fetch.sh), or by hand from
<http://corpus-texmex.irisa.fr/>.

| dataset | base vectors | dim | queries | download |
|---|---|---|---|---|
| `siftsmall` | 10,000 | 128 | 100 | 5 MB |
| `sift` (SIFT1M) | 1,000,000 | 128 | 10,000 | 168 MB |
| `gist` (GIST1M) | 1,000,000 | 960 | 1,000 | 2.6 GB |

Each archive expands to a directory holding `*_base.fvecs`, `*_query.fvecs`,
`*_groundtruth.ivecs` and `*_learn.fvecs`.

## Why the ground truth files matter

`*_groundtruth.ivecs` holds the true 100 nearest neighbours of every query,
computed independently by the benchmark's authors. Reproducing it exactly is an
**external** check on our brute-force oracle — and since every recall figure in
this project is measured against that oracle, it is the only check that can catch
an error which would otherwise propagate silently into every ANN result.

```bash
cargo run --release --example ann_groundtruth                              # siftsmall
cargo run --release --example ann_groundtruth benchmark/datasets/sift/sift 300
```

### Verified 2026-07-27

**siftsmall** (10k vectors): recall **1.000000** at k = 1, 10 and 100, with
100/100 queries matching exactly.

**SIFT1M** (1M vectors, first 300 queries):

| k | recall | exact id matches | ties | genuine mismatches |
|---|---|---|---|---|
| 1 | 0.993333 | 298/300 | 2 | **0** |
| 10 | 0.999667 | 299/300 | 1 | **0** |
| 100 | 0.999900 | 297/300 | 3 | **0** |

The handful of id disagreements are **exact ties** — our result sits at precisely
the same distance as the published one, so both rankings are correct and the two
implementations simply broke the tie differently. The runner checks this by
comparing *distances* rather than ids, which is the only way to tell a tie from a
real error.

It is not a floating-point artefact either: SIFT vectors are integers 0–255, so a
squared distance is an integer of at most ~8.3M, well under `2^24`, and is
therefore exact in `f32`.

**Brute-force baselines on SIFT1M**, which Phases 5 and 6 have to beat:

- **488.3 MiB** of `f32` vectors resident — the memory figure the quantizer must
  reduce.
- **62 ms per query, ~16 queries/sec** — one million distance computations each.
  This is the speed the graph index must improve on.

## Formats

`.fvecs` and `.ivecs` are a flat sequence of self-describing records:

```text
[i32 dimension][dimension × f32]     .fvecs
[i32 dimension][dimension × i32]     .ivecs
```

Little-endian, with the dimension repeated in every record. `src/ann/fvecs.rs`
uses that repetition as a corruption check and rejects files whose records
disagree.

## A note on GIST

GIST vectors are 960 dimensions of `f32` — **3840 bytes each**. That is the value
size the Phase 3 storage benchmarks found costs 22.65× write amplification
against the ~100-byte values LSM compaction research is measured at. GIST is
therefore the more revealing dataset for this project's storage half, and the
more expensive one to run.
