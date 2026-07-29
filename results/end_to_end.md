# End to end (Phase 6)

The two halves of the project joined: full-precision vectors in the LSM-tree,
quantized codes in the IVF index, and search that uses the index to *propose*
candidates and storage to *score* them.

Reproduce with:

```
cargo run --release --example end_to_end benchmark/datasets/sift/sift 100
cargo run --release --example index_comparison
```

## Headline: re-ranking against storage beats the index alone

SIFT1M, 1M vectors × 128 dims, B = 5, 1000 clusters, `nprobe` fixed at 32, 100
queries. Only the re-rank budget varies, so the row at 10 candidates is the
index's own ordering with nothing re-ranked.

| candidates | recall@10 | ms/query |
|---|---|---|
| 10 (index alone) | 0.9420 | 13.63 |
| **20** | **0.9870** | **13.38** |
| 50 | 0.9870 | 14.92 |
| 100 | 0.9870 | 16.65 |
| 200 | 0.9870 | 20.75 |
| 500 | 0.9870 | 29.58 |

**Ten extra candidates buy 4.5 points of recall at no measurable cost.** The
IVF scan at `nprobe = 32` costs ~13 ms; re-ranking twenty vectors against
storage disappears into it.

The reason the gain is so cheap is structural. Re-ranking eliminates *ordering*
errors inside the candidate set completely — every returned distance is computed
from the stored full-precision vector, so a code that mis-ranked two neighbours
cannot cost anything. What survives is only the failure to *propose*. The
quantizer's job collapses from "rank correctly" to "put the right vectors in a
set 2× larger than the answer", which is a far weaker requirement.

For context, Phase 5 measured this same index topping out at **0.965 while
probing all 1024 clusters**. Re-ranking beats that ceiling while probing 3.6% of
the data.

## And then it stops dead

Recall is flat at 0.9870 from 20 candidates to 500. This is the limit of the
technique stated precisely: **re-ranking can only reorder what the index
proposed.** The missing 1.3% of true neighbours sit in clusters that
`nprobe = 32` never opened, and no re-rank budget reaches them.

So the two knobs are not interchangeable, and the table says which to reach for:

- past 20 candidates, more re-ranking buys **0.0000** recall for 2.2× the latency
- recall beyond 0.987 has to be bought with `nprobe`, i.e. with candidate
  generation, not with storage reads

On siftsmall the same sweep reaches **1.0000** at 20 candidates and hides this
entirely, because `nprobe = 32` of 100 clusters is a third of the data rather
than a thirtieth. The plateau is only visible at scale — which is the argument
for testing on SIFT1M rather than the 10k set.

## Cost of the durable path

| | |
|---|---|
| ingest, 1M vectors | 698 s and 918 s on two runs (~1,100–1,400 vectors/s) |
| on disk | 508.5 MiB |
| index resident | 76.8 MiB |

Ingest goes through the real write path — WAL, memtable, flush, compaction — and
`rebuild_index` then scans every vector back out and encodes it. The two runs
differ by 31% on identical inputs; this is a laptop with background load, so the
ingest figure is an order of magnitude, not a benchmark.

The footprint split is the point of the design: **77 MiB has to stay resident,
508 MiB does not.** The index is what answers the query; storage is consulted
only for the ~20 candidates that survive.

## All four indexes on one axis

siftsmall, 10k × 128, 100 queries, recall@10. One process, one ground truth, one
timing loop, so the rows are comparable to each other.

| index | knob | recall@10 | QPS | build s | MiB |
|---|---|---|---|---|---|
| brute force | – | 1.0000 | 2,305 | 0.0 | 4.9 |
| IVF+RaBitQ | 1 | 0.5080 | 65,155 | 2.0 | 0.8 |
| IVF+RaBitQ | 4 | 0.8390 | 21,055 | 2.0 | 0.8 |
| IVF+RaBitQ | 8 | 0.9200 | 10,982 | 2.0 | 0.8 |
| IVF+RaBitQ | 16 | 0.9560 | 3,187 | 2.0 | 0.8 |
| IVF+RaBitQ | 32 | 0.9550 | 1,573 | 2.0 | 0.8 |
| IVF+RaBitQ | 64 | 0.9550 | 786 | 2.0 | 0.8 |
| graph | 10 | 0.9280 | 16,157 | 3.1 | 4.9 |
| graph | 20 | 0.9780 | 15,603 | 3.1 | 4.9 |
| graph | 40 | 0.9940 | 11,610 | 3.1 | 4.9 |
| graph | 80 | **1.0000** | **6,686** | 3.1 | 4.9 |
| graph | 160 | 1.0000 | 4,116 | 3.1 | 4.9 |
| SymphonyQG | 10 | 0.5410 | 13,373 | 55.1 | 24.4 |
| SymphonyQG | 20 | 0.7690 | 7,902 | 55.1 | 24.4 |
| SymphonyQG | 40 | 0.8300 | 5,356 | 55.1 | 24.4 |
| SymphonyQG | 80 | 0.8960 | 2,828 | 55.1 | 24.4 |
| SymphonyQG | 160 | 0.9190 | 1,557 | 55.1 | 24.4 |

`knob` is `nprobe` for IVF and beam width for the graphs. MiB is what must stay
resident to answer a query: raw vectors for brute force and the graph, packed
codes for IVF, and both for SymphonyQG.

### The same sweep at 100k

The first 100k vectors of SIFT1M, 200 queries. A capped base set is a different
dataset, so these rows are comparable to each other and **not** to the siftsmall
table above.

| index | knob | recall@10 | QPS | build s | MiB |
|---|---|---|---|---|---|
| brute force | – | 1.0000 | 66 | 0.0 | 48.8 |
| IVF+RaBitQ | 1 | 0.4090 | 12,357 | 74.5 | 7.8 |
| IVF+RaBitQ | 4 | 0.7710 | 4,257 | 74.5 | 7.8 |
| IVF+RaBitQ | 8 | 0.8810 | 2,168 | 74.5 | 7.8 |
| IVF+RaBitQ | 16 | 0.9440 | 773 | 74.5 | 7.8 |
| IVF+RaBitQ | 32 | 0.9600 | 355 | 74.5 | 7.8 |
| IVF+RaBitQ | 64 | 0.9620 | 193 | 74.5 | 7.8 |
| IVF+RaBitQ | 128 | 0.9625 | 96 | 74.5 | 7.8 |
| graph | 10 | 0.8230 | 10,786 | 57.8 | 48.8 |
| graph | 20 | 0.9265 | 6,984 | 57.8 | 48.8 |
| graph | 40 | 0.9725 | 2,726 | 57.8 | 48.8 |
| graph | 80 | 0.9940 | 1,729 | 57.8 | 48.8 |
| graph | 160 | **0.9990** | **938** | 57.8 | 48.8 |
| SymphonyQG | 10 | 0.6085 | 12,342 | 554.6 | 244.1 |
| SymphonyQG | 20 | 0.8515 | 6,951 | 554.6 | 244.1 |
| SymphonyQG | 40 | 0.9325 | 4,106 | 554.6 | 244.1 |
| SymphonyQG | 80 | 0.9535 | 2,445 | 554.6 | 244.1 |
| SymphonyQG | 160 | 0.9740 | 1,078 | 554.6 | 244.1 |

**The graph's advantage over brute force grows with the dataset**, which is the
whole point of an index: 2.9× at 10k, **14× at 100k** (938 QPS at 0.9990 recall
against brute force's 66). Brute force is linear in the data; graph search is
not.

**IVF's ceiling rises with the dataset too** — 0.956 at 10k, 0.9625 at 100k —
because 316 clusters over 100k vectors give tighter residuals than 100 clusters
over 10k. The plateau shape is unchanged: 32 → 64 → 128 buys 0.0025 recall for
3.7× the latency.

### The graph wins on accuracy, IVF wins on memory

They sit in opposite corners and neither dominates:

- **The graph reaches exact results** — 1.0000 at 6,686 QPS, 2.9× faster than
  brute force with identical answers — because it scores exact distances and
  uses structure only to decide *what* to score.
- **IVF cannot get there at any setting.** It plateaus at 0.956 (16 → 32 → 64
  barely move), because 5-bit codes have a ceiling that more probing cannot
  raise. What it buys is **6× less memory**: 0.8 MiB against 4.9.

This is exactly the trade Phase 6 dissolves. The engine takes IVF's cheap,
compact candidate generation and buys back the accuracy with exact re-ranking
from storage — which is why the end-to-end run reaches 1.0000 on siftsmall
holding a 0.8 MiB index rather than a 4.9 MiB one.

### SymphonyQG is dominated here, and that is a real result

On this hardware SymphonyQG loses to the plain graph on every axis at both
sizes — lower recall at matched QPS, ~5× the memory, and 18× (10k) to 9.6×
(100k) the build time.

It does close a lot of ground at 100k. Its reachable recall goes from 0.9190 to
0.9740, and at beam 10 it now matches the graph's throughput (12,342 against
10,786 QPS). But matched against the graph at equal recall it is still behind:

| recall@10 | graph QPS | SymphonyQG QPS |
|---|---|---|
| ~0.93 | 6,984 (beam 20) | 4,106 (beam 40) |
| ~0.97 | 2,726 (beam 40) | 1,078 (beam 160) |

It needs 4× the beam width to reach the recall the graph gets at beam 40, and
gives up 2.5× throughput doing it.

This is not a bug, and it is not the paper being wrong. It is the consequence of
one missing piece, already documented in Phase 5: **scoring quantized codes in
scalar Rust is slower than computing the exact f32 distance.** SymphonyQG's whole
proposition is that fusing codes into the graph makes each hop cheap enough to
pay for the accuracy lost — and that cheapness comes from a SIMD FastScan kernel
we did not implement. Without it we pay the accuracy (4-bit codes) and collect
none of the speed, so the structure faithfully reproduced has nothing to trade.

The memory figure is the same story from the other side: codes are replicated
once per in-edge, so at degree 32 the index holds **more than the raw vectors it
is meant to compress**, and implicit re-ranking needs those raw vectors anyway.

Labelled honestly: the algorithm is a faithful reproduction; the *benchmark* is
not a reproduction of the paper's, because the kernel its claim rests on is
absent. Published SymphonyQG numbers should not be compared against this table.

## Measurement notes

Two flaws were found and fixed while producing the tables above, both of which
had already produced wrong-looking numbers:

1. **Cold page cache.** The first SIFT1M sweep reported 10 candidates costing
   18.83 ms against 20 candidates' 13.63 ms — strictly more work for less time.
   The first timed configuration was absorbing the cost of faulting in a 508 MiB
   store. `end_to_end` now runs an untimed warm pass first.
2. **Timing windows below the clock's noise floor.** At 20k+ QPS, 100 queries
   finish in ~5 ms, and the comparison table reported SymphonyQG at beam 20 as
   *faster* than beam 10. `index_comparison` now repeats each configuration for
   at least a second and at least five passes, and reports the **fastest** pass:
   every noise source here (background load, frequency scaling, thermal limits)
   can only add time, so the minimum is the closest estimate of the true cost.
   All three series are strictly monotonic under this statistic; none were
   before it.

Everything is single-threaded and scalar. No figure here is comparable to a
published one; they are comparable **within a table**, which is what the tables
are for.

### One number I cannot yet account for

QPS should not be compared across the two tables, and one pair shows why.
Brute force at 100k costs **15.1 ms/query**. Phase 5 measured brute force on the
full SIFT1M at **62 ms/query** — 10× the data for 4.1× the time. Since brute
force is exactly linear in the data, one of the two is off by ~2.4×, and I have
not isolated which.

Candidates I have not ruled out: the two figures come from different examples
built at different times, and the 1M measurement predates the best-of-N timing
change. It does not affect any conclusion drawn above — every comparison in this
document is between rows measured in the same process against the same data —
but it does mean the cross-table scaling claims should be read as within-table
trends, not as a measured scaling law. Settling it needs brute force re-measured
at both sizes under the current harness.
