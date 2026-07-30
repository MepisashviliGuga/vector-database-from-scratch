# DEG: hybrid vector search (Phase 7)

Paper 05, the Dynamic Edge Navigation Graph. **The algebra reproduces and is
verified against the paper. The end-to-end claim does not reproduce here, and
this document says so and localises why.**

Reproduce with:

```
cargo run --release --example deg_alpha_sweep
cargo run --release --example deg_alpha_sweep benchmark/datasets/sift/sift 100000 100 100 32 100
```

## The problem

A Hybrid Vector Query gives every object two feature vectors and lets each
*query* choose the weight between them (§3.1, Eq 1):

```
Dist(q,o) = α·δe(q,o) + (1−α)·δs(q,o),   α ∈ [0,1] chosen per query
```

The two vectors are not the difficulty. The difficulty is that α arrives *with
the query*, while every graph index prunes its edges using distances assumed
fixed at build time. §3.3 measures the two prior approaches and finds each strong
over part of the α range and weak elsewhere — not from poor implementation, but
because both are structurally committed to a weight they do not yet know.

## What makes it tractable

`Dist` is **affine in α**: written `δs + α(δe − δs)`, a fixed pair of objects
traces a straight line. So every distance *comparison* is two lines crossing at
one point, and "for which α would the RNG rule prune this edge?" has a closed
form. That collapses an infinite family of graphs — one per α — into one interval
set stored per edge.

## What reproduces

**The pruning algebra (§4.3).** All three of Table 1's worked examples reproduce,
including Example 1's conclusion that edge (x₃,y₃) is pruned on exactly
`[2/3, 1]` via the intersection of `[2/3,1]` and `[1/3,1]`. Beyond golden values,
2,000 random triangles are checked at 101 α each against Eq 2/3 evaluated
directly, so a sign error cannot hide behind agreeing constants.

**Theorem 4.1 (§4.2).** Verified rather than assumed: 300 random point sets × 51
α, comparing the best hybrid distance achievable within frontier layer 1 against
the best over all points. Compared on distances, not ids, so ties cannot produce a
false failure. The converse is checked too — nothing undominated is ever dropped.

**The §4.5 early exit.** Since `δs ≥ 0`, `α·δe` alone lower-bounds the hybrid
distance, so a node can be rejected without reading the second modality. 25
queries × 11 α produce byte-identical results with the optimisation on and off.

**The metric premise (§3.2).** RNG pruning needs a metric. At fixed α the hybrid
distance is one — a non-negative weighted sum of two metrics — checked over 24³
triples at 11 α rather than argued.

**Merging's failure mode (§3.3).** The one baseline behaviour that reproduces
cleanly. Separate per-modality indexes are strong at the extremes and collapse in
the middle, exactly the shape of Fig 2 (10k objects, beam 250):

| α | 0.0 | 0.3 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|---|
| Merging recall@10 | 0.9870 | 0.1730 | 0.1830 | 0.3290 | 0.9800 |

Each index sees one modality, so mid-range queries — where both matter — get
candidates that are good in one and arbitrary in the other.

## Three corrections and gaps in the paper

**1. Case 4 of §4.3 is wrong at a boundary, and it fires on the paper's own
data.** The range for `B < 0, A < 0` is given as `[min(1, B/A), 1]`. When
`B/A ≥ 1` no α ≤ 1 satisfies `α > B/A`, so the answer is empty — but the clamp
returns the point `[1,1]`, and a non-empty pruning range *prunes an edge that
should be kept*. Table 1's first example has `A = 0` in exact arithmetic and
−4×10⁻⁸ in f32, routing it to Case 4 with `B/A = 1.25×10⁷`, so the published
formula prunes the very edge the paper states is never pruned. Corrected, with
tests pinning both the correction and the float-noise path.

**2. `B = 0` is not covered by any of the four cases**, all of which require `B`
strictly signed. It is reachable whenever two second-modality distances coincide —
common at low `m`, and two of the paper's five datasets use `m = 2` — and
`min(1, B/A)` divides by zero when `A = 0` too. Solved directly: `A < 0` admits
every `α > 0`, `A ≥ 0` admits none.

**3. The "active range" is not a range.** Algorithm 2 unions the pruning ranges of
every selected neighbour and takes the complement, which is generally several
disjoint intervals. Implemented as an interval *set*; collapsing to a hull would
re-admit α values the rule rejects.

Minor: Algorithm 2 line 5 initialises `r^x` but lines 8–9 use `r[x]`; same
variable. And §4.3 states the strategy does not extend past two modalities, since
the active range becomes a hyperplane — the authors' own limit, not a
simplification here.

## Labelled simplification

`emax`/`smax` are estimated from a seeded sample of pairs, not computed exactly.
§3.1 defines them as the maximum over *all* pairs — O(N²), 500 billion pairs at
the paper's 1M scale — and does not say how it obtains them. Safe because they
only rescale the modalities: every index and query distance divides by the same
two constants, so nothing is inconsistent; what shifts is the exchange rate α
expresses. Exhaustive is used automatically when N is small.

## What does not reproduce

**DEG does not beat Fusion**, and at the operating points tested the fixed-α
baseline does not degrade the way §3.3 reports.

100k SIFT objects, 2-D synthetic second modality, beam 100, degree 32, pool 100:

| α | DEG | Fusion | Merging |
|---|---|---|---|
| 0.0 | 0.2490 | 0.2170 | 0.3230 |
| 0.1 | 0.3250 | 0.3620 | 0.1070 |
| 0.3 | 0.3420 | **0.4270** | 0.0650 |
| 0.5 | 0.3140 | 0.4090 | 0.1020 |
| 0.7 | 0.3010 | 0.3640 | 0.1740 |
| 0.9 | 0.2790 | 0.2970 | 0.2990 |
| 1.0 | 0.2780 | 0.2380 | 0.3610 |

DEG wins at the extremes (α = 0 and 1) and loses through the middle. That is
*directionally* the right shape — DEG's curve is flatter, Fusion's peaks near its
build α of 0.5 — but the absolute numbers are far too low to claim anything.

### The diagnostic: the graphs themselves are bad

**At α = 1.0 the hybrid problem reduces exactly to single-modality SIFT search.**
Phase 6 measured our own plain `GraphIndex` on the identical 100k SIFT vectors
(`results/end_to_end.md`):

| index | beam | recall@10 |
|---|---|---|
| plain `GraphIndex`, single modality | 20 | 0.9265 |
| plain `GraphIndex`, single modality | 80 | **0.9940** |
| DEG, hybrid, at α = 1.0 | 100 | 0.2780 |
| Fusion, hybrid, at α = 1.0 | 100 | 0.2380 |

Same vectors, same distance at α = 1, a *larger* beam — and a third of the
recall. So this is not "hybrid search is intrinsically harder". The hybrid
construction is producing badly connected graphs, and every conclusion about DEG
versus Fusion is drawn on top of that defect.

Two suspects, neither yet isolated:

1. **GPS as the candidate generator.** It returns Pareto *layers*, which for an
   independent second modality are full of objects far in `δe` but close in `δs`.
   Its greedy step expands the nearest *layer*, which is not the same as
   descending toward the query in the metric the search will use. A candidate pool
   selected this way may simply contain few good `δe` neighbours.
2. **Edge seeds (§4.4).** Every search starts from nodes deliberately *farthest*
   from the centroid. At 100k the graph diameter is larger and much of the beam is
   spent travelling inward. A plain graph starts from an arbitrary interior vertex.

The honest summary of scale: the hybrid graphs need roughly **10× the beam** a
plain graph needs for comparable recall, and that ratio is the defect, not a
property of the problem.

### Operating point, and why 10k could not settle it

Recall is dominated by beam far more than by pruning policy:

| objects | beam | beam ÷ objects | DEG | Fusion |
|---|---|---|---|---|
| 10,000 | 48 | 0.5% | 0.44–0.61 | 0.39–0.61 |
| 10,000 | 250 | 2.5% | 0.90–0.98 | 0.95–0.98 |
| 100,000 | 100 | 0.1% | 0.25–0.34 | 0.22–0.43 |

At 10k/beam 250 every arm scores 0.90+ — the search explores 2.5% of the dataset,
so graph quality stops mattering and Fusion looks perfectly flat (0.952–0.982).
That is the *opposite* of the paper's finding, and it is an artefact: the paper
uses 500K–10M objects, where the same beam is a thousandth of the data. A small
dataset cannot test this claim, in either direction.

## Cost

100k objects, independent modalities:

| | DEG | Fusion | Merging |
|---|---|---|---|
| build | 437.7 s | 276.5 s | 586.8 s |
| mean out-degree | 13.0 | — | — |
| active ranges | 15.53 MiB | 0 | 0 |
| vectors | 49.59 MiB | 49.59 MiB | 49.59 MiB |

§4.5 calls the active-range overhead negligible against the vectors. Measured, it
is **31% of the vector bytes** — small but not negligible, and it would grow with
out-degree.

DEG's build is 1.6× Fusion's, and the reason is structural: Algorithm 3 line 10
re-prunes a target's entire neighbour set on every incoming reverse edge.

## A bug worth recording

Reverse edges initially inherited the forward edge's active range. A range is only
meaningful at the vertex that computed it — the range on `a → b` comes from the
triangle around `a`, tested against `a`'s other neighbours, and says nothing about
when `b → a` should be live. Copying it left most reverse edges carrying a range
for the wrong triangle, skipping them at α where they should have been traversed.

Algorithm 3 line 10's re-prune is what prevents this, and it is not free: fixing
it dropped mean out-degree from 16.3 to 11.9, because re-pruning a vertex on every
reverse edge repeatedly applies an order-dependent heuristic to its own output and
erodes edges each pass. That erosion is inherent to the algorithm as published.

Two of my own tests were ill-posed before the code was wrong, and both are
recorded in the module docs rather than quietly deleted:

- Asserting that stored ranges equal a recomputation from a vertex's final edge
  set. They cannot: Algorithm 2 tests a candidate only against neighbours
  *already* committed, so ranges depend on selection order and are not
  recoverable afterwards. Containment fails in both directions too. Replaced by
  the test that actually detects range-copying — reciprocal edges must carry
  *different* ranges.
- Asserting most edges stay traversable at any α; it failed at 36% for α = 0.
  That is correct behaviour: at α = 0 the metric is the 2-D modality alone, and an
  RNG over a plane averages about six neighbours.

## Data

DEG needs two vectors per object; SIFT has one. The second modality is a
generated 2-D coordinate — a shape the paper itself uses, not an invention:
Ins-SG and Twitter-US pair text embeddings with geographic coordinates at `m = 2`
(Table 2). Two regimes, both seeded and reproducible:

- **independent** — coordinate drawn independently. The modalities disagree, α
  genuinely matters, and a fixed-α index has something to get wrong.
- **correlated** — coordinate from a fixed random projection of the vector. A
  control: the modalities largely agree, so α should matter little, and it
  doesn't — DEG and Fusion track each other within a few points across the range.

## Status

Faithful and verified: the α algebra, Pareto frontiers and Theorem 4.1,
Algorithm 2's selection, the §4.5 early exit, the metric premise.

Implemented but not validated end to end: Algorithm 1 (GPS), Algorithm 3's build,
§4.4's edge seeds. The α-sweep shows the composition under-performing a plain
single-modality graph on the same vectors, so the claim DEG makes is untested
here rather than confirmed or refuted.

Next step to settle it: measure DEG's recall at α = 1 against `GraphIndex` on
identical vectors while swapping one component at a time — first the seeds
(edge seeds versus an interior entry point), then the candidate generator (GPS
versus ordinary beam search). Whichever swap closes the gap identifies the
defect.
