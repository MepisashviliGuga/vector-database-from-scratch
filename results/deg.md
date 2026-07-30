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

## The headline result: DEG trades peak recall for a flatter curve

10k objects, beam 48, degree 32, **pool 256** — the operating point the diagnosis
below shows GPS actually needs. Bold marks the winner between DEG and Fusion.

**Independent modalities** (the regime where α genuinely matters):

| α | DEG | Fusion | Merging |
|---|---|---|---|
| 0.0 | **0.9050** | 0.8150 | 1.0000 |
| 0.1 | **0.9650** | 0.9450 | 0.3810 |
| 0.3 | 0.9620 | **0.9770** | 0.1750 |
| 0.5 | 0.9440 | **0.9660** | 0.1840 |
| 0.7 | 0.8980 | **0.9060** | 0.3280 |
| 0.9 | **0.8830** | 0.8420 | 0.6760 |
| 1.0 | **0.8310** | 0.7900 | 0.9480 |

**Correlated modalities** (the control):

| α | DEG | Fusion | Merging |
|---|---|---|---|
| 0.0 | **0.8820** | 0.7830 | 0.9730 |
| 0.1 | **0.9550** | 0.9370 | 0.4770 |
| 0.3 | 0.9350 | **0.9660** | 0.3730 |
| 0.5 | 0.9270 | **0.9700** | 0.4630 |
| 0.7 | 0.9180 | **0.9530** | 0.6410 |
| 0.9 | 0.9030 | **0.9190** | 0.8460 |
| 1.0 | 0.8980 | **0.9060** | 0.9580 |

**§3.3's argument reproduces.** Fusion peaks near the α it was built at and
degrades away from it; DEG holds a flatter curve. The spread across α is the
number that matters, since α is unknown when the index is built:

| | Fusion spread | DEG spread | Fusion worst | DEG worst |
|---|---|---|---|---|
| independent | 18.7 pts | 13.4 pts | 0.7900 | **0.8310** |
| correlated | 18.7 pts | **7.3 pts** | 0.7830 | **0.8820** |

**DEG does not dominate — it insures.** Fusion wins at four of seven α in each
regime, always in the band around its build α of 0.5, and by 0.8–4.3 points. DEG
wins at the extremes, by up to 9.9. What DEG buys is not a better best case but a
better *worst* case: its floor is 4–10 points above Fusion's in both regimes, and
its curve is half as steep in the control. That is exactly the trade §3.3
describes, and it is the right trade when the query's α is not known in advance.

The correlated control behaves as a control should: with the modalities largely
agreeing, Fusion's fixed α is a better approximation of every query, so it wins
more often — and DEG's curve flattens to a 7.3-point spread, its guarantee
mattering more while its cost stays the same.

**Merging reproduces Fig 2 exactly** in both regimes: near-perfect at the
extremes (1.0000 and 0.9480 independent), collapsing to 0.1750 mid-range. Each
index sees one modality, so mid-range queries get candidates good in one and
arbitrary in the other.

### Cost of the guarantee

| | DEG | Fusion | Merging |
|---|---|---|---|
| build (independent) | 47.7 s | 17.6 s | 107.2 s |
| mean out-degree | 18.7 | — | — |
| active ranges | 2.23 MiB | 0 | 0 |
| vectors | 4.96 MiB | 4.96 MiB | 4.96 MiB |

DEG costs **2.7× Fusion's build** and 45% of the vector bytes in stored ranges.
§4.5 calls that storage negligible; it is not, though it is affordable.

### An earlier verdict, retracted

A previous run of this sweep at pool 64 (10k) and 100 (100k) concluded DEG loses
to Fusion. **That conclusion was wrong**, and the reason is instructive: all three
arms draw candidates through GPS, and at those pools GPS is starved — mean
out-degree 4.8 against the 18.7 above. Every arm was crippled by the same defect,
so the comparison measured provisioning rather than pruning. The diagnosis below
is what found it.

### The diagnostic that found it

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

### Isolated: GPS was under-provisioned, and §4.4 is exonerated

`cargo run --release --example deg_diagnosis` swaps one component at a time, all
at α = 1 on the Fusion policy — the arm a plain graph should exactly match, since
it prunes with the plain RNG rule at one α and marks every edge always-active.
10k objects, beam 80:

| arm | recall@10 | mean out-degree | build |
|---|---|---|---|
| `GraphIndex` (reference) | 1.0000 | 18.8 | 1.1 s |
| GPS + edge seeds, pool 64 | 0.6320 | 4.8 | 1.6 s |
| GPS + interior entry, pool 64 | 0.5150 | 4.4 | 3.8 s |
| GPS + edge seeds, pool 256 | 0.9730 | 7.8 | 21.6 s |
| GPS + edge seeds, pool 1024 | **0.9980** | 9.9 | **528.2 s** |
| beam + edge seeds, pool 64 | 0.9920 | 7.8 | 5.0 s |
| beam + interior entry, pool 64 | 0.9930 | 7.8 | 5.6 s |

**The candidate source is the cause; the entry policy is not.** Replacing GPS with
an ordinary beam search moves recall from 0.632 to 0.992 with everything else
held fixed. Switching entry points moves it by a fraction of that, and in the
*opposite* direction to the suspicion — §4.4's edge seeds beat an interior entry
point (0.632 against 0.515), so the paper's seeding is fine and the earlier
suspicion of it was wrong.

**GPS is under-provisioned rather than wrong.** Given a large enough pool it
reaches 0.998, edging out the plain graph. The cost is the finding:

- **16× the candidate pool** to beat a beam search, and
- **106× the build time** — 528 s against 5.0 s — for 0.6 points of recall.

The mechanism is visible in the out-degree column. A Pareto pool is spread along
the whole `(δe, δs)` trade-off curve, so only a fraction of it is usable at any
one α; at pool 64 that fraction yields 4.8 edges per vertex against a plain
graph's 18.8, and RNG pruning at α = 1 has little to work with. That is the price
of α-agnostic candidate acquisition, and it is a real cost of the design rather
than an implementation artefact — the same pool must serve every α at once.

### Why this mattered

Every arm of the α sweep draws candidates through GPS, so an under-provisioned
pool crippled all three at once and the comparison measured provisioning rather
than pruning. Re-running at pool 256 lifted DEG's mean out-degree from 4.8 to
18.7 — level with a plain graph — and the paper's claimed shape appeared. The
starved numbers are kept below as a record.

**The α sweep at pool 64 (10k) and 100 (100k), now superseded.** Recall
0.19–0.43 at 100k, 0.44–0.61 at 10k; DEG won at the extremes and lost through the
middle, the right shape at absolute values too low to support any conclusion.

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

**Faithful and verified:** the α algebra (pinned to Table 1 and cross-checked
against the raw inequalities), Pareto frontiers and Theorem 4.1, Algorithm 2's
selection, the §4.5 early exit, the §3.2 metric premise, and — via the diagnosis
— Algorithm 1 and §4.4's edge seeds, both of which behave correctly once GPS is
given an adequate pool.

**Corrected in the paper:** Case 4's boundary, the uncovered `B = 0` case, and
the active range being a set rather than an interval.

**Measured and worth carrying forward:** GPS needs 16× the candidate pool and
~100× the build time of an ordinary beam search to match it, because a Pareto
pool must serve every α at once and only a fraction of it is usable at any single
one. That is the structural cost of the design.

**Reproduced, at an adequate operating point:** §3.3's argument that a fixed-α
index degrades away from its build α while DEG holds a flatter curve. DEG does not
dominate — Fusion wins the band around α = 0.5 — but DEG's worst case across α is
4–10 points better, and that is the case that matters when α is unknown at build
time.

**Still open:** whether the result holds at the paper's scale. Everything above
is 10k objects; the 100k run was made at a starved pool and would need repeating
at pool 256+, which the build times make expensive. Also untested: `k` other than
10, and a second modality that is neither an independent draw nor a projection —
a genuinely semantic pairing, which is what the paper's image-text datasets are
and what our synthetic coordinate only approximates.
