# Key–value separation

The fix for the largest finding in this project, and the one no source paper
covers.

```
cargo run --release --example kv_separation
```

## The problem

Phase 3 swept value size and found write amplification rising sharply with it
(`results/README.md`). The cause is structural: compaction rewrites the **whole
entry**, so a byte of value pays the full rewrite cost at every level its key
descends. A 3,840-byte GIST vector is rewritten in full each time its key moves
down, however many times that happens.

The four papers this project reproduces all benchmark ~100-byte values, where the
key and its bookkeeping dominate and value size is noise. At vector sizes the
ordering reverses, and **no choice of growth scheme or merge policy recovers the
difference** — the cost is in what compaction moves, not in when it runs.

## The fix

WiscKey's: write the value once to an append-only log and store a 13-byte tagged
pointer in the tree. Compaction then rewrites pointers, not payloads, so a value
is written once no matter how far its key travels.

`src/storage/vlog.rs` is the log; `LsmConfig::value_log_threshold` turns it on.
Values below the threshold stay inline, since moving them would cost a second
read to save nothing.

## Result

20,000 / 10,000 / 4,000 keys at each size, uniform key distribution, 64 KiB
memtable, vertical growth at ratio 4, leveling — the configuration Phase 3
measured in.

| value bytes | WA off | WA on | tree only, on | improvement |
|---|---|---|---|---|
| 100 | 4.92× | 1.49× | 0.59× | 3.3× |
| 512 (SIFT) | 7.95× | 1.08× | 0.10× | 7.4× |
| **3,840 (GIST)** | **19.84×** | **1.00×** | **0.01×** | **19.8×** |

**At vector sizes, write amplification collapses to the theoretical floor.** 1.00×
means the engine writes exactly the bytes it was given, once. The `tree only`
column is the LSM's own share: at GIST size it rewrites **1% of the user bytes**,
so compaction stops being a meaningful cost at all.

The benefit scales with value size — 3.3× at 100 B, 19.8× at 3,840 B — which is
exactly why the papers' ~100 B benchmarks do not surface it.

`WA on` counts **every byte written**, value log included. Separation moves bytes
rather than removing them, and a figure that ignored where they went would
flatter itself by construction. `LsmStats::write_amplification` sums SSTable and
value-log bytes for this reason; `tree_write_amplification` reports the tree's
share alone.

## What it costs

Nothing here is free, and all three costs were measured rather than assumed.

### Reads roughly double

| value bytes | read off | read on |
|---|---|---|
| 100 | 9.6 µs | 16.3 µs |
| 512 | 17.6 µs | 27.4 µs |
| 3,840 | 20.7 µs | 51.5 µs |

A lookup that used to end at the SSTable block now follows a pointer into the
log. Bloom filters and the sparse index still prune, so only hits pay it. The
penalty grows with value size because the extra read is proportional to the
payload.

### Space grows, and this is the missing garbage collector

| value bytes | disk off | disk on |
|---|---|---|
| 100 | 1.5 MiB | 2.4 MiB |
| 512 | 3.4 MiB | 5.1 MiB |
| 3,840 | 9.4 MiB | 14.8 MiB |

Separation uses **1.6× the space** at GIST size. This is not framing overhead —
it is orphaned records. The workload draws keys uniformly from a fixed space, so
many keys are overwritten; without separation, compaction reclaims the old value,
while the value log keeps it forever.

**Garbage collection is deliberately not implemented**, and this table is what
that costs. WiscKey reclaims space by scanning the log tail, checking each record
against the tree, and re-appending the live ones. Adding it would not change the
write-amplification result — which is what this exists to measure — but an
update-heavy production workload would need it. Labelled rather than hidden.

### Scan locality is lost

Values sit in **insertion order, not key order**, so a range scan that reads
values walks the log randomly. Point lookups are unaffected. This is WiscKey's
known weakness and it is not addressed here.

## The trade, stated plainly

Key–value separation converts a **write** problem into a **read and space**
problem. At 100-byte values that is a poor trade: 3.3× less writing for 1.7× more
reading and 1.6× more space. At vector sizes it is overwhelming: **19.8× less
writing** for 2.5× more reading.

Which is the point. A vector database stores vectors, and the compaction
literature is calibrated on values forty times smaller.

## Caveats

- **Insert-only workload.** Reads are measured separately, after the writes. A
  mixed workload would interleave them and the read penalty would compete with
  compaction for I/O.
- **No garbage collection**, so the space column is a floor for update-heavy
  workloads, not a typical figure.
- **The threshold is part of the on-disk format.** Small values stay inline, so
  separated mode tags every stored value; reopening a directory with a different
  setting than it was written with will fail loudly rather than misread. A
  production engine would record it in the manifest.
- Phase 3 reported **22.65×** at 3,840 B against the 19.84× here. Same
  configuration and same regime; the remaining gap is the workload mix — Phase 3
  ran a read/write mix through the full workload generator, this sweep writes
  then reads. The finding is the collapse to 1.00×, not the exact baseline.
