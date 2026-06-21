# OPTIMIZE.md — reducing roundtrips and bytes for S3-backed search

## Goal

HypVector is meant to run fully serverless: the index is one Parquet file on
S3 (or any HTTP range source), and *all* compute happens in the client over the
network. The cost function we are optimizing is therefore:

```
query cost ≈ (number of dependent network roundtrips) × (cold latency ~100–250 ms each)
           + (bytes transferred) / (bandwidth)
```

Three levers, in priority order of impact for *cold* object-storage reads:

1. **Roundtrips** — each dependent fetch is ~100–250 ms cold. Fewer, larger,
   parallel range GETs beat many small serial ones.
2. **Bytes on wire** — dominated by the float32 `vector` column. Quantization is
   the only big lever; Parquet codecs barely move embeddings.
3. **Query latency** — keep or improve client-side scan/rerank speed.

This file is a backlog of investigations. Each item states what it is, the
concrete expected win, the implementation cost, and a way to validate it.
PLAN_AUTO.md covers the *already-shipped* auto-tuning decisions; this file is
about the next frontier.

---

## What we already do (baseline — don't re-investigate)

Several things the literature recommends are already in the code. Stating them
so we don't waste an experiment re-discovering them:

- **IVF-style binary k-means clustering**, `round(√N/2)` clusters by default,
  centroids + per-cluster counts in Parquet KV metadata (`src/cluster.js`,
  `src/writeVectors.js`).
- **Rows sorted by cluster**, and **each cluster written as its own row group**
  (`rowGroupSize` = array of per-cluster counts). A probed list is already a
  contiguous row range.
- **Clusters renumbered by a greedy Hamming walk** (`reorderClustersByHamming`)
  so the nearest clusters to any query tend to land in adjacent id ranges,
  which `mergeRanges` then coalesces into fewer reads.
- **Two-phase search**: phase-1 Hamming scan over the 1-bit `vector_bin`
  column, phase-2 float32 rerank over `rerankFactor × topK` candidates
  (`src/search/rerank.js`).
- **`useOffsetIndex: true` in phase 2 and the id fetch**, with run coalescing
  (64-row gap tolerance) so scattered candidates become a few range GETs.
- **Uncompressed PLAIN** float32 (correct default — see Experiment B).

So the IVF instinct, the contiguous-list layout, and offset-index page seeking
in the rerank phase are done. The open work is below.

### Already tried and removed — do not rebuild as-is

Two quantization schemes were built, benchmarked, and **deleted** as net
negatives. The shared lesson governs everything in Tier 2:

- **int8 cascade tier** (commit `e3e37f8`): an int8 column between phase-1
  binary and phase-2 float32. Saved only ~0.3 MB of phase-2 reads but added
  ~38 MB of file size and ~22 extra fetches per query. Net negative.
- **IVF-PQ** (commit `92e09bc`, documented in PLAN_AUTO.md): lost on every axis
  except raw phase-1 bytes; at 3072-dim it read fewer phase-1 bytes but at 66%
  recall and 2–6× wall-time.

**The lesson:** any quantizer that *adds a tier while keeping the full float32
`vector` column* optimizes the cheap part. Phase-2 float fetches dominate
bytes-read regardless, so shrinking phase-1 codes saves nothing meaningful, and
a new column only adds size and fetches. **The only quantization that can win is
a float-free lossy mode** — codes only, approximate final scores, no float32
column at all — for a multiplicatively smaller file. That reframes Tier 2 below:
the bar is "replace float32," never "add a tier beside it."

---

## Tier 1 — highest leverage

### Experiment A: RaBitQ in place of raw sign bits (bytes-neutral recall win)

**What.** Our `vector_bin` column is the raw sign bit per dimension. RaBitQ
(Gao & Long, SIGMOD 2024, arxiv 2405.12497) keeps the *same 1 bit/dim, same
32× size* but first applies a random orthogonal rotation (Johnson–Lindenstrauss)
and uses an *unbiased* distance estimator with a provable `O(1/√D)` error bound.
It is the same byte cost as what we ship, with a strictly better phase-1
estimator that doesn't collapse on hard distributions the way PQ can.

**Win.** Higher phase-1 recall at fixed candidate budget → we can lower
`rerankFactor` (fewer phase-2 bytes) at equal end recall, or raise recall at
fixed `rerankFactor`. Pure upside at the same on-wire size for `vector_bin`.

**Cost.** Medium. Need: a fixed random rotation (seedable, stored in KV
metadata so the reader reproduces it), encode = rotate then sign, and a phase-1
scorer that uses the RaBitQ estimator instead of raw Hamming. The rotation is
the only new moving part; everything else is our existing pipeline. Reference:
github.com/VectorDB-NTU/RaBitQ-Library.

**Validate.** Reuse `scripts/validate-params.js` recall harness: compare
recall@10/@100 of raw-sign vs RaBitQ at identical `rerankFactor` and probe, on
wiki (384-dim) and a 1024-dim corpus. Win = higher recall, or equal recall at
lower `rerankFactor`.

### Experiment C: phase-1 offset-index page skipping — RESOLVED (no change)

**Outcome (2026-06-21): already handled by design; not an opportunity.**

The premise was wrong. Phase 1 deliberately reads whole binary column chunks,
and `rerank.js:51-57` documents why: the binary column is `dim/8` bytes/row, so
per-page `useOffsetIndex` seeking costs an extra roundtrip to read the offset
index without saving meaningful bytes. The 32 KB binary page size exists for
*phase 2* candidate seeking, not phase 1. Moreover there is a `prefetchBinary`
path (`src/prefetch.js`) that loads the entire small binary column into RAM
once, making phase 1 *zero-network* — strictly better than page-seeking it.
Nothing to do here.

### Experiment D: nprobe — RESOLVED (cap the fraction at scale)

**Outcome (2026-06-21): keep the fraction, but add an absolute cap.**

Measured probe sweeps (`scripts/validate-params.js probe`) on wiki (384-dim,
N=20k–156k) and tpuf (1024-dim, N=250k/1M), clusters at the shipped √N/2:

- **The "switch to absolute probe" idea is refuted.** A fixed absolute count
  lets recall *slide* as N grows, because clusters grow as √N/2 so a constant
  count is a shrinking fraction. probe=16 → 91% @20k, 79% @80k, 81% @156k.
  The 0.25 *fraction* holds recall steady (91→90→93%) across 8× scale — it is
  the correct parameterization here, not absolute count. (This is why the
  literature's "~16–32 probes" rule doesn't transfer: it assumes
  nlist≈C·√N with large C; we use √N/2, far fewer/bigger lists.)

- **But the fraction over-probes at large N.** At 1M (500 clusters), probed
  list count vs cost/recall:

  | lists | fetches | MB read | recall@10 |
  |------:|--------:|--------:|----------:|
  | 48 | 118 | 17.8 | 89.0% |
  | 64 | 137 | 21.8 | 91.0% |
  | 80 | 155 | 25.7 | 92.0% |
  | 96 | 172 | 29.9 | 92.5% |
  | **125 (=0.25 frac)** | **202** | **37.0** | **93.0%** |

  Recall knees at ~80 lists (92%). The fraction's last 1pp (92→93%) costs +47
  fetches and +11 MB — ~30% more roundtrips and bytes for marginal recall.

**Recommended change:** `probe = min(ceil(fraction × nlist), cap)` with
`cap ≈ 80–96`. The cap only binds above ~400k vectors (where 0.25·√N/2 > 80),
so all current small/medium-N behavior is unchanged; at 1M it trims ~25% of
roundtrips and ~30% of bytes for ~1pp recall. Backward-compatible, low risk.
Open question: exact cap value (80 vs 96) and whether it's user-overridable.

---

## Tier 2 — meaningful, more work

### Experiment E: float-free lossy mode (the only quantization that can win)

**What.** A search mode with **no float32 column at all** — final scores come
from a multi-bit code. Candidate codec: Extended RaBitQ (SIGMOD 2025, arxiv
2409.09913), B bits/dim, reported **B=5 → >95% recall at 6.4×, B=7 → >99% at
4.5×**, beating scalar quantization at equal bits and good enough that there is
nothing to rerank against. This is the *float-free lossy* feature PLAN_AUTO
named as "the only way quantization pays off," now with a codec that might
actually hit the recall bar.

**Win.** Multiplicatively smaller file — the float32 `vector` column is ~3/4 of
the bytes and the bulk of phase-2 reads. Removing it (not shrinking it, not
adding a tier beside it) is the single biggest bytes-on-wire reduction
available. This is a *different feature* from today's exact-rerank index, with
its own recall/size contract, not a drop-in tier.

**Cost.** High. New multi-bit codec, new scorer, a new file mode, and a clear
API story that this trades exactness for ~5–6× smaller files. Reuses the RaBitQ
rotation from Experiment A. Gate strictly behind its own benchmark.

**Validate.** This is the make-or-break number for the whole quantization line:
does a *float-free* index hold ≥95% recall@10 on real corpora (384- and
1024-dim, ≥500k)? Compare file size, MB read, and recall against today's
binary+float32. If float-free can't clear the recall bar, quantization stays
shelved — adding a tier beside float32 is already proven net-negative (see
"Already tried and removed").

> **Rejected: int8 / any tier beside float32.** An int8 cascade tier was built
> and removed (`e3e37f8`) for exactly the "optimizes the cheap part" reason.
> Do not re-propose int8, PQ, or RaBitQ *as an added column* — only as a
> float32 *replacement* per Experiment E.

### Experiment G: two-level centroid index for large nlist

**What.** Centroids live in KV metadata and are scanned linearly to rank
clusters (`ranges.js`). Fine for √N/2 clusters at small N; at 1M+ vectors
(~700+ clusters, growing) that linear scan and the metadata size both grow.
SPANN's answer: a small index *over the centroids* so finding the K nearest
clusters is sub-ms, plus optional boundary-vector replication into a few nearby
lists to lift recall without raising nprobe.

**Win.** Keeps cluster selection cheap as nlist grows, and bounds KV-metadata
size. Mostly a scale concern (>1M).

**Cost.** Medium–high. New in-file structure for centroids; replication
inflates data ~20%. Only worth it once nlist is large enough that linear
centroid scan or metadata bloat actually shows up.

**Validate.** Measure centroid-scan time and KV-metadata bytes vs N; only pursue
if either becomes material at target corpus sizes.

---

## Tier 3 — measure first, likely small or negative

### Experiment B: float column encoding & compression (probably a no-op)

**What.** We ship PLAIN + UNCOMPRESSED float32. Candidates: BYTE_STREAM_SPLIT
encoding, and zstd/snappy compression.

**Expected.** **Small or zero.** Unit-norm float32 embeddings are near-
incompressible: the mantissa is ~7.3 bits/byte, so lossless ratios sit around
1.08–1.20×, and snappy/zstd cost decode latency for ~5–10%. BYTE_STREAM_SPLIT
averages ~30% on *scientific* floats but is unproven on embeddings and has gone
*negative* on some data. This is why UNCOMPRESSED is the current default and the
right one.

**Cost.** Low to test (writer flags), but **gate any change behind a write-time
sample A/B** — never always-on. Most likely outcome: confirm the default and
move on.

**Validate.** On real corpora, write the float column under PLAIN, PLAIN+zstd,
and BYTE_STREAM_SPLIT; compare file size and phase-2 decode time. Adopt only if
a corpus shows a clear net win.

### Experiment H: cold-open roundtrip floor

**What.** Confirm the very first fetch sequence is minimal: over-read the file
tail (~64 KB) in one GET to grab the footer + KV metadata in a single roundtrip,
then the page index, then coalesced data ranges. Target ~3 roundtrips for a
selective cold query.

**Win.** Shaves fixed startup latency off every cold query. Small but every
query pays it.

**Cost.** Low. Mostly verifying what hyparquet already does on a real S3/HTTP
source and adding a tail over-read hint if it issues a separate tiny footer GET.

**Validate.** Count `fetches` on a cold `asyncBuffer` for a single query; aim to
drive the fixed overhead to ~2 (footer+index) before data reads begin.

### Explicitly out of scope / rejected

- **Bloom filters** — answer equality/membership only, never nearest-neighbor.
  Irrelevant to the vector path.
- **Dictionary encoding** — embeddings are continuous/high-cardinality; Parquet
  falls back to PLAIN anyway, and PLAIN is what we want for SIMD scan.
- **Graph indexes (HNSW / DiskANN / Vamana)** — dozens of *serial dependent*
  hops per query, each a cold fetch. Catastrophic on object storage. IVF is the
  correct family for S3 and we already use it. Do not pursue graph indexes for
  the cold tier.

---

## Suggested sequencing

1. ~~**D (nprobe)** and **C (phase-1 offset index)**~~ — DONE (2026-06-21).
   C was already handled by design (no change). D → keep the fraction but add an
   absolute cap (~80–96) that bounds over-probing above ~400k vectors. The cap
   is the one remaining code change from this tier; everything else here was a
   confirm-the-default.
2. **A (RaBitQ pre-rank)** — bytes-neutral recall win; also builds the rotation
   machinery E depends on.
3. **B (encoding A/B)** — quick confirm-or-reject, probably confirms the default.
4. **E (float-free lossy mode)** — the only quantization that can win, since a
   tier beside float32 is already proven net-negative. One make-or-break recall
   benchmark decides whether the whole quantization line is alive.
5. **G (two-level centroids)** and **H (cold-open floor)** — scale and
   fixed-overhead polish, pursue when measurements say they matter.

All experiments validate through `scripts/validate-params.js` (extend its
subcommands) using `recall@10` / `recall@100`, average `fetches`, and average
`MB read` on real corpora (wiki 384-dim, a 1024-dim set, and a ≥500k corpus for
scale). A change ships only when it improves bytes or roundtrips at equal-or-
better recall.
