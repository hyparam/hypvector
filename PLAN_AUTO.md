# Auto-tuning plan

Goal: make hypvector's knobs disappear for the common case. Caller passes `vectors`, `query`, `topK`. Everything else is either picked from the inputs or burned into the file at write time.

For every parameter in [PARAMETERS.md](PARAMETERS.md), we pick exactly one of four strategies:

- **Fixed**: one value that's better than alternatives across realistic regimes. No knob exposed (or expose only as an escape hatch).
- **Derive(inputs)**: compute at call time from things we already have: `N`, `dimension`, `topK`.
- **KV-metadata**: write-time decision is recorded in the parquet, search reads it transparently. No restatement at query time.
- **Document**: keep the knob, but tell people clearly when to reach for it. Falls back to a sensible default.

Each parameter below has a current state, a target strategy, and the experiments needed to lock in the strategy.

## Decision table

### Write-side

| Param | Today | Target | Why / experiment |
|---|---|---|---|
| `dimension` | required | **Required** | Caller's model dictates this. No automation possible. |
| `metric` | `'cosine'` default, in KV | **KV-metadata** (done) | Defaults to `'cosine'`, stored in KV, read transparently at search. |
| `normalize` | `false` arg | **KV-metadata, default `true` (not yet flipped)** | Cosine + normalized = dot, which dominates everywhere. Every benchmark ran normalized with no downside, and the README/quickstart already pass `true`. Open: flip the *code* default so callers can omit it. Harmless if vectors are already unit-length. |
| `binary` | **Auto (shipped)** | **Derive(N): on at N ≥ 10k** | Shipped: auto-on at `defaultAutoBinaryThreshold = 10000` (~1.5% extra bytes for ~50× fewer bytes-read in phase 2). Below threshold, exact scan is fine. Small-N crossover still unmeasured (see open experiments). |
| `clusters` | **Auto (shipped)** | **Derive(N): `round(√N/2)`** | Shipped: `round(√N/2)` when binary auto-on (`writeVectors.js`). The sweep below locked in `√N/2` over `√N` (better latency, same recall on both corpora). Caller can still pass an explicit count or `0`. |
| `clusterIterations` | `6` | **Fixed (6)** | The existing ablations show diminishing returns past 6. Hide the knob. |
| `clusterSeed` | `1` | **Fixed (1)** | Determinism is the only reason this exists. No reason to expose. |
| `codec` | `'UNCOMPRESSED'` | **Fixed** | Already ablated (`scripts/test-encoding.js`). Float embeddings don't compress; SNAPPY/ZSTD costs latency. Hide. |
| `pageSize` | `1 MB` / `32 KB` when binary | **Derive(binary)** | Already automatic: keep the rule, hide the knob from the public API unless a test rig needs it. |
| `rowGroupSize` | `10000` / per-cluster | **Derive(clusters)** | Already automatic: clustered files use per-cluster row groups, unclustered uses 10k. Hide the knob. |

### Search-side

| Param | Today | Target | Why / experiment |
|---|---|---|---|
| `topK` | `10` | **Required**, default 10 | Caller intent. Keep. |
| `query`, `source`, `metadata`, `binary`, `signal` | n/a | **Required / passthrough** | These aren't tuning knobs. |
| `metric` | from KV | **KV-metadata** (already) | Already automatic. The argument exists only as an override; demote to "rarely needed". |
| `rerankFactor` | `10` | **Fixed (10), document override** | Sweeps below kept the default at 10 (already saturated at 100k LLM-log; +2pp on wiki only at rf=30). The `~max(10, N/3000)` rule lives in the README as "raise if you see sub-target recall", not as a derived default. |
| `probe` | `0.25` | **Fixed (0.25), document override** | Sweeps below kept 0.25: LLM-log tempted 0.10, but wiki showed 0.10 → 84% recall (−9pp). 0.25 is tuned for the harder distribution. Expose only when the caller wants more/less recall. |

## Method

The open questions (clusters formula, rerank/probe defaults) were settled with one dataset and one harness, re-runnable as perf work continues:

- **Dataset**: `AmanPriyanshu/tool-reasoning-sft-...` (Hugging Face). LLM tool/code logs, structurally unlike wiki titles, so defaults that overfit wiki show up here. Embedded 384-dim MiniLM, normalized, to compare directly with the wiki baseline.
- **Harness**: `scripts/sweep-llmlog.js` (takes an optional file arg, e.g. a wiki parquet) sweeps `clusters` / `rerankFactor` / `probe` and reports recall@10, ms/query, fetches, MB read.

Results below; conclusions folded into **Final defaults**.

## Empirical results: LLM logs, 100k × 384-dim MiniLM

From `scripts/sweep-llmlog.js`. Corpus is 100k messages from the tool-reasoning-sft dataset, embedded with `Xenova/all-MiniLM-L6-v2`, normalized. 20 in-corpus queries; reference top-10 from exact full scan.

### Clusters sweep (probe=0.25, rerankFactor=10)

| clusters | size MB | ms | fetches | MB read | recall |
|---:|---:|---:|---:|---:|---:|
| 0 (no clustering, binary only) | 160.6 | 31.9 | 104 | 8.37 | 93.0% |
| 158 (≈ √N/2) | 160.7 | **8.4** | 71 | 3.80 | 94.0% |
| 316 (≈ √N) | 160.7 | 9.4 | 105 | 3.29 | 94.0% |
| 632 (≈ 2√N) | 160.8 | 13.1 | 187 | 3.18 | 94.0% |
| 1264 (≈ 4√N) | 161.1 | 20.0 | 346 | 2.99 | 94.0% |

**Reads**: clustering wins big, with a 4× speedup over unclustered. The latency optimum is `√N/2`, not `√N`, because with `probe=0.25` more clusters means more row-ranges to fetch. The MB-read optimum keeps dropping with more clusters (tighter ranges), so the right `clusters` value depends on whether you optimize wall-time or bandwidth.

### rerankFactor sweep (clusters=316, probe=0.25)

| rerankFactor | ms | fetches | MB read | recall |
|---:|---:|---:|---:|---:|
| **10** | **9.4** | 105 | 3.30 | **94.0%** |
| 30 | 16.3 | 138 | 5.73 | 94.5% |
| 33 (N/3000 rule) | 17.7 | 142 | 6.08 | 94.5% |
| 100 | 39.6 | 188 | 12.83 | 94.5% |
| 300 | 100.1 | 226 | 26.12 | 94.5% |

**Read**: at 100k the `N/3000` rule from the README is overcautious for this corpus; `rf=10` is already at 94% recall, and bumping to 33 buys 0.5pp at +8ms. The rule was tuned on synthetic 1M data where binary collisions dominate; LLM logs are well-clustered enough that the default 10 holds longer.

### Probe sweep (clusters=316, rerankFactor=10)

| probe | ms | fetches | MB read | recall |
|---:|---:|---:|---:|---:|
| 0.05 | 4.4 | 35 | 2.21 | 93.0% |
| **0.10** | **5.4** | 55 | 2.54 | **94.0%** |
| 0.25 (current default) | 9.0 | 105 | 3.29 | 94.0% |
| 0.50 | 15.1 | 185 | 4.53 | 94.0% |
| 1.00 | 27.2 | 343 | 6.91 | 94.0% |

**Read**: `probe=0.10` matches the recall of `probe=0.25` at ~60% of the latency. The 0.25 default is overcautious, at least for this corpus and `clusters ≈ √N`.

### Reading (LLM-log alone)

Taken on its own, this corpus suggested `probe → 0.10` (same recall, ~40% faster), `rerankFactor` stays 10 (already saturated), and `clusters → √N/2`. The wiki sanity check below **reverses the probe call** (see Final defaults). One caveat that holds: all three sweeps cap at ~94% recall, suspiciously flat because the corpus has many near-duplicate messages, so top-10 is "easy". recall@100 would discriminate better.

## Sanity check: wiki, 156k × 384-dim MiniLM

From `scripts/sweep-llmlog.js` against the 156k wiki corpus. Same sweeps, same code path, 20 in-corpus queries.

### Clusters (probe=0.25, rerankFactor=10)

| clusters | ms | fetches | MB read | recall |
|---:|---:|---:|---:|---:|
| 0 (no clustering, binary only) | 42.0 | 87 | 11.6 | 97.0% |
| 198 (≈ √N/2) | **13.5** | 122 | 5.6 | 93.0% |
| 395 (≈ √N) | 14.6 | 182 | 5.4 | 93.0% |
| 790 (≈ 2√N) | 19.2 | 283 | 5.0 | 92.5% |
| 1580 (≈ 4√N) | 30.9 | 491 | 5.0 | 94.5% |

### rerankFactor (clusters=395, probe=0.25)

| rerankFactor | ms | recall |
|---:|---:|---:|
| 10 | 14.6 | 93.0% |
| 30 | 27.2 | **95.0%** |
| 52 (N/3000 rule) | 39.1 | 95.5% |
| 100 | 69.2 | 96.5% |
| 300 | 173.5 | 96.5% |

### Probe (clusters=395, rerankFactor=10)

| probe | ms | recall |
|---:|---:|---:|
| 0.05 | 6.4 | **72.5%** ← regression |
| 0.10 | 8.3 | **84.0%** ← regression |
| 0.25 (default) | 14.5 | 93.0% |
| 0.50 | 23.8 | 96.5% |
| 1.00 | 41.6 | 97.0% |

### What the sanity check changed

The wiki numbers reverse two of the three LLM-log recommendations:

| Knob | LLM log says | Wiki says | Final |
|---|---|---|---|
| `clusters` | √N/2 wins on ms | √N/2 wins on ms (same recall as √N) | **Adopt `√N/2`** |
| `probe` default | 0.10 enough for 94% | 0.10 = 84% (regression of 9pp) | **Keep 0.25 as default** |
| `rerankFactor` | 10 is fine | 10→30 gains 2pp recall on wiki | **Keep 10 as default**, document `~max(10, N/3000)` as the recall-pressure knob (the README rule was right) |

The disagreement on `probe` is the most interesting finding: LLM-log retrievals are dominated by near-duplicate tool/code messages, so even probe=0.05 finds 9 of the 10 "right" answers because there are many right answers per query. Wiki has more diverse content, so cluster probing actually matters. **The 0.25 default is correct precisely because it's tuned for the harder distribution.** Don't change it.

### Final defaults (post-sanity-check)

- `clusters` write-time default: `Math.round(Math.sqrt(N) / 2)` (when binary is on). Saves wall-time at the same or near-same recall on both corpora.
- `binary` write-time default: on when `N ≥ ~10k` (not yet measured at small N; assumption based on existing wiki ablation showing it's a clear win past hundreds of thousands).
- `probe` search default: stays at `0.25`. The LLM-log data tempted us to drop it; the wiki data showed why we shouldn't.
- `rerankFactor` search default: stays at `10`. The `N/3000` rule moves into the documentation as "raise this if you observe sub-target recall", not as a default.

### Still-open experiments

- Repeat the clusters sweep at 500k and 1M LLM-log row counts to confirm `√N/2` across sizes.
- Recall@100 to discriminate the LLM-log 94% ceiling.
- Find the small-N crossover where the binary column stops being worth ~1.5% bytes.


## Product quantization: evaluated, removed

An IVF-PQ path was built, swept at 384-dim and 3072-dim (`sweep-pq.js`, `hidim-pq.js`), and **removed** (commit `92e09bc`). The lesson, kept so we don't rebuild it:

- **384-dim**: tuned PQ lost on every axis except raw bytes-read at low recall, and binary+cluster's `probe` knob beats it there too (probe=0.05 → 4.4 ms / 93% / 2.21 MB).
- **3072-dim**: PQ read fewer phase-1 bytes (9.2 vs 15.6 MB, so the bandwidth hypothesis was real) but at catastrophic recall (66% vs 95.6%) and 2-6× wall-time.
- **Why it can't win as built**: PQ-then-float-rerank keeps the full float32 column, which dominates the file at high dim (369 of 381 MB at 3072-dim). Shrinking the phase-1 code saves nothing meaningful; phase-2 float fetches dominate bytes-read regardless. PQ optimizes the cheap part.
- **The only way PQ pays off** is a *float-free lossy* mode (codes only, approximate scores), for a ~100× smaller file. That's a different feature, justified by its own benchmark, and isn't built.

## Embedding model comparison

From `scripts/sweep-models.js`, 300 conversations → 5,412 messages, 300 labeled (user-question → answer) pairs. Task: embed the whole corpus, query with each user message, check whether its answer ranks in the top-10 (self-match excluded). This measures *embedding quality* for LLM-log retrieval, independent of the index.

| model | dim | size MB | ms/q | hits@1 | hits@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| **MiniLM-L6** | 384 | 8.4 | 2.5 | 33.7% | 40.7% | 0.363 |
| bge-small | 384 | 8.4 | 2.3 | 30.3% | 40.7% | 0.347 |
| bge-base | 768 | 16.7 | 4.0 | 31.0% | 40.7% | 0.351 |
| jina-code | 768 | 16.7 | 3.9 | 12.3% | 34.0% | 0.192 |
| oai-3-small (→384) | 384 | 8.4 | 2.7 | 33.0% | 41.0% | 0.360 |
| oai-3-small | 1536 | 33.3 | 8.0 | 33.3% | 41.3% | 0.364 |
| oai-3-large | 3072 | 66.6 | 15.1 | **34.3%** | **42.0%** | **0.373** |

(`oai-*` from the OpenAI API via `OPENAI_API_KEY`; higher dims use the `dimensions` Matryoshka-truncation param.)

Lessons:
- **Model pedigree barely moves the needle; dimension dominates cost.** MiniLM-L6 → `text-embedding-3-large` gains ~1pp (hits@1 33.7 → 34.3) for 8× the file size and 6× the latency. Dim-matched at 384, oai-3-small *ties* MiniLM. → **Keep MiniLM-L6 as the documented default; recommend 384-dim models (or Matryoshka truncation, which preserves the quality).**
- **A code encoder actively hurts here**: jina-code dropped hits@1 to 12.3%, because the eval is NL→NL (prose question → prose answer). It might win on NL→code retrieval, which is worth a separate eval before recommending.
- The eval (user→answer within-conversation) is a rough proxy; treat ~41% as a *relative* yardstick, not a quality bar.

## End state for the public API

The common case is now (one open item: flip the `normalize` default to `true`):

```js
await writeVectors({
  writer: fileWriter('vectors.parquet'),
  dimension: 384,
  normalize: true, // still required explicitly; flipping the default is the last open write-side item
  vectors: embed(docs),
}) // binary auto at N≥10k, clusters≈√N/2 (both automatic)

const results = await searchVectors({
  source: 'vectors.parquet',
  query,
  topK: 10,
}) // rerankFactor=10 and probe=0.25 are fixed defaults (not derived); override for recall pressure
```

The advanced knobs (`rerankFactor`, `probe`, `binary` write-flag, `clusters`) stay available but live in an "Advanced" subsection of the README, not the quick start.
