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
| `metric` | `'cosine'` arg | **KV-metadata** (already) | Already stored. Make `'cosine'` the default and stop asking. |
| `normalize` | `false` arg | **KV-metadata, default `true`** | Cosine + normalized = dot, which dominates everywhere. We should flip the default and just normalize if the caller doesn't say otherwise. Cheap, harmless if vectors are already unit-length. **Needed**: confirm there's no observable downside on the LLM log corpus. |
| `binary` | `false` arg | **Derive(N, dimension): on when worth it** | At ~1.5% extra bytes for ~50× fewer bytes-read in phase 2, binary is almost always worth it past ~10k vectors. **Needed**: write-time check using `N`; turn on automatically for `N ≥ ~10k`. Below that, exact scan is fine. Ablate on LLM log to confirm threshold. |
| `clusters` | `0` arg | **Derive(N)** | Roughly `clusters ≈ sqrt(N)` is the IVF folklore rule (and matches our 128 for 156k = ~395 floor). **Needed**: sweep `clusters ∈ {0, sqrt(N)/2, sqrt(N), 2·sqrt(N), 4·sqrt(N)}` on LLM logs at 50k / 100k / 500k. Lock in a formula. |
| `clusterIterations` | `6` | **Fixed (6)** | The existing ablations show diminishing returns past 6. Hide the knob. |
| `clusterSeed` | `1` | **Fixed (1)** | Determinism is the only reason this exists. No reason to expose. |
| `codec` | `'UNCOMPRESSED'` | **Fixed** | Already ablated (`scripts/test-encoding.js`, `data/enc_*`). Float embeddings don't compress; SNAPPY/ZSTD costs latency. Hide. |
| `pageSize` | `1 MB` / `32 KB` when binary | **Derive(binary)** | Already automatic: keep the rule, hide the knob from the public API unless a test rig needs it. |
| `rowGroupSize` | `10000` / per-cluster | **Derive(clusters)** | Already automatic: clustered files use per-cluster row groups, unclustered uses 10k. Hide the knob. |

### Search-side

| Param | Today | Target | Why / experiment |
|---|---|---|---|
| `topK` | `10` | **Required**, default 10 | Caller intent. Keep. |
| `query`, `source`, `metadata`, `binary`, `signal` | n/a | **Required / passthrough** | These aren't tuning knobs. |
| `metric` | from KV | **KV-metadata** (already) | Already automatic. The argument exists only as an override; demote to "rarely needed". |
| `rerankFactor` | `10` | **Derive(N, topK)** | The README already documents `~max(10, N/3000)`. Make this the default: read `N` from KV and compute. Caller can still override for the recall/latency knob. **Needed**: confirm the `N/3000` rule on LLM logs at 100k / 500k / 1M. The wiki benchmark only validates it at 1M synthetic. |
| `probe` | `0.25` | **Derive(N, clusters)** | Probe is tightly coupled to recall. **Needed**: sweep `probe ∈ {0.05, 0.1, 0.25, 0.5, 1.0}` on LLM logs, plot recall vs. ms. If the recall@10 curve is well-behaved (monotonic, knee in a predictable place), pick a default that gives ≥90% recall; expose `probe` only when caller wants more/less recall. |

## What we need to actually run

Most parameters above resolve via existing evidence (the README ablations) or trivial code changes. The genuinely open questions all need the **same dataset** and the **same sweep harness**:

1. **The dataset**: `AmanPriyanshu/tool-reasoning-sft-CODING-jupyter-agent-dataset-sft-tool-use-agent-data-cleaned-rectified` from Hugging Face. LLM tool/code logs: repetitive, long-tailed, structurally different from wiki titles. If our defaults look wrong on this, we know they're tuned to wiki.
2. **Embed at 384-dim with MiniLM**, normalized: same model as the wiki baseline, so numbers compare directly.
3. **Sweep, at 50k / 100k / 500k row subsets**:
   - `clusters ∈ {0, sqrt(N)/2, sqrt(N), 2·sqrt(N), 4·sqrt(N)}` (write-side, expensive)
   - `rerankFactor ∈ {0, 10, 30, 100, max(10, N/3000), 300}` (cheap; redo per query set)
   - `probe ∈ {0.05, 0.1, 0.25, 0.5, 1.0}` (cheap; same)
4. **Report** recall@10, ms/query, fetches, MB read, using the same table format as the existing README ablation, so they're directly comparable.

If LLM log results agree with wiki, we adopt the `sqrt(N)` / `N/3000` / `probe=0.25` defaults and document. If they disagree, we keep the knobs as "tune for your corpus" and write up the difference.

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

### What this changes in the plan

1. **`probe` default → 0.10** (was 0.25). Same recall, ~40% faster. Worth re-confirming on wiki to make sure we're not regressing there.
2. **`rerankFactor` default → keep 10**, not `max(10, N/3000)`. At 100k LLM log, 10 is already saturated. The `N/3000` rule should be reframed as "scale up only if you observe recall below your target", not a default.
3. **`clusters` rule → `√N/2`**, not `√N`. Better latency at the same recall on this corpus. Sanity-check on wiki before locking in.
4. **All three sweeps recall-cap at 94%.** This is suspiciously flat across configs; likely the corpus has many near-duplicate tool/code messages, so top-10 is "easy". A second pass with stricter recall@1 or recall@100 metrics would be more discriminating, but the relative *ranking* across params should hold.

## Sanity check: wiki, 156k × 384-dim MiniLM

From `scripts/sweep-llmlog.js data/wiki_en.vectors.parquet`. Same sweeps, same code path, 20 in-corpus queries.

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


## PQ tuning: does IVF-PQ ever beat binary+cluster?

From `scripts/sweep-pq.js` + `scripts/sweep-pq-probe.js` on the 100k LLM-log corpus (384-dim). Swept `pqSegments × pqCentroids × ivfClusters`, then probe/rerankFactor on the best config.

Best PQ configs vs. the binary+cluster default (clusters=√N/2 = 158 → 8.4 ms / 94% recall / 3.8 MB):

| Path | Config | ms | recall | MB read |
|---|---|---:|---:|---:|
| **binary+cluster (default)** | clusters=158 | **8.4** | **94%** | 3.80 |
| PQ — fastest decent | s32/c64/ivf128 | 12.6 | 90.5% | 3.65 |
| PQ — best recall @ probe 0.25 | s64/c256/ivf316 | 28.9 | 94% | 4.60 |
| PQ — full probe + rf=30 | s64/c64/ivf128 | 74.0 | 94.5% | 11.2 |
| PQ — bandwidth optimum | s16/c64/ivf316 | 14.1 | 73.5% | **2.48** |

PQ's recall ceiling is ~94% even at probe=1.0 (residual codes lose top-10 signal); matching binary+cluster's recall costs 1.5–9× the latency. **At 384-dim, tuned PQ still loses on every axis except raw bytes-read at low recall** — and binary+cluster's `probe` knob beats it there too (probe=0.05 → 4.4 ms / 93% / 2.21 MB).

### High dimension: tested, PQ still loses

The remaining hope for PQ was high dimension — the binary column grows as `dim/8` while a PQ code stays at `pqSegments` bytes, so PQ's phase-1 scan should read far less. Tested at **3072-dim** (`text-embedding-3-large`), 30k LLM-log messages, via `scripts/hidim-pq.js`:

| variant | file MB | ms | fetches | MB read | recall |
|---|---:|---:|---:|---:|---:|
| **binary+cluster** | 381.2 | **11.6** | 48 | 15.6 | **95.6%** |
| pq s32/c64 | 374.5 | 22.0 | 53 | **9.2** | 66.0% |
| pq s64/c64 | 375.5 | 22.3 | 54 | 9.8 | 73.4% |
| pq s96/c256 | 379.6 | 64.8 | 57 | 14.9 | 87.2% |

PQ *does* read fewer bytes (9.2 vs 15.6 MB) — the bandwidth hypothesis was real — but at catastrophic recall loss (66% vs 95.6%) and 2–6× the wall-time (building PQ distance tables across IVF cells is CPU-heavy at high dim). **So no: PQ does not win at OpenAI scale.**

The reason is structural and kills the whole premise: at 3072-dim the **float32 rerank column is 12,288 bytes/row**, so it dominates the file (369 of 381 MB). The binary column is only 384 bytes/row — already negligible — so shrinking phase-1 to a 32-byte PQ code saves nothing meaningful on total size (374 vs 381 MB), and phase-2 *float* fetches (which both paths keep, for exact rerank) dominate bytes-read regardless. PQ optimizes the cheap part.

**The actual high-dim cost driver is keeping the full float column at all.** The only way PQ pays off is a *lossy* mode — PQ codes with **no** float column, accepting approximate scores — which would shrink a 381 MB file to ~3 MB. That's a different feature (lossy/quantized-only storage) than the current PQ-then-float-rerank, and isn't built. Conclusion: **drop or de-emphasize the current IVF-PQ path; if PQ comes back, it should be as a float-free lossy mode, justified by its own benchmark.**

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

(`oai-*` from the OpenAI embeddings API via `OPENAI_API_KEY`; higher dims use the `dimensions` Matryoshka-truncation param. API embedding is ~150–375 msg/s vs MiniLM's 55.)

**The headline: embedding model choice barely moves the needle on this task, and dimension cost dominates.** From MiniLM-L6 (free, local, 384-dim) to `text-embedding-3-large` (OpenAI's best, 3072-dim), hits@1 moves 33.7% → 34.3% and hits@10 moves 40.7% → 42.0% — within noise. But oai-3-large costs **8× the file size (66.6 vs 8.4 MB) and 6× the per-query latency (15.1 vs 2.5 ms)** for that ~1 pp. Dim-matched at 384, OpenAI's small model *ties* MiniLM exactly (33.0% / 41.0%). So the only thing that materially changes the hypvector cost profile is the embedding **dimension**, not the model's pedigree.

**The code-specialized model actively hurts.** `jinaai/jina-embeddings-v2-base-code` (768-dim) dropped hits@1 from 33.7% to 12.3% and embeds ~10× slower (4–5 msg/s). Reason: **the eval task is natural-language → natural-language** — a prose user question ("Which feature has the most outliers?") retrieving a prose answer. A code encoder tunes its space for code structure (code↔code, code↔docstring), the wrong objective for NL Q&A. It would likely win on a *different* task — NL intent → retrieve the `tool_call`/code cell — but that's not what user→answer retrieval measures.

Takeaways:
- **Keep MiniLM-L6 as the documented default.** Nothing tested beats it on quality-per-byte; the SOTA paid model adds ~1 pp at 8× the storage.
- **Dimension is the real cost lever.** If a user brings a 1536- or 3072-dim model, the linear-scan file and query both grow proportionally (visible above: ms/q 2.5 → 8.0 → 15.1, size 8.4 → 33.3 → 66.6 MB). This is exactly the regime where dimensionality reduction (Matryoshka truncation — oai-3-small→384 keeps the quality) or PQ compression earns its keep. **Recommend 384-dim models, or truncating, in the docs.**
- Model choice is task-dependent: a code encoder may still help when retrieving *code/tool messages* rather than NL answers — worth a separate code-retrieval eval before recommending one there.
- The eval (user→answer within-conversation) is a rough proxy; treat the absolute ~41% as a *relative* yardstick, not a quality bar.

## End state for the public API

After the experiments above, the common case should look like:

```js
await writeVectors({
  writer: fileWriter('vectors.parquet'),
  dimension: 384,
  vectors: embed(docs),
}) // normalize=true, binary if N≥~10k, clusters≈sqrt(N), all automatic

const results = await searchVectors({
  source: 'vectors.parquet',
  query,
  topK: 10,
}) // rerankFactor and probe derived from KV count
```

The advanced knobs (`rerankFactor`, `probe`, `binary` write-flag, `clusters`) stay available, but they move into an "Advanced" subsection of the README, not the quick start.
