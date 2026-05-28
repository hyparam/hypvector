# Auto-tuning plan

Goal: make hypvector's knobs disappear for the common case. Caller passes `vectors`, `query`, `topK`. Everything else is either picked from the inputs or burned into the file at write time.

For every parameter in [PARAMETERS.md](PARAMETERS.md), we pick exactly one of four strategies:

- **Fixed** — one value that's better than alternatives across realistic regimes. No knob exposed (or expose only as an escape hatch).
- **Derive(inputs)** — compute at call time from things we already have: `N`, `dimension`, `topK`.
- **KV-metadata** — write-time decision is recorded in the parquet, search reads it transparently. No restatement at query time.
- **Document** — keep the knob, but tell people clearly when to reach for it. Falls back to a sensible default.

Each parameter below has a current state, a target strategy, and the experiments needed to lock in the strategy.

## Decision table

### Write-side

| Param | Today | Target | Why / experiment |
|---|---|---|---|
| `dimension` | required | **Required** | Caller's model dictates this. No automation possible. |
| `metric` | `'cosine'` arg | **KV-metadata** (already) | Already stored. Make `'cosine'` the default and stop asking. |
| `normalize` | `false` arg | **KV-metadata, default `true`** | Cosine + normalized = dot, which dominates everywhere. We should flip the default and just normalize if the caller doesn't say otherwise. Cheap, harmless if vectors are already unit-length. **Needed**: confirm there's no observable downside on the LLM log corpus. |
| `binary` | `false` arg | **Derive(N, dimension): on when worth it** | At ~1.5% extra bytes for ~50× fewer bytes-read in phase 2, binary is almost always worth it past ~10k vectors. **Needed**: write-time check using `N` — turn on automatically for `N ≥ ~10k`; below that, exact scan is fine. Ablate on LLM log to confirm threshold. |
| `clusters` | `0` arg | **Derive(N)** | Roughly `clusters ≈ sqrt(N)` is the IVF folklore rule (and matches our 128 for 156k = ~395 floor). **Needed**: sweep `clusters ∈ {0, sqrt(N)/2, sqrt(N), 2·sqrt(N), 4·sqrt(N)}` on LLM logs at 50k / 100k / 500k. Lock in a formula. |
| `clusterIterations` | `6` | **Fixed (6)** | The existing ablations show diminishing returns past 6. Hide the knob. |
| `clusterSeed` | `1` | **Fixed (1)** | Determinism is the only reason this exists. No reason to expose. |
| `codec` | `'UNCOMPRESSED'` | **Fixed** | Already ablated (`scripts/test-encoding.js`, `data/enc_*`). Float embeddings don't compress; SNAPPY/ZSTD costs latency. Hide. |
| `pageSize` | `1 MB` / `32 KB` when binary | **Derive(binary)** | Already automatic — keep the rule, hide the knob from the public API unless a test rig needs it. |
| `rowGroupSize` | `10000` / per-cluster | **Derive(clusters)** | Already automatic — clustered files use per-cluster row groups, unclustered uses 10k. Hide the knob. |

### Search-side

| Param | Today | Target | Why / experiment |
|---|---|---|---|
| `topK` | `10` | **Required**, default 10 | Caller intent. Keep. |
| `query`, `source`, `metadata`, `binary`, `signal` | — | **Required / passthrough** | These aren't tuning knobs. |
| `metric` | from KV | **KV-metadata** (already) | Already automatic. The argument exists only as an override; demote to "rarely needed". |
| `rerankFactor` | `10` | **Derive(N, topK)** | The README already documents `~max(10, N/3000)`. Make this the default — read `N` from KV and compute. Caller can still override for the recall/latency knob. **Needed**: confirm the `N/3000` rule on LLM logs at 100k / 500k / 1M. The wiki benchmark only validates it at 1M synthetic. |
| `probe` | `0.25` | **Derive(N, clusters)** | Probe is tightly coupled to recall. **Needed**: sweep `probe ∈ {0.05, 0.1, 0.25, 0.5, 1.0}` on LLM logs, plot recall vs. ms. If the recall@10 curve is well-behaved (monotonic, knee in a predictable place), pick a default that gives ≥90% recall; expose `probe` only when caller wants more/less recall. |

## What we need to actually run

Most parameters above resolve via existing evidence (the README ablations) or trivial code changes. The genuinely open questions all need the **same dataset** and the **same sweep harness**:

1. **The dataset**: `AmanPriyanshu/tool-reasoning-sft-CODING-jupyter-agent-dataset-sft-tool-use-agent-data-cleaned-rectified` from Hugging Face. LLM tool/code logs — repetitive, long-tailed, structurally different from wiki titles. If our defaults look wrong on this, we know they're tuned to wiki.
2. **Embed at 384-dim with MiniLM**, normalized — same model as the wiki baseline, so numbers compare directly.
3. **Sweep, at 50k / 100k / 500k row subsets**:
   - `clusters ∈ {0, sqrt(N)/2, sqrt(N), 2·sqrt(N), 4·sqrt(N)}` (write-side, expensive)
   - `rerankFactor ∈ {0, 10, 30, 100, max(10, N/3000), 300}` (cheap; redo per query set)
   - `probe ∈ {0.05, 0.1, 0.25, 0.5, 1.0}` (cheap; same)
4. **Report** recall@10, ms/query, fetches, MB read — same table format as the existing README ablation, so they're directly comparable.

If LLM log results agree with wiki, we adopt the `sqrt(N)` / `N/3000` / `probe=0.25` defaults and document. If they disagree, we keep the knobs as "tune for your corpus" and write up the difference.

## Empirical results — LLM logs, 100k × 384-dim MiniLM

From `scripts/sweep-llmlog.js`. Corpus is 100k messages from the tool-reasoning-sft dataset, embedded with `Xenova/all-MiniLM-L6-v2`, normalized. 20 in-corpus queries; reference top-10 from exact full scan.

### Clusters sweep (probe=0.25, rerankFactor=10)

| clusters | size MB | ms | fetches | MB read | recall |
|---:|---:|---:|---:|---:|---:|
| 0 (no clustering, binary only) | 160.6 | 31.9 | 104 | 8.37 | 93.0% |
| 158 (≈ √N/2) | 160.7 | **8.4** | 71 | 3.80 | 94.0% |
| 316 (≈ √N) | 160.7 | 9.4 | 105 | 3.29 | 94.0% |
| 632 (≈ 2√N) | 160.8 | 13.1 | 187 | 3.18 | 94.0% |
| 1264 (≈ 4√N) | 161.1 | 20.0 | 346 | 2.99 | 94.0% |

**Reads**: clustering wins big — 4× speedup over unclustered. The latency optimum is `√N/2`, not `√N`, because with `probe=0.25` more clusters means more row-ranges to fetch. The MB-read optimum keeps dropping with more clusters (tighter ranges), so the right `clusters` value depends on whether you optimize wall-time or bandwidth.

### rerankFactor sweep (clusters=316, probe=0.25)

| rerankFactor | ms | fetches | MB read | recall |
|---:|---:|---:|---:|---:|
| **10** | **9.4** | 105 | 3.30 | **94.0%** |
| 30 | 16.3 | 138 | 5.73 | 94.5% |
| 33 (N/3000 rule) | 17.7 | 142 | 6.08 | 94.5% |
| 100 | 39.6 | 188 | 12.83 | 94.5% |
| 300 | 100.1 | 226 | 26.12 | 94.5% |

**Read**: at 100k the `N/3000` rule from the README is overcautious for this corpus — `rf=10` is already at 94% recall, and bumping to 33 buys 0.5pp at +8ms. The rule was tuned on synthetic 1M data where binary collisions dominate; LLM logs are well-clustered enough that the default 10 holds longer.

### Probe sweep (clusters=316, rerankFactor=10)

| probe | ms | fetches | MB read | recall |
|---:|---:|---:|---:|---:|
| 0.05 | 4.4 | 35 | 2.21 | 93.0% |
| **0.10** | **5.4** | 55 | 2.54 | **94.0%** |
| 0.25 (current default) | 9.0 | 105 | 3.29 | 94.0% |
| 0.50 | 15.1 | 185 | 4.53 | 94.0% |
| 1.00 | 27.2 | 343 | 6.91 | 94.0% |

**Read**: `probe=0.10` matches the recall of `probe=0.25` at ~60% of the latency. The 0.25 default is overcautious — at least for this corpus and `clusters ≈ √N`.

### What this changes in the plan

1. **`probe` default → 0.10** (was 0.25). Same recall, ~40% faster. Worth re-confirming on wiki to make sure we're not regressing there.
2. **`rerankFactor` default → keep 10**, not `max(10, N/3000)`. At 100k LLM log, 10 is already saturated. The `N/3000` rule should be reframed as "scale up only if you observe recall below your target", not a default.
3. **`clusters` rule → `√N/2`**, not `√N`. Better latency at the same recall on this corpus. Sanity-check on wiki before locking in.
4. **All three sweeps recall-cap at 94%.** This is suspiciously flat across configs — likely the corpus has many near-duplicate tool/code messages, so top-10 is "easy". A second pass with stricter recall@1 or recall@100 metrics would be more discriminating, but the relative *ranking* across params should hold.

## Sanity check — wiki, 156k × 384-dim MiniLM

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
- `binary` write-time default: on when `N ≥ ~10k` (not yet measured at small N — assumption based on existing wiki ablation showing it's a clear win past hundreds of thousands).
- `probe` search default: stays at `0.25`. The LLM-log data tempted us to drop it; the wiki data showed why we shouldn't.
- `rerankFactor` search default: stays at `10`. The `N/3000` rule moves into the documentation as "raise this if you observe sub-target recall", not as a default.

### Still-open experiments

- Repeat the clusters sweep at 500k and 1M LLM-log row counts to confirm `√N/2` across sizes.
- Recall@100 to discriminate the LLM-log 94% ceiling.
- Find the small-N crossover where the binary column stops being worth ~1.5% bytes.


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

The advanced knobs (`rerankFactor`, `probe`, `binary` write-flag, `clusters`) stay available — but they move into an "Advanced" subsection of the README, not the quick start.
