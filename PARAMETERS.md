# PARAMETERS

Every user-facing knob in hypvector, what it does, and where the value lives at query time. The companion file [PLAN_AUTO.md](PLAN_AUTO.md) tracks how each one becomes automatic.

## Write-side (`writeVectors`)

| Param | Default | What it does |
|---|---|---|
| `dimension` | required | Length of each vector. Stored in KV `hypvector.dimension`. |
| `metric` | `'cosine'` | Intended similarity metric. Hint stored in KV; search reads it. |
| `normalize` | `false` | L2-normalize on write; lets cosine score via dot product. Stored in KV `hypvector.normalized`. |
| `binary` | auto (on at `N ≥ 10000`) | Also write a 1-bit-per-dim sign column (`vector_bin`) for the Hamming phase-1 rerank path. Adds ~`dim/8` bytes/row (~1.5% at 384-dim). Pass `false` to force-off. |
| `clusters` | auto (`round(sqrt(N)/2)` in full auto mode) | Number of k-means clusters. Implies `binary: true`. Rows are reordered by cluster id; centroids + per-cluster counts go into KV. Enables phase-1 cluster skipping. Pass `0` to force-off, or an integer to set explicitly. |
| `clusterIterations` | `6` | k-means iterations over the 1-bit codes. |
| `clusterSeed` | `1` | RNG seed for deterministic clustering. |
| `codec` | `'UNCOMPRESSED'` | Parquet codec. SNAPPY/ZSTD rarely shrink float embeddings and cost query latency. |
| `pageSize` | `1 MB` (or 32 KB when `binary`) | Parquet page size. Smaller pages let `useOffsetIndex` fetch tighter byte ranges during rerank phase 2. |
| `rowGroupSize` | `10000` (or per-cluster sizes when clustered) | Rows per row group. When clustering, each cluster becomes its own row group. |

## Search-side (`searchVectors`)

| Param | Default | What it does |
|---|---|---|
| `query` | required | The query vector. Must match `dimension`. |
| `source` | required | URL, file path, AsyncBuffer, or array of any of those (parallel multi-file search). |
| `topK` | `10` | Number of nearest neighbors to return. |
| `metric` | from KV | Override the stored metric. Almost never needed. |
| `rerankFactor` | `10` | Candidate pool size = `topK × rerankFactor`. `0` forces exact full scan. Higher = more recall, more bytes fetched. Suggested `~max(10, N/3000)`. |
| `probe` | `0.25` | Fraction (or integer count) of clusters to scan in phase 1. Lower = faster, lower recall. Ignored if file has no centroids. |
| `binary` | none | Pre-fetched binary column (from `prefetchBinary`). When provided, phase-1 Hamming scan runs from memory. |
| `metadata` | none | Pre-parsed parquet metadata, reused across queries. Pure latency win. |
| `signal` | none | AbortSignal. |
| `asyncBufferFactory` | `cachedAsyncBuffer` wrapper | How to open a string `source`. |
| `compressors` | none | Custom decompressor map. |

## Where each parameter is decided

- **Stored in KV metadata, read implicitly at query time**: `dimension`, `metric`, `normalized`, `binary` (presence of column), `clusters`, centroids, cluster counts. The caller never restates these on search.
- **Search-side, must be passed every query**: `topK`, `rerankFactor`, `probe`. These are the per-query trade-offs and the main targets for auto-tuning.
- **Pure performance (no correctness implications)**: `pageSize`, `rowGroupSize`, `codec`, `metadata` reuse, `binary` prefetch, `asyncBufferFactory`. Defaults already cover the common case; ablations exist for `pageSize` and `codec`.
- **Build-time only**: `clusterIterations`, `clusterSeed`. Set once at write.
