# HypVector

[![mit license](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)

Store embedding vectors compactly in Parquet and query them directly over HTTP range requests using [`hyparquet`](https://github.com/hyparam/hyparquet) and [`hyparquet-writer`](https://github.com/hyparam/hyparquet-writer).

## Why?

Most vector databases require a server. HypVector treats a Parquet file on S3 (or local disk) as the database, so any client can run similarity search without infrastructure. The file is self-describing — query parameters (dimension, metric, normalization, cluster centroids) are stored in Parquet key-value metadata.

At 156k 384-dim wiki embeddings (249 MB), a single top-10 query reads **~6 MB across ~160 ranged HTTP fetches** with ~91% recall against an exact full scan. Over a localhost HTTP server with 20 ms of injected per-request latency, the rerank path lands at **~140 ms/query** vs ~360 ms for an exact full scan.

## How it works

Three columns: `id` (STRING), `vector` (`FIXED_LEN_BYTE_ARRAY(4 × dim)`, raw float32 bytes, `UNCOMPRESSED`), and — when `binary: true` — `vector_bin` (`FIXED_LEN_BYTE_ARRAY(dim/8)`, 1 bit per dim).

**Exact search path** (no binary column, or `rerankFactor: 0`): single pass over the float32 column via `parquetRead({ onChunk })`. Each row-group's decoded `Uint8Array[]` shares a backing buffer, so we view it as one aligned `Float32Array` and stride by `dim` — zero per-row allocations.

**Binary + cluster + rerank path** (default when `binary: true`):

1. **Build-time clustering** (when `clusters > 0`): k-means on the 1-bit codes using Hamming distance and bit-majority voting. Cluster ids are then renumbered via a greedy nearest-neighbor walk so that adjacent ids = similar centroids — this makes the top-N nearest clusters at query time tend to land in fewer contiguous row ranges. Rows are sorted by the new cluster id. Centroids and per-cluster row counts go into KV metadata.
2. **Phase 1 — cluster pruning**: rank clusters by Hamming(query, centroid), pick the top `probe` fraction, and Hamming-scan only those clusters' row ranges. With 32 KB pages and `useOffsetIndex`, hyparquet fetches only the pages covering each cluster's rows.
3. **Phase 2 — float32 rerank**: collect the top `topK × rerankFactor` candidate row indices, coalesce them into contiguous runs (merging gaps ≤ 64 rows), and issue one ranged `parquetRead` per run for the `vector` column only. Score under the exact metric.
4. **Phase 3 — id lookup**: fetch the `id` column for *only* the top-K winners (the id column is variable-length and reading it for every candidate doubles phase-2 cost).

A `cachedAsyncBuffer` deduplicates footer / offset-index byte ranges across all the parallel `parquetRead` calls.

For pre-normalized vectors with `metric: 'cosine'`, the search normalizes the query once and scores via dot product to skip the per-candidate sqrt loop.

## CLI usage

```bash
npx hypvector vectors.parquet
```

Prints format version, vector count, dimension, metric, whether a binary column is present, cluster count, and storage overhead.

## Producing vectors

HypVector is BYO-embedding: you decide which model produces the vectors. It just stores `{ id, vector }` pairs and queries them. The only contracts are:

1. **Same model on write and query.** Embeddings from different models aren't comparable.
2. **Same `dimension`** for every record (must match the `dimension` you pass to `writeVectors`).
3. **`normalize: true`** is the right default for any model whose vectors aren't already unit-length and you intend to query with cosine — it saves the per-candidate sqrt at query time. If your model already normalizes (most modern sentence-transformer models do), still pass `normalize: true` so the flag is recorded in KV metadata.

The natural shape is an async generator that yields embedded records as you batch them through your embedder. Two end-to-end sketches:

### Local model (Transformers.js)

```javascript
import { pipeline } from '@huggingface/transformers'
import { fileWriter } from 'hyparquet-writer'
import { writeVectors } from 'hypvector'

const extract = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2')

async function* embed(docs, batchSize = 32) {
  for (let i = 0; i < docs.length; i += batchSize) {
    const batch = docs.slice(i, i + batchSize)
    const out = await extract(batch.map(d => d.text), { pooling: 'mean', normalize: true })
    for (let j = 0; j < batch.length; j += 1) {
      yield { id: batch[j].id, vector: out.data.slice(j * 384, (j + 1) * 384) }
    }
  }
}

await writeVectors({
  writer: fileWriter('vectors.parquet'),
  dimension: 384,
  normalize: true,
  binary: true,
  clusters: 128,
  vectors: embed(docs),
})
```

### Remote API (OpenAI / Cohere / Voyage / etc.)

```javascript
async function* embed(docs, batchSize = 96) {
  for (let i = 0; i < docs.length; i += batchSize) {
    const batch = docs.slice(i, i + batchSize)
    const res = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: { authorization: `Bearer ${process.env.OPENAI_API_KEY}`, 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'text-embedding-3-small', input: batch.map(d => d.text) }),
    })
    const { data } = await res.json()
    for (let j = 0; j < batch.length; j += 1) {
      yield { id: batch[j].id, vector: Float32Array.from(data[j].embedding) }
    }
  }
}
```

See `scripts/embed.js` for a working version that streams 156k wiki rows through MiniLM and writes the result.

## Write vectors

```javascript
import { fileWriter } from 'hyparquet-writer'
import { writeVectors } from 'hypvector'

const writer = fileWriter('vectors.parquet')
await writeVectors({
  writer,
  dimension: 384,
  normalize: true,    // L2-normalize on write; lets search skip sqrt for cosine
  binary: true,       // also write 1-bit-per-dim sign column for binary+rerank search
  clusters: 128,      // k-means clusters for phase-1 pruning (implies binary: true)
  vectors: myEmbedder(), // any sync or async iterable of { id, vector }
})
```

`vectors` accepts any sync or async iterable of `{ id, vector }`. When `binary: true`, the default `pageSize` drops to 32 KB so that `useOffsetIndex` reads in phase 2 fetch tight ranges. Override with explicit `pageSize` / `codec` / `rowGroupSize` if needed.

## Read vectors

```javascript
import { asyncBufferFromFile } from 'hyparquet'
import { readVectors } from 'hypvector'

const file = await asyncBufferFromFile('vectors.parquet')
for await (const { id, vector } of readVectors({ file })) {
  console.log(id, vector.slice(0, 4))
}
```

## Search

Embed the query with the **same model** used at write time, then pass the resulting vector:

```javascript
import { searchVectors } from 'hypvector'

const [{ data: queryVec }] = await extract(['how do neural networks learn?'], { pooling: 'mean', normalize: true })

const results = await searchVectors({
  url: 'https://example.com/vectors.parquet',
  query: queryVec,    // Float32Array of length `dimension`
  topK: 10,
  rerankFactor: 10,   // candidate pool = topK * rerankFactor (default 10). Set to 0 to force exact full scan.
  probe: 0.25,        // fraction of clusters to scan in phase 1 (default 0.25). Set to 1 to scan all clusters; pass an integer > 1 for an absolute count.
})

for (const { id, score } of results) {
  console.log(score, id)
}
```

- `metric: 'cosine' | 'dot' | 'euclidean'` overrides the metric stored in the file.
- For local files, pass a file path as `url` (auto-detected) or supply your own `asyncBufferFactory`.
- When no `sourceFile` is provided, the default factory wraps the buffer in `cachedAsyncBuffer` so repeated reads of the footer / offset indexes are served from memory.

## File layout

| Column | Type | Bytes per row | When written |
|---|---|---|---|
| `id` | `STRING` (UTF8) | variable | always |
| `vector` | `FIXED_LEN_BYTE_ARRAY(4 × dim)` | `4 × dim` | always |
| `vector_bin` | `FIXED_LEN_BYTE_ARRAY(dim/8)` | `dim/8` | when `binary: true` |

Key-value metadata:

| Key | Value |
|---|---|
| `hypvector.version` | format version (currently `0`) |
| `hypvector.dimension` | length of each vector |
| `hypvector.metric` | `cosine` \| `dot` \| `euclidean` |
| `hypvector.normalized` | `true` if vectors were L2-normalized on write |
| `hypvector.binary` | `true` if the `vector_bin` column is present |
| `hypvector.count` | number of vectors |
| `hypvector.clusters` | number of k-means clusters (0 if not clustered) |
| `hypvector.centroids` | base64-encoded centroid binary codes (`clusters × dim/8` bytes); present when `clusters > 0` |
| `hypvector.clusterCounts` | base64-encoded `Uint32Array` of per-cluster row counts; present when `clusters > 0` |

## Scale guidance

The default `rerankFactor` of 10 is tuned for the hundreds-of-thousands range. As `N` grows, more binary candidates tie at the same Hamming distance and a wider phase-1 pool is needed to keep recall up. On a 1M synthetic dataset (256 true clusters, Gaussian noise):

| `rerankFactor` | candidates fetched | ms | recall@10 |
|---:|---:|---:|---:|
| 10  | 100 | 41  | 18% |
| 30  | 300 | 58  | 32% |
| 100 | 1,000 | 155 | 68% |
| 300 | 3,000 | 443 | 98% |

Rough rule: `rerankFactor ≈ max(10, N / 3000)`. At 1M that's ~333, giving ~98% recall at ~440 ms — still about an order of magnitude faster than the 950 ms exact scan.

## Performance (156k 384-dim wiki, local file)

From `scripts/ablation.js` (write-side optimizations):

| Variant | File MB | Query ms | Fetches | MB read | Recall@10 |
|---|---:|---:|---:|---:|---:|
| base (`vector` + `id`) — forced exact scan | 241.5 | 108 | 33 | 242.0 | 100% |
| `+ binary` (phase 1 + 2 rerank) | 249.3 | 48 | 136 | 11.7 | 93% |
| `+ cluster` (default; `probe=0.25`, `clusters=128`) | 249.4 | 15 | 162 | 6.2 | 91% |

From `scripts/bench-http.js` (localhost HTTP server with +20 ms per-request RTT, same 156k file):

| Search | ms/query |
|---|---:|
| Exact full scan | 362 |
| Rerank `probe=0.5` | 152 |
| Rerank `probe=0.25` (default) | 139 |
| Rerank `probe=0.1` | 129 |

From `scripts/ablation-search.js` (same data, toggling search-side knobs):

| Search variant | Query ms | Fetches | MB read |
|---|---:|---:|---:|
| baseline (all opts on) | 22 | 100 | 5.5 |
| `-coalesce` (one `parquetRead` per candidate) | 34 | 133 | 4.9 |
| `-deferId` (fetch ids alongside vectors) | 50 | 117 | 5.8 |

Trade query speed for recall via the `probe` knob (fraction of clusters scanned):

| `probe` | ms | fetches | MB | recall |
|---:|---:|---:|---:|---:|
| 0.05 |  9 | 47 | 3.7 | 78% |
| 0.10 | 11 | 59 | 4.2 | 84% |
| 0.25 (default) | 16 | 79 | 5.2 | 91% |
| 0.50 | 21 | 94 | 6.5 | 94% |
| 1.00 (all clusters) | 29 | 70 | 8.3 | 94% |
