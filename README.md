# HypVector

[![mit license](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)

Store embedding vectors compactly in Parquet and query them directly over HTTP range requests using [`hyparquet`](https://github.com/hyparam/hyparquet) and [`hyparquet-writer`](https://github.com/hyparam/hyparquet-writer).

## Why?

Most vector databases require a server. HypVector treats a Parquet file on S3 (or local disk) as the database, so any client can run similarity search without infrastructure. The file is self-describing — query parameters (dimension, metric, normalization, layout) are stored in Parquet key-value metadata.

At 100k 384-dim wiki embeddings, a single query reads ~12 MB of a 159 MB file with ~96% recall against an exact full scan.

## How it works

Each vector is stored as raw float32 bytes in a `FIXED_LEN_BYTE_ARRAY(4 × dim)` column, `UNCOMPRESSED` by default (SNAPPY/ZSTD don't shrink dense embeddings and cost ~3× query latency). The file also contains an `id` column and optionally a 1-bit-per-dim `vector_bin` column for accelerated rerank search.

**Exact search path** (no binary column, or `rerankFactor: 0`): single pass over the float32 column via `parquetRead({ onChunk })`. Each row-group's decoded `Uint8Array[]` shares a backing buffer, so we view it as one aligned `Float32Array` and stride by `dim` — zero per-row allocations.

**Binary + rerank path** (binary column present, default):
1. **Phase 1** — full scan of the 1-bit `vector_bin` column with Hamming distance (XOR + popcount over `Uint32Array` chunks). Picks top `topK × rerankFactor` candidate row indices. Bytes: `dim/8` per row.
2. **Phase 2** — for each candidate, a single-row `parquetRead({ rowStart: i, rowEnd: i+1, useOffsetIndex: true })` issued in parallel. With `useOffsetIndex`, hyparquet fetches only the data page containing that row. With small pages (default 64 KB when `binary: true`) and a `cachedAsyncBuffer` deduplicating footer/offset-index reads across the K parallel fetches, total bytes ≈ `K × 1 page` instead of the full vector column.
3. Rerank the candidates under the exact metric, return top K.

For pre-normalized vectors with `metric: 'cosine'`, the search normalizes the query once and scores via dot product to skip the per-candidate sqrt loop.

## CLI usage

```bash
npx hypvector vectors.parquet
```

Prints format version, vector count, dimension, metric, whether a binary column is present, and storage overhead.

## Write vectors

```javascript
import { fileWriter } from 'hyparquet-writer'
import { writeVectors } from 'hypvector'

const writer = fileWriter('vectors.parquet')
await writeVectors({
  writer,
  dimension: 384,
  normalize: true,   // L2-normalize on write; lets search skip sqrt for cosine
  binary: true,      // also write 1-bit sign column for binary+rerank search
  vectors: [
    { id: 'doc-1', vector: new Float32Array(384) /* ... */ },
    { id: 'doc-2', vector: new Float32Array(384) /* ... */ },
  ],
})
```

`vectors` accepts any sync or async iterable of `{ id, vector }`. When `binary: true`, the default `pageSize` drops to 64 KB so that `useOffsetIndex` reads in phase 2 fetch only ~one page per candidate. Override with explicit `pageSize` / `codec` if needed.

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

```javascript
import { searchVectors } from 'hypvector'

const results = await searchVectors({
  url: 'https://example.com/vectors.parquet',
  query: new Float32Array(384) /* ... */,
  topK: 10,
  rerankFactor: 10,   // candidate pool size = topK * rerankFactor (default 10). Set to 0 to force the exact full-scan path.
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
