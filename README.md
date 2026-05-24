# HypVector

[![mit license](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)

Store embedding vectors compactly in Parquet and query them directly over HTTP range requests using [`hyparquet`](https://github.com/hyparam/hyparquet) and [`hyparquet-writer`](https://github.com/hyparam/hyparquet-writer).

## Why?

Most vector databases require a server. HypVector treats a Parquet file on S3 (or local disk) as the database, so any client can run similarity search without infrastructure.

This is the naive v0: each vector is stored as raw float32 bytes in a `BYTE_ARRAY` column, and search is a linear scan. Future versions will add quantization, ANN indexes, and partitioning.

## CLI usage

```bash
npx hypvector vectors.parquet
```

Prints the format version, vector count, dimension, metric, and storage overhead.

## Write vectors

```javascript
import { fileWriter } from 'hyparquet-writer'
import { writeVectors } from 'hypvector'

const writer = fileWriter('vectors.parquet')
await writeVectors({
  writer,
  dimension: 384,
  vectors: [
    { id: 'doc-1', vector: new Float32Array(384) /* ... */ },
    { id: 'doc-2', vector: new Float32Array(384) /* ... */ },
  ],
})
```

`vectors` accepts any sync or async iterable of `{ id, vector }`.

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
})

for (const { id, score } of results) {
  console.log(score, id)
}
```

Pass `metric: 'cosine' | 'dot' | 'euclidean'` to override the default. For local files, set `url` to a path or supply your own `asyncBufferFactory`.
