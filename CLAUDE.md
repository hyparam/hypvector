# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

HypVector is a library for storing and querying embedding vectors in Parquet files. It targets serverless similarity search: clients fetch a Parquet file (over HTTP range requests or from local disk) and run search directly, without a vector database.

## Build and Test Commands

```bash
npm test            # run tests
npm run coverage    # tests with coverage
npm run lint        # eslint
npm run lint:fix    # eslint --fix
npm run benchmark   # write + search benchmark
```

## Architecture

### Storage layout

Vectors are stored in a single Parquet file with two columns:

- `id` (STRING): caller-supplied identifier, coerced to string
- `vector` (FIXED_LEN_BYTE_ARRAY, `type_length = 4 * dimension`): raw little-endian float32 bytes

Format-level info lives in Parquet KV metadata so readers don't need out-of-band coordination:

- `hypvector.version`: index format version
- `hypvector.dimension`: length of each vector
- `hypvector.metric`: intended similarity metric (`cosine` | `dot` | `euclidean`)
- `hypvector.normalized`: whether vectors were L2-normalized on write
- `hypvector.count`: vector count

### Core modules (`src/`)

- `writeVectors.js`: packs each vector to float32 bytes and writes to a Parquet `BYTE_ARRAY` column. Accepts sync or async iterables.
- `readVectors.js`: async generator that yields `{ id, vector }` records, unpacking bytes back to `Float32Array`.
- `searchVectors.js`: linear-scan top-k search. Streams every vector, computes the chosen metric, keeps a bounded result set.
- `utils.js`: `cosineSimilarity`, `dotProduct`, `euclideanDistance`, `l2Normalize`, plus `packFloat32` / `unpackFloat32` / `parseKvMetadata`.
- `constants.js`: version and defaults.

### Known limitations (intentional for v0)

- **Linear scan only**: no ANN index, no partitioning, no inverted lists.
- **Full file read for search**: every query reads the entire `vector` column.
- **No quantization**: float32 only; int8 / binary / product quantization are future experiments.
- **PLAIN encoding**: no `BYTE_STREAM_SPLIT` or other float-friendly encoding yet.
- **No batching API**: `writeVectors` materializes all packed bytes before writing.

These are intentional starting points; each one is a candidate for a future experiment.

## Code Style

- JavaScript with JSDoc type annotations (not TypeScript)
- Type definitions in `.d.ts` files (`src/types.d.ts`, `src/index.d.ts`)
- ES modules (`type: "module"` in package.json)
- Single quotes, no semicolons (enforced by eslint)
- All functions need JSDoc with `@param`, `@returns`, and `@import` for types

## Testing

- Vitest as test runner
- Test files in `test/` with `.test.js` suffix
- Generated test files go under `test/files/` and are cleaned up in `afterEach`
- `test/helpers.js` provides a deterministic `makeVectors(count, dimension, seed)` and a `countingBuffer` fetch wrapper

## Dependencies

- **hyparquet**: Parquet reading
- **hyparquet-writer**: Parquet writing
- **hyparquet-compressors**: Compression codecs

All three are maintained by Hyperparam.
