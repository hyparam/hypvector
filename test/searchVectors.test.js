import { existsSync, unlinkSync } from 'node:fs'
import { asyncBufferFromFile, parquetMetadataAsync } from 'hyparquet'
import { fileWriter } from 'hyparquet-writer'
import { afterEach, describe, expect, it } from 'vitest'
import { prefetchBinary } from '../src/prefetch.js'
import { searchVectors } from '../src/searchVectors.js'
import { parseKvMetadata } from '../src/utils.js'
import { writeVectors } from '../src/writeVectors.js'
import { makeVectors } from './helpers.js'

const TEST_FILE = 'test/files/search.parquet'

describe('searchVectors', () => {
  afterEach(() => {
    if (existsSync(TEST_FILE)) unlinkSync(TEST_FILE)
  })

  it('finds the nearest vector when querying with itself', async () => {
    const dimension = 32
    const vectors = makeVectors(100, dimension, 7)
    const writer = fileWriter(TEST_FILE)
    await writeVectors({ writer, vectors, dimension })

    const target = vectors[42]
    const results = await searchVectors({
      source: TEST_FILE,
      query: target.vector,
      topK: 5,
    })

    expect(results.length).toBe(5)
    expect(results[0].id).toBe(target.id)
    expect(results[0].rowIndex).toBe(42)
    expect(results[0].score).toBeCloseTo(1, 4)
    // Results are sorted best-first
    for (let i = 1; i < results.length; i += 1) {
      expect(results[i - 1].score).toBeGreaterThanOrEqual(results[i].score)
    }
  })

  it('supports euclidean metric override', async () => {
    const dimension = 4
    const writer = fileWriter(TEST_FILE)
    await writeVectors({
      writer,
      dimension,
      vectors: [
        { id: 'a', vector: [1, 0, 0, 0] },
        { id: 'b', vector: [0, 1, 0, 0] },
        { id: 'c', vector: [0.9, 0.1, 0, 0] },
      ],
    })

    const results = await searchVectors({
      source: TEST_FILE,
      query: [1, 0, 0, 0],
      topK: 3,
      metric: 'euclidean',
    })

    expect(results[0].id).toBe('a')
    expect(results[0].score).toBeCloseTo(0)
    expect(results[1].id).toBe('c')
  })

  it('returns the exact match when binary + rerank is enabled', async () => {
    const dimension = 64
    const vectors = makeVectors(200, dimension, 11)
    const writer = fileWriter(TEST_FILE)
    await writeVectors({ writer, vectors, dimension, normalize: true, binary: true })

    const file = await asyncBufferFromFile(TEST_FILE)
    const metadata = await parquetMetadataAsync(file)
    const meta = parseKvMetadata(metadata)
    expect(meta.hasBinary).toBe(true)

    const target = vectors[77]
    const results = await searchVectors({
      source: file,
      metadata,
      query: target.vector,
      topK: 5,
    })

    expect(results.length).toBe(5)
    expect(results[0].id).toBe(target.id)
    expect(results[0].rowIndex).toBe(77)
    expect(results[0].score).toBeCloseTo(1, 4)
  })

  it('produces identical results with prefetched binary as without', async () => {
    const dimension = 64
    const vectors = makeVectors(500, dimension, 11)
    const writer = fileWriter(TEST_FILE)
    await writeVectors({ writer, vectors, dimension, normalize: true, binary: true, clusters: 8 })

    const file = await asyncBufferFromFile(TEST_FILE)
    const metadata = await parquetMetadataAsync(file)
    const binary = await prefetchBinary({ source: file, metadata })

    const bytesPerRow = (dimension + 7) >> 3
    expect(binary.byteLength).toBe(500 * bytesPerRow)

    for (const idx of [3, 199, 444]) {
      const query = vectors[idx].vector
      const baseline = await searchVectors({ source: file, metadata, query, topK: 5 })
      const prefetched = await searchVectors({ source: file, metadata, query, topK: 5, binary })
      expect(prefetched.map(r => r.id)).toEqual(baseline.map(r => r.id))
      for (let i = 0; i < baseline.length; i += 1) {
        expect(prefetched[i].score).toBeCloseTo(baseline[i].score, 5)
      }
    }
  })

  it('accepts a single-element source array as equivalent to a bare source', async () => {
    const dimension = 32
    const vectors = makeVectors(80, dimension, 5)
    const writer = fileWriter(TEST_FILE)
    await writeVectors({ writer, vectors, dimension })

    const query = vectors[12].vector
    const bare = await searchVectors({ source: TEST_FILE, query, topK: 3 })
    const wrapped = await searchVectors({ source: [TEST_FILE], query, topK: 3 })
    expect(wrapped.map(r => r.id)).toEqual(bare.map(r => r.id))
    // sourceIndex is only attached when truly multi-source (length > 1) to keep
    // results identical between the bare and array-of-one forms.
    expect(bare[0].sourceIndex).toBeUndefined()
    expect(wrapped[0].sourceIndex).toBeUndefined()
  })

  it('merges results across multiple sources matching a single-file search', async () => {
    const dimension = 32
    const fileA = 'test/files/search-a.parquet'
    const fileB = 'test/files/search-b.parquet'
    const all = makeVectors(100, dimension, 9)
    const halfA = all.slice(0, 50)
    const halfB = all.slice(50)

    await writeVectors({ writer: fileWriter(TEST_FILE), vectors: all, dimension, normalize: true })
    await writeVectors({ writer: fileWriter(fileA), vectors: halfA, dimension, normalize: true })
    await writeVectors({ writer: fileWriter(fileB), vectors: halfB, dimension, normalize: true })

    try {
      const query = all[37].vector
      const combined = await searchVectors({ source: TEST_FILE, query, topK: 5 })
      const split = await searchVectors({ source: [fileA, fileB], query, topK: 5 })

      // Same IDs in the same order — vectors[37] lands in fileA so the top hit must be from source 0.
      expect(split.map(r => r.id)).toEqual(combined.map(r => r.id))
      expect(split[0].sourceIndex).toBe(0)
      for (let i = 0; i < combined.length; i += 1) {
        expect(split[i].score).toBeCloseTo(combined[i].score, 5)
      }
    } finally {
      if (existsSync(fileA)) unlinkSync(fileA)
      if (existsSync(fileB)) unlinkSync(fileB)
    }
  })

  it('rejects metadata/binary arrays whose length does not match the source array', async () => {
    const dimension = 16
    const vectors = makeVectors(20, dimension, 1)
    await writeVectors({ writer: fileWriter(TEST_FILE), vectors, dimension })

    await expect(searchVectors({
      source: [TEST_FILE, TEST_FILE], query: vectors[0].vector, topK: 1, metadata: [undefined],
    })).rejects.toThrow(/metadata.*length/)
  })

  it('produces the same top-1 with and without rerank', async () => {
    const dimension = 64
    const vectors = makeVectors(200, dimension, 23)
    const writer = fileWriter(TEST_FILE)
    await writeVectors({ writer, vectors, dimension, normalize: true, binary: true })

    const file = await asyncBufferFromFile(TEST_FILE)
    const metadata = await parquetMetadataAsync(file)

    // Probe with a random stored vector — its own row should be top-1 either way.
    const probe = vectors[42].vector
    const exact = await searchVectors({
      source: file, metadata, query: probe, topK: 3, rerankFactor: 0,
    })
    const rerank = await searchVectors({
      source: file, metadata, query: probe, topK: 3, rerankFactor: 10,
    })
    expect(exact[0].id).toBe(rerank[0].id)
    expect(exact[0].score).toBeCloseTo(rerank[0].score, 4)
  })

  it('uses PQ approximate scoring with exact rerank', async () => {
    const dimension = 32
    const vectors = makeVectors(240, dimension, 31)
    const writer = fileWriter(TEST_FILE)
    await writeVectors({
      writer,
      vectors,
      dimension,
      normalize: true,
      pq: true,
      pqSegments: 8,
      pqCentroids: 8,
      pqIterations: 3,
      pqSampleSize: 80,
    })

    const file = await asyncBufferFromFile(TEST_FILE)
    const metadata = await parquetMetadataAsync(file)
    const meta = parseKvMetadata(metadata)
    expect(meta.hasPq).toBe(true)

    const target = vectors[123]
    const results = await searchVectors({
      source: file,
      metadata,
      query: target.vector,
      topK: 5,
      rerankFactor: 40,
    })

    expect(results.length).toBe(5)
    expect(results[0].id).toBe(target.id)
    expect(results[0].score).toBeCloseTo(1, 4)
  })
})
