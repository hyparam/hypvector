import { existsSync, unlinkSync } from 'node:fs'
import { fileWriter } from 'hyparquet-writer'
import { afterEach, describe, expect, it } from 'vitest'
import { searchVectors } from '../src/searchVectors.js'
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
      url: TEST_FILE,
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
      url: TEST_FILE,
      query: [1, 0, 0, 0],
      topK: 3,
      metric: 'euclidean',
    })

    expect(results[0].id).toBe('a')
    expect(results[0].score).toBeCloseTo(0)
    expect(results[1].id).toBe('c')
  })
})
