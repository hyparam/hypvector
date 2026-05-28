import { existsSync, unlinkSync } from 'node:fs'
import { asyncBufferFromFile, parquetMetadataAsync } from 'hyparquet'
import { fileWriter } from 'hyparquet-writer'
import { afterEach, describe, expect, it } from 'vitest'
import { writeVectors } from '../src/writeVectors.js'
import { makeVectors } from './helpers.js'

const TEST_FILE = 'test/files/write.parquet'

describe('writeVectors', () => {
  afterEach(() => {
    if (existsSync(TEST_FILE)) unlinkSync(TEST_FILE)
  })

  it('writes vectors with the expected schema and kv metadata', async () => {
    const dimension = 8
    const vectors = makeVectors(50, dimension)
    const writer = fileWriter(TEST_FILE)

    await writeVectors({ writer, vectors, dimension })

    expect(existsSync(TEST_FILE)).toBe(true)
    const file = await asyncBufferFromFile(TEST_FILE)
    const meta = await parquetMetadataAsync(file)
    expect(Number(meta.num_rows)).toBe(50)

    const kv = meta.key_value_metadata
    /**
     * @param {string} key
     * @returns {string | undefined}
     */
    function find(key) {
      return kv?.find(e => e.key === key)?.value
    }
    expect(find('hypvector.version')).toBe('0')
    expect(find('hypvector.dimension')).toBe('8')
    expect(find('hypvector.metric')).toBe('cosine')
    expect(find('hypvector.normalized')).toBe('false')
    expect(find('hypvector.count')).toBe('50')
  })

  it('rejects vectors with the wrong dimension', async () => {
    const writer = fileWriter(TEST_FILE)
    await expect(writeVectors({
      writer,
      dimension: 4,
      vectors: [{ id: 'a', vector: [1, 2, 3] }],
    })).rejects.toThrow(/length 3, expected 4/)
  })

  it('normalizes vectors when requested', async () => {
    const dimension = 4
    const writer = fileWriter(TEST_FILE)
    await writeVectors({
      writer,
      dimension,
      normalize: true,
      vectors: [{ id: 'a', vector: [3, 0, 0, 4] }],
    })
    const file = await asyncBufferFromFile(TEST_FILE)
    const meta = await parquetMetadataAsync(file)
    /**
     * @param {string} key
     * @returns {string | undefined}
     */
    function find(key) {
      return meta.key_value_metadata?.find(e => e.key === key)?.value
    }
    expect(find('hypvector.normalized')).toBe('true')
  })

  it('auto-disables binary and clusters when N is below the threshold', async () => {
    const dimension = 32
    const writer = fileWriter(TEST_FILE)
    await writeVectors({ writer, dimension, vectors: makeVectors(50, dimension) })
    const file = await asyncBufferFromFile(TEST_FILE)
    const meta = await parquetMetadataAsync(file)
    /**
     * @param {string} key
     * @returns {string | undefined}
     */
    function find(key) {
      return meta.key_value_metadata?.find(e => e.key === key)?.value
    }
    expect(find('hypvector.binary')).toBe('false')
    expect(find('hypvector.clusters')).toBe('0')
  })

  it('auto-enables binary and auto-picks clusters at large N', async () => {
    const dimension = 32
    const writer = fileWriter(TEST_FILE)
    await writeVectors({ writer, dimension, vectors: makeVectors(10000, dimension) })
    const file = await asyncBufferFromFile(TEST_FILE)
    const meta = await parquetMetadataAsync(file)
    /**
     * @param {string} key
     * @returns {string | undefined}
     */
    function find(key) {
      return meta.key_value_metadata?.find(e => e.key === key)?.value
    }
    expect(find('hypvector.binary')).toBe('true')
    // round(sqrt(10000)/2) = 50
    expect(find('hypvector.clusters')).toBe('50')
  })
})
