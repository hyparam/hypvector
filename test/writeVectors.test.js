import { existsSync, unlinkSync } from 'node:fs'
import { asyncBufferFromFile, parquetMetadataAsync } from 'hyparquet'
import { fileWriter } from 'hyparquet-writer'
import { afterEach, describe, expect, it } from 'vitest'
import { readVectors } from '../src/readVectors.js'
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
    expect(find('hypvector.normalized')).toBe('true')
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

  it('forces the binary column when clusters are requested below the auto-binary threshold', async () => {
    const dimension = 32 // binaryBytes must be a multiple of 4 for k-means
    const writer = fileWriter(TEST_FILE)
    // binary omitted (auto, would be off at N=300) but clusters explicitly on:
    // clustering needs the binary codes, so the column must be written anyway.
    await writeVectors({ writer, dimension, vectors: makeVectors(300, dimension), clusters: 4 })
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
    expect(find('hypvector.clusters')).toBe('4')
    expect(meta.schema.some(e => e.name === 'vector_bin')).toBe(true)
  })

  it('streams (no buffering) when binary is explicit and clusters are off, roundtripping exactly', async () => {
    const dimension = 16
    const source = makeVectors(2500, dimension)
    const writer = fileWriter(TEST_FILE)

    // binary: false takes the streaming fast path: row-group-sized batches are
    // packed and flushed without materializing the whole dataset.
    await writeVectors({ writer, dimension, vectors: source, binary: false, normalize: false })

    const file = await asyncBufferFromFile(TEST_FILE)
    const meta = await parquetMetadataAsync(file)
    expect(Number(meta.num_rows)).toBe(source.length)
    /**
     * @param {string} key
     * @returns {string | undefined}
     */
    function find(key) {
      return meta.key_value_metadata?.find(e => e.key === key)?.value
    }
    expect(find('hypvector.binary')).toBe('false')
    expect(find('hypvector.clusters')).toBe('0')
    // No vector_bin column when binary is disabled.
    expect(meta.schema.some(e => e.name === 'vector_bin')).toBe(false)

    // Every stored vector matches its source bit-for-bit (raw float32 bytes).
    const byId = new Map(source.map(r => [r.id, r.vector]))
    let seen = 0
    for await (const record of readVectors({ file, metadata: meta })) {
      expect(record.vector).toEqual(byId.get(String(record.id)))
      seen += 1
    }
    expect(seen).toBe(source.length)
  })

  it('streams an explicit binary column without clustering', async () => {
    const dimension = 24
    const writer = fileWriter(TEST_FILE)
    // Explicit binary: true below the auto threshold => binary column, no clustering.
    await writeVectors({ writer, dimension, vectors: makeVectors(300, dimension), binary: true })

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
    expect(find('hypvector.clusters')).toBe('0')
    expect(meta.schema.some(e => e.name === 'vector_bin')).toBe(true)
  })
})
