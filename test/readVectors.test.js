import { existsSync, unlinkSync } from 'node:fs'
import { asyncBufferFromFile } from 'hyparquet'
import { fileWriter } from 'hyparquet-writer'
import { afterEach, describe, expect, it } from 'vitest'
import { readVectors } from '../src/readVectors.js'
import { writeVectors } from '../src/writeVectors.js'
import { makeVectors } from './helpers.js'

const TEST_FILE = 'test/files/roundtrip.parquet'

describe('readVectors', () => {
  afterEach(() => {
    if (existsSync(TEST_FILE)) unlinkSync(TEST_FILE)
  })

  it('round-trips vectors exactly', async () => {
    const dimension = 16
    const original = makeVectors(25, dimension, 42)
    const writer = fileWriter(TEST_FILE)
    await writeVectors({ writer, vectors: original, dimension })

    const file = await asyncBufferFromFile(TEST_FILE)
    const read = []
    for await (const record of readVectors({ file })) {
      read.push(record)
    }

    expect(read.length).toBe(original.length)
    for (let i = 0; i < original.length; i += 1) {
      expect(read[i].id).toBe(original[i].id)
      expect(read[i].vector.length).toBe(dimension)
      for (let j = 0; j < dimension; j += 1) {
        expect(read[i].vector[j]).toBeCloseTo(original[i].vector[j], 5)
      }
    }
  })
})
