import { parquetWrite } from 'hyparquet-writer'
import {
  defaultIdColumn,
  defaultRowGroupSize,
  defaultVectorColumn,
  hypVectorVersion,
} from './constants.js'
import { l2Normalize, packFloat32 } from './utils.js'

/**
 * @import { WriteVectorsOptions } from './types.js'
 */

/**
 * Write embedding vectors to a parquet file.
 *
 * Naive v0 layout:
 *   - `id`: STRING (id of each vector, coerced to string)
 *   - `vector`: BYTE_ARRAY (raw little-endian float32 bytes, length = 4 * dimension)
 *
 * Metadata about the format is stored in parquet KV metadata so readers can
 * unpack vectors without out-of-band coordination.
 *
 * @param {WriteVectorsOptions} options
 * @returns {Promise<void>}
 */
export async function writeVectors({
  writer,
  vectors,
  dimension,
  rowGroupSize = defaultRowGroupSize,
  metric = 'cosine',
  normalize = false,
}) {
  if (!Number.isInteger(dimension) || dimension <= 0) {
    throw new Error(`invalid dimension: ${dimension}`)
  }

  /** @type {string[]} */
  const ids = []
  /** @type {Uint8Array[]} */
  const packed = []

  for await (const record of vectors) {
    const { id, vector } = record
    if (!vector || vector.length !== dimension) {
      throw new Error(`vector for id=${id} has length ${vector?.length}, expected ${dimension}`)
    }
    const v = normalize ? l2Normalize(vector) : vector
    ids.push(String(id))
    packed.push(packFloat32(v))
  }

  const kvMetadata = [
    { key: 'hypvector.version', value: String(hypVectorVersion) },
    { key: 'hypvector.dimension', value: String(dimension) },
    { key: 'hypvector.metric', value: metric },
    { key: 'hypvector.normalized', value: String(normalize) },
    { key: 'hypvector.count', value: String(ids.length) },
  ]

  await parquetWrite({
    writer,
    rowGroupSize,
    kvMetadata,
    columnData: [
      { name: defaultIdColumn, data: ids, type: 'STRING' },
      { name: defaultVectorColumn, data: packed, type: 'BYTE_ARRAY' },
    ],
  })
}
