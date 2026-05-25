import { parquetWrite, schemaFromColumnData } from 'hyparquet-writer'
import {
  defaultIdColumn,
  defaultRowGroupSize,
  defaultVectorColumn,
  hypVectorVersion,
} from './constants.js'
import { l2Normalize, packFloat32 } from './utils.js'

/**
 * @import { WriteVectorsOptions } from './types.js'
 * @import { ColumnSource } from 'hyparquet-writer'
 */

/**
 * Write embedding vectors to a parquet file.
 *
 * v0 layout:
 *   - `id`: STRING (id of each vector, coerced to string)
 *   - `vector`: FIXED_LEN_BYTE_ARRAY(4 * dimension) (raw little-endian float32 bytes)
 *
 * FIXED_LEN_BYTE_ARRAY avoids the 4-byte length prefix that BYTE_ARRAY writes
 * per row, and lets readers/writers know the row width up-front.
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
  codec = 'UNCOMPRESSED',
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

  /** @type {ColumnSource[]} */
  const columnData = [
    { name: defaultIdColumn, data: ids },
    { name: defaultVectorColumn, data: packed },
  ]
  const schema = schemaFromColumnData({
    columnData: [{ ...columnData[0], type: 'STRING' }, columnData[1]],
    schemaOverrides: {
      [defaultVectorColumn]: {
        name: defaultVectorColumn,
        type: 'FIXED_LEN_BYTE_ARRAY',
        type_length: dimension * 4,
        repetition_type: 'REQUIRED',
      },
    },
  })

  await parquetWrite({
    writer,
    schema,
    rowGroupSize,
    kvMetadata,
    columnData,
    codec,
  })
}
