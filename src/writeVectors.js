import { parquetWrite, schemaFromColumnData } from 'hyparquet-writer'
import {
  defaultBinaryColumn,
  defaultBinaryPageSize,
  defaultIdColumn,
  defaultRowGroupSize,
  defaultVectorColumn,
  hypVectorVersion,
} from './constants.js'
import { l2Normalize, packBinary, packFloat32 } from './utils.js'

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
  binary = false,
  pageSize,
}) {
  if (!Number.isInteger(dimension) || dimension <= 0) {
    throw new Error(`invalid dimension: ${dimension}`)
  }

  // When writing a binary rerank column, default to small pages so that
  // useOffsetIndex in phase 2 fetches only ~one page per candidate row.
  const effectivePageSize = pageSize ?? (binary ? defaultBinaryPageSize : undefined)

  const binaryBytes = (dimension + 7) >> 3

  /** @type {string[]} */
  const ids = []
  /** @type {Uint8Array[]} */
  const packed = []
  /** @type {Uint8Array[] | null} */
  const packedBin = binary ? [] : null

  for await (const record of vectors) {
    const { id, vector } = record
    if (!vector || vector.length !== dimension) {
      throw new Error(`vector for id=${id} has length ${vector?.length}, expected ${dimension}`)
    }
    const v = normalize ? l2Normalize(vector) : vector
    ids.push(String(id))
    packed.push(packFloat32(v))
    if (packedBin) packedBin.push(packBinary(v, dimension))
  }

  const kvMetadata = [
    { key: 'hypvector.version', value: String(hypVectorVersion) },
    { key: 'hypvector.dimension', value: String(dimension) },
    { key: 'hypvector.metric', value: metric },
    { key: 'hypvector.normalized', value: String(normalize) },
    { key: 'hypvector.binary', value: String(binary) },
    { key: 'hypvector.count', value: String(ids.length) },
  ]

  /** @type {ColumnSource[]} */
  const columnData = [
    { name: defaultIdColumn, data: ids },
    { name: defaultVectorColumn, data: packed },
  ]
  /** @type {Record<string, import('hyparquet').SchemaElement>} */
  const schemaOverrides = {
    [defaultVectorColumn]: {
      name: defaultVectorColumn,
      type: 'FIXED_LEN_BYTE_ARRAY',
      type_length: dimension * 4,
      repetition_type: 'REQUIRED',
    },
  }
  if (packedBin) {
    columnData.push({ name: defaultBinaryColumn, data: packedBin })
    schemaOverrides[defaultBinaryColumn] = {
      name: defaultBinaryColumn,
      type: 'FIXED_LEN_BYTE_ARRAY',
      type_length: binaryBytes,
      repetition_type: 'REQUIRED',
    }
  }

  const schemaInput = columnData.map(c => c.name === defaultIdColumn ? { ...c, type: /** @type {const} */ ('STRING') } : c)
  const schema = schemaFromColumnData({ columnData: schemaInput, schemaOverrides })

  await parquetWrite({
    writer,
    schema,
    rowGroupSize,
    kvMetadata,
    columnData,
    codec,
    ...(effectivePageSize !== undefined ? { pageSize: effectivePageSize } : {}),
  })
}
