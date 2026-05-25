import { parquetWrite, schemaFromColumnData } from 'hyparquet-writer'
import { binaryKMeans, reorderClustersByHamming } from './cluster.js'
import {
  defaultBinaryColumn,
  defaultBinaryPageSize,
  defaultClusterIterations,
  defaultIdColumn,
  defaultRowGroupSize,
  defaultVectorColumn,
  hypVectorVersion,
} from './constants.js'
import { l2Normalize, packBinary, packFloat32 } from './utils.js'

/**
 * @import { WriteVectorsOptions } from './types.js'
 * @import { ColumnSource } from 'hyparquet-writer'
 * @import { SchemaElement } from 'hyparquet'
 */

/**
 * Write embedding vectors to a parquet file.
 *
 * Columns:
 *   - `id`: STRING (caller-supplied id, coerced to string)
 *   - `vector`: FIXED_LEN_BYTE_ARRAY(4 * dimension) raw little-endian float32 bytes
 *   - `vector_bin`: FIXED_LEN_BYTE_ARRAY(dim/8) — written when `binary: true`
 *   - `cluster_id`: INT32 — written when `clusters > 0`; rows are sorted by it
 *     so row-group min/max stats let the searcher skip whole row groups.
 *
 * Format metadata is stored in parquet KV metadata so readers can unpack
 * vectors and use centroids without out-of-band coordination.
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
  clusters = 0,
  clusterIterations = defaultClusterIterations,
  clusterSeed = 1,
}) {
  if (!Number.isInteger(dimension) || dimension <= 0) {
    throw new Error(`invalid dimension: ${dimension}`)
  }
  if (clusters > 0 && !binary) {
    // Clustering operates on binary codes; require the binary column too.
    binary = true
  }

  const effectivePageSize = pageSize ?? (binary ? defaultBinaryPageSize : undefined)
  const binaryBytes = dimension + 7 >> 3

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

  /** @type {Uint8Array[] | null} */
  let centroids = null
  /** @type {Uint32Array | null} */
  let clusterCounts = null
  if (clusters > 0 && packedBin) {
    const { assignments, centroids: cs } = binaryKMeans(
      packedBin, binaryBytes, clusters, clusterIterations, clusterSeed
    )
    // Renumber cluster ids so adjacent ids = similar centroids. Lets the
    // top-N nearest clusters at query time collapse to fewer scan ranges.
    const remap = reorderClustersByHamming(cs)
    const reorderedCentroids = new Array(cs.length)
    for (let oldId = 0; oldId < cs.length; oldId += 1) {
      reorderedCentroids[remap[oldId]] = cs[oldId]
    }
    centroids = reorderedCentroids
    // Sort rows by the NEW cluster id.
    const order = new Int32Array(ids.length)
    for (let i = 0; i < ids.length; i += 1) order[i] = i
    const sorted = Array.from(order).sort((a, b) => remap[assignments[a]] - remap[assignments[b]])
    const idsOut = new Array(ids.length)
    const packedOut = new Array(ids.length)
    const packedBinOut = new Array(ids.length)
    clusterCounts = new Uint32Array(cs.length)
    for (let i = 0; i < sorted.length; i += 1) {
      const src = sorted[i]
      idsOut[i] = ids[src]
      packedOut[i] = packed[src]
      packedBinOut[i] = packedBin[src]
      clusterCounts[remap[assignments[src]]] += 1
    }
    ids.length = 0
    ids.push(...idsOut)
    packed.length = 0
    packed.push(...packedOut)
    packedBin.length = 0
    packedBin.push(...packedBinOut)
  }

  const kvMetadata = [
    { key: 'hypvector.version', value: String(hypVectorVersion) },
    { key: 'hypvector.dimension', value: String(dimension) },
    { key: 'hypvector.metric', value: metric },
    { key: 'hypvector.normalized', value: String(normalize) },
    { key: 'hypvector.binary', value: String(binary) },
    { key: 'hypvector.count', value: String(ids.length) },
    { key: 'hypvector.clusters', value: String(centroids ? centroids.length : 0) },
  ]
  if (centroids && clusterCounts) {
    // Pack centroids as one contiguous Uint8Array, then base64-encode.
    const buf = new Uint8Array(centroids.length * binaryBytes)
    for (let c = 0; c < centroids.length; c += 1) buf.set(centroids[c], c * binaryBytes)
    kvMetadata.push({ key: 'hypvector.centroids', value: encodeBase64(buf) })

    // Per-cluster row counts. Cluster k spans [cumsum[k], cumsum[k+1]).
    kvMetadata.push({
      key: 'hypvector.clusterCounts',
      value: encodeBase64(new Uint8Array(clusterCounts.buffer, clusterCounts.byteOffset, clusterCounts.byteLength)),
    })
  }

  /** @type {ColumnSource[]} */
  const columnData = [
    { name: defaultIdColumn, data: ids },
    { name: defaultVectorColumn, data: packed },
  ]
  /** @type {Record<string, SchemaElement>} */
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
    ...effectivePageSize !== undefined ? { pageSize: effectivePageSize } : {},
  })
}

/**
 * Base64 encode a Uint8Array. Uses Node's Buffer when available, falls back
 * to btoa for browser environments.
 *
 * @param {Uint8Array} bytes
 * @returns {string}
 */
function encodeBase64(bytes) {
  if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64')
  let s = ''
  for (let i = 0; i < bytes.length; i += 1) s += String.fromCharCode(bytes[i])
  return btoa(s)
}
