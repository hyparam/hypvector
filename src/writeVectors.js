import { parquetWrite, schemaFromColumnData } from 'hyparquet-writer'
import { binaryKMeans, reorderClustersByHamming } from './cluster.js'
import {
  defaultAutoBinaryThreshold,
  defaultBinaryColumn,
  defaultBinaryPageSize,
  defaultClusterIterations,
  defaultIdColumn,
  defaultRowGroupSize,
  defaultVectorColumn,
  hypVectorVersion,
} from './constants.js'
import { encodeBase64, l2Normalize, packBinary, packFloat32 } from './utils.js'

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
 *
 * When `clusters > 0`, rows are reordered by binary cluster id and the
 * centroids plus per-cluster row counts go into KV metadata.
 *
 * @param {WriteVectorsOptions} options
 * @returns {Promise<void>}
 */
export async function writeVectors({
  writer,
  vectors,
  dimension,
  rowGroupSize,
  metric = 'cosine',
  normalize = false,
  codec = 'UNCOMPRESSED',
  binary,
  pageSize,
  clusters,
  clusterIterations = defaultClusterIterations,
  clusterSeed = 1,
}) {
  if (!Number.isInteger(dimension) || dimension <= 0) {
    throw new Error(`invalid dimension: ${dimension}`)
  }
  if (clusters !== undefined && clusters > 0 && binary === false) {
    throw new Error('writeVectors: clusters > 0 requires binary !== false')
  }

  // Auto mode (`binary` / `clusters` omitted): pack binary codes opportunistically
  // so we can decide once N is known. The dim/8 bytes per vector are negligible
  // compared to the float32 buffer we're already materializing.
  const autoBinary = binary === undefined
  const collectBinary = autoBinary || binary === true || clusters !== undefined && clusters > 0

  const binaryBytes = dimension + 7 >> 3

  /** @type {string[]} */
  const ids = []
  /** @type {Uint8Array[]} */
  const packed = []
  /** @type {Uint8Array[] | null} */
  let packedBin = collectBinary ? [] : null

  for await (const record of vectors) {
    const { id, vector } = record
    if (!vector || vector.length !== dimension) {
      throw new Error(`vector for id=${id} has length ${vector?.length}, expected ${dimension}`)
    }
    const v = normalize
      ? l2Normalize(vector)
      : vector instanceof Float32Array ? vector : Float32Array.from(vector)
    ids.push(String(id))
    packed.push(packFloat32(v))
    if (packedBin) packedBin.push(packBinary(v, dimension))
  }

  // Resolve auto defaults now that we know N. Auto-clusters only fires
  // when the caller also let `binary` auto — explicit `binary: true` means
  // "add the column, don't reshuffle rows".
  if (autoBinary) binary = ids.length >= defaultAutoBinaryThreshold
  binary = binary === true
  if (!binary) packedBin = null
  const clusterCount = clusters ?? (autoBinary && binary ? Math.max(1, Math.round(Math.sqrt(ids.length) / 2)) : 0)
  if (clusterCount > 0 && !binary) {
    // Clustering operates on binary codes; require the binary column too.
    // Only reachable when caller explicitly set clusters > 0 in auto-binary
    // mode; the explicit `clusters>0 && binary===false` case threw above.
    binary = true
  }

  const effectivePageSize = pageSize ?? (binary ? defaultBinaryPageSize : undefined)

  /** @type {Uint8Array[] | null} */
  let centroids = null
  /** @type {Uint32Array | null} */
  let clusterCounts = null
  if (clusterCount > 0 && packedBin) {
    const { assignments, centroids: cs } = binaryKMeans(
      packedBin, binaryBytes, clusterCount, clusterIterations, clusterSeed
    )
    // Renumber cluster ids so adjacent ids = similar centroids. Lets the
    // top-N nearest clusters at query time collapse to fewer scan ranges.
    const remap = reorderClustersByHamming(cs)
    const reorderedCentroids = new Array(cs.length)
    for (let oldId = 0; oldId < cs.length; oldId += 1) {
      reorderedCentroids[remap[oldId]] = cs[oldId]
    }
    centroids = reorderedCentroids
    clusterCounts = new Uint32Array(cs.length)
    for (let i = 0; i < ids.length; i += 1) clusterCounts[remap[assignments[i]]] += 1
    const sorted = sortedRowOrder(ids.length, (a, b) => remap[assignments[a]] - remap[assignments[b]])
    permuteInPlace(sorted, [ids, packed, packedBin])
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
  /** @type {ColumnSource[]} */
  const schemaInput = columnData.map(c => c.name === defaultIdColumn ? { ...c, type: /** @type {const} */ 'STRING' } : c)
  const schema = schemaFromColumnData({ columnData: schemaInput, schemaOverrides })

  // When clustering, each cluster becomes its own row group so phase-1
  // binary scans and phase-2 candidate fetches stay within a single column
  // chunk per cluster — drops fetches roughly proportional to clusters
  // probed. Caller-supplied rowGroupSize wins if explicitly passed.
  const effectiveRowGroupSize = rowGroupSize ?? (
    clusterCounts ? Array.from(clusterCounts) : defaultRowGroupSize
  )

  await parquetWrite({
    writer,
    schema,
    rowGroupSize: effectiveRowGroupSize,
    kvMetadata,
    columnData,
    codec,
    ...effectivePageSize !== undefined ? { pageSize: effectivePageSize } : {},
  })
}

/**
 * Build a row index array [0..n) sorted by the given comparator.
 *
 * @param {number} n
 * @param {(a: number, b: number) => number} compare
 * @returns {number[]}
 */
function sortedRowOrder(n, compare) {
  const order = new Array(n)
  for (let i = 0; i < n; i += 1) order[i] = i
  return order.sort(compare)
}

/**
 * Permute each non-null array by `sorted`, in-place. Each column's element
 * at position i becomes the element previously at position sorted[i].
 * (push(...arr) blows the call stack at ~100k elements.)
 *
 * @param {number[]} sorted
 * @param {(Array<any> | Uint8Array[] | null)[]} columns
 * @returns {void}
 */
function permuteInPlace(sorted, columns) {
  const n = sorted.length
  for (const col of columns) {
    if (!col) continue
    const out = new Array(n)
    for (let i = 0; i < n; i += 1) out[i] = col[sorted[i]]
    for (let i = 0; i < n; i += 1) col[i] = out[i]
  }
}
