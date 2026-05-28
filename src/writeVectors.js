import { parquetWrite, schemaFromColumnData } from 'hyparquet-writer'
import { binaryKMeans, reorderClustersByHamming } from './cluster.js'
import {
  defaultAutoBinaryThreshold,
  defaultBinaryColumn,
  defaultBinaryPageSize,
  defaultClusterIterations,
  defaultIdColumn,
  defaultIvfClusters,
  defaultIvfIterations,
  defaultIvfSampleSize,
  defaultPqCentroids,
  defaultPqColumn,
  defaultPqIterations,
  defaultPqSampleSize,
  defaultPqSegments,
  defaultRowGroupSize,
  defaultVectorColumn,
  hypVectorVersion,
} from './constants.js'
import { buildIvfPq } from './pq.js'
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
 *   - `vector_pq`: FIXED_LEN_BYTE_ARRAY(pqSegments) — written when `pq: true`
 *
 * When `clusters > 0`, rows are reordered by binary cluster id. When
 * `pq: true`, rows are reordered by IVF list id. These layouts are mutually
 * exclusive because both define contiguous row ranges for search.
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
  pq = false,
  pqSegments = defaultPqSegments,
  pqCentroids = defaultPqCentroids,
  pqIterations = defaultPqIterations,
  pqSampleSize = defaultPqSampleSize,
  pqSeed = 1,
  ivfClusters = defaultIvfClusters,
  ivfIterations = defaultIvfIterations,
  ivfSampleSize = defaultIvfSampleSize,
}) {
  if (!Number.isInteger(dimension) || dimension <= 0) {
    throw new Error(`invalid dimension: ${dimension}`)
  }
  if (pq && clusters !== undefined && clusters > 0) {
    throw new Error('writeVectors: `pq` uses IVF row ordering; do not combine it with binary `clusters`')
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
  /** @type {Float32Array[] | null} */
  const vectorsForPq = pq ? [] : null

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
    if (vectorsForPq) vectorsForPq.push(v)
  }

  /** @type {Uint8Array[] | null} */
  let packedPq = null
  /** @type {Float32Array | null} */
  let pqCodebooks = null
  /** @type {Float32Array | null} */
  let ivfCentroids = null
  /** @type {Uint32Array | null} */
  let ivfCounts = null
  /** @type {Int32Array | null} */
  let ivfAssignments = null
  let pqSegmentsOut = 0
  let pqCentroidsOut = 0
  let ivfClustersOut = 0
  if (vectorsForPq) {
    const built = buildIvfPq({
      vectors: vectorsForPq,
      dimension,
      ivfClusters,
      ivfIterations,
      ivfSampleSize,
      pqSegments,
      pqCentroids,
      pqIterations,
      pqSampleSize,
      seed: pqSeed,
    })
    packedPq = built.codes
    pqCodebooks = built.codebooks
    ivfCentroids = built.ivfCentroids
    ivfCounts = built.ivfCounts
    ivfAssignments = built.assignments
    pqSegmentsOut = built.pqSegments
    pqCentroidsOut = built.pqCentroids
    ivfClustersOut = built.ivfClusters
  }

  if (packedPq && ivfAssignments) {
    const order = new Int32Array(ids.length)
    for (let i = 0; i < ids.length; i += 1) order[i] = i
    const sorted = Array.from(order).sort((a, b) => ivfAssignments[a] - ivfAssignments[b])
    const idsOut = new Array(ids.length)
    const packedOut = new Array(ids.length)
    const packedBinOut = packedBin ? new Array(ids.length) : null
    const packedPqOut = new Array(ids.length)
    for (let i = 0; i < sorted.length; i += 1) {
      const src = sorted[i]
      idsOut[i] = ids[src]
      packedOut[i] = packed[src]
      if (packedBinOut && packedBin) packedBinOut[i] = packedBin[src]
      packedPqOut[i] = packedPq[src]
    }
    for (let i = 0; i < ids.length; i += 1) {
      ids[i] = idsOut[i]
      packed[i] = packedOut[i]
      if (packedBinOut && packedBin) packedBin[i] = packedBinOut[i]
      packedPq[i] = packedPqOut[i]
    }
  }

  // Resolve auto defaults now that we know N. Auto-clusters only fires
  // when the caller also let `binary` auto — explicit `binary: true` means
  // "add the column, don't reshuffle rows".
  if (autoBinary) binary = ids.length >= defaultAutoBinaryThreshold
  if (!binary) packedBin = null
  const clusterCount = clusters ?? (autoBinary && binary ? Math.max(1, Math.round(Math.sqrt(ids.length) / 2)) : 0)
  if (clusterCount > 0 && !binary) {
    // Clustering operates on binary codes; require the binary column too.
    // Only reachable when caller explicitly set clusters > 0 in auto-binary
    // mode; the explicit `clusters>0 && binary===false` case threw above.
    binary = true
  }

  const effectivePageSize = pageSize ?? (binary || pq ? defaultBinaryPageSize : undefined)

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
    // Sort rows by the NEW cluster id.
    const order = new Int32Array(ids.length)
    for (let i = 0; i < ids.length; i += 1) order[i] = i
    const sorted = Array.from(order).sort((a, b) => remap[assignments[a]] - remap[assignments[b]])
    const idsOut = new Array(ids.length)
    const packedOut = new Array(ids.length)
    const packedBinOut = new Array(ids.length)
    const packedPqOut = packedPq ? new Array(ids.length) : null
    clusterCounts = new Uint32Array(cs.length)
    for (let i = 0; i < sorted.length; i += 1) {
      const src = sorted[i]
      idsOut[i] = ids[src]
      packedOut[i] = packed[src]
      packedBinOut[i] = packedBin[src]
      if (packedPqOut && packedPq) packedPqOut[i] = packedPq[src]
      clusterCounts[remap[assignments[src]]] += 1
    }
    // In-place swap (push(...arr) blows the call stack at ~100k elements).
    for (let i = 0; i < ids.length; i += 1) {
      ids[i] = idsOut[i]
      packed[i] = packedOut[i]
      packedBin[i] = packedBinOut[i]
      if (packedPqOut && packedPq) packedPq[i] = packedPqOut[i]
    }
  }

  const kvMetadata = [
    { key: 'hypvector.version', value: String(hypVectorVersion) },
    { key: 'hypvector.dimension', value: String(dimension) },
    { key: 'hypvector.metric', value: metric },
    { key: 'hypvector.normalized', value: String(normalize) },
    { key: 'hypvector.binary', value: String(binary) },
    { key: 'hypvector.pq', value: String(Boolean(packedPq)) },
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
  if (packedPq && pqCodebooks) {
    kvMetadata.push({ key: 'hypvector.pq.segments', value: String(pqSegmentsOut) })
    kvMetadata.push({ key: 'hypvector.pq.centroids', value: String(pqCentroidsOut) })
    kvMetadata.push({
      key: 'hypvector.pq.codebooks',
      value: encodeBase64(new Uint8Array(pqCodebooks.buffer, pqCodebooks.byteOffset, pqCodebooks.byteLength)),
    })
    if (ivfCentroids && ivfCounts) {
      kvMetadata.push({ key: 'hypvector.ivf.clusters', value: String(ivfClustersOut) })
      kvMetadata.push({
        key: 'hypvector.ivf.centroids',
        value: encodeBase64(new Uint8Array(ivfCentroids.buffer, ivfCentroids.byteOffset, ivfCentroids.byteLength)),
      })
      kvMetadata.push({
        key: 'hypvector.ivf.counts',
        value: encodeBase64(new Uint8Array(ivfCounts.buffer, ivfCounts.byteOffset, ivfCounts.byteLength)),
      })
    }
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
  if (packedPq) {
    columnData.push({ name: defaultPqColumn, data: packedPq })
    schemaOverrides[defaultPqColumn] = {
      name: defaultPqColumn,
      type: 'FIXED_LEN_BYTE_ARRAY',
      type_length: pqSegmentsOut,
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
    ivfCounts ? Array.from(ivfCounts) : clusterCounts ? Array.from(clusterCounts) : defaultRowGroupSize
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
