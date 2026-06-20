import { parquetWrite, parquetWriteRows, schemaFromColumnData } from 'hyparquet-writer'
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
 * @import { VectorRecord, WriteVectorsOptions } from './types.js'
 * @import { ColumnSource, Writer } from 'hyparquet-writer'
 * @import { CompressionCodec, KeyValue, SchemaElement } from 'hyparquet'
 */

/**
 * Write embedding vectors to a parquet file.
 *
 * Columns:
 *   - `id`: STRING (caller-supplied id, coerced to string)
 *   - `vector`: FIXED_LEN_BYTE_ARRAY(4 * dimension) raw little-endian float32 bytes
 *   - `vector_bin`: FIXED_LEN_BYTE_ARRAY(dim/8), written when `binary: true`
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

  const binaryBytes = dimension + 7 >> 3
  const willCluster = clusters !== undefined && clusters > 0

  // Streaming fast path: when `binary` is set explicitly and no clustering is
  // requested, the schema is fully determined up front and rows are emitted in
  // input order, so each row-group-sized batch can be packed and flushed
  // without ever holding the whole dataset. Peak memory is one row group, not
  // O(N). Auto-binary (binary omitted) needs N to choose the column set, and
  // clustering needs a global k-means + row reorder, so both fall through to
  // the buffered path below.
  if (binary !== undefined && !willCluster) {
    return streamVectors({
      writer,
      vectors,
      dimension,
      binaryBytes,
      binary,
      normalize,
      metric,
      codec,
      rowGroupSize: rowGroupSize ?? defaultRowGroupSize,
      pageSize: pageSize ?? (binary ? defaultBinaryPageSize : undefined),
    })
  }

  // Buffered path: auto-binary and clustering both need the whole dataset in
  // memory: auto-binary to count N before choosing the column set, clustering
  // to k-means the binary codes and reorder rows so each cluster is contiguous.
  const autoBinary = binary === undefined

  /** @type {string[]} */
  const ids = []
  /** @type {Uint8Array[]} */
  const packed = []
  /** @type {Uint8Array[]} */
  const packedBin = []

  for await (const record of vectors) {
    const v = toFloat32(record.vector, dimension, normalize, record.id)
    ids.push(String(record.id))
    packed.push(packFloat32(v))
    packedBin.push(packBinary(v, dimension))
  }

  // Resolve auto defaults now that we know N. Auto-clusters only fires
  // when the caller also let `binary` auto; explicit `binary: true` means
  // "add the column, don't reshuffle rows".
  if (autoBinary) binary = ids.length >= defaultAutoBinaryThreshold
  binary = binary === true
  const clusterCount = clusters ?? (autoBinary && binary ? Math.max(1, Math.round(Math.sqrt(ids.length) / 2)) : 0)
  // Clustering operates on the binary codes, so it implies the binary column
  // even when auto-binary would have left it off at small N (explicit
  // `clusters > 0` with a sub-threshold corpus).
  if (clusterCount > 0) binary = true

  const effectivePageSize = pageSize ?? (binary ? defaultBinaryPageSize : undefined)

  /** @type {Uint8Array[] | null} */
  let centroids = null
  /** @type {Uint32Array | null} */
  let clusterCounts = null
  if (clusterCount > 0) {
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

  const kvMetadata = baseKvMetadata({ dimension, metric, normalize, binary })
  kvMetadata.push({ key: 'hypvector.clusters', value: String(centroids ? centroids.length : 0) })
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
  if (binary) columnData.push({ name: defaultBinaryColumn, data: packedBin })

  // When clustering, each cluster becomes its own row group so phase-1
  // binary scans and phase-2 candidate fetches stay within a single column
  // chunk per cluster, dropping fetches roughly proportional to clusters
  // probed. Caller-supplied rowGroupSize wins if explicitly passed.
  const effectiveRowGroupSize = rowGroupSize ?? (
    clusterCounts ? Array.from(clusterCounts) : defaultRowGroupSize
  )

  await parquetWrite({
    writer,
    schema: vectorSchema({ dimension, binary, binaryBytes }),
    rowGroupSize: effectiveRowGroupSize,
    kvMetadata,
    columnData,
    codec,
    ...effectivePageSize !== undefined ? { pageSize: effectivePageSize } : {},
  })
}

/**
 * Streaming writer for the no-cluster, explicit-binary case. Packs and flushes
 * one row-group-sized batch at a time through {@link parquetWriteRows}, so peak
 * memory is bounded by the row-group size rather than the dataset size. The
 * schema and KV metadata are fully known up front (row count is recovered from
 * the parquet footer's `num_rows`, so nothing here depends on N).
 *
 * @param {object} options
 * @param {Writer} options.writer
 * @param {Iterable<VectorRecord> | AsyncIterable<VectorRecord>} options.vectors
 * @param {number} options.dimension
 * @param {number} options.binaryBytes
 * @param {boolean} options.binary
 * @param {boolean} options.normalize
 * @param {string} options.metric
 * @param {CompressionCodec} options.codec
 * @param {number | number[]} options.rowGroupSize
 * @param {number} [options.pageSize]
 * @returns {Promise<void>}
 */
async function streamVectors({ writer, vectors, dimension, binaryBytes, binary, normalize, metric, codec, rowGroupSize, pageSize }) {
  /** @type {Omit<ColumnSource, 'data'>[]} */
  const columns = [{ name: defaultIdColumn }, { name: defaultVectorColumn }]
  if (binary) columns.push({ name: defaultBinaryColumn })

  const kvMetadata = baseKvMetadata({ dimension, metric, normalize, binary })
  kvMetadata.push({ key: 'hypvector.clusters', value: '0' })

  /**
   * Map each input record to a parquet row, packing on the fly so only one
   * row group's worth of packed bytes is ever live at once.
   * @returns {AsyncGenerator<Record<string, string | Uint8Array>>}
   */
  async function* rows() {
    for await (const record of vectors) {
      const v = toFloat32(record.vector, dimension, normalize, record.id)
      /** @type {Record<string, string | Uint8Array>} */
      const row = {
        [defaultIdColumn]: String(record.id),
        [defaultVectorColumn]: packFloat32(v),
      }
      if (binary) row[defaultBinaryColumn] = packBinary(v, dimension)
      yield row
    }
  }

  await parquetWriteRows({
    writer,
    rows: rows(),
    columns,
    schema: vectorSchema({ dimension, binary, binaryBytes }),
    rowGroupSize,
    kvMetadata,
    codec,
    ...pageSize !== undefined ? { pageSize } : {},
  })
}

/**
 * Validate one record's vector and return it as a Float32Array, L2-normalized
 * when requested. Reuses the caller's Float32Array in place when possible.
 *
 * @param {Float32Array | number[]} vector
 * @param {number} dimension
 * @param {boolean} normalize
 * @param {string | number} id
 * @returns {Float32Array}
 */
function toFloat32(vector, dimension, normalize, id) {
  if (!vector || vector.length !== dimension) {
    throw new Error(`vector for id=${id} has length ${vector?.length}, expected ${dimension}`)
  }
  return normalize
    ? l2Normalize(vector)
    : vector instanceof Float32Array ? vector : Float32Array.from(vector)
}

/**
 * KV metadata shared by both write paths: everything knowable without N. The
 * vector count is intentionally omitted: it's exactly the parquet footer's
 * `num_rows`, which readers already use (see parseKvMetadata).
 *
 * @param {{ dimension: number, metric: string, normalize: boolean, binary: boolean }} options
 * @returns {KeyValue[]}
 */
function baseKvMetadata({ dimension, metric, normalize, binary }) {
  return [
    { key: 'hypvector.version', value: String(hypVectorVersion) },
    { key: 'hypvector.dimension', value: String(dimension) },
    { key: 'hypvector.metric', value: metric },
    { key: 'hypvector.normalized', value: String(normalize) },
    { key: 'hypvector.binary', value: String(binary) },
  ]
}

/**
 * Build the parquet schema for the vector columns. Independent of row count and
 * data values (types are forced via overrides / the id STRING hint), so it
 * works for both the buffered and streaming paths.
 *
 * @param {{ dimension: number, binary: boolean, binaryBytes: number }} options
 * @returns {SchemaElement[]}
 */
function vectorSchema({ dimension, binary, binaryBytes }) {
  /** @type {Record<string, SchemaElement>} */
  const schemaOverrides = {
    [defaultVectorColumn]: {
      name: defaultVectorColumn,
      type: 'FIXED_LEN_BYTE_ARRAY',
      type_length: dimension * 4,
      repetition_type: 'REQUIRED',
    },
  }
  /** @type {ColumnSource[]} */
  const columnData = [
    { name: defaultIdColumn, type: 'STRING', data: [] },
    { name: defaultVectorColumn, data: [] },
  ]
  if (binary) {
    columnData.push({ name: defaultBinaryColumn, data: [] })
    schemaOverrides[defaultBinaryColumn] = {
      name: defaultBinaryColumn,
      type: 'FIXED_LEN_BYTE_ARRAY',
      type_length: binaryBytes,
      repetition_type: 'REQUIRED',
    }
  }
  return schemaFromColumnData({ columnData, schemaOverrides })
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
