import { asyncBufferFromFile, asyncBufferFromUrl, parquetMetadataAsync, parquetRead } from 'hyparquet'
import { defaultIdColumn, defaultVectorColumn } from './constants.js'
import {
  cosineSimilarity,
  dotProduct,
  euclideanDistance,
  l2Normalize,
  parseKvMetadata,
} from './utils.js'

/**
 * @import { SearchResult, SearchVectorsOptions, DistanceMetric } from './types.js'
 * @import { AsyncBuffer, DecodedArray } from 'hyparquet'
 */

/**
 * Find the top-k nearest neighbors to a query vector.
 *
 * Reads vector and id columns via parquetRead's onChunk callback so we avoid
 * the per-row JS object materialization that parquetReadObjects pays. Within
 * a row group, FIXED_LEN_BYTE_ARRAY decode hands us a Uint8Array[] backed by
 * a single contiguous buffer; we score it via aligned Float32Array views
 * with zero per-row allocations.
 *
 * @param {SearchVectorsOptions} options
 * @returns {Promise<SearchResult[]>}
 */
export async function searchVectors({
  query,
  url,
  topK = 10,
  metric,
  signal,
  asyncBufferFactory,
  sourceFile,
  sourceMetadata,
}) {
  const factory = asyncBufferFactory ?? defaultAsyncBufferFactory
  const file = sourceFile ?? await factory({ url, signal })
  const metadata = sourceMetadata ?? await parquetMetadataAsync(file)
  const meta = parseKvMetadata(metadata)

  if (query.length !== meta.dimension) {
    throw new Error(`query has dimension ${query.length}, file expects ${meta.dimension}`)
  }
  const requestedMetric = metric ?? meta.metric
  const dim = meta.dimension
  let queryF32 = query instanceof Float32Array ? query : Float32Array.from(query)

  // When stored vectors are pre-normalized, cosine == dot(query/||query||, candidate).
  // Pre-normalize the query once and score with dot product to skip the per-candidate
  // sqrt/normalize hot loop inside cosineSimilarity.
  let scoringMetric = requestedMetric
  if (requestedMetric === 'cosine' && meta.normalized) {
    queryF32 = l2Normalize(queryF32)
    scoringMetric = 'dot'
  }
  // Report scores under the user-requested metric (cosine and dot agree numerically
  // here because both query and candidates are unit vectors).
  const reportedMetric = requestedMetric

  /** @type {{ rowIndex: number, score: number }[]} */
  const heap = []
  /** @type {{ start: number, ids: string[] }[]} */
  const idChunks = []

  await parquetRead({
    file,
    metadata,
    columns: [defaultIdColumn, defaultVectorColumn],
    onChunk: ({ columnName, columnData, rowStart }) => {
      if (columnName === defaultVectorColumn) {
        scoreChunk(columnData, rowStart, dim, queryF32, scoringMetric, heap, topK)
      } else if (columnName === defaultIdColumn) {
        idChunks.push({ start: rowStart, ids: /** @type {string[]} */ (columnData) })
      }
    },
  })

  const ordered = sortHeap(heap, reportedMetric)
  return ordered.map(({ rowIndex, score }) => ({
    id: lookupId(idChunks, rowIndex) ?? String(rowIndex),
    score,
    rowIndex,
  }))
}

/**
 * Score every row in a chunk and update the running top-k heap.
 *
 * @param {DecodedArray} columnData
 * @param {number} rowStart
 * @param {number} dim
 * @param {Float32Array} query
 * @param {DistanceMetric} metric
 * @param {{ rowIndex: number, score: number }[]} heap
 * @param {number} topK
 */
function scoreChunk(columnData, rowStart, dim, query, metric, heap, topK) {
  const rows = /** @type {Uint8Array[]} */ (columnData)
  if (rows.length === 0) return

  // All rows in a chunk share a backing ArrayBuffer laid out contiguously.
  // If the chunk start is 4-byte aligned, view it as one big Float32Array
  // and stride by `dim`. Otherwise, copy each row into a reused scratch.
  const first = rows[0]
  if (first.byteOffset % 4 === 0) {
    const flat = new Float32Array(first.buffer, first.byteOffset, rows.length * dim)
    for (let i = 0; i < rows.length; i += 1) {
      const offset = i * dim
      const candidate = flat.subarray(offset, offset + dim)
      const score = computeScore(query, candidate, metric)
      pushHeap(heap, { rowIndex: rowStart + i, score }, topK, metric)
    }
    return
  }

  const scratch = new Float32Array(dim)
  const scratchBytes = new Uint8Array(scratch.buffer)
  for (let i = 0; i < rows.length; i += 1) {
    scratchBytes.set(rows[i])
    const score = computeScore(query, scratch, metric)
    pushHeap(heap, { rowIndex: rowStart + i, score }, topK, metric)
  }
}

/**
 * Find the id for a given row index within the collected id chunks.
 *
 * @param {{ start: number, ids: string[] }[]} chunks
 * @param {number} rowIndex
 * @returns {string | undefined}
 */
function lookupId(chunks, rowIndex) {
  for (const { start, ids } of chunks) {
    if (rowIndex >= start && rowIndex < start + ids.length) {
      return ids[rowIndex - start]
    }
  }
  return undefined
}

/**
 * Default AsyncBuffer factory: uses node fs for local paths and HTTP fetch otherwise.
 *
 * @param {{ url: string, signal?: AbortSignal }} options
 * @returns {Promise<AsyncBuffer>}
 */
function defaultAsyncBufferFactory({ url, signal }) {
  if (url.startsWith('http://') || url.startsWith('https://')) {
    /** @type {RequestInit} */
    const requestInit = signal ? { signal } : {}
    return asyncBufferFromUrl({ url, requestInit })
  }
  return asyncBufferFromFile(url)
}

/**
 * Compute the score for a candidate vector under the chosen metric.
 *
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {DistanceMetric} metric
 * @returns {number}
 */
function computeScore(a, b, metric) {
  if (metric === 'cosine') return cosineSimilarity(a, b)
  if (metric === 'dot') return dotProduct(a, b)
  if (metric === 'euclidean') return euclideanDistance(a, b)
  throw new Error(`unsupported metric: ${metric}`)
}

/**
 * Return true if `a` is "better" than `b` under the metric.
 * For cosine/dot, higher is better. For euclidean, lower is better.
 *
 * @param {number} a
 * @param {number} b
 * @param {DistanceMetric} metric
 * @returns {boolean}
 */
function isBetter(a, b, metric) {
  if (metric === 'euclidean') return a < b
  return a > b
}

/**
 * Naive bounded "heap": keep an unsorted array of at most topK items and
 * track the worst by linear scan. Plenty fast for v0 + small topK.
 *
 * @param {{ rowIndex: number, score: number }[]} heap
 * @param {{ rowIndex: number, score: number }} candidate
 * @param {number} topK
 * @param {DistanceMetric} metric
 */
function pushHeap(heap, candidate, topK, metric) {
  if (heap.length < topK) {
    heap.push(candidate)
    return
  }
  let worstIdx = 0
  for (let i = 1; i < heap.length; i += 1) {
    if (isBetter(heap[worstIdx].score, heap[i].score, metric)) {
      worstIdx = i
    }
  }
  if (isBetter(candidate.score, heap[worstIdx].score, metric)) {
    heap[worstIdx] = candidate
  }
}

/**
 * Sort results best-first under the chosen metric.
 *
 * @param {{ rowIndex: number, score: number }[]} results
 * @param {DistanceMetric} metric
 * @returns {{ rowIndex: number, score: number }[]}
 */
function sortHeap(results, metric) {
  const dir = metric === 'euclidean' ? 1 : -1
  return results.slice().sort((a, b) => dir * (a.score - b.score))
}
