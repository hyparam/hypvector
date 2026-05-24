import { asyncBufferFromFile, asyncBufferFromUrl, parquetMetadataAsync } from 'hyparquet'
import { readVectors } from './readVectors.js'
import {
  cosineSimilarity,
  dotProduct,
  euclideanDistance,
  parseKvMetadata,
} from './utils.js'

/**
 * @import { SearchResult, SearchVectorsOptions, DistanceMetric } from './types.js'
 */

/**
 * Find the top-k nearest neighbors to a query vector.
 *
 * Naive v0: linear scan over every stored vector. This streams rows from the
 * source parquet, computes the chosen similarity metric, and keeps a
 * top-k heap of the best results.
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
  const usedMetric = metric ?? meta.metric

  /** @type {SearchResult[]} */
  const heap = []
  let rowIndex = 0

  for await (const record of readVectors({ file, metadata, includeMetadata: true })) {
    const score = computeScore(query, record.vector, usedMetric)
    pushHeap(heap, { id: record.id, score, rowIndex, metadata: record.metadata }, topK, usedMetric)
    rowIndex += 1
  }

  return sortResults(heap, usedMetric)
}

/**
 * Default AsyncBuffer factory: uses node fs for local paths and HTTP fetch otherwise.
 *
 * @param {{ url: string, signal?: AbortSignal }} options
 * @returns {Promise<import('hyparquet').AsyncBuffer>}
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
 * @param {Float32Array | number[]} a
 * @param {Float32Array | number[]} b
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
 *
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
 * @param {SearchResult[]} heap
 * @param {SearchResult} candidate
 * @param {number} topK
 * @param {DistanceMetric} metric
 */
function pushHeap(heap, candidate, topK, metric) {
  if (heap.length < topK) {
    heap.push(candidate)
    return
  }
  // Find current worst entry
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
 * @param {SearchResult[]} results
 * @param {DistanceMetric} metric
 * @returns {SearchResult[]}
 */
function sortResults(results, metric) {
  const dir = metric === 'euclidean' ? 1 : -1
  return results.slice().sort((a, b) => dir * (a.score - b.score))
}
