import { cosineSimilarity, dotProduct, euclideanDistance } from '../utils.js'

/**
 * @import { DistanceMetric } from '../types.js'
 */

/**
 * Compute the score for a candidate vector under the chosen metric.
 *
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {DistanceMetric} metric
 * @returns {number}
 */
export function computeScore(a, b, metric) {
  if (metric === 'cosine') return cosineSimilarity(a, b)
  if (metric === 'dot') return dotProduct(a, b)
  if (metric === 'euclidean') return euclideanDistance(a, b)
  throw new Error(`unsupported metric: ${metric}`)
}

/**
 * Return true if `a` is "better" than `b` under the metric.
 *
 * @param {number} a
 * @param {number} b
 * @param {DistanceMetric} metric
 * @returns {boolean}
 */
export function isBetter(a, b, metric) {
  if (metric === 'euclidean') return a < b
  return a > b
}

/**
 * Total order on scored entries: better score wins; ties broken by lower
 * rowIndex. The tiebreak makes selection independent of the order chunks
 * arrive (parallel reads complete in nondeterministic order), so identical
 * queries always return identical results.
 *
 * @param {{ rowIndex: number, score: number }} a
 * @param {{ rowIndex: number, score: number }} b
 * @param {DistanceMetric} metric
 * @returns {boolean}
 */
function betterEntry(a, b, metric) {
  if (a.score !== b.score) return isBetter(a.score, b.score, metric)
  return a.rowIndex < b.rowIndex
}

/**
 * Bounded heap by score (uses metric to decide "better"). Linear-scan to
 * find the worst entry; fine for the small topK we care about.
 *
 * @param {{ rowIndex: number, score: number }[]} heap
 * @param {{ rowIndex: number, score: number }} candidate
 * @param {number} topK
 * @param {DistanceMetric} metric
 */
export function pushHeap(heap, candidate, topK, metric) {
  if (heap.length < topK) {
    heap.push(candidate)
    return
  }
  let worstIdx = 0
  for (let i = 1; i < heap.length; i += 1) {
    if (betterEntry(heap[worstIdx], heap[i], metric)) {
      worstIdx = i
    }
  }
  if (betterEntry(candidate, heap[worstIdx], metric)) {
    heap[worstIdx] = candidate
  }
}

/**
 * Bounded heap for Hamming candidates (lower hamming is better). Ties on
 * hamming are broken by lower rowIndex so the kept candidate set does not
 * depend on the (parallel, nondeterministic) order chunks are scored.
 *
 * @param {{ rowIndex: number, hamming: number }[]} heap
 * @param {{ rowIndex: number, hamming: number }} candidate
 * @param {number} candidatesK
 */
export function pushHammingHeap(heap, candidate, candidatesK) {
  if (heap.length < candidatesK) {
    heap.push(candidate)
    return
  }
  let worstIdx = 0
  for (let i = 1; i < heap.length; i += 1) {
    const h = heap[i]; const w = heap[worstIdx]
    if (h.hamming > w.hamming || (h.hamming === w.hamming && h.rowIndex > w.rowIndex)) worstIdx = i
  }
  const w = heap[worstIdx]
  if (candidate.hamming < w.hamming || (candidate.hamming === w.hamming && candidate.rowIndex < w.rowIndex)) {
    heap[worstIdx] = candidate
  }
}

/**
 * Sort results best-first under the chosen metric, ties broken by lower
 * rowIndex for a deterministic order.
 *
 * @param {{ rowIndex: number, score: number }[]} results
 * @param {DistanceMetric} metric
 * @returns {{ rowIndex: number, score: number }[]}
 */
export function sortHeap(results, metric) {
  const dir = metric === 'euclidean' ? 1 : -1
  return results.slice().sort((a, b) => dir * (a.score - b.score) || a.rowIndex - b.rowIndex)
}
