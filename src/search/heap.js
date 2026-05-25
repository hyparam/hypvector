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
    if (isBetter(heap[worstIdx].score, heap[i].score, metric)) {
      worstIdx = i
    }
  }
  if (isBetter(candidate.score, heap[worstIdx].score, metric)) {
    heap[worstIdx] = candidate
  }
}

/**
 * Bounded heap for Hamming candidates (lower hamming is better).
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
    if (heap[i].hamming > heap[worstIdx].hamming) worstIdx = i
  }
  if (candidate.hamming < heap[worstIdx].hamming) {
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
export function sortHeap(results, metric) {
  const dir = metric === 'euclidean' ? 1 : -1
  return results.slice().sort((a, b) => dir * (a.score - b.score))
}
