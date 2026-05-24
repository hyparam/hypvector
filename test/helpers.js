/**
 * Wraps an AsyncBuffer to count the number of fetches made.
 *
 * @import {AsyncBuffer} from 'hyparquet'
 * @param {AsyncBuffer} asyncBuffer
 * @returns {AsyncBuffer & {fetches: number, bytes: number}}
 */
export function countingBuffer(asyncBuffer) {
  return {
    ...asyncBuffer,
    fetches: 0,
    bytes: 0,
    slice(start, end) {
      this.fetches++
      this.bytes += (end ?? asyncBuffer.byteLength) - start
      return asyncBuffer.slice(start, end)
    },
  }
}

/**
 * Generate deterministic pseudo-random vectors for tests.
 * Uses a simple LCG so output is stable across runs.
 *
 * @param {number} count
 * @param {number} dimension
 * @param {number} [seed]
 * @returns {{ id: string, vector: Float32Array }[]}
 */
export function makeVectors(count, dimension, seed = 1) {
  let state = seed >>> 0 || 1
  function next() {
    const stepped = Math.imul(state, 1664525) + 1013904223
    state = stepped >>> 0
    return state / 0x100000000
  }
  const out = []
  for (let i = 0; i < count; i += 1) {
    const v = new Float32Array(dimension)
    for (let j = 0; j < dimension; j += 1) {
      v[j] = next() * 2 - 1
    }
    out.push({ id: `vec-${i}`, vector: v })
  }
  return out
}
