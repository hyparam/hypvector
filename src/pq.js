/**
 * Product quantization helpers.
 *
 * Codebooks are stored segment-major. For segment s with bounds
 * [bounds[s], bounds[s + 1]), the codebook block starts at
 * `centroids * bounds[s]` and contains `centroids * segmentDim` float32s.
 */

/**
 * @import { DistanceMetric, HypVectorMetadata } from './types.js'
 */

/**
 * Build product-quantized codes for a set of vectors.
 *
 * @param {object} options
 * @param {Float32Array[]} options.vectors
 * @param {number} options.dimension
 * @param {number} options.segments
 * @param {number} options.centroids
 * @param {number} options.iterations
 * @param {number} options.sampleSize
 * @param {number} options.seed
 * @returns {{ codes: Uint8Array[], codebooks: Float32Array, segments: number, centroids: number }}
 */
export function buildPq({ vectors, dimension, segments, centroids, iterations, sampleSize, seed }) {
  if (!Number.isInteger(segments) || segments <= 0) {
    throw new Error(`pqSegments must be a positive integer, got ${segments}`)
  }
  if (!Number.isInteger(centroids) || centroids <= 1 || centroids > 256) {
    throw new Error(`pqCentroids must be an integer in [2, 256], got ${centroids}`)
  }
  const effectiveSegments = Math.min(segments, dimension)
  const bounds = pqSegmentBounds(dimension, effectiveSegments)
  const sample = sampleIndices(vectors.length, sampleSize)
  const codebooks = new Float32Array(centroids * dimension)

  for (let s = 0; s < effectiveSegments; s += 1) {
    trainSegment({
      vectors,
      sample,
      start: bounds[s],
      end: bounds[s + 1],
      centroids,
      iterations,
      seed: seed + s * 1009,
      out: codebooks,
    })
  }

  const codes = new Array(vectors.length)
  for (let i = 0; i < vectors.length; i += 1) {
    codes[i] = encodePqVector(vectors[i], codebooks, dimension, effectiveSegments, centroids)
  }

  return { codes, codebooks, segments: effectiveSegments, centroids }
}

/**
 * Return segment boundaries that cover [0, dimension).
 *
 * @param {number} dimension
 * @param {number} segments
 * @returns {Uint32Array}
 */
export function pqSegmentBounds(dimension, segments) {
  const bounds = new Uint32Array(segments + 1)
  for (let s = 0; s <= segments; s += 1) {
    bounds[s] = Math.floor(s * dimension / segments)
  }
  return bounds
}

/**
 * Encode one vector against trained PQ codebooks.
 *
 * @param {Float32Array} vector
 * @param {Float32Array} codebooks
 * @param {number} dimension
 * @param {number} segments
 * @param {number} centroids
 * @returns {Uint8Array}
 */
export function encodePqVector(vector, codebooks, dimension, segments, centroids) {
  const bounds = pqSegmentBounds(dimension, segments)
  const code = new Uint8Array(segments)
  for (let s = 0; s < segments; s += 1) {
    const start = bounds[s]
    const end = bounds[s + 1]
    code[s] = nearestCentroid(vector, codebooks, start, end, centroids)
  }
  return code
}

/**
 * Build per-segment lookup tables for approximate PQ scoring.
 *
 * For euclidean search the table stores squared L2 contributions and lower
 * values are better. For dot/cosine search it stores dot-product
 * contributions and higher values are better.
 *
 * @param {Float32Array} query
 * @param {HypVectorMetadata} meta
 * @param {DistanceMetric} metric
 * @returns {{ table: Float32Array, approxMetric: DistanceMetric }}
 */
export function buildPqTables(query, meta, metric) {
  if (!meta.hasPq || !meta.pqCodebooks || !meta.pqSegments || !meta.pqCentroids) {
    throw new Error('PQ metadata is missing')
  }
  const table = new Float32Array(meta.pqSegments * meta.pqCentroids)
  const bounds = pqSegmentBounds(meta.dimension, meta.pqSegments)
  for (let s = 0; s < meta.pqSegments; s += 1) {
    const start = bounds[s]
    const end = bounds[s + 1]
    const dim = end - start
    const block = meta.pqCentroids * start
    for (let c = 0; c < meta.pqCentroids; c += 1) {
      const centroid = block + c * dim
      let score = 0
      if (metric === 'euclidean') {
        for (let d = 0; d < dim; d += 1) {
          const delta = query[start + d] - meta.pqCodebooks[centroid + d]
          score += delta * delta
        }
      } else {
        for (let d = 0; d < dim; d += 1) {
          score += query[start + d] * meta.pqCodebooks[centroid + d]
        }
      }
      table[s * meta.pqCentroids + c] = score
    }
  }
  return { table, approxMetric: metric === 'euclidean' ? 'euclidean' : 'dot' }
}

/**
 * Train one subspace codebook with k-means over a deterministic sample.
 *
 * @param {object} options
 * @param {Float32Array[]} options.vectors
 * @param {Int32Array} options.sample
 * @param {number} options.start
 * @param {number} options.end
 * @param {number} options.centroids
 * @param {number} options.iterations
 * @param {number} options.seed
 * @param {Float32Array} options.out
 */
function trainSegment({ vectors, sample, start, end, centroids, iterations, seed, out }) {
  const dim = end - start
  const block = centroids * start
  const sampleCount = sample.length
  if (sampleCount === 0) return

  for (let c = 0; c < centroids; c += 1) {
    const src = vectors[sample[Math.floor(c * sampleCount / centroids)]]
    out.set(src.subarray(start, end), block + c * dim)
  }

  for (let iter = 0; iter < iterations; iter += 1) {
    const counts = new Int32Array(centroids)
    const sums = new Float32Array(centroids * dim)

    for (let i = 0; i < sampleCount; i += 1) {
      const vector = vectors[sample[i]]
      const best = nearestCentroid(vector, out, start, end, centroids)
      counts[best] += 1
      const sumOff = best * dim
      for (let d = 0; d < dim; d += 1) sums[sumOff + d] += vector[start + d]
    }

    for (let c = 0; c < centroids; c += 1) {
      const dst = block + c * dim
      if (counts[c] === 0) {
        const src = vectors[sample[reseedIndex(seed, iter, c, sampleCount)]]
        out.set(src.subarray(start, end), dst)
        continue
      }
      const inv = 1 / counts[c]
      const sumOff = c * dim
      for (let d = 0; d < dim; d += 1) out[dst + d] = sums[sumOff + d] * inv
    }
  }
}

/**
 * Find the nearest centroid for one segment under squared L2.
 *
 * @param {Float32Array} vector
 * @param {Float32Array} codebooks
 * @param {number} start
 * @param {number} end
 * @param {number} centroids
 * @returns {number}
 */
function nearestCentroid(vector, codebooks, start, end, centroids) {
  const dim = end - start
  const block = centroids * start
  let best = 0
  let bestDist = Infinity
  for (let c = 0; c < centroids; c += 1) {
    const off = block + c * dim
    let dist = 0
    for (let d = 0; d < dim; d += 1) {
      const delta = vector[start + d] - codebooks[off + d]
      dist += delta * delta
      if (dist >= bestDist) break
    }
    if (dist < bestDist) {
      bestDist = dist
      best = c
    }
  }
  return best
}

/**
 * Deterministic evenly-spaced sample indices.
 *
 * @param {number} count
 * @param {number} sampleSize
 * @returns {Int32Array}
 */
function sampleIndices(count, sampleSize) {
  const n = Math.min(count, Math.max(1, sampleSize))
  const out = new Int32Array(n)
  for (let i = 0; i < n; i += 1) out[i] = Math.floor(i * count / n)
  return out
}

/**
 * @param {number} seed
 * @param {number} iter
 * @param {number} centroid
 * @param {number} sampleCount
 * @returns {number}
 */
function reseedIndex(seed, iter, centroid, sampleCount) {
  let s = (seed ^ Math.imul(iter + 1, 2654435761) ^ Math.imul(centroid + 1, 2246822519)) >>> 0
  s = Math.imul(s, 1664525) + 1013904223 >>> 0
  return s % sampleCount
}
