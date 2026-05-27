import { defaultClusterProbeFraction } from './constants.js'

/**
 * @import { DistanceMetric, HypVectorMetadata } from './types.js'
 */

/**
 * Build an IVF-PQ index:
 *   1. train a coarse float k-means quantizer
 *   2. assign every vector to its nearest IVF centroid
 *   3. train global residual PQ codebooks
 *   4. encode every vector's residual as one byte per PQ segment
 *
 * @param {object} options
 * @param {Float32Array[]} options.vectors
 * @param {number} options.dimension
 * @param {number} options.ivfClusters
 * @param {number} options.ivfIterations
 * @param {number} options.ivfSampleSize
 * @param {number} options.pqSegments
 * @param {number} options.pqCentroids
 * @param {number} options.pqIterations
 * @param {number} options.pqSampleSize
 * @param {number} options.seed
 * @returns {{
 *   codes: Uint8Array[],
 *   assignments: Int32Array,
 *   codebooks: Float32Array,
 *   ivfCentroids: Float32Array,
 *   ivfCounts: Uint32Array,
 *   pqSegments: number,
 *   pqCentroids: number,
 *   ivfClusters: number,
 * }}
 */
export function buildIvfPq({
  vectors,
  dimension,
  ivfClusters,
  ivfIterations,
  ivfSampleSize,
  pqSegments,
  pqCentroids,
  pqIterations,
  pqSampleSize,
  seed,
}) {
  validateParams({ dimension, ivfClusters, pqSegments, pqCentroids })
  const effectivePqSegments = Math.min(pqSegments, dimension)
  const effectiveIvfClusters = Math.min(ivfClusters, vectors.length)

  const coarse = trainKMeans({
    vectors,
    dimension,
    k: effectiveIvfClusters,
    iterations: ivfIterations,
    sampleSize: ivfSampleSize,
    seed,
  })
  const rawAssignments = assignVectors(vectors, coarse.centroids, dimension, coarse.k)
  const compact = compactIvf(coarse.centroids, rawAssignments, dimension, coarse.k)

  const codebooks = trainResidualPq({
    vectors,
    assignments: compact.assignments,
    ivfCentroids: compact.centroids,
    dimension,
    segments: effectivePqSegments,
    centroids: pqCentroids,
    iterations: pqIterations,
    sampleSize: pqSampleSize,
    seed,
  })

  const codes = new Array(vectors.length)
  for (let i = 0; i < vectors.length; i += 1) {
    codes[i] = encodeResidualPqVector(
      vectors[i],
      compact.assignments[i],
      compact.centroids,
      codebooks,
      dimension,
      effectivePqSegments,
      pqCentroids
    )
  }

  return {
    codes,
    assignments: compact.assignments,
    codebooks,
    ivfCentroids: compact.centroids,
    ivfCounts: compact.counts,
    pqSegments: effectivePqSegments,
    pqCentroids,
    ivfClusters: compact.counts.length,
  }
}

/**
 * Return IVF row ranges selected by coarse centroid similarity.
 *
 * @param {HypVectorMetadata} meta
 * @param {Float32Array} query
 * @param {DistanceMetric} metric
 * @param {number | undefined} probe
 * @returns {{ list: number, rowStart: number, rowEnd: number }[]}
 */
export function selectIvfRanges(meta, query, metric, probe) {
  if (!meta.ivfCentroids || !meta.ivfCounts || !meta.ivfClusters) {
    return [{ list: -1, rowStart: 0, rowEnd: meta.count }]
  }

  const offsets = new Uint32Array(meta.ivfClusters + 1)
  for (let i = 0; i < meta.ivfClusters; i += 1) offsets[i + 1] = offsets[i] + meta.ivfCounts[i]

  const scores = new Array(meta.ivfClusters)
  for (let list = 0; list < meta.ivfClusters; list += 1) {
    scores[list] = { list, score: coarseScore(query, meta.ivfCentroids, list, meta.dimension, metric) }
  }
  const dir = metric === 'euclidean' ? 1 : -1
  scores.sort((a, b) => dir * (a.score - b.score))

  const probeValue = probe === undefined ? defaultClusterProbeFraction : probe
  const targetLists = probeValue > 1
    ? Math.min(Math.ceil(probeValue), meta.ivfClusters)
    : Math.max(1, Math.ceil(meta.ivfClusters * probeValue))

  return scores.slice(0, targetLists)
    .map(({ list }) => ({ list, rowStart: offsets[list], rowEnd: offsets[list + 1] }))
    .filter(r => r.rowEnd > r.rowStart)
}

/**
 * Build a per-code lookup table for one IVF list.
 *
 * @param {Float32Array} query
 * @param {HypVectorMetadata} meta
 * @param {DistanceMetric} metric
 * @param {number} list
 * @returns {{ table: Float32Array, approxMetric: DistanceMetric }}
 */
export function buildIvfPqTable(query, meta, metric, list) {
  if (!meta.pqCodebooks || !meta.pqSegments || !meta.pqCentroids || !meta.ivfCentroids) {
    throw new Error('IVF-PQ metadata is missing')
  }
  const table = new Float32Array(meta.pqSegments * meta.pqCentroids)
  const bounds = pqSegmentBounds(meta.dimension, meta.pqSegments)
  const coarseBase = list * meta.dimension
  for (let s = 0; s < meta.pqSegments; s += 1) {
    const start = bounds[s]
    const end = bounds[s + 1]
    const dim = end - start
    const codebookBlock = meta.pqCentroids * start
    for (let c = 0; c < meta.pqCentroids; c += 1) {
      const codebook = codebookBlock + c * dim
      let score = 0
      if (metric === 'euclidean') {
        for (let d = 0; d < dim; d += 1) {
          const reconstructed = meta.ivfCentroids[coarseBase + start + d] + meta.pqCodebooks[codebook + d]
          const delta = query[start + d] - reconstructed
          score += delta * delta
        }
      } else {
        for (let d = 0; d < dim; d += 1) {
          const reconstructed = meta.ivfCentroids[coarseBase + start + d] + meta.pqCodebooks[codebook + d]
          score += query[start + d] * reconstructed
        }
      }
      table[s * meta.pqCentroids + c] = score
    }
  }
  return { table, approxMetric: metric === 'euclidean' ? 'euclidean' : 'dot' }
}

/**
 * @param {number} dimension
 * @param {number} segments
 * @returns {Uint32Array}
 */
export function pqSegmentBounds(dimension, segments) {
  const bounds = new Uint32Array(segments + 1)
  for (let s = 0; s <= segments; s += 1) bounds[s] = Math.floor(s * dimension / segments)
  return bounds
}

/**
 * @param {object} options
 * @param {number} options.dimension
 * @param {number} options.ivfClusters
 * @param {number} options.pqSegments
 * @param {number} options.pqCentroids
 */
function validateParams({ dimension, ivfClusters, pqSegments, pqCentroids }) {
  if (!Number.isInteger(ivfClusters) || ivfClusters <= 0) {
    throw new Error(`ivfClusters must be a positive integer, got ${ivfClusters}`)
  }
  if (!Number.isInteger(pqSegments) || pqSegments <= 0) {
    throw new Error(`pqSegments must be a positive integer, got ${pqSegments}`)
  }
  if (pqSegments > dimension) {
    throw new Error(`pqSegments ${pqSegments} cannot exceed dimension ${dimension}`)
  }
  if (!Number.isInteger(pqCentroids) || pqCentroids <= 1 || pqCentroids > 256) {
    throw new Error(`pqCentroids must be an integer in [2, 256], got ${pqCentroids}`)
  }
}

/**
 * @param {object} options
 * @param {Float32Array[]} options.vectors
 * @param {number} options.dimension
 * @param {number} options.k
 * @param {number} options.iterations
 * @param {number} options.sampleSize
 * @param {number} options.seed
 * @returns {{ centroids: Float32Array, k: number }}
 */
function trainKMeans({ vectors, dimension, k, iterations, sampleSize, seed }) {
  const sample = sampleIndices(vectors.length, sampleSize)
  const effectiveK = Math.min(k, sample.length)
  const centroids = new Float32Array(effectiveK * dimension)

  for (let c = 0; c < effectiveK; c += 1) {
    const src = vectors[sample[Math.floor(c * sample.length / effectiveK)]]
    centroids.set(src, c * dimension)
  }

  for (let iter = 0; iter < iterations; iter += 1) {
    const counts = new Int32Array(effectiveK)
    const sums = new Float32Array(effectiveK * dimension)

    for (let i = 0; i < sample.length; i += 1) {
      const vector = vectors[sample[i]]
      const best = nearestVectorCentroid(vector, centroids, dimension, effectiveK)
      counts[best] += 1
      const sumOff = best * dimension
      for (let d = 0; d < dimension; d += 1) sums[sumOff + d] += vector[d]
    }

    for (let c = 0; c < effectiveK; c += 1) {
      const dst = c * dimension
      if (counts[c] === 0) {
        centroids.set(vectors[sample[reseedIndex(seed, iter, c, sample.length)]], dst)
        continue
      }
      const inv = 1 / counts[c]
      for (let d = 0; d < dimension; d += 1) centroids[dst + d] = sums[dst + d] * inv
    }
  }

  return { centroids, k: effectiveK }
}

/**
 * @param {Float32Array[]} vectors
 * @param {Float32Array} centroids
 * @param {number} dimension
 * @param {number} k
 * @returns {Int32Array}
 */
function assignVectors(vectors, centroids, dimension, k) {
  const assignments = new Int32Array(vectors.length)
  for (let i = 0; i < vectors.length; i += 1) {
    assignments[i] = nearestVectorCentroid(vectors[i], centroids, dimension, k)
  }
  return assignments
}

/**
 * Remove empty IVF lists and remap assignments.
 *
 * @param {Float32Array} centroids
 * @param {Int32Array} assignments
 * @param {number} dimension
 * @param {number} k
 * @returns {{ centroids: Float32Array, assignments: Int32Array, counts: Uint32Array }}
 */
function compactIvf(centroids, assignments, dimension, k) {
  const rawCounts = new Uint32Array(k)
  for (let i = 0; i < assignments.length; i += 1) rawCounts[assignments[i]] += 1

  const remap = new Int32Array(k)
  remap.fill(-1)
  let outK = 0
  for (let c = 0; c < k; c += 1) {
    if (rawCounts[c] > 0) remap[c] = outK++
  }

  const outCentroids = new Float32Array(outK * dimension)
  const outCounts = new Uint32Array(outK)
  for (let c = 0; c < k; c += 1) {
    const dst = remap[c]
    if (dst < 0) continue
    outCentroids.set(centroids.subarray(c * dimension, (c + 1) * dimension), dst * dimension)
    outCounts[dst] = rawCounts[c]
  }

  const outAssignments = new Int32Array(assignments.length)
  for (let i = 0; i < assignments.length; i += 1) outAssignments[i] = remap[assignments[i]]
  return { centroids: outCentroids, assignments: outAssignments, counts: outCounts }
}

/**
 * @param {object} options
 * @param {Float32Array[]} options.vectors
 * @param {Int32Array} options.assignments
 * @param {Float32Array} options.ivfCentroids
 * @param {number} options.dimension
 * @param {number} options.segments
 * @param {number} options.centroids
 * @param {number} options.iterations
 * @param {number} options.sampleSize
 * @param {number} options.seed
 * @returns {Float32Array}
 */
function trainResidualPq({ vectors, assignments, ivfCentroids, dimension, segments, centroids, iterations, sampleSize, seed }) {
  const bounds = pqSegmentBounds(dimension, segments)
  const sample = sampleIndices(vectors.length, sampleSize)
  const codebooks = new Float32Array(centroids * dimension)

  for (let s = 0; s < segments; s += 1) {
    trainResidualSegment({
      vectors,
      assignments,
      ivfCentroids,
      sample,
      start: bounds[s],
      end: bounds[s + 1],
      dimension,
      centroids,
      iterations,
      seed: seed + s * 1009,
      out: codebooks,
    })
  }

  return codebooks
}

/**
 * @param {object} options
 * @param {Float32Array[]} options.vectors
 * @param {Int32Array} options.assignments
 * @param {Float32Array} options.ivfCentroids
 * @param {Int32Array} options.sample
 * @param {number} options.start
 * @param {number} options.end
 * @param {number} options.dimension
 * @param {number} options.centroids
 * @param {number} options.iterations
 * @param {number} options.seed
 * @param {Float32Array} options.out
 */
function trainResidualSegment({ vectors, assignments, ivfCentroids, sample, start, end, dimension, centroids, iterations, seed, out }) {
  const dim = end - start
  const block = centroids * start

  for (let c = 0; c < centroids; c += 1) {
    const row = sample[Math.floor(c * sample.length / centroids)]
    writeResidual(vectors[row], assignments[row], ivfCentroids, dimension, start, end, out, block + c * dim)
  }

  for (let iter = 0; iter < iterations; iter += 1) {
    const counts = new Int32Array(centroids)
    const sums = new Float32Array(centroids * dim)

    for (let i = 0; i < sample.length; i += 1) {
      const row = sample[i]
      const best = nearestResidualCentroid(vectors[row], assignments[row], ivfCentroids, out, dimension, start, end, centroids)
      counts[best] += 1
      const coarse = assignments[row] * dimension
      const sumOff = best * dim
      for (let d = 0; d < dim; d += 1) {
        sums[sumOff + d] += vectors[row][start + d] - ivfCentroids[coarse + start + d]
      }
    }

    for (let c = 0; c < centroids; c += 1) {
      const dst = block + c * dim
      if (counts[c] === 0) {
        const row = sample[reseedIndex(seed, iter, c, sample.length)]
        writeResidual(vectors[row], assignments[row], ivfCentroids, dimension, start, end, out, dst)
        continue
      }
      const inv = 1 / counts[c]
      const sumOff = c * dim
      for (let d = 0; d < dim; d += 1) out[dst + d] = sums[sumOff + d] * inv
    }
  }
}

/**
 * @param {Float32Array} vector
 * @param {number} assignment
 * @param {Float32Array} ivfCentroids
 * @param {Float32Array} codebooks
 * @param {number} dimension
 * @param {number} segments
 * @param {number} centroids
 * @returns {Uint8Array}
 */
function encodeResidualPqVector(vector, assignment, ivfCentroids, codebooks, dimension, segments, centroids) {
  const bounds = pqSegmentBounds(dimension, segments)
  const code = new Uint8Array(segments)
  for (let s = 0; s < segments; s += 1) {
    code[s] = nearestResidualCentroid(vector, assignment, ivfCentroids, codebooks, dimension, bounds[s], bounds[s + 1], centroids)
  }
  return code
}

/**
 * @param {Float32Array} vector
 * @param {Float32Array} centroids
 * @param {number} dimension
 * @param {number} k
 * @returns {number}
 */
function nearestVectorCentroid(vector, centroids, dimension, k) {
  let best = 0
  let bestDist = Infinity
  for (let c = 0; c < k; c += 1) {
    const off = c * dimension
    let dist = 0
    for (let d = 0; d < dimension; d += 1) {
      const delta = vector[d] - centroids[off + d]
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
 * @param {Float32Array} vector
 * @param {number} assignment
 * @param {Float32Array} ivfCentroids
 * @param {Float32Array} codebooks
 * @param {number} dimension
 * @param {number} start
 * @param {number} end
 * @param {number} centroids
 * @returns {number}
 */
function nearestResidualCentroid(vector, assignment, ivfCentroids, codebooks, dimension, start, end, centroids) {
  const dim = end - start
  const block = centroids * start
  const coarse = assignment * dimension
  let best = 0
  let bestDist = Infinity
  for (let c = 0; c < centroids; c += 1) {
    const off = block + c * dim
    let dist = 0
    for (let d = 0; d < dim; d += 1) {
      const residual = vector[start + d] - ivfCentroids[coarse + start + d]
      const delta = residual - codebooks[off + d]
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
 * @param {Float32Array} vector
 * @param {number} assignment
 * @param {Float32Array} ivfCentroids
 * @param {number} dimension
 * @param {number} start
 * @param {number} end
 * @param {Float32Array} out
 * @param {number} outOffset
 */
function writeResidual(vector, assignment, ivfCentroids, dimension, start, end, out, outOffset) {
  const coarse = assignment * dimension
  for (let d = start; d < end; d += 1) out[outOffset + d - start] = vector[d] - ivfCentroids[coarse + d]
}

/**
 * @param {Float32Array} query
 * @param {Float32Array} centroids
 * @param {number} list
 * @param {number} dimension
 * @param {DistanceMetric} metric
 * @returns {number}
 */
function coarseScore(query, centroids, list, dimension, metric) {
  const off = list * dimension
  let score = 0
  if (metric === 'euclidean') {
    for (let d = 0; d < dimension; d += 1) {
      const delta = query[d] - centroids[off + d]
      score += delta * delta
    }
    return score
  }
  for (let d = 0; d < dimension; d += 1) score += query[d] * centroids[off + d]
  return score
}

/**
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
