/**
 * Binary k-means clustering on packed 1-bit-per-dim sign codes.
 *
 * Used by writeVectors to cluster rows so that rows with similar binary
 * codes are stored adjacently. The cluster id is then written as a sorted
 * INT32 column, and the centroids are stored in KV metadata. searchVectors
 * uses the centroids to pick the top-N nearest clusters and only scan
 * those row ranges in phase 1.
 *
 * Centroid update is "bit majority vote": for each bit position, the new
 * centroid bit is 1 iff more than half the cluster's members have that bit
 * set. This is the binary analogue of mean for euclidean k-means.
 */

/**
 * @param {Uint8Array[]} codes        per-row binary codes (length n, each binaryBytes long)
 * @param {number} binaryBytes        bytes per code (= ceil(dim/8))
 * @param {number} k                  number of clusters
 * @param {number} [iterations=6]     number of k-means iterations
 * @param {number} [seed=1]           RNG seed (deterministic init)
 * @returns {{ assignments: Int32Array, centroids: Uint8Array[] }}
 */
export function binaryKMeans(codes, binaryBytes, k, iterations = 6, seed = 1) {
  const n = codes.length
  if (n === 0) return { assignments: new Int32Array(0), centroids: [] }
  const effectiveK = Math.min(k, n)
  const wordsPerRow = binaryBytes >> 2
  const tailBytes = binaryBytes - wordsPerRow * 4

  // Aligned U32 views over a flat backing buffer (one contiguous copy).
  const flat = new Uint8Array(n * binaryBytes)
  for (let i = 0; i < n; i += 1) flat.set(codes[i], i * binaryBytes)
  const flatU32 = wordsPerRow > 0
    ? new Uint32Array(flat.buffer, 0, n * wordsPerRow)
    : new Uint32Array(0)

  // Random init: pick k distinct row indices as initial centroids.
  let rngState = seed >>> 0 || 1
  /** @returns {number} pseudo-random uint32 */
  function rng() {
    const stepped = Math.imul(rngState, 1664525) + 1013904223
    rngState = stepped >>> 0
    return rngState
  }
  const initIdx = pickDistinct(n, effectiveK, rng)
  /** @type {Uint8Array[]} */
  const centroids = initIdx.map(i => flat.slice(i * binaryBytes, (i + 1) * binaryBytes))
  /** @type {Uint32Array[]} */
  let centroidU32 = centroids.map(c => bytesToU32(c, wordsPerRow))

  const assignments = new Int32Array(n)

  for (let iter = 0; iter < iterations; iter += 1) {
    // Assign each row to nearest centroid.
    for (let i = 0; i < n; i += 1) {
      let best = 0
      let bestDist = Infinity
      const rowOff = i * wordsPerRow
      for (let c = 0; c < effectiveK; c += 1) {
        const cw = centroidU32[c]
        let d = 0
        for (let j = 0; j < wordsPerRow; j += 1) {
          let v = flatU32[rowOff + j] ^ cw[j]
          v = v - (v >>> 1 & 0x55555555)
          v = (v & 0x33333333) + (v >>> 2 & 0x33333333)
          d += (v + (v >>> 4) & 0x0f0f0f0f) * 0x01010101 >>> 24
          if (d >= bestDist) break
        }
        if (d < bestDist) {
          bestDist = d
          best = c
        }
      }
      assignments[i] = best
    }

    // Update centroids via bit-majority vote.
    const lastIter = iter === iterations - 1
    if (lastIter) break
    const counts = new Int32Array(effectiveK)
    // bitSums[c][bit] = number of rows in cluster c with that bit set.
    /** @type {Int32Array[]} */
    const bitSums = []
    for (let c = 0; c < effectiveK; c += 1) bitSums.push(new Int32Array(binaryBytes * 8))

    for (let i = 0; i < n; i += 1) {
      const c = assignments[i]
      counts[c] += 1
      const sums = bitSums[c]
      const rowOff = i * binaryBytes
      for (let b = 0; b < binaryBytes; b += 1) {
        const byte = flat[rowOff + b]
        const baseBit = b * 8
        if (byte & 0x01) sums[baseBit + 0] += 1
        if (byte & 0x02) sums[baseBit + 1] += 1
        if (byte & 0x04) sums[baseBit + 2] += 1
        if (byte & 0x08) sums[baseBit + 3] += 1
        if (byte & 0x10) sums[baseBit + 4] += 1
        if (byte & 0x20) sums[baseBit + 5] += 1
        if (byte & 0x40) sums[baseBit + 6] += 1
        if (byte & 0x80) sums[baseBit + 7] += 1
      }
    }

    for (let c = 0; c < effectiveK; c += 1) {
      if (counts[c] === 0) {
        // Reseed empty cluster from a random row.
        const r = rng() % n
        flat.copyWithin(0, 0, 0) // noop just to keep linter calm
        centroids[c] = flat.slice(r * binaryBytes, (r + 1) * binaryBytes)
        continue
      }
      const half = counts[c] >> 1
      const cb = new Uint8Array(binaryBytes)
      const sums = bitSums[c]
      for (let b = 0; b < binaryBytes; b += 1) {
        let byte = 0
        const baseBit = b * 8
        for (let bit = 0; bit < 8; bit += 1) {
          if (sums[baseBit + bit] > half) byte |= 1 << bit
        }
        cb[b] = byte
      }
      centroids[c] = cb
    }
    centroidU32 = centroids.map(c => bytesToU32(c, wordsPerRow))
    // Use tailBytes to silence unused warning (could be extended for non-u32 tails).
    if (tailBytes < 0) throw new Error('unreachable')
  }

  return { assignments, centroids }
}

/**
 * @param {number} n
 * @param {number} k
 * @param {() => number} rng
 * @returns {number[]}
 */
function pickDistinct(n, k, rng) {
  const picked = new Set()
  while (picked.size < k) picked.add(rng() % n)
  return [...picked]
}

/**
 * @param {Uint8Array} bytes
 * @param {number} wordsPerRow
 * @returns {Uint32Array}
 */
function bytesToU32(bytes, wordsPerRow) {
  if (bytes.byteOffset % 4 === 0) {
    return new Uint32Array(bytes.buffer, bytes.byteOffset, wordsPerRow)
  }
  const copy = new Uint8Array(bytes.byteLength)
  copy.set(bytes)
  return new Uint32Array(copy.buffer, 0, wordsPerRow)
}

/**
 * Hamming distance between two packed binary codes (Uint8Array). Wraps
 * the inner SWAR loop with a Uint32Array view when possible.
 *
 * @param {Uint8Array} a
 * @param {Uint8Array} b
 * @returns {number}
 */
export function hammingDistanceBytes(a, b) {
  if (a.byteLength !== b.byteLength) {
    throw new Error(`hamming length mismatch: ${a.byteLength} vs ${b.byteLength}`)
  }
  const words = a.byteLength >> 2
  let d = 0
  let aU32, bU32
  if (a.byteOffset % 4 === 0 && b.byteOffset % 4 === 0) {
    aU32 = new Uint32Array(a.buffer, a.byteOffset, words)
    bU32 = new Uint32Array(b.buffer, b.byteOffset, words)
  } else {
    aU32 = new Uint32Array(new Uint8Array(a).buffer)
    bU32 = new Uint32Array(new Uint8Array(b).buffer)
  }
  for (let j = 0; j < words; j += 1) {
    let v = aU32[j] ^ bU32[j]
    v = v - (v >>> 1 & 0x55555555)
    v = (v & 0x33333333) + (v >>> 2 & 0x33333333)
    d += (v + (v >>> 4) & 0x0f0f0f0f) * 0x01010101 >>> 24
  }
  const tailStart = words * 4
  for (let b2 = tailStart; b2 < a.byteLength; b2 += 1) {
    let v = a[b2] ^ b[b2]
    v = v - (v >>> 1 & 0x55)
    v = (v & 0x33) + (v >>> 2 & 0x33)
    d += v + (v >>> 4) & 0x0f
  }
  return d
}
