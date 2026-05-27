/**
 * @import { HypVectorMetadata } from './types.js'
 * @import { FileMetaData, KeyValue } from 'hyparquet'
 */

/**
 * Compute the dot product of two equal-length vectors.
 *
 * @param {Float32Array | number[]} a
 * @param {Float32Array | number[]} b
 * @returns {number}
 */
export function dotProduct(a, b) {
  if (a.length !== b.length) {
    throw new Error(`vector length mismatch: ${a.length} vs ${b.length}`)
  }
  let sum = 0
  for (let i = 0; i < a.length; i += 1) {
    sum += a[i] * b[i]
  }
  return sum
}

/**
 * Compute the cosine similarity of two equal-length vectors.
 * Returns a value in [-1, 1]; higher means more similar.
 *
 * @param {Float32Array | number[]} a
 * @param {Float32Array | number[]} b
 * @returns {number}
 */
export function cosineSimilarity(a, b) {
  if (a.length !== b.length) {
    throw new Error(`vector length mismatch: ${a.length} vs ${b.length}`)
  }
  let dot = 0
  let normA = 0
  let normB = 0
  for (let i = 0; i < a.length; i += 1) {
    const av = a[i]
    const bv = b[i]
    dot += av * bv
    normA += av * av
    normB += bv * bv
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB)
  if (denom === 0) return 0
  return dot / denom
}

/**
 * Compute the Euclidean (L2) distance between two equal-length vectors.
 *
 * @param {Float32Array | number[]} a
 * @param {Float32Array | number[]} b
 * @returns {number}
 */
export function euclideanDistance(a, b) {
  if (a.length !== b.length) {
    throw new Error(`vector length mismatch: ${a.length} vs ${b.length}`)
  }
  let sum = 0
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i]
    sum += d * d
  }
  return Math.sqrt(sum)
}

/**
 * Return an L2-normalized copy of the vector as Float32Array.
 *
 * @param {Float32Array | number[]} v
 * @returns {Float32Array}
 */
export function l2Normalize(v) {
  let norm = 0
  for (let i = 0; i < v.length; i += 1) {
    norm += v[i] * v[i]
  }
  norm = Math.sqrt(norm)
  const out = new Float32Array(v.length)
  if (norm === 0) return out
  for (let i = 0; i < v.length; i += 1) {
    out[i] = v[i] / norm
  }
  return out
}

/**
 * Pack a vector into a Uint8Array of raw little-endian float32 bytes.
 *
 * @param {Float32Array | number[]} v
 * @returns {Uint8Array}
 */
export function packFloat32(v) {
  const f32 = v instanceof Float32Array ? v : Float32Array.from(v)
  return new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength)
}

/**
 * Pack a vector into a 1-bit-per-dimension binary code (sign bits).
 * Bit i of byte (i >> 3) is 1 iff v[i] >= 0. Length is ceil(dim/8) bytes.
 *
 * @param {Float32Array | number[]} v
 * @param {number} [dim] optional explicit dimension (defaults to v.length)
 * @returns {Uint8Array}
 */
export function packBinary(v, dim = v.length) {
  const bytes = new Uint8Array(dim + 7 >> 3)
  for (let i = 0; i < dim; i += 1) {
    if (v[i] >= 0) bytes[i >> 3] |= 1 << (i & 7)
  }
  return bytes
}

/**
 * Unpack a Uint8Array of raw little-endian float32 bytes into a Float32Array.
 * Always returns a fresh aligned copy, since the source buffer may be
 * misaligned for a Float32Array view.
 *
 * @param {Uint8Array} bytes
 * @returns {Float32Array}
 */
export function unpackFloat32(bytes) {
  const out = new Float32Array(bytes.byteLength / 4)
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  for (let i = 0; i < out.length; i += 1) {
    out[i] = view.getFloat32(i * 4, true)
  }
  return out
}

/**
 * Parse hypvector key-value metadata from a parquet file.
 *
 * @param {FileMetaData} metadata
 * @returns {HypVectorMetadata}
 */
export function parseKvMetadata(metadata) {
  /** @type {Record<string, string>} */
  const kv = {}
  for (const entry of metadata.key_value_metadata ?? []) {
    if (entry.value !== undefined) kv[entry.key] = entry.value
  }
  const version = parseInt(kv['hypvector.version'] ?? '0', 10)
  const dimension = parseInt(kv['hypvector.dimension'] ?? '0', 10)
  const metric = /** @type {HypVectorMetadata['metric']} */ (kv['hypvector.metric'] ?? 'cosine')
  const normalized = kv['hypvector.normalized'] === 'true'
  const hasBinary = kv['hypvector.binary'] === 'true'
  const hasPq = kv['hypvector.pq'] === 'true'
  const count = Number(metadata.num_rows)
  const clusters = parseInt(kv['hypvector.clusters'] ?? '0', 10)
  if (!dimension) {
    throw new Error('Not a hypvector parquet file: missing hypvector.dimension metadata')
  }
  /** @type {HypVectorMetadata} */
  const out = { version, dimension, metric, normalized, hasBinary, hasPq, count, clusters }
  if (clusters > 0 && kv['hypvector.centroids']) {
    const binaryBytes = dimension + 7 >> 3
    const bytes = decodeBase64(kv['hypvector.centroids'])
    if (bytes.byteLength !== clusters * binaryBytes) {
      throw new Error(`centroids length mismatch: ${bytes.byteLength} vs ${clusters * binaryBytes}`)
    }
    out.centroids = []
    for (let c = 0; c < clusters; c += 1) {
      out.centroids.push(bytes.slice(c * binaryBytes, (c + 1) * binaryBytes))
    }
  }
  if (clusters > 0 && kv['hypvector.clusterCounts']) {
    const bytes = decodeBase64(kv['hypvector.clusterCounts'])
    // The encoded buffer may not be 4-byte-aligned; copy to ensure alignment.
    const aligned = new Uint8Array(bytes.byteLength)
    aligned.set(bytes)
    out.clusterCounts = new Uint32Array(aligned.buffer, 0, clusters)
  }
  if (hasPq) {
    const pqSegments = parseInt(kv['hypvector.pq.segments'] ?? '0', 10)
    const pqCentroids = parseInt(kv['hypvector.pq.centroids'] ?? '0', 10)
    if (!pqSegments || !pqCentroids) {
      throw new Error('PQ metadata is missing segment or centroid count')
    }
    out.pqSegments = pqSegments
    out.pqCentroids = pqCentroids
    if (kv['hypvector.pq.codebooks']) {
      const bytes = decodeBase64(kv['hypvector.pq.codebooks'])
      const expectedBytes = pqCentroids * dimension * 4
      if (bytes.byteLength !== expectedBytes) {
        throw new Error(`PQ codebooks length mismatch: ${bytes.byteLength} vs ${expectedBytes}`)
      }
      const aligned = new Uint8Array(bytes.byteLength)
      aligned.set(bytes)
      out.pqCodebooks = new Float32Array(aligned.buffer, 0, pqCentroids * dimension)
    }
  }
  return out
}

/**
 * @param {string} s
 * @returns {Uint8Array}
 */
export function decodeBase64(s) {
  if (typeof Buffer !== 'undefined') return new Uint8Array(Buffer.from(s, 'base64'))
  const bin = atob(s)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i += 1) out[i] = bin.charCodeAt(i)
  return out
}

/**
 * Base64 encode a Uint8Array. Uses Node's Buffer when available, falls back
 * to btoa for browser environments.
 *
 * @param {Uint8Array} bytes
 * @returns {string}
 */
export function encodeBase64(bytes) {
  if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64')
  let s = ''
  for (let i = 0; i < bytes.length; i += 1) s += String.fromCharCode(bytes[i])
  return btoa(s)
}
