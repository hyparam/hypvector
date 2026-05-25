import { computeScore, pushHammingHeap, pushHeap } from './heap.js'

/**
 * @import { DistanceMetric } from '../types.js'
 * @import { DecodedArray } from 'hyparquet'
 */

/**
 * Score every row in a vector chunk and update the top-k heap.
 *
 * @param {DecodedArray} columnData
 * @param {number} rowStart
 * @param {number} dim
 * @param {Float32Array} query
 * @param {DistanceMetric} metric
 * @param {{ rowIndex: number, score: number }[]} heap
 * @param {number} topK
 */
export function scoreVectorChunk(columnData, rowStart, dim, query, metric, heap, topK) {
  const rows = /** @type {Uint8Array[]} */ (columnData)
  if (rows.length === 0) return
  const first = rows[0]
  if (first.byteOffset % 4 === 0) {
    const flat = new Float32Array(first.buffer, first.byteOffset, rows.length * dim)
    for (let i = 0; i < rows.length; i += 1) {
      const candidate = flat.subarray(i * dim, (i + 1) * dim)
      pushHeap(heap, { rowIndex: rowStart + i, score: computeScore(query, candidate, metric) }, topK, metric)
    }
    return
  }
  const scratch = new Float32Array(dim)
  const scratchBytes = new Uint8Array(scratch.buffer)
  for (let i = 0; i < rows.length; i += 1) {
    scratchBytes.set(rows[i])
    pushHeap(heap, { rowIndex: rowStart + i, score: computeScore(query, scratch, metric) }, topK, metric)
  }
}

/**
 * Hamming-score every row in a binary chunk and update the candidate heap.
 *
 * @param {DecodedArray} columnData
 * @param {number} rowStart
 * @param {number} bytesPerRow
 * @param {Uint32Array} queryU32
 * @param {{ rowIndex: number, hamming: number }[]} heap
 * @param {number} candidatesK
 */
export function hammingScoreChunk(columnData, rowStart, bytesPerRow, queryU32, heap, candidatesK) {
  const rows = /** @type {Uint8Array[]} */ (columnData)
  if (rows.length === 0) return
  const wordsPerRow = bytesPerRow >> 2
  const first = rows[0]
  const aligned = first.byteOffset % 4 === 0
  const flat = aligned ? new Uint32Array(first.buffer, first.byteOffset, rows.length * wordsPerRow) : null
  const scratchU32 = aligned ? null : new Uint32Array(wordsPerRow)
  const scratchBytes = scratchU32 ? new Uint8Array(scratchU32.buffer) : null

  for (let i = 0; i < rows.length; i += 1) {
    /** @type {Uint32Array} */
    let candidate
    if (flat) {
      candidate = flat.subarray(i * wordsPerRow, (i + 1) * wordsPerRow)
    } else if (scratchBytes && scratchU32) {
      scratchBytes.set(rows[i])
      candidate = scratchU32
    } else {
      continue
    }
    let d = 0
    for (let j = 0; j < wordsPerRow; j += 1) {
      let v = candidate[j] ^ queryU32[j]
      v = v - (v >>> 1 & 0x55555555)
      v = (v & 0x33333333) + (v >>> 2 & 0x33333333)
      d += (v + (v >>> 4) & 0x0f0f0f0f) * 0x01010101 >>> 24
    }
    pushHammingHeap(heap, { rowIndex: rowStart + i, hamming: d }, candidatesK)
  }
}

/**
 * Score a contiguous row range of an in-memory binary buffer. Used after
 * prefetchBinary: phase 1 reads from RAM instead of fetching parquet pages.
 * The buffer is laid out row-major: row i occupies [i * bytesPerRow, (i+1) * bytesPerRow).
 * Allocate the buffer 4-byte aligned (true for any Uint8Array backed by a
 * fresh ArrayBuffer or Uint32Array.buffer) so the U32 view path is hot.
 *
 * @param {Uint8Array} buffer
 * @param {number} rowStart inclusive
 * @param {number} rowEnd exclusive
 * @param {number} bytesPerRow
 * @param {Uint32Array} queryU32
 * @param {{ rowIndex: number, hamming: number }[]} heap
 * @param {number} candidatesK
 */
export function hammingScoreFlatRange(buffer, rowStart, rowEnd, bytesPerRow, queryU32, heap, candidatesK) {
  if (rowEnd <= rowStart) return
  const wordsPerRow = bytesPerRow >> 2
  const startByte = rowStart * bytesPerRow
  const byteLen = (rowEnd - rowStart) * bytesPerRow
  if ((buffer.byteOffset + startByte) % 4 === 0) {
    const flat = new Uint32Array(buffer.buffer, buffer.byteOffset + startByte, byteLen >> 2)
    for (let i = 0; i < rowEnd - rowStart; i += 1) {
      let d = 0
      const base = i * wordsPerRow
      for (let j = 0; j < wordsPerRow; j += 1) {
        let v = flat[base + j] ^ queryU32[j]
        v = v - (v >>> 1 & 0x55555555)
        v = (v & 0x33333333) + (v >>> 2 & 0x33333333)
        d += (v + (v >>> 4) & 0x0f0f0f0f) * 0x01010101 >>> 24
      }
      pushHammingHeap(heap, { rowIndex: rowStart + i, hamming: d }, candidatesK)
    }
    return
  }
  // Misaligned fallback (rare; bytesPerRow is typically dim/8 with dim ≥ 32).
  const scratchU32 = new Uint32Array(wordsPerRow)
  const scratchBytes = new Uint8Array(scratchU32.buffer)
  for (let i = 0; i < rowEnd - rowStart; i += 1) {
    scratchBytes.set(buffer.subarray(startByte + i * bytesPerRow, startByte + (i + 1) * bytesPerRow))
    let d = 0
    for (let j = 0; j < wordsPerRow; j += 1) {
      let v = scratchU32[j] ^ queryU32[j]
      v = v - (v >>> 1 & 0x55555555)
      v = (v & 0x33333333) + (v >>> 2 & 0x33333333)
      d += (v + (v >>> 4) & 0x0f0f0f0f) * 0x01010101 >>> 24
    }
    pushHammingHeap(heap, { rowIndex: rowStart + i, hamming: d }, candidatesK)
  }
}

/**
 * Return a Uint32Array view of a Uint8Array. Copies if the source byteOffset
 * isn't 4-byte aligned (Uint32Array requires alignment).
 *
 * @param {Uint8Array} bytes
 * @returns {Uint32Array}
 */
export function bytesToAlignedU32(bytes) {
  if (bytes.byteOffset % 4 === 0 && bytes.byteLength % 4 === 0) {
    return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2)
  }
  const copy = new Uint8Array(bytes.byteLength)
  copy.set(bytes)
  return new Uint32Array(copy.buffer, 0, bytes.byteLength >> 2)
}
