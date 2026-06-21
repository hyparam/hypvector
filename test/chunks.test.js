import { describe, expect, it } from 'vitest'
import { hammingScoreChunk } from '../src/search/chunks.js'

/**
 * Reference Hamming distance between two equal-length byte rows.
 * @param {Uint8Array} a
 * @param {Uint8Array} b
 * @returns {number}
 */
function hamming(a, b) {
  let d = 0
  for (let i = 0; i < a.length; i += 1) {
    let v = a[i] ^ b[i]
    while (v) { d += v & 1; v >>>= 1 }
  }
  return d
}

describe('hammingScoreChunk', () => {
  const bytesPerRow = 8 // 64-bit rows, 2 words
  const nRows = 5
  // Deterministic row patterns and a query.
  const patterns = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [255, 0, 0, 0, 0, 0, 0, 0],
    [255, 255, 255, 255, 255, 255, 255, 255],
    [1, 2, 4, 8, 16, 32, 64, 128],
    [170, 85, 170, 85, 170, 85, 170, 85],
  ].map(p => Uint8Array.from(p))
  const queryBytes = Uint8Array.from([15, 240, 0, 255, 1, 1, 1, 1])
  const queryU32 = new Uint32Array(queryBytes.buffer.slice(0))
  const expected = patterns.map(p => hamming(p, queryBytes))

  /**
   * Collect the {rowIndex, hamming} pairs the function pushes, sorted by row.
   * @param {Uint8Array[]} rows
   * @returns {{ rowIndex: number, hamming: number }[]}
   */
  function scoreAll(rows) {
    /** @type {{ rowIndex: number, hamming: number }[]} */
    const heap = []
    hammingScoreChunk(rows, 0, bytesPerRow, queryU32, heap, rows.length)
    return heap.slice().sort((a, b) => a.rowIndex - b.rowIndex)
  }

  it('scores a contiguous single-buffer chunk correctly (fast path)', () => {
    // Rows are tightly-packed, 4-byte-aligned slices of one buffer.
    const backing = new Uint8Array(nRows * bytesPerRow)
    for (let i = 0; i < nRows; i += 1) backing.set(patterns[i], i * bytesPerRow)
    const rows = Array.from({ length: nRows }, (_, i) => backing.subarray(i * bytesPerRow, (i + 1) * bytesPerRow))

    const result = scoreAll(rows)
    expect(result.map(r => r.hamming)).toEqual(expected)
  })

  it('scores a non-contiguous multi-buffer chunk correctly (regression: no crash, no corruption)', () => {
    // Each row is its own buffer — the layout a clustered row-range read can
    // produce when assembled from several parquet pages. The old flat-view
    // fast path either threw RangeError or scored the wrong bytes here.
    const rows = patterns.map(p => Uint8Array.from(p))
    expect(rows[1].buffer).not.toBe(rows[0].buffer)

    const result = scoreAll(rows)
    expect(result.map(r => r.hamming)).toEqual(expected)
  })

  it('agrees between contiguous and non-contiguous layouts', () => {
    const backing = new Uint8Array(nRows * bytesPerRow)
    for (let i = 0; i < nRows; i += 1) backing.set(patterns[i], i * bytesPerRow)
    const contiguous = Array.from({ length: nRows }, (_, i) => backing.subarray(i * bytesPerRow, (i + 1) * bytesPerRow))
    const separate = patterns.map(p => Uint8Array.from(p))

    expect(scoreAll(contiguous)).toEqual(scoreAll(separate))
  })
})
