import { describe, expect, it } from 'vitest'
import { pushHammingHeap, pushHeap, sortHeap } from '../src/search/heap.js'

/**
 * Run a sequence of candidates through a bounded heap and return the kept
 * rowIndices, sorted ascending for comparison.
 * @param {{ rowIndex: number, hamming: number }[]} candidates
 * @param {number} k
 * @returns {number[]}
 */
function keepHamming(candidates, k) {
  /** @type {{ rowIndex: number, hamming: number }[]} */
  const heap = []
  for (const c of candidates) pushHammingHeap(heap, c, k)
  return heap.map(e => e.rowIndex).sort((a, b) => a - b)
}

describe('heap tie-breaking is deterministic', () => {
  it('pushHammingHeap keeps the lowest rowIndices when hamming ties, regardless of insertion order', () => {
    // Five rows all at the same hamming distance; keep 3 -> must keep rows 0,1,2.
    const rows = [0, 1, 2, 3, 4].map(rowIndex => ({ rowIndex, hamming: 7 }))
    const forward = keepHamming(rows, 3)
    const reversed = keepHamming([...rows].reverse(), 3)
    const shuffled = keepHamming([rows[3], rows[0], rows[4], rows[2], rows[1]], 3)
    expect(forward).toEqual([0, 1, 2])
    expect(reversed).toEqual([0, 1, 2])
    expect(shuffled).toEqual([0, 1, 2])
  })

  it('pushHammingHeap prefers strictly nearer candidates over ties', () => {
    const rows = [
      { rowIndex: 5, hamming: 2 },
      { rowIndex: 9, hamming: 9 },
      { rowIndex: 1, hamming: 9 },
      { rowIndex: 8, hamming: 1 },
    ]
    // keep 2 -> the two nearest by hamming (1 then 2): rows 8 and 5.
    expect(keepHamming(rows, 2)).toEqual([5, 8])
    expect(keepHamming([...rows].reverse(), 2)).toEqual([5, 8])
  })

  it('pushHeap keeps best score, breaking ties by lower rowIndex (order-independent)', () => {
    // cosine: higher score is better. Tied at 0.9 -> keep lower rowIndices.
    const rows = [0, 1, 2, 3].map(rowIndex => ({ rowIndex, score: 0.9 }))
    /** @type {{ rowIndex: number, score: number }[]} */
    const a = []
    for (const c of rows) pushHeap(a, c, 2, 'cosine')
    /** @type {{ rowIndex: number, score: number }[]} */
    const b = []
    for (const c of [...rows].reverse()) pushHeap(b, c, 2, 'cosine')
    expect(a.map(e => e.rowIndex).sort((x, y) => x - y)).toEqual([0, 1])
    expect(b.map(e => e.rowIndex).sort((x, y) => x - y)).toEqual([0, 1])
  })

  it('sortHeap orders tied scores by ascending rowIndex', () => {
    const results = [
      { rowIndex: 4, score: 0.5 },
      { rowIndex: 1, score: 0.9 },
      { rowIndex: 7, score: 0.9 },
      { rowIndex: 2, score: 0.9 },
    ]
    expect(sortHeap(results, 'cosine').map(e => e.rowIndex)).toEqual([1, 2, 7, 4])
    // euclidean: lower score is better, ties still ascend by rowIndex.
    expect(sortHeap(results, 'euclidean').map(e => e.rowIndex)).toEqual([4, 1, 2, 7])
  })
})
