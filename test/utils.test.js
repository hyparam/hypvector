import { describe, expect, it } from 'vitest'
import {
  cosineSimilarity,
  dotProduct,
  euclideanDistance,
  l2Normalize,
} from '../src/utils.js'

describe('dotProduct', () => {
  it('computes dot product', () => {
    expect(dotProduct([1, 2, 3], [4, 5, 6])).toBe(32)
  })
  it('throws on length mismatch', () => {
    expect(() => dotProduct([1, 2], [1, 2, 3])).toThrow(/length mismatch/)
  })
})

describe('cosineSimilarity', () => {
  it('returns 1 for identical vectors', () => {
    expect(cosineSimilarity([1, 0, 0], [1, 0, 0])).toBeCloseTo(1)
  })
  it('returns 0 for orthogonal vectors', () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0)
  })
  it('returns -1 for opposite vectors', () => {
    expect(cosineSimilarity([1, 2, 3], [-1, -2, -3])).toBeCloseTo(-1)
  })
  it('returns 0 when one vector is zero', () => {
    expect(cosineSimilarity([0, 0, 0], [1, 2, 3])).toBe(0)
  })
})

describe('euclideanDistance', () => {
  it('returns 0 for identical vectors', () => {
    expect(euclideanDistance([1, 2, 3], [1, 2, 3])).toBe(0)
  })
  it('computes L2 distance', () => {
    expect(euclideanDistance([0, 0], [3, 4])).toBe(5)
  })
})

describe('l2Normalize', () => {
  it('produces a unit vector', () => {
    const n = l2Normalize([3, 4])
    expect(n[0]).toBeCloseTo(0.6)
    expect(n[1]).toBeCloseTo(0.8)
    expect(Math.hypot(n[0], n[1])).toBeCloseTo(1)
  })
  it('handles zero vector', () => {
    const n = l2Normalize([0, 0, 0])
    expect(Array.from(n)).toEqual([0, 0, 0])
  })
})
