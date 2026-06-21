import { describe, expect, it } from 'vitest'
import { defaultClusterProbeCap } from '../src/constants.js'
import { selectClusterRowRanges } from '../src/search/ranges.js'

/**
 * Build synthetic cluster metadata with `nClusters` clusters of one row each,
 * so the total rows covered by the returned ranges equals the number of
 * clusters probed (independent of how ranges merge).
 * @param {number} nClusters
 * @returns {import('../src/types.js').HypVectorMetadata}
 */
function makeMeta(nClusters) {
  /** @type {Uint8Array[]} */
  const centroids = []
  for (let c = 0; c < nClusters; c += 1) {
    // Distinct 16-bit patterns so cluster ranking by Hamming is well-defined.
    centroids.push(Uint8Array.from([c & 0xff, c >> 8 & 0xff]))
  }
  const clusterCounts = new Uint32Array(nClusters).fill(1)
  return {
    version: 0, dimension: 16, metric: 'cosine', normalized: true,
    hasBinary: true, count: nClusters, clusters: nClusters, centroids, clusterCounts,
  }
}

/**
 * Sum of rows covered by a list of row ranges.
 * @param {{ rowStart: number, rowEnd: number }[]} ranges
 * @returns {number}
 */
function rowsCovered(ranges) {
  let sum = 0
  for (const r of ranges) sum += r.rowEnd - r.rowStart
  return sum
}

describe('selectClusterRowRanges probe cap', () => {
  const query = Uint8Array.from([0, 0])

  it('caps the default fraction at the absolute ceiling for large nlist', () => {
    // 0.25 * 500 = 125 clusters, but the default cap (96) should bind.
    const ranges = selectClusterRowRanges(makeMeta(500), query, undefined)
    expect(rowsCovered(ranges)).toBe(defaultClusterProbeCap)
  })

  it('does not cap when the default fraction is below the ceiling', () => {
    // 0.25 * 100 = 25 clusters, well under the cap — unchanged behavior.
    const ranges = selectClusterRowRanges(makeMeta(100), query, undefined)
    expect(rowsCovered(ranges)).toBe(25)
  })

  it('honors an explicit fraction literally (no cap)', () => {
    // Explicit 0.25 on 500 clusters → 125, NOT capped.
    const ranges = selectClusterRowRanges(makeMeta(500), query, 0.25)
    expect(rowsCovered(ranges)).toBe(125)
  })

  it('honors an explicit absolute count above the cap', () => {
    const ranges = selectClusterRowRanges(makeMeta(500), query, 200)
    expect(rowsCovered(ranges)).toBe(200)
  })

  it('never probes more clusters than exist', () => {
    const ranges = selectClusterRowRanges(makeMeta(40), query, undefined)
    expect(rowsCovered(ranges)).toBe(10) // ceil(0.25 * 40)
  })
})
