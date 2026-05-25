import { hammingDistanceBytes } from '../cluster.js'
import { defaultClusterProbeFraction } from '../constants.js'

/**
 * @import { HypVectorMetadata } from '../types.js'
 */

/**
 * Pick exact contiguous row ranges based on cluster nearness to the query.
 * Uses `clusterCounts` KV metadata: since rows are sorted by cluster id,
 * cluster k occupies [cumsum[k], cumsum[k+1]). We pick the top-N nearest
 * clusters (by Hamming centroid distance), then merge their contiguous
 * row ranges so useOffsetIndex fetches only the pages that cover them.
 *
 * @param {HypVectorMetadata} meta
 * @param {Uint8Array} queryBin
 * @param {number | undefined} probe
 * @returns {{ rowStart: number, rowEnd: number }[]}
 */
export function selectClusterRowRanges(meta, queryBin, probe) {
  const centroids = meta.centroids ?? []
  const counts = meta.clusterCounts
  if (centroids.length === 0 || !counts) return [{ rowStart: 0, rowEnd: meta.count }]

  // Cumulative offsets so cluster k spans [offset[k], offset[k+1]).
  const offsets = new Uint32Array(centroids.length + 1)
  for (let c = 0; c < centroids.length; c += 1) offsets[c + 1] = offsets[c] + counts[c]

  // Rank clusters by Hamming to query.
  const clusterDist = new Array(centroids.length)
  for (let c = 0; c < centroids.length; c += 1) {
    clusterDist[c] = { cluster: c, hamming: hammingDistanceBytes(queryBin, centroids[c]) }
  }
  clusterDist.sort((a, b) => a.hamming - b.hamming)

  const probeFraction = probe === undefined ? defaultClusterProbeFraction : probe
  // probe in (0, 1] is a fraction of clusters (1.0 = all clusters);
  // probe > 1 is an absolute count.
  const targetClusters = probeFraction > 1
    ? Math.min(Math.ceil(probeFraction), centroids.length)
    : Math.max(1, Math.ceil(centroids.length * probeFraction))

  const wanted = clusterDist.slice(0, targetClusters).map(c => c.cluster).sort((a, b) => a - b)
  /** @type {{ rowStart: number, rowEnd: number }[]} */
  const ranges = []
  for (const c of wanted) {
    ranges.push({ rowStart: offsets[c], rowEnd: offsets[c + 1] })
  }
  return mergeRanges(ranges)
}

/**
 * Merge adjacent/overlapping ranges.
 *
 * @param {{ rowStart: number, rowEnd: number }[]} ranges (already in order)
 * @returns {{ rowStart: number, rowEnd: number }[]}
 */
export function mergeRanges(ranges) {
  /** @type {{ rowStart: number, rowEnd: number }[]} */
  const out = []
  for (const r of ranges) {
    const last = out[out.length - 1]
    if (last && r.rowStart <= last.rowEnd) {
      if (r.rowEnd > last.rowEnd) last.rowEnd = r.rowEnd
    } else {
      out.push({ ...r })
    }
  }
  return out
}

/**
 * Group a sorted list of row indices into contiguous runs, merging runs
 * whose gap is <= maxGap. Each run becomes one parquetRead call.
 *
 * @param {number[]} rows (sorted ascending)
 * @param {number} maxGap
 * @returns {{ rowStart: number, rowEnd: number }[]}
 */
export function coalesceRuns(rows, maxGap) {
  if (rows.length === 0) return []
  /** @type {{ rowStart: number, rowEnd: number }[]} */
  const runs = []
  let start = rows[0]
  let end = rows[0] + 1
  for (let i = 1; i < rows.length; i += 1) {
    if (rows[i] - end <= maxGap) {
      end = rows[i] + 1
    } else {
      runs.push({ rowStart: start, rowEnd: end })
      start = rows[i]
      end = rows[i] + 1
    }
  }
  runs.push({ rowStart: start, rowEnd: end })
  return runs
}
