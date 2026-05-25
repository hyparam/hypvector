import { asyncBufferFromFile, asyncBufferFromUrl, cachedAsyncBuffer, parquetMetadataAsync, parquetRead } from 'hyparquet'
import { hammingDistanceBytes } from './cluster.js'
import { defaultBinaryColumn, defaultClusterProbeFraction, defaultIdColumn, defaultVectorColumn } from './constants.js'
import {
  cosineSimilarity,
  dotProduct,
  euclideanDistance,
  l2Normalize,
  packBinary,
  parseKvMetadata,
} from './utils.js'

/**
 * @import { HypVectorMetadata, SearchResult, SearchVectorsOptions, DistanceMetric } from './types.js'
 * @import { AsyncBuffer, DecodedArray, FileMetaData, RowGroup } from 'hyparquet'
 */

/**
 * Find the top-k nearest neighbors to a query vector.
 *
 * Three paths, in order of preference:
 *   - Clustered + binary + rerank (file has centroids): phase 1 scans only
 *     the top-N nearest clusters' row ranges (skipping whole row groups),
 *     phase 2 fetches the candidate float32 vectors and reranks.
 *   - Binary + rerank (binary column present, no clustering): full Hamming
 *     scan in phase 1, then per-candidate float32 fetch + rerank in phase 2.
 *   - Exact (no binary column, or rerankFactor=0): single pass over the
 *     float32 column, scoring every row.
 *
 * @param {SearchVectorsOptions} options
 * @returns {Promise<SearchResult[]>}
 */
export async function searchVectors({
  query,
  url,
  topK = 10,
  metric,
  rerankFactor = 10,
  probe,
  signal,
  asyncBufferFactory,
  sourceFile,
  sourceMetadata,
}) {
  const factory = asyncBufferFactory ?? defaultAsyncBufferFactory
  const file = sourceFile ?? await factory({ url, signal })
  const metadata = sourceMetadata ?? await parquetMetadataAsync(file)
  const meta = parseKvMetadata(metadata)

  if (query.length !== meta.dimension) {
    throw new Error(`query has dimension ${query.length}, file expects ${meta.dimension}`)
  }
  const requestedMetric = metric ?? meta.metric
  let queryF32 = query instanceof Float32Array ? query : Float32Array.from(query)

  // When stored vectors are pre-normalized, cosine == dot(query/||query||, candidate).
  let scoringMetric = requestedMetric
  if (requestedMetric === 'cosine' && meta.normalized) {
    queryF32 = l2Normalize(queryF32)
    scoringMetric = 'dot'
  }

  if (meta.hasBinary && rerankFactor > 0) {
    return searchRerank({
      file, metadata, meta, queryF32, scoringMetric, reportedMetric: requestedMetric, topK, rerankFactor, probe,
    })
  }
  return searchExact({
    file, metadata, meta, queryF32, scoringMetric, reportedMetric: requestedMetric, topK,
  })
}

/**
 * Exact full-scan path (no binary column).
 *
 * @param {object} options
 * @param {AsyncBuffer} options.file
 * @param {FileMetaData} options.metadata
 * @param {HypVectorMetadata} options.meta
 * @param {Float32Array} options.queryF32
 * @param {DistanceMetric} options.scoringMetric
 * @param {DistanceMetric} options.reportedMetric
 * @param {number} options.topK
 * @returns {Promise<SearchResult[]>}
 */
async function searchExact({ file, metadata, meta, queryF32, scoringMetric, reportedMetric, topK }) {
  /** @type {{ rowIndex: number, score: number }[]} */
  const heap = []
  /** @type {{ start: number, ids: string[] }[]} */
  const idChunks = []

  await parquetRead({
    file,
    metadata,
    columns: [defaultIdColumn, defaultVectorColumn],
    onChunk: ({ columnName, columnData, rowStart }) => {
      if (columnName === defaultVectorColumn) {
        scoreVectorChunk(columnData, rowStart, meta.dimension, queryF32, scoringMetric, heap, topK)
      } else if (columnName === defaultIdColumn) {
        idChunks.push({ start: rowStart, ids: /** @type {string[]} */ (columnData) })
      }
    },
  })

  return sortHeap(heap, reportedMetric).map(({ rowIndex, score }) => ({
    id: lookupId(idChunks, rowIndex) ?? String(rowIndex),
    score,
    rowIndex,
  }))
}

/**
 * Binary + rerank path. When the file has cluster centroids, phase 1
 * restricts the scan to row ranges of the top-N nearest clusters and
 * phase 2 issues coalesced reads spanning each contiguous candidate run.
 * Without clustering, falls back to a full binary scan and per-candidate
 * point reads in phase 2.
 *
 * @param {object} options
 * @param {AsyncBuffer} options.file
 * @param {FileMetaData} options.metadata
 * @param {HypVectorMetadata} options.meta
 * @param {Float32Array} options.queryF32
 * @param {DistanceMetric} options.scoringMetric
 * @param {DistanceMetric} options.reportedMetric
 * @param {number} options.topK
 * @param {number} options.rerankFactor
 * @param {number | undefined} options.probe
 * @returns {Promise<SearchResult[]>}
 */
async function searchRerank({ file, metadata, meta, queryF32, scoringMetric, reportedMetric, topK, rerankFactor, probe }) {
  const dim = meta.dimension
  const binaryBytes = dim + 7 >> 3
  const candidatesK = Math.max(topK * rerankFactor, topK)

  const queryBin = packBinary(queryF32, dim)
  const queryBinU32 = bytesToAlignedU32(queryBin)

  /** @type {{ rowIndex: number, hamming: number }[]} */
  const candidateHeap = []

  // Decide which row ranges to scan in phase 1. With cluster metadata we
  // know each cluster's exact contiguous row range; otherwise full scan.
  const scanRanges = meta.centroids && meta.clusterCounts
    ? selectClusterRowRanges(meta, queryBin, probe)
    : [{ rowStart: 0, rowEnd: Number(metadata.num_rows) }]

  // Phase 1: Hamming scan over selected ranges of the binary column.
  await Promise.all(scanRanges.map(({ rowStart, rowEnd }) => parquetRead({
    file,
    metadata,
    columns: [defaultBinaryColumn],
    rowStart,
    rowEnd,
    useOffsetIndex: true,
    onChunk: ({ columnName, columnData, rowStart: chunkStart }) => {
      if (columnName !== defaultBinaryColumn) return
      hammingScoreChunk(columnData, chunkStart, binaryBytes, queryBinU32, candidateHeap, candidatesK)
    },
  })))

  if (candidateHeap.length === 0) return []

  const candidateRows = [...new Set(candidateHeap.map(c => c.rowIndex))].sort((a, b) => a - b)
  const wantedRows = new Set(candidateRows)
  const runs = coalesceRuns(candidateRows, 64)

  /** @type {{ rowIndex: number, score: number }[]} */
  const scored = []

  await Promise.all(runs.map(async ({ rowStart, rowEnd }) => {
    /** @type {Map<number, Float32Array>} */
    const local = new Map()
    await parquetRead({
      file,
      metadata,
      columns: [defaultVectorColumn],
      rowStart,
      rowEnd,
      useOffsetIndex: true,
      onChunk: ({ columnName, columnData, rowStart: chunkStart }) => {
        if (columnName !== defaultVectorColumn) return
        for (let i = 0; i < columnData.length; i += 1) {
          const rowIndex = chunkStart + i
          if (!wantedRows.has(rowIndex)) continue
          const bytes = /** @type {Uint8Array[]} */ (columnData)[i]
          /** @type {Float32Array} */
          let vector
          if (bytes.byteOffset % 4 === 0) {
            vector = new Float32Array(bytes.buffer, bytes.byteOffset, dim)
          } else {
            vector = new Float32Array(dim)
            new Uint8Array(vector.buffer).set(bytes)
          }
          local.set(rowIndex, vector)
        }
      },
    })
    for (const [rowIndex, vector] of local) {
      scored.push({ rowIndex, score: computeScore(queryF32, vector, scoringMetric) })
    }
  }))

  const dir = reportedMetric === 'euclidean' ? 1 : -1
  scored.sort((a, b) => dir * (a.score - b.score))
  const winners = scored.slice(0, topK)

  // Phase 3: fetch ids for just the top-K winners.
  const ids = await fetchIds(file, metadata, winners.map(w => w.rowIndex))
  return winners.map((w, i) => ({ id: ids[i], score: w.score, rowIndex: w.rowIndex }))
}

/**
 * Read the id column for a set of row indices. Coalesces into runs so a
 * few small parquetRead calls cover all winners. Returns ids in the same
 * order as the input rowIndices.
 *
 * @param {AsyncBuffer} file
 * @param {FileMetaData} metadata
 * @param {number[]} rowIndices
 * @returns {Promise<string[]>}
 */
async function fetchIds(file, metadata, rowIndices) {
  if (rowIndices.length === 0) return []
  const sorted = [...new Set(rowIndices)].sort((a, b) => a - b)
  const wanted = new Set(sorted)
  const runs = coalesceRuns(sorted, 64)
  const decoder = new TextDecoder()
  /** @type {Map<number, string>} */
  const byRow = new Map()

  await Promise.all(runs.map(({ rowStart, rowEnd }) => parquetRead({
    file,
    metadata,
    columns: [defaultIdColumn],
    rowStart,
    rowEnd,
    useOffsetIndex: true,
    onChunk: ({ columnName, columnData, rowStart: chunkStart }) => {
      if (columnName !== defaultIdColumn) return
      for (let i = 0; i < columnData.length; i += 1) {
        const rowIndex = chunkStart + i
        if (!wanted.has(rowIndex)) continue
        const raw = /** @type {any} */ (columnData)[i]
        byRow.set(rowIndex, typeof raw === 'string' ? raw : decoder.decode(raw))
      }
    },
  })))

  return rowIndices.map(r => byRow.get(r) ?? String(r))
}

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
function selectClusterRowRanges(meta, queryBin, probe) {
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
function mergeRanges(ranges) {
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
function coalesceRuns(rows, maxGap) {
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
function scoreVectorChunk(columnData, rowStart, dim, query, metric, heap, topK) {
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
function hammingScoreChunk(columnData, rowStart, bytesPerRow, queryU32, heap, candidatesK) {
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
 * Bounded heap for Hamming candidates (lower hamming is better).
 *
 * @param {{ rowIndex: number, hamming: number }[]} heap
 * @param {{ rowIndex: number, hamming: number }} candidate
 * @param {number} candidatesK
 */
function pushHammingHeap(heap, candidate, candidatesK) {
  if (heap.length < candidatesK) {
    heap.push(candidate)
    return
  }
  let worstIdx = 0
  for (let i = 1; i < heap.length; i += 1) {
    if (heap[i].hamming > heap[worstIdx].hamming) worstIdx = i
  }
  if (candidate.hamming < heap[worstIdx].hamming) {
    heap[worstIdx] = candidate
  }
}

/**
 * Return a Uint32Array view of a Uint8Array. Copies if the source byteOffset
 * isn't 4-byte aligned (Uint32Array requires alignment).
 *
 * @param {Uint8Array} bytes
 * @returns {Uint32Array}
 */
function bytesToAlignedU32(bytes) {
  if (bytes.byteOffset % 4 === 0 && bytes.byteLength % 4 === 0) {
    return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2)
  }
  const copy = new Uint8Array(bytes.byteLength)
  copy.set(bytes)
  return new Uint32Array(copy.buffer, 0, bytes.byteLength >> 2)
}

/**
 * Find the id for a given row index within the collected id chunks.
 *
 * @param {{ start: number, ids: string[] }[]} chunks
 * @param {number} rowIndex
 * @returns {string | undefined}
 */
function lookupId(chunks, rowIndex) {
  for (const { start, ids } of chunks) {
    if (rowIndex >= start && rowIndex < start + ids.length) {
      return ids[rowIndex - start]
    }
  }
  return undefined
}

/**
 * Default AsyncBuffer factory: uses node fs for local paths and HTTP fetch otherwise.
 *
 * @param {{ url: string, signal?: AbortSignal }} options
 * @returns {Promise<AsyncBuffer>}
 */
async function defaultAsyncBufferFactory({ url, signal }) {
  /** @type {AsyncBuffer} */
  let raw
  if (url.startsWith('http://') || url.startsWith('https://')) {
    /** @type {RequestInit} */
    const requestInit = signal ? { signal } : {}
    raw = await asyncBufferFromUrl({ url, requestInit })
  } else {
    raw = await asyncBufferFromFile(url)
  }
  // Cache slices so repeated reads of the same byte range (footer, offset
  // indexes, overlapping pages) don't re-fetch.
  return cachedAsyncBuffer(raw)
}

/**
 * Compute the score for a candidate vector under the chosen metric.
 *
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {DistanceMetric} metric
 * @returns {number}
 */
function computeScore(a, b, metric) {
  if (metric === 'cosine') return cosineSimilarity(a, b)
  if (metric === 'dot') return dotProduct(a, b)
  if (metric === 'euclidean') return euclideanDistance(a, b)
  throw new Error(`unsupported metric: ${metric}`)
}

/**
 * Return true if `a` is "better" than `b` under the metric.
 *
 * @param {number} a
 * @param {number} b
 * @param {DistanceMetric} metric
 * @returns {boolean}
 */
function isBetter(a, b, metric) {
  if (metric === 'euclidean') return a < b
  return a > b
}

/**
 * Bounded heap by score (uses metric to decide "better").
 *
 * @param {{ rowIndex: number, score: number }[]} heap
 * @param {{ rowIndex: number, score: number }} candidate
 * @param {number} topK
 * @param {DistanceMetric} metric
 */
function pushHeap(heap, candidate, topK, metric) {
  if (heap.length < topK) {
    heap.push(candidate)
    return
  }
  let worstIdx = 0
  for (let i = 1; i < heap.length; i += 1) {
    if (isBetter(heap[worstIdx].score, heap[i].score, metric)) {
      worstIdx = i
    }
  }
  if (isBetter(candidate.score, heap[worstIdx].score, metric)) {
    heap[worstIdx] = candidate
  }
}

/**
 * Sort results best-first under the chosen metric.
 *
 * @param {{ rowIndex: number, score: number }[]} results
 * @param {DistanceMetric} metric
 * @returns {{ rowIndex: number, score: number }[]}
 */
function sortHeap(results, metric) {
  const dir = metric === 'euclidean' ? 1 : -1
  return results.slice().sort((a, b) => dir * (a.score - b.score))
}
