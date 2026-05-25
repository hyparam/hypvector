import { asyncBufferFromFile, asyncBufferFromUrl, cachedAsyncBuffer, parquetMetadataAsync, parquetRead } from 'hyparquet'
import { defaultBinaryColumn, defaultIdColumn, defaultVectorColumn } from './constants.js'
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
 * @import { AsyncBuffer, DecodedArray, FileMetaData } from 'hyparquet'
 */

/**
 * Find the top-k nearest neighbors to a query vector.
 *
 * Two paths:
 *   - Exact (no binary column, or rerankFactor=0): single pass over the
 *     float32 vector column, scoring every row.
 *   - Binary + rerank (binary column present and rerankFactor>0):
 *     phase 1 scans the 1-bit-per-dim column with Hamming distance to pick
 *     topK * rerankFactor candidates, phase 2 reads the float32 vectors for
 *     the candidate row range and reranks under the exact metric.
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
      file, metadata, meta, queryF32, scoringMetric, reportedMetric: requestedMetric, topK, rerankFactor,
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
 * Binary + rerank path.
 *
 * Phase 1 scans the binary column with Hamming distance to pick candidates.
 * Phase 2 reads the float32 vectors for just the candidate row range and
 * reranks under the exact metric.
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
 * @returns {Promise<SearchResult[]>}
 */
async function searchRerank({ file, metadata, meta, queryF32, scoringMetric, reportedMetric, topK, rerankFactor }) {
  const dim = meta.dimension
  const binaryBytes = (dim + 7) >> 3
  const candidatesK = Math.max(topK * rerankFactor, topK)

  // Phase 1: Hamming scan over binary column.
  const queryBin = packBinary(queryF32, dim)
  const queryBinU32 = bytesToAlignedU32(queryBin)
  /** @type {{ rowIndex: number, hamming: number }[]} */
  const candidateHeap = []

  await parquetRead({
    file,
    metadata,
    columns: [defaultBinaryColumn],
    onChunk: ({ columnName, columnData, rowStart }) => {
      if (columnName !== defaultBinaryColumn) return
      hammingScoreChunk(columnData, rowStart, binaryBytes, queryBinU32, candidateHeap, candidatesK)
    },
  })

  if (candidateHeap.length === 0) return []

  // Phase 2: per-candidate single-row reads with useOffsetIndex. Each
  // parquetRead fetches only the data page containing that row. With small
  // page sizes this is the network-optimal pattern; the cached buffer dedups
  // repeated reads of footer + offset indexes across the K parallel calls.
  const candidateRows = [...new Set(candidateHeap.map(c => c.rowIndex))].sort((a, b) => a - b)
  const decoder = new TextDecoder()

  const scored = await Promise.all(candidateRows.map(async rowIndex => {
    /** @type {Float32Array | null} */
    let vector = null
    /** @type {string | undefined} */
    let id
    await parquetRead({
      file,
      metadata,
      columns: [defaultIdColumn, defaultVectorColumn],
      rowStart: rowIndex,
      rowEnd: rowIndex + 1,
      useOffsetIndex: true,
      onChunk: ({ columnName, columnData, rowStart: chunkStart }) => {
        const idx = rowIndex - chunkStart
        if (idx < 0 || idx >= columnData.length) return
        if (columnName === defaultIdColumn) {
          const raw = /** @type {any} */ (columnData)[idx]
          id = typeof raw === 'string' ? raw : decoder.decode(raw)
        } else if (columnName === defaultVectorColumn) {
          const bytes = /** @type {Uint8Array[]} */ (columnData)[idx]
          if (bytes.byteOffset % 4 === 0) {
            vector = new Float32Array(bytes.buffer, bytes.byteOffset, dim)
          } else {
            vector = new Float32Array(dim)
            new Uint8Array(vector.buffer).set(bytes)
          }
        }
      },
    })
    if (!vector) return null
    return {
      rowIndex,
      id: id ?? String(rowIndex),
      score: computeScore(queryF32, vector, scoringMetric),
    }
  }))

  const valid = /** @type {{ rowIndex: number, score: number, id: string }[]} */ (
    scored.filter(r => r !== null)
  )
  const dir = reportedMetric === 'euclidean' ? 1 : -1
  valid.sort((a, b) => dir * (a.score - b.score))
  return valid.slice(0, topK).map(({ id, score, rowIndex }) => ({ id, score, rowIndex }))
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
