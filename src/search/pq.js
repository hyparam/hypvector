import { parquetRead } from 'hyparquet'
import { defaultIdColumn, defaultPqColumn, defaultVectorColumn } from '../constants.js'
import { buildPqTables } from '../pq.js'
import { packBinary } from '../utils.js'
import { computeScore, pushHeap } from './heap.js'
import { coalesceRuns, selectClusterRowRanges } from './ranges.js'

/**
 * @import { DistanceMetric, HypVectorMetadata, SearchResult } from '../types.js'
 * @import { AsyncBuffer, Compressors, FileMetaData } from 'hyparquet'
 */

/**
 * Product-quantized rerank path. Phase 1 scans compact PQ codes over the
 * selected cluster ranges, phase 2 fetches float32 vectors for the best
 * approximate candidates, and phase 3 fetches ids for the final winners.
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
 * @param {Compressors} [options.compressors]
 * @returns {Promise<SearchResult[]>}
 */
export async function searchPq({ file, metadata, meta, queryF32, scoringMetric, reportedMetric, topK, rerankFactor, probe, compressors }) {
  const candidatesK = Math.max(topK * rerankFactor, topK)
  const { table, approxMetric } = buildPqTables(queryF32, meta, scoringMetric)

  /** @type {{ rowIndex: number, score: number }[]} */
  const candidateHeap = []

  const scanRanges = meta.centroids && meta.clusterCounts
    ? selectClusterRowRanges(meta, packBinary(queryF32, meta.dimension), probe)
    : [{ rowStart: 0, rowEnd: Number(metadata.num_rows) }]

  await Promise.all(scanRanges.map(({ rowStart, rowEnd }) => parquetRead({
    file,
    metadata,
    compressors,
    columns: [defaultPqColumn],
    rowStart,
    rowEnd,
    onChunk: ({ columnName, columnData, rowStart: chunkStart }) => {
      if (columnName !== defaultPqColumn) return
      scorePqChunk(columnData, chunkStart, meta, table, approxMetric, candidateHeap, candidatesK)
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
      compressors,
      columns: [defaultVectorColumn],
      rowStart,
      rowEnd,
      useOffsetIndex: true,
      onChunk: ({ columnName, columnData, rowStart: chunkStart }) => {
        if (columnName !== defaultVectorColumn) return
        const rows = /** @type {Uint8Array[]} */ (columnData)
        for (let i = 0; i < rows.length; i += 1) {
          const rowIndex = chunkStart + i
          if (!wantedRows.has(rowIndex)) continue
          const bytes = rows[i]
          /** @type {Float32Array} */
          let vector
          if (bytes.byteOffset % 4 === 0) {
            vector = new Float32Array(bytes.buffer, bytes.byteOffset, meta.dimension)
          } else {
            vector = new Float32Array(meta.dimension)
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

  const ids = await fetchIds(file, metadata, winners.map(w => w.rowIndex), compressors)
  return winners.map((w, i) => ({ id: ids[i], score: w.score, rowIndex: w.rowIndex }))
}

/**
 * @param {import('hyparquet').DecodedArray} columnData
 * @param {number} rowStart
 * @param {HypVectorMetadata} meta
 * @param {Float32Array} table
 * @param {DistanceMetric} approxMetric
 * @param {{ rowIndex: number, score: number }[]} heap
 * @param {number} candidatesK
 */
function scorePqChunk(columnData, rowStart, meta, table, approxMetric, heap, candidatesK) {
  const rows = /** @type {Uint8Array[]} */ (columnData)
  const segments = meta.pqSegments ?? 0
  const centroids = meta.pqCentroids ?? 0
  for (let i = 0; i < rows.length; i += 1) {
    const code = rows[i]
    let score = 0
    for (let s = 0; s < segments; s += 1) {
      score += table[s * centroids + code[s]]
    }
    pushHeap(heap, { rowIndex: rowStart + i, score }, candidatesK, approxMetric)
  }
}

/**
 * Read the id column for a set of row indices. Returns ids in the same
 * order as the input row indices.
 *
 * @param {AsyncBuffer} file
 * @param {FileMetaData} metadata
 * @param {number[]} rowIndices
 * @param {Compressors} [compressors]
 * @returns {Promise<string[]>}
 */
async function fetchIds(file, metadata, rowIndices, compressors) {
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
    compressors,
    columns: [defaultIdColumn],
    rowStart,
    rowEnd,
    useOffsetIndex: true,
    onChunk: ({ columnName, columnData, rowStart: chunkStart }) => {
      if (columnName !== defaultIdColumn) return
      const rows = /** @type {any[]} */ (columnData)
      for (let i = 0; i < rows.length; i += 1) {
        const rowIndex = chunkStart + i
        if (!wanted.has(rowIndex)) continue
        const raw = rows[i]
        byRow.set(rowIndex, typeof raw === 'string' ? raw : decoder.decode(raw))
      }
    },
  })))

  return rowIndices.map(r => byRow.get(r) ?? String(r))
}
