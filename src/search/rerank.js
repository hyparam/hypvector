import { parquetRead } from 'hyparquet'
import { defaultBinaryColumn, defaultIdColumn, defaultVectorColumn } from '../constants.js'
import { packBinary } from '../utils.js'
import { bytesToAlignedU32, hammingScoreChunk, hammingScoreFlatRange } from './chunks.js'
import { computeScore } from './heap.js'
import { coalesceRuns, selectClusterRowRanges } from './ranges.js'

/**
 * @import { DistanceMetric, HypVectorMetadata, SearchResult } from '../types.js'
 * @import { AsyncBuffer, Compressors, FileMetaData } from 'hyparquet'
 */

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
 * @param {Uint8Array} [options.binary] in-memory binary column from prefetchBinary; when present, phase 1 runs from RAM
 * @param {Compressors} [options.compressors]
 * @returns {Promise<SearchResult[]>}
 */
export async function searchRerank({ file, metadata, meta, queryF32, scoringMetric, reportedMetric, topK, rerankFactor, probe, binary, compressors }) {
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
  // With a prefetched in-memory buffer, score the row ranges directly from
  // RAM. Otherwise, parquetRead each range — the binary column is small
  // (dim/8 bytes/row), so per-page seeking via useOffsetIndex costs an
  // extra RT without saving meaningful bytes; read whole column chunks
  // instead. (Phase 2's float32 column is ~32x larger per row, so it
  // keeps useOffsetIndex below.)
  if (binary) {
    for (const { rowStart, rowEnd } of scanRanges) {
      hammingScoreFlatRange(binary, rowStart, rowEnd, binaryBytes, queryBinU32, candidateHeap, candidatesK)
    }
  } else {
    await Promise.all(scanRanges.map(({ rowStart, rowEnd }) => parquetRead({
      file,
      metadata,
      compressors,
      columns: [defaultBinaryColumn],
      rowStart,
      rowEnd,
      onChunk: ({ columnName, columnData, rowStart: chunkStart }) => {
        if (columnName !== defaultBinaryColumn) return
        hammingScoreChunk(columnData, chunkStart, binaryBytes, queryBinU32, candidateHeap, candidatesK)
      },
    })))
  }

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
        for (let i = 0; i < columnData.length; i += 1) {
          const rowIndex = chunkStart + i
          if (!wantedRows.has(rowIndex)) continue
          const bytes = columnData[i]
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
  // Tie-break by rowIndex so the winners are independent of the order the
  // parallel candidate reads completed in (deterministic results).
  scored.sort((a, b) => dir * (a.score - b.score) || a.rowIndex - b.rowIndex)
  const winners = scored.slice(0, topK)

  // Phase 3: fetch ids for just the top-K winners.
  const ids = await fetchIds(file, metadata, winners.map(w => w.rowIndex), compressors)
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
      for (let i = 0; i < columnData.length; i += 1) {
        const rowIndex = chunkStart + i
        if (!wanted.has(rowIndex)) continue
        const raw = columnData[i]
        byRow.set(rowIndex, typeof raw === 'string' ? raw : decoder.decode(raw))
      }
    },
  })))

  return rowIndices.map(r => byRow.get(r) ?? String(r))
}
