import { parquetRead } from 'hyparquet'
import { defaultIdColumn, defaultVectorColumn } from '../constants.js'
import { scoreVectorChunk } from './chunks.js'
import { sortHeap } from './heap.js'

/**
 * @import { DistanceMetric, HypVectorMetadata, SearchResult } from '../types.js'
 * @import { AsyncBuffer, Compressors, FileMetaData } from 'hyparquet'
 */

/**
 * Exact full-scan path: read the float32 column from end to end, scoring
 * every row against the query and keeping a bounded top-K heap.
 *
 * @param {object} options
 * @param {AsyncBuffer} options.file
 * @param {FileMetaData} options.metadata
 * @param {HypVectorMetadata} options.meta
 * @param {Float32Array} options.queryF32
 * @param {DistanceMetric} options.scoringMetric
 * @param {DistanceMetric} options.reportedMetric
 * @param {number} options.topK
 * @param {Compressors} [options.compressors]
 * @returns {Promise<SearchResult[]>}
 */
export async function searchExact({ file, metadata, meta, queryF32, scoringMetric, reportedMetric, topK, compressors }) {
  /** @type {{ rowIndex: number, score: number }[]} */
  const heap = []
  /** @type {{ start: number, ids: string[] }[]} */
  const idChunks = []

  await parquetRead({
    file,
    metadata,
    compressors,
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
