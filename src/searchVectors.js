import { parquetMetadataAsync } from 'hyparquet'
import { defaultAsyncBufferFactory } from './asyncBufferFactory.js'
import { searchExact } from './search/exact.js'
import { searchRerank } from './search/rerank.js'
import { l2Normalize, parseKvMetadata } from './utils.js'

/**
 * @import { SearchResult, SearchVectorsOptions } from './types.js'
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
  source,
  metadata: providedMetadata,
  topK = 10,
  metric,
  rerankFactor = 10,
  probe,
  signal,
  asyncBufferFactory,
  compressors,
}) {
  if (source === undefined || source === null) {
    throw new Error('searchVectors: `source` is required (URL, file path, or AsyncBuffer)')
  }
  const file = typeof source === 'string'
    ? await (asyncBufferFactory ?? defaultAsyncBufferFactory)({ source, signal })
    : source
  const metadata = providedMetadata ?? await parquetMetadataAsync(file)
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
      file, metadata, meta, queryF32, scoringMetric, reportedMetric: requestedMetric, topK, rerankFactor, probe, compressors,
    })
  }
  return searchExact({
    file, metadata, meta, queryF32, scoringMetric, reportedMetric: requestedMetric, topK, compressors,
  })
}
