import { parquetMetadataAsync } from 'hyparquet'
import { defaultAsyncBufferFactory } from './asyncBufferFactory.js'
import { searchExact } from './search/exact.js'
import { searchPq } from './search/pq.js'
import { searchRerank } from './search/rerank.js'
import { l2Normalize, parseKvMetadata } from './utils.js'

/**
 * @import { SearchResult, SearchVectorsOptions } from './types.js'
 * @import { AsyncBuffer, FileMetaData } from 'hyparquet'
 */

/**
 * Find the top-k nearest neighbors to a query vector.
 *
 * `source` may be a single URL/path/AsyncBuffer or an array of them. With
 * an array, each file is searched in parallel and results are heap-merged
 * to the global top-K. Per-source `metadata` and `binary` (from
 * `prefetchBinary`) may be passed as same-length arrays; pass `undefined`
 * in any slot to fall back to the default behavior for that source.
 *
 * With `algorithm: 'auto'`, paths are picked per source in this order:
 *   - Clustered + binary + rerank (file has centroids): phase 1 scans only
 *     the top-N nearest clusters' row ranges (skipping whole row groups),
 *     phase 2 fetches the candidate float32 vectors and reranks.
 *   - Binary + rerank (binary column present, no clustering): full Hamming
 *     scan in phase 1, then per-candidate float32 fetch + rerank in phase 2.
 *   - PQ + rerank (PQ column present, no binary column): scan compact PQ
 *     codes in phase 1, then per-candidate float32 fetch + rerank in phase 2.
 *   - Exact (no binary column, or rerankFactor=0): single pass over the
 *     float32 column, scoring every row.
 *
 * @param {SearchVectorsOptions} options
 * @returns {Promise<SearchResult[]>}
 */
export async function searchVectors({
  query,
  source,
  metadata,
  topK = 10,
  metric,
  rerankFactor = 10,
  probe,
  binary,
  algorithm = 'auto',
  signal,
  asyncBufferFactory,
  compressors,
}) {
  if (source === undefined || source === null) {
    throw new Error('searchVectors: `source` is required (URL, file path, or AsyncBuffer)')
  }
  const sources = Array.isArray(source) ? source : [source]
  if (sources.length === 0) {
    throw new Error('searchVectors: `source` array is empty')
  }
  if (Array.isArray(metadata) && metadata.length !== sources.length) {
    throw new Error(`searchVectors: \`metadata\` array length ${metadata.length} does not match \`source\` array length ${sources.length}`)
  }
  if (Array.isArray(binary) && binary.length !== sources.length) {
    throw new Error(`searchVectors: \`binary\` array length ${binary.length} does not match \`source\` array length ${sources.length}`)
  }
  const metadatas = Array.isArray(metadata) ? metadata : sources.map(() => metadata)
  const binaries = Array.isArray(binary) ? binary : sources.map(() => binary)

  const multi = sources.length > 1
  const perSource = await Promise.all(sources.map((src, i) => searchOne({
    query,
    source: src,
    metadata: metadatas[i],
    binary: binaries[i],
    topK,
    metric,
    rerankFactor,
    probe,
    signal,
    algorithm,
    asyncBufferFactory,
    compressors,
    sourceIndex: multi ? i : undefined,
  })))

  if (!multi) return perSource[0].results

  // Merge: each per-source array is already sorted best-first under the same
  // direction (we assert below). Flatten, sort, slice — N is small.
  const { direction } = perSource[0]
  for (let i = 1; i < perSource.length; i += 1) {
    if (perSource[i].direction !== direction) {
      throw new Error('searchVectors: sources have inconsistent metric directions')
    }
  }
  const merged = perSource.flatMap(p => p.results)
  merged.sort((a, b) => direction * (a.score - b.score))
  return merged.slice(0, topK)
}

/**
 * Search a single source. Returns the sorted results plus the score
 * direction (1 = ascending / lower-is-better for euclidean, -1 = descending
 * for cosine/dot). The direction lets the caller merge cross-source results
 * without re-parsing metadata.
 *
 * @param {object} opts
 * @param {Float32Array | number[]} opts.query
 * @param {string | AsyncBuffer} opts.source
 * @param {FileMetaData} [opts.metadata]
 * @param {Uint8Array} [opts.binary]
 * @param {number} opts.topK
 * @param {import('./types.js').DistanceMetric} [opts.metric]
 * @param {number} opts.rerankFactor
 * @param {number} [opts.probe]
 * @param {import('./types.js').SearchAlgorithm} opts.algorithm
 * @param {AbortSignal} [opts.signal]
 * @param {(options: { source: string, signal?: AbortSignal }) => Promise<AsyncBuffer>} [opts.asyncBufferFactory]
 * @param {import('hyparquet').Compressors} [opts.compressors]
 * @param {number} [opts.sourceIndex]
 * @returns {Promise<{ results: SearchResult[], direction: 1 | -1 }>}
 */
async function searchOne({
  query, source, metadata: providedMetadata, binary, topK, metric, rerankFactor, probe, algorithm, signal, asyncBufferFactory, compressors, sourceIndex,
}) {
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

  /** @type {SearchResult[]} */
  let results
  if (algorithm === 'exact' || rerankFactor === 0) {
    results = await searchExact({
      file, metadata, meta, queryF32, scoringMetric, reportedMetric: requestedMetric, topK, compressors,
    })
  } else if (algorithm === 'pq') {
    if (!meta.hasPq) throw new Error('searchVectors: algorithm `pq` requested, but file has no PQ column')
    results = await searchPq({
      file, metadata, meta, queryF32, scoringMetric, reportedMetric: requestedMetric, topK, rerankFactor, probe, compressors,
    })
  } else if (algorithm === 'binary') {
    if (!meta.hasBinary) throw new Error('searchVectors: algorithm `binary` requested, but file has no binary column')
    results = await searchRerank({
      file, metadata, meta, queryF32, scoringMetric, reportedMetric: requestedMetric, topK, rerankFactor, probe, binary, compressors,
    })
  } else if (algorithm !== 'auto') {
    throw new Error(`searchVectors: unsupported algorithm ${algorithm}`)
  } else if (meta.hasBinary) {
    results = await searchRerank({
      file, metadata, meta, queryF32, scoringMetric, reportedMetric: requestedMetric, topK, rerankFactor, probe, binary, compressors,
    })
  } else if (meta.hasPq) {
    results = await searchPq({
      file, metadata, meta, queryF32, scoringMetric, reportedMetric: requestedMetric, topK, rerankFactor, probe, compressors,
    })
  } else {
    results = await searchExact({
      file, metadata, meta, queryF32, scoringMetric, reportedMetric: requestedMetric, topK, compressors,
    })
  }

  if (sourceIndex !== undefined) {
    for (const r of results) r.sourceIndex = sourceIndex
  }

  const direction = requestedMetric === 'euclidean' ? 1 : -1
  return { results, direction }
}
