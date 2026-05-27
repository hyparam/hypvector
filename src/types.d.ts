import type { AsyncBuffer, CompressionCodec, Compressors, FileMetaData } from 'hyparquet'
import type { Writer } from 'hyparquet-writer'

export type DistanceMetric = 'cosine' | 'dot' | 'euclidean'
export type SearchAlgorithm = 'auto' | 'exact' | 'binary' | 'pq'

export interface VectorRecord {
  id: string | number
  vector: Float32Array | number[]
  metadata?: Record<string, any>
}

export interface WriteVectorsOptions {
  writer: Writer // output parquet writer
  vectors: Iterable<VectorRecord> | AsyncIterable<VectorRecord>
  dimension: number // length of every vector (must match)
  rowGroupSize?: number // rows per row group (default: 10000)
  metric?: DistanceMetric // hint stored in kv metadata (default: 'cosine')
  normalize?: boolean // l2-normalize vectors on write (default: false)
  codec?: CompressionCodec // parquet codec (default: 'UNCOMPRESSED'; SNAPPY rarely shrinks float embeddings and costs ~2-3x query latency. ZSTD on write isn't supported here — hyparquet-compressors only ships decompressors.)
  binary?: boolean // also write a 1-bit-per-dim sign-bit column for binary+rerank search (default: false; adds ~1.5% file size at 384-dim)
  pageSize?: number // target page size in bytes (default: 1 MB). Smaller pages let `useOffsetIndex` fetch tighter ranges in rerank phase 2 at the cost of more page-header overhead.
  clusters?: number // run binary k-means with this many clusters, sort rows by cluster id, and store centroids in KV metadata. Enables phase 1 row-group skipping at query time. Implies binary=true. Recommended: 64-256 for 100k vectors.
  clusterIterations?: number // k-means iterations (default: 6)
  clusterSeed?: number // RNG seed for deterministic clustering (default: 1)
  pq?: boolean // write an IVF-PQ index: float IVF centroids, residual PQ codes, and residual PQ codebooks. Search uses approximate IVF-PQ scoring before exact float32 rerank.
  pqSegments?: number // number of PQ sub-vectors / bytes per code (default: 32, capped to dimension)
  pqCentroids?: number // centroids per sub-vector, 2-256 (default: 64)
  pqIterations?: number // k-means iterations per PQ sub-vector (default: 8)
  pqSampleSize?: number // deterministic training sample size per sub-vector (default: 4096)
  pqSeed?: number // RNG seed for empty-codebook reseeding (default: 1)
  ivfClusters?: number // number of IVF coarse clusters / row groups for PQ files (default: 128)
  ivfIterations?: number // k-means iterations for IVF centroids (default: 6)
  ivfSampleSize?: number // deterministic IVF training sample size (default: 4096)
}

export interface ReadVectorsOptions {
  file: AsyncBuffer // file reader for the source parquet file
  metadata?: FileMetaData // optional parquet metadata
  rowStart?: number // inclusive start row index
  rowEnd?: number // exclusive end row index
  includeMetadata?: boolean // include extra columns in yielded records (default: true)
}

export interface SearchVectorsOptions {
  query: Float32Array | number[] // the query vector

  /**
   * One source or an array of sources. Each source is a URL, file path, or
   * an already-opened AsyncBuffer. With an array, each file is searched in
   * parallel and the global top-K is returned with `result.sourceIndex` set.
   * All sources must share the same dimension and metric.
   */
  source: string | AsyncBuffer | Array<string | AsyncBuffer>

  /**
   * Pre-parsed parquet metadata; skips the footer fetch on every call.
   * Reuse across queries for best throughput. When `source` is an array,
   * pass a same-length array of (FileMetaData | undefined) — undefined
   * slots fall back to fetching the footer for that source.
   */
  metadata?: FileMetaData | Array<FileMetaData | undefined>

  topK?: number // number of nearest neighbors to return (default: 10)
  metric?: DistanceMetric // override the stored metric

  /**
   * Pre-fetched binary column from `prefetchBinary`. When provided, phase 1
   * Hamming scan runs entirely from memory and the binary parquet fetches
   * are skipped. The buffer must be `count × dim/8` bytes in the same row
   * order as the file. Reuse across queries. When `source` is an array,
   * pass a same-length array of (Uint8Array | undefined).
   */
  binary?: Uint8Array | Array<Uint8Array | undefined>

  /**
   * When the file has a binary column, controls two-phase search:
   *   - rerankFactor > 0: phase 1 scans 1-bit codes (Hamming), keeps top
   *     (topK * rerankFactor) candidates, phase 2 fetches their float32
   *     vectors and reranks by exact metric. Default: 10.
   *   - rerankFactor = 0: forces the exact full-scan path (skip binary).
   * Tune up as N grows: binary Hamming saturates, so bigger datasets
   * need a wider candidate pool. ~N/3000 is a good starting point.
   * Ignored when the file has no binary column.
   */
  rerankFactor?: number

  /**
   * Search strategy. `auto` preserves the current default priority:
   * binary+rerank when a binary column exists, PQ+rerank when only PQ exists,
   * otherwise exact full scan. Use `pq` to benchmark or force the PQ path.
   */
  algorithm?: SearchAlgorithm

  /**
   * When the file is clustered, this controls how many clusters phase 1
   * actually scans. Can be expressed as:
   *   - an integer >= 1 (number of clusters)
   *   - a float in (0, 1] (fraction of total clusters)
   * Lower values are faster but reduce recall. Default: 0.25 (scan 25% of clusters).
   * Ignored when the file has no centroids.
   */
  probe?: number

  // fetch options
  signal?: AbortSignal
  asyncBufferFactory?: (options: { source: string, signal?: AbortSignal }) => Promise<AsyncBuffer> // only consulted when `source` is a string
  compressors?: Compressors
}

export interface PrefetchBinaryOptions {
  source: string | AsyncBuffer // URL, file path, or open AsyncBuffer
  metadata?: FileMetaData // pre-parsed parquet metadata; skips a footer fetch
  signal?: AbortSignal
  asyncBufferFactory?: (options: { source: string, signal?: AbortSignal }) => Promise<AsyncBuffer> // only consulted when `source` is a string
  compressors?: Compressors
}

export interface SearchResult {
  id: string | number
  score: number // similarity score (higher = better for cosine/dot, lower = better for euclidean)
  rowIndex: number // original row index in the source parquet
  sourceIndex?: number // index into the `source` array; set only when multi-source search was used
  metadata?: Record<string, any>
}

/**
 * Metadata about a hypvector parquet file.
 * Parsed from the parquet KV metadata.
 */
export interface HypVectorMetadata {
  version: number // index format version
  dimension: number // length of each vector
  metric: DistanceMetric // intended distance metric
  normalized: boolean // whether vectors were l2-normalized on write
  hasBinary: boolean // whether a `vector_bin` sign-bit column is present
  hasPq: boolean // whether a `vector_pq` product-quantized code column is present
  count: number // number of vectors
  clusters: number // number of k-means clusters used to sort rows (0 = not clustered)
  centroids?: Uint8Array[] // binary centroids (length == clusters), each binaryBytes long
  clusterCounts?: Uint32Array // number of rows in each cluster; cluster k spans [cumsum[k], cumsum[k+1])
  pqSegments?: number // number of PQ sub-vectors / bytes per code
  pqCentroids?: number // number of PQ centroids per sub-vector
  pqMode?: 'ivf' // PQ index mode
  pqCodebooks?: Float32Array // segment-major residual codebooks, length pqCentroids * dimension
  ivfClusters?: number // number of non-empty IVF lists
  ivfCentroids?: Float32Array // IVF centroids, length ivfClusters * dimension
  ivfCounts?: Uint32Array // number of rows in each IVF list; list k spans [cumsum[k], cumsum[k+1])
}
