import type { AsyncBuffer, asyncBufferFromUrl, CompressionCodec, Compressors, FileMetaData } from 'hyparquet'
import type { Writer } from 'hyparquet-writer'

export type DistanceMetric = 'cosine' | 'dot' | 'euclidean'

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
  codec?: CompressionCodec // parquet codec (default: 'UNCOMPRESSED'; SNAPPY/ZSTD rarely shrink float embeddings and cost ~2-3x query latency)
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
  url: string // URL or file path to the source parquet file
  topK?: number // number of nearest neighbors to return (default: 10)
  metric?: DistanceMetric // override the stored metric

  // fetch options
  signal?: AbortSignal
  asyncBufferFactory?: typeof asyncBufferFromUrl
  sourceFile?: AsyncBuffer
  sourceMetadata?: FileMetaData
  compressors?: Compressors
}

export interface SearchResult {
  id: string | number
  score: number // similarity score (higher = better for cosine/dot, lower = better for euclidean)
  rowIndex: number // original row index in the source parquet
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
  count: number // number of vectors
}
