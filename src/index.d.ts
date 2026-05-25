import type {
  ReadVectorsOptions,
  SearchResult,
  SearchVectorsOptions,
  VectorRecord,
  WriteVectorsOptions,
} from './types.js'

export type {
  DistanceMetric,
  HypVectorMetadata,
  ReadVectorsOptions,
  SearchResult,
  SearchVectorsOptions,
  VectorRecord,
  WriteVectorsOptions,
} from './types.js'

export const hypVectorVersion: number

/**
 * Write embedding vectors to a parquet file.
 * Vectors are stored as raw float32 bytes in a BYTE_ARRAY column.
 */
export function writeVectors(options: WriteVectorsOptions): Promise<void>

/**
 * Stream vector records from a parquet file written by hypvector.
 */
export function readVectors(options: ReadVectorsOptions): AsyncGenerator<VectorRecord, void, unknown>

/**
 * Find the top-k nearest neighbors to a query vector by similarity.
 *
 * Picks the cheapest available path: clustered binary + float32 rerank when
 * the file has centroids, binary + rerank when it has a binary column, or
 * an exact full-scan when neither is present (or when `rerankFactor: 0`).
 */
export function searchVectors(options: SearchVectorsOptions): Promise<SearchResult[]>

/**
 * Cosine similarity between two vectors (higher = more similar).
 */
export function cosineSimilarity(a: Float32Array | number[], b: Float32Array | number[]): number

/**
 * Dot product of two vectors.
 */
export function dotProduct(a: Float32Array | number[], b: Float32Array | number[]): number

/**
 * Euclidean (L2) distance between two vectors.
 */
export function euclideanDistance(a: Float32Array | number[], b: Float32Array | number[]): number

/**
 * Return an l2-normalized copy of the vector.
 */
export function l2Normalize(v: Float32Array | number[]): Float32Array
