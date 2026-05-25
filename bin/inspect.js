import { parseKvMetadata } from '../src/utils.js'
import { loadParquet } from './load.js'

/**
 * Print statistics about a hypvector parquet file.
 *
 * @param {object} options
 * @param {string} options.path
 */
export async function inspect({ path }) {
  const { file, metadata } = await loadParquet(path)
  const meta = parseKvMetadata(metadata)

  const bytesPerVector = meta.dimension * 4
  const rawSize = meta.count * bytesPerVector
  const ratio = rawSize > 0 ? file.byteLength / rawSize : 0

  console.log(`File: ${path}`)
  console.log(`File: ${file.byteLength.toLocaleString()} bytes`)
  console.log(`Vectors: ${meta.count.toLocaleString()}`)
  console.log(`Dimension: ${meta.dimension}`)
  console.log(`Metric: ${meta.metric}`)
  console.log(`Normalized: ${meta.normalized}`)
  console.log(`Binary column: ${meta.hasBinary}`)
  console.log(`Row groups: ${metadata.row_groups.length.toLocaleString()}`)
  console.log(`Raw float32 size: ${rawSize.toLocaleString()} bytes`)
  console.log(`Overhead: ${(ratio * 100).toFixed(1)}% of raw`)
}
