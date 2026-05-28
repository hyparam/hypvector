import { parquetMetadataAsync, parquetRead } from 'hyparquet'
import { defaultAsyncBufferFactory } from './asyncBufferFactory.js'
import { defaultBinaryColumn } from './constants.js'
import { parseKvMetadata } from './utils.js'

/**
 * @import { PrefetchBinaryOptions } from './types.js'
 */

/**
 * Eagerly fetch the entire binary (1-bit-per-dim) column into a single
 * Uint8Array. Pass the result as `binary` to `searchVectors` and phase 1
 * runs from memory — every subsequent query skips the binary parquet
 * fetches entirely.
 *
 * Cost: ~count × dim/8 bytes (e.g. 7.5 MB for 156k × 384-dim). Recouped
 * after a handful of queries on any high-RTT path.
 *
 * @param {PrefetchBinaryOptions} options
 * @returns {Promise<Uint8Array>}
 */
export async function prefetchBinary({
  source,
  metadata: providedMetadata,
  signal,
  asyncBufferFactory,
  compressors,
}) {
  if (source === undefined || source === null) {
    throw new Error('prefetchBinary: `source` is required (URL, file path, or AsyncBuffer)')
  }
  const file = typeof source === 'string'
    ? await (asyncBufferFactory ?? defaultAsyncBufferFactory)({ source, signal })
    : source
  const metadata = providedMetadata ?? await parquetMetadataAsync(file)
  const meta = parseKvMetadata(metadata)
  if (!meta.hasBinary) throw new Error('prefetchBinary: file has no binary column')

  const bytesPerRow = meta.dimension + 7 >> 3
  const buffer = new Uint8Array(meta.count * bytesPerRow)

  await parquetRead({
    file,
    metadata,
    compressors,
    columns: [defaultBinaryColumn],
    onChunk: ({ columnName, columnData, rowStart }) => {
      if (columnName !== defaultBinaryColumn) return
      const baseByte = rowStart * bytesPerRow
      for (let i = 0; i < columnData.length; i += 1) {
        buffer.set(columnData[i], baseByte + i * bytesPerRow)
      }
    },
  })

  return buffer
}
