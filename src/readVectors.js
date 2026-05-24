import { parquetMetadataAsync, parquetReadObjects } from 'hyparquet'
import { defaultIdColumn, defaultVectorColumn } from './constants.js'
import { parseKvMetadata, unpackFloat32 } from './utils.js'

/**
 * @import { ReadVectorsOptions, VectorRecord } from './types.js'
 */

/**
 * Stream vector records from a hypvector parquet file.
 *
 * @param {ReadVectorsOptions} options
 * @returns {AsyncGenerator<VectorRecord, void, unknown>}
 */
export async function* readVectors({
  file,
  metadata,
  rowStart,
  rowEnd,
  includeMetadata = true,
}) {
  const meta = metadata ?? await parquetMetadataAsync(file)
  const { dimension } = parseKvMetadata(meta)
  const numRows = Number(meta.num_rows)
  const start = rowStart ?? 0
  const end = rowEnd ?? numRows

  const rows = await parquetReadObjects({
    file,
    metadata: meta,
    rowStart: start,
    rowEnd: end,
    utf8: false,
  })

  const decoder = new TextDecoder()
  for (const row of rows) {
    if (!row) continue
    const rawId = row[defaultIdColumn]
    const id = rawId instanceof Uint8Array ? decoder.decode(rawId) : rawId
    const bytes = row[defaultVectorColumn]
    if (!(bytes instanceof Uint8Array)) continue
    const vector = unpackFloat32(bytes)
    if (vector.length !== dimension) {
      throw new Error(`row id=${id} has vector length ${vector.length}, expected ${dimension}`)
    }
    /** @type {VectorRecord} */
    const record = { id, vector }
    if (includeMetadata) {
      /** @type {Record<string, any>} */
      const extra = {}
      let hasExtra = false
      for (const key of Object.keys(row)) {
        if (key === defaultIdColumn || key === defaultVectorColumn) continue
        extra[key] = row[key]
        hasExtra = true
      }
      if (hasExtra) record.metadata = extra
    }
    yield record
  }
}
