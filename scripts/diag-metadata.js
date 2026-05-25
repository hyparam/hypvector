/**
 * Report parquet metadata + index sizes for a hypvector file.
 *
 *   - footer:        the FileMetaData thrift block at the end of the file
 *   - offset indexes: per-column-chunk arrays of (offset, length, first-row)
 *                     for every page, used by `useOffsetIndex` to fetch tight
 *                     ranges
 *   - column indexes: per-column-chunk min/max stats per page (we don't ask
 *                     for these, but report if present)
 *
 * Usage: node scripts/diag-metadata.js [file]
 */
import { promises as fs } from 'node:fs'
import { asyncBufferFromFile, parquetMetadataAsync } from 'hyparquet'
import { parseKvMetadata } from '../src/utils.js'

const FILE = process.argv[2] ?? 'data/wiki_en.vectors.clustered.parquet'
const stat = await fs.stat(FILE)
const file = await asyncBufferFromFile(FILE)
const metadata = await parquetMetadataAsync(file)
const meta = parseKvMetadata(metadata)

console.log(`File: ${FILE}  size: ${(stat.size / 1e6).toFixed(2)} MB`)
console.log(`Rows: ${meta.count.toLocaleString()}  Row groups: ${metadata.row_groups.length}  Columns: ${metadata.row_groups[0]?.columns.length}`)
console.log(`Clusters: ${meta.clusters}`)

// Footer = the thrift block at the end. metadata_length is its byte length.
const footerLen = metadata.metadata_length
console.log(`\nFooter (FileMetaData thrift): ${(footerLen / 1024).toFixed(1)} KB`)

// KV metadata
const kvBytes = (metadata.key_value_metadata ?? []).reduce(
  (s, kv) => s + kv.key.length + (kv.value?.length ?? 0), 0,
)
console.log(`KV metadata payload: ${(kvBytes / 1024).toFixed(1)} KB`)
for (const kv of metadata.key_value_metadata ?? []) {
  console.log(`  ${kv.key.padEnd(28)} ${(kv.value?.length ?? 0).toLocaleString()} bytes`)
}

// Offset indexes / column indexes (per column chunk in each row group)
let offsetIdxBytes = 0
let columnIdxBytes = 0
let offsetIdxCount = 0
let columnIdxCount = 0
const byColumn = new Map()
for (const rg of metadata.row_groups) {
  for (const col of rg.columns) {
    const name = col.meta_data?.path_in_schema.join('.') ?? '?'
    if (!byColumn.has(name)) byColumn.set(name, { offset: 0, column: 0, oCount: 0, cCount: 0 })
    const e = byColumn.get(name)
    if (col.offset_index_length) {
      offsetIdxBytes += col.offset_index_length
      offsetIdxCount += 1
      e.offset += col.offset_index_length
      e.oCount += 1
    }
    if (col.column_index_length) {
      columnIdxBytes += col.column_index_length
      columnIdxCount += 1
      e.column += col.column_index_length
      e.cCount += 1
    }
  }
}

console.log(`\nOffset indexes: ${offsetIdxCount} chunks  total ${(offsetIdxBytes / 1024).toFixed(1)} KB`)
console.log(`Column indexes: ${columnIdxCount} chunks  total ${(columnIdxBytes / 1024).toFixed(1)} KB`)

console.log('\nPer-column breakdown:')
console.log(`${'column'.padEnd(16)} ${'offset-idx'.padStart(12)} ${'column-idx'.padStart(12)}`)
for (const [name, e] of byColumn) {
  console.log(`${name.padEnd(16)} ${(e.offset / 1024).toFixed(1).padStart(10)} KB ${(e.column / 1024).toFixed(1).padStart(10)} KB`)
}

const overhead = footerLen + offsetIdxBytes + columnIdxBytes
console.log(`\nTotal index/metadata overhead: ${(overhead / 1024).toFixed(1)} KB (${(overhead / stat.size * 100).toFixed(2)}% of file)`)
