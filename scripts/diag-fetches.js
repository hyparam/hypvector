/**
 * Diagnostic: run a single query against a clustered file and log every
 * byte range hyparquet asks for, so we can see what's actually being
 * fetched in phase 1 vs phase 2 vs metadata overhead.
 */
import { asyncBufferFromFile, cachedAsyncBuffer, parquetMetadataAsync } from 'hyparquet'
import { readVectors } from '../src/readVectors.js'
import { searchVectors } from '../src/searchVectors.js'
import { parseKvMetadata } from '../src/utils.js'

/**
 * @import { AsyncBuffer } from 'hyparquet'
 */

const filename = process.argv[2] ?? 'data/wiki_en.vectors.clustered.parquet'

const file = await asyncBufferFromFile(filename)
const metadata = await parquetMetadataAsync(file)
const meta = parseKvMetadata(metadata)
console.log(`File: ${filename}  vectors=${meta.count}  size=${file.byteLength}`)

// Pull one query vector
let query
let i = 0
for await (const r of readVectors({ file, metadata })) {
  if (i === Math.floor(meta.count / 2)) { query = r.vector; break }
  i += 1
}

// Annotate vector col / binary col / id col / cluster col page ranges
const colRanges = new Map() // colname -> [byteStart, byteEnd]
for (const rg of metadata.row_groups) {
  for (const col of rg.columns) {
    const name = col.meta_data?.path_in_schema.join('.')
    const start = Number(col.meta_data?.data_page_offset ?? 0)
    const len = Number(col.meta_data?.total_compressed_size ?? 0)
    if (!colRanges.has(name)) colRanges.set(name, [])
    colRanges.get(name).push([start, start + len])
  }
}

function classify(start, end) {
  for (const [name, ranges] of colRanges) {
    for (const [s, e] of ranges) {
      if (start >= s && end <= e) return name
      if (start < e && end > s) return `${name}(partial)`
    }
  }
  if (start > file.byteLength - 200000) return 'footer/index'
  return 'unknown'
}

const fetches = []
const raw = await asyncBufferFromFile(filename)
/** @type {AsyncBuffer} */
const wrapped = {
  byteLength: raw.byteLength,
  slice: (start, end) => {
    fetches.push({ start, end: end ?? raw.byteLength })
    return raw.slice(start, end)
  },
}
const cached = cachedAsyncBuffer(wrapped)

const results = await searchVectors({
  source: cached, metadata, query, topK: 10, probe: 0.25,
})

// Group fetches by classification
const grouped = new Map()
for (const f of fetches) {
  const size = f.end - f.start
  const k = classify(f.start, f.end)
  if (!grouped.has(k)) grouped.set(k, { count: 0, bytes: 0 })
  grouped.get(k).count += 1
  grouped.get(k).bytes += size
}

let totalBytes = 0
for (const f of fetches) totalBytes += f.end - f.start
console.log(`\nTotal: ${fetches.length} fetches, ${totalBytes} bytes`)
console.log('\nBy column:')
for (const [k, v] of grouped) {
  console.log(`  ${k.padEnd(30)} ${v.count.toString().padStart(4)} fetches  ${(v.bytes / 1024).toFixed(1).padStart(8)} KB`)
}
console.log('\nTop 5 candidates:')
for (const r of results.slice(0, 5)) console.log(`  ${r.id}  ${r.score.toFixed(4)}`)
