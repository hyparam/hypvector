/**
 * Rewrite a hypvector parquet file with k-means clustering applied so
 * row-group-skipping search can run in phase 1.
 *
 * Usage:
 *   node scripts/recluster.js [input] [output] [clusters]
 *
 * Defaults to reclustering data/wiki_en.vectors.parquet → .clustered.parquet
 * with 128 clusters.
 */
import { promises as fs } from 'node:fs'
import { asyncBufferFromFile, parquetMetadataAsync } from 'hyparquet'
import { fileWriter } from 'hyparquet-writer'
import { readVectors } from '../src/readVectors.js'
import { parseKvMetadata } from '../src/utils.js'
import { writeVectors } from '../src/writeVectors.js'

const SRC = process.argv[2] ?? 'data/wiki_en.vectors.parquet'
const DST = process.argv[3] ?? 'data/wiki_en.vectors.clustered.parquet'
const CLUSTERS = parseInt(process.argv[4] ?? '128', 10)
const PAGE_SIZE = process.argv[5] ? parseInt(process.argv[5], 10) : undefined
const ROW_GROUP_SIZE = process.argv[6] ? parseInt(process.argv[6], 10) : undefined

const srcStat = await fs.stat(SRC).catch(() => undefined)
if (!srcStat) {
  console.error(`Missing ${SRC}.`)
  process.exit(1)
}

const file = await asyncBufferFromFile(SRC)
const metadata = await parquetMetadataAsync(file)
const meta = parseKvMetadata(metadata)
console.log(`Source: ${SRC} (${meta.count.toLocaleString()} vectors × ${meta.dimension}-dim)`)
console.log('Reading vectors...')

const readStart = performance.now()
/** @type {{ id: string | number, vector: Float32Array }[]} */
const records = []
for await (const record of readVectors({ file, metadata, includeMetadata: false })) {
  records.push(record)
}
console.log(`Read ${records.length.toLocaleString()} vectors in ${((performance.now() - readStart) / 1000).toFixed(1)}s`)

console.log(`Clustering into ${CLUSTERS} clusters and writing → ${DST}`)
const writeStart = performance.now()
const writer = fileWriter(DST)
await writeVectors({
  writer,
  dimension: meta.dimension,
  metric: meta.metric,
  normalize: meta.normalized,
  binary: true,
  clusters: CLUSTERS,
  ...PAGE_SIZE !== undefined ? { pageSize: PAGE_SIZE } : {},
  ...ROW_GROUP_SIZE !== undefined ? { rowGroupSize: ROW_GROUP_SIZE } : {},
  vectors: records,
})

const dstStat = await fs.stat(DST)
const seconds = (performance.now() - writeStart) / 1000
console.log(`Wrote ${DST}: ${dstStat.size.toLocaleString()} bytes in ${seconds.toFixed(1)}s`)
const overhead = ((dstStat.size - srcStat.size) / srcStat.size * 100).toFixed(2)
console.log(`Size delta vs source: ${overhead}%`)
