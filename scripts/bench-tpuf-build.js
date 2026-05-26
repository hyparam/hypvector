/**
 * Build a hypvector parquet that matches the turbopuffer `vector-10m` workload
 * spec for apples-to-apples comparison. At 1M scale here (10 Cohere shards).
 *
 * Source:  Cohere/wikipedia-2023-11-embed-multilingual-v3 (en), 1024-dim
 * Output:  data/tpuf-bench-1m.parquet
 *
 * Run with extra heap; ~4GB of packed float32 bytes live in memory before
 * parquet write commits them:
 *   node --max-old-space-size=16384 scripts/bench-tpuf-build.js
 */
import { createWriteStream, promises as fs } from 'node:fs'
import { pipeline } from 'node:stream/promises'
import { asyncBufferFromFile, parquetMetadataAsync, parquetReadObjects } from 'hyparquet'
import { fileWriter } from 'hyparquet-writer'
import { writeVectors } from '../src/writeVectors.js'

const SHARDS = parseInt(process.argv[2] ?? '10', 10)
const SHARD_DIR = 'data/cohere-shards'
const DEST = `data/tpuf-bench-${SHARDS}00k.parquet`
const DIM = 1024
const CLUSTERS = 256
const URL_BASE = 'https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3/resolve/main/en'

await fs.mkdir(SHARD_DIR, { recursive: true })

/**
 * Download a single shard if missing.
 * @param {number} i shard index
 * @returns {Promise<string>} local file path
 */
async function ensureShard(i) {
  const name = String(i).padStart(4, '0') + '.parquet'
  const dest = `${SHARD_DIR}/${name}`
  const existing = await fs.stat(dest).catch(() => undefined)
  if (existing) return dest
  const url = `${URL_BASE}/${name}`
  console.log(`  downloading ${url}`)
  const res = await fetch(url)
  if (!res.ok) throw new Error(`fetch ${url} failed: ${res.status}`)
  if (!res.body) throw new Error(`no body for ${url}`)
  await pipeline(res.body, createWriteStream(dest))
  return dest
}

console.log(`Downloading ${SHARDS} shards in parallel...`)
const tDownload = performance.now()
const paths = await Promise.all(Array.from({ length: SHARDS }, (_, i) => ensureShard(i)))
console.log(`  ${SHARDS} shards ready in ${((performance.now() - tDownload) / 1000).toFixed(1)}s`)

/**
 * Stream {id, vector} records from each shard in sequence.
 * @returns {AsyncGenerator<{id: string, vector: Float32Array}>}
 */
async function* iterRecords() {
  let yielded = 0
  for (const path of paths) {
    const file = await asyncBufferFromFile(path)
    const metadata = await parquetMetadataAsync(file)
    const rows = Number(metadata.num_rows)
    const BATCH = 1000
    for (let i = 0; i < rows; i += BATCH) {
      const end = Math.min(i + BATCH, rows)
      const objs = await parquetReadObjects({
        file, metadata, rowStart: i, rowEnd: end, columns: ['_id', 'emb'],
      })
      for (const r of objs) {
        // emb is { list: [{item: float}, ...] } in this parquet's nested schema
        const { emb } = r
        // hyparquet decodes list<float> as a plain Float32Array or number[] — both work for writeVectors.
        if (!emb || emb.length !== DIM) {
          throw new Error(`row ${yielded}: emb length ${emb?.length} != ${DIM}`)
        }
        yield { id: String(r._id), vector: emb }
        yielded += 1
      }
    }
    process.stdout.write(`\r  read ${yielded.toLocaleString()} vectors `)
  }
  process.stdout.write('\n')
}

console.log(`Writing ${DEST} (binary + clusters=${CLUSTERS})...`)
const tWrite = performance.now()
const writer = fileWriter(DEST)
await writeVectors({
  writer,
  dimension: DIM,
  vectors: iterRecords(),
  metric: 'cosine',
  normalize: true,
  binary: true,
  clusters: CLUSTERS,
})
const stat = await fs.stat(DEST)
const secs = (performance.now() - tWrite) / 1000
console.log(`  wrote ${(stat.size / 1e6).toFixed(1)} MB in ${secs.toFixed(1)}s`)
