/**
 * Ablation test: write progressively-featured variants of the wiki vector
 * file and benchmark each against a fixed query set. Each row in the
 * report shows the marginal contribution of one optimization.
 *
 * Variants:
 *   A) base               vector + id only  (search must use exact full scan)
 *   B) +binary            adds vector_bin column (binary phase 1 + per-cand phase 2 reads)
 *   C) +cluster           B plus k-means clustering + centroids/counts KV
 *   D) +PQ                C plus vector_pq column + PQ codebooks
 *
 * Page size is held at 32 KB for B-D so we isolate the feature contribution
 * from the page-size knob.
 */
import { promises as fs } from 'node:fs'
import { asyncBufferFromFile, cachedAsyncBuffer, parquetMetadataAsync } from 'hyparquet'
import { fileWriter } from 'hyparquet-writer'
import { readVectors } from '../src/readVectors.js'
import { searchVectors } from '../src/searchVectors.js'
import { parseKvMetadata } from '../src/utils.js'
import { writeVectors } from '../src/writeVectors.js'

/**
 * @import { AsyncBuffer } from 'hyparquet'
 */

const SRC = process.argv[2] ?? 'data/wiki_en.vectors.parquet'

const file = await asyncBufferFromFile(SRC)
const metadata = await parquetMetadataAsync(file)
const meta = parseKvMetadata(metadata)
console.log(`Source: ${SRC} (${meta.count.toLocaleString()} × ${meta.dimension}-dim)`)

console.log('Reading all vectors...')
const records = []
for await (const record of readVectors({ file, metadata, includeMetadata: false })) {
  records.push(record)
}

const variants = [
  { name: 'A_base', label: 'A) base (vec only)', opts: { binary: false } },
  { name: 'B_binary', label: 'B) +binary', opts: { binary: true } },
  { name: 'C_cluster', label: 'C) +cluster', opts: { binary: true, clusters: 128 } },
  { name: 'D_pq', label: 'D) +cluster+PQ', opts: { binary: true, clusters: 128, pq: true }, search: { algorithm: 'pq' } },
]

for (const v of variants) {
  v.path = `data/abl_${v.name}.parquet`
  const exists = await fs.stat(v.path).catch(() => undefined)
  if (!exists) {
    const writer = fileWriter(v.path)
    await writeVectors({
      writer,
      dimension: meta.dimension,
      metric: meta.metric,
      normalize: meta.normalized,
      vectors: records,
      ...v.opts,
    })
  }
  v.size = (await fs.stat(v.path)).size
}

const QUERY_COUNT = 10
const queries = []
const step = Math.max(1, Math.floor(meta.count / (QUERY_COUNT + 1)))
let i = 0
let nextPick = step
for await (const r of readVectors({ file, metadata })) {
  if (i === nextPick) {
    queries.push(r.vector)
    nextPick += step
    if (queries.length >= QUERY_COUNT) break
  }
  i += 1
}

/**
 * Wrap an AsyncBuffer with byte / fetch counters.
 *
 * @param {AsyncBuffer} buf
 * @returns {AsyncBuffer & { bytes: number, fetches: number }}
 */
function instrument(buf) {
  const slice = buf.slice.bind(buf)
  const w = {
    byteLength: buf.byteLength,
    bytes: 0,
    fetches: 0,
    slice(s, e) {
      w.bytes += (e ?? buf.byteLength) - s
      w.fetches += 1
      return slice(s, e)
    },
  }
  return w
}

async function bench(path, extra) {
  const times = [], bytesA = [], fetchesA = [], tops = []
  for (const q of queries) {
    const raw = instrument(await asyncBufferFromFile(path))
    const cached = cachedAsyncBuffer(raw)
    const start = performance.now()
    const r = await searchVectors({ source: cached, query: q, topK: 10, ...extra })
    times.push(performance.now() - start)
    bytesA.push(raw.bytes); fetchesA.push(raw.fetches); tops.push(r.map(x => String(x.id)))
  }
  return {
    ms: avg(times), mb: avg(bytesA) / 1e6, fetches: avg(fetchesA), tops,
  }
}

/**
 * @param {number[]} a
 * @returns {number}
 */
function avg(a) {
  let s = 0
  for (let i = 0; i < a.length; i += 1) s += a[i]
  return s / a.length
}

// Reference top-10 = exact full scan on base file
const ref = await bench(variants[0].path, { rerankFactor: 0 })

console.log(`\n${'variant'.padEnd(28)} ${'size MB'.padStart(8)} ${'ms'.padStart(7)} ${'fetches'.padStart(8)} ${'MB read'.padStart(9)} ${'recall'.padStart(8)}`)
console.log('-'.repeat(75))

for (const v of variants) {
  const opts = {}
  // For base file, rerankFactor=0 forces exact path. For others, default rerank/probe.
  if (v.name === 'A_base') opts.rerankFactor = 0
  Object.assign(opts, v.search)
  const r = await bench(v.path, opts)
  let hits = 0, total = 0
  for (let q = 0; q < ref.tops.length; q += 1) {
    const refSet = new Set(ref.tops[q])
    for (const id of r.tops[q]) if (refSet.has(id)) hits += 1
    total += refSet.size
  }
  const recall = hits / total
  console.log(`${v.label.padEnd(28)} ${(v.size / 1e6).toFixed(1).padStart(8)} ${r.ms.toFixed(1).padStart(7)} ${r.fetches.toFixed(0).padStart(8)} ${r.mb.toFixed(2).padStart(9)} ${(recall * 100).toFixed(1).padStart(7)}%`)
}
