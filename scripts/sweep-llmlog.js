/**
 * Auto-tuning sweep for LLM log embeddings.
 *
 * Three sweeps, each driven by the experiments listed in PLAN_AUTO.md:
 *
 *   1) clusters ∈ {0, sqrt(N)/2, sqrt(N), 2·sqrt(N), 4·sqrt(N)} — write-side
 *   2) rerankFactor ∈ {10, 30, 100, max(10,N/3000), 300} — query-side
 *   3) probe ∈ {0.05, 0.1, 0.25, 0.5, 1.0} — query-side
 *
 * Reference top-10 = exact full scan against the base file. All clustered
 * files share write-time options aside from the clusters knob.
 *
 * Usage: node scripts/sweep-llmlog.js [data/llmlog.vectors.parquet]
 */
import { promises as fs } from 'node:fs'
import { asyncBufferFromFile, cachedAsyncBuffer, parquetMetadataAsync } from 'hyparquet'
import { fileWriter } from 'hyparquet-writer'
import { readVectors } from '../src/readVectors.js'
import { searchVectors } from '../src/searchVectors.js'
import { parseKvMetadata } from '../src/utils.js'
import { writeVectors } from '../src/writeVectors.js'

/** @import { AsyncBuffer } from 'hyparquet' */

const SRC = process.argv[2] ?? 'data/llmlog.vectors.parquet'
const QUERY_COUNT = 20

const file = await asyncBufferFromFile(SRC)
const metadata = await parquetMetadataAsync(file)
const meta = parseKvMetadata(metadata)
const N = meta.count
console.log(`Source: ${SRC} (${N.toLocaleString()} × ${meta.dimension}-dim, metric=${meta.metric})`)

console.log('Reading all vectors...')
const records = []
for await (const record of readVectors({ file, metadata, includeMetadata: false })) {
  records.push(record)
}

// Held-out queries: evenly spaced from the corpus. Same vector is excluded
// from its own top-K via id mismatch on rerank (we don't bother — the
// query == doc effect just inflates recall identically across variants).
const queries = []
const step = Math.max(1, Math.floor(N / (QUERY_COUNT + 1)))
for (let i = 0, pick = step; i < records.length && queries.length < QUERY_COUNT; i += 1) {
  if (i === pick) { queries.push(records[i].vector); pick += step }
}

const sqrtN = Math.round(Math.sqrt(N))
const clusterValues = [0, Math.round(sqrtN / 2), sqrtN, 2 * sqrtN, 4 * sqrtN]
const rerankValues = [10, 30, 100, Math.max(10, Math.round(N / 3000)), 300]
const probeValues = [0.05, 0.1, 0.25, 0.5, 1.0]

const SRC_BASE = SRC.replace(/\.parquet$/, '').split('/').pop()
function pathFor(c) { return `data/${SRC_BASE}_sweep_c${c}.parquet` }

// (Re)write one file per cluster value.
for (const c of clusterValues) {
  const path = pathFor(c)
  if (await fs.stat(path).catch(() => undefined)) {
    console.log(`  ${path} exists, skipping write`)
    continue
  }
  console.log(`Writing ${path} (clusters=${c})...`)
  const writer = fileWriter(path)
  const start = performance.now()
  await writeVectors({
    writer,
    dimension: meta.dimension,
    metric: meta.metric,
    normalize: meta.normalized,
    vectors: records,
    binary: c > 0 ? true : true, // always include binary so rerank path exists
    clusters: c,
  })
  console.log(`  wrote in ${((performance.now() - start) / 1000).toFixed(1)}s`)
}

/**
 * @param {AsyncBuffer} buf
 * @returns {AsyncBuffer & { bytes: number, fetches: number }}
 */
function instrument(buf) {
  const slice = buf.slice.bind(buf)
  const w = {
    byteLength: buf.byteLength, bytes: 0, fetches: 0,
    slice(s, e) { w.bytes += (e ?? buf.byteLength) - s; w.fetches += 1; return slice(s, e) },
  }
  return w
}

/**
 * @param {string} path
 * @param {object} extra
 * @returns {Promise<{ ms: number, mb: number, fetches: number, tops: string[][] }>}
 */
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
  return { ms: avg(times), mb: avg(bytesA) / 1e6, fetches: avg(fetchesA), tops }
}

function avg(a) { let s = 0; for (const x of a) s += x; return s / a.length }

function recallAgainst(refTops, tops) {
  let hits = 0, total = 0
  for (let i = 0; i < refTops.length; i += 1) {
    const refSet = new Set(refTops[i])
    for (const id of tops[i]) if (refSet.has(id)) hits += 1
    total += refSet.size
  }
  return hits / total
}

console.log('\nReference: exact full scan on c=0 file')
const ref = await bench(pathFor(0), { rerankFactor: 0 })

// --- Sweep 1: clusters ---
console.log(`\n=== clusters sweep (N=${N.toLocaleString()}, sqrt(N)=${sqrtN}) ===`)
console.log(`${'clusters'.padStart(10)} ${'size MB'.padStart(8)} ${'ms'.padStart(7)} ${'fetches'.padStart(8)} ${'MB read'.padStart(9)} ${'recall'.padStart(8)}`)
console.log('-'.repeat(60))
for (const c of clusterValues) {
  const path = pathFor(c)
  const { size } = await fs.stat(path)
  const opts = c === 0 ? { rerankFactor: 10 } : {} // default probe/rerank for clustered
  const r = await bench(path, opts)
  const rec = recallAgainst(ref.tops, r.tops)
  console.log(`${String(c).padStart(10)} ${(size / 1e6).toFixed(1).padStart(8)} ${r.ms.toFixed(1).padStart(7)} ${r.fetches.toFixed(0).padStart(8)} ${r.mb.toFixed(2).padStart(9)} ${(rec * 100).toFixed(1).padStart(7)}%`)
}

// --- Sweep 2: rerankFactor on c=sqrt(N) file ---
const cMid = sqrtN
console.log(`\n=== rerankFactor sweep (clusters=${cMid}, probe=default) ===`)
console.log(`${'rerankFactor'.padStart(13)} ${'ms'.padStart(7)} ${'fetches'.padStart(8)} ${'MB read'.padStart(9)} ${'recall'.padStart(8)}`)
console.log('-'.repeat(55))
for (const rf of rerankValues) {
  const r = await bench(pathFor(cMid), { rerankFactor: rf })
  const rec = recallAgainst(ref.tops, r.tops)
  console.log(`${String(rf).padStart(13)} ${r.ms.toFixed(1).padStart(7)} ${r.fetches.toFixed(0).padStart(8)} ${r.mb.toFixed(2).padStart(9)} ${(rec * 100).toFixed(1).padStart(7)}%`)
}

// --- Sweep 3: probe on c=sqrt(N) file ---
console.log(`\n=== probe sweep (clusters=${cMid}, rerankFactor=10) ===`)
console.log(`${'probe'.padStart(7)} ${'ms'.padStart(7)} ${'fetches'.padStart(8)} ${'MB read'.padStart(9)} ${'recall'.padStart(8)}`)
console.log('-'.repeat(50))
for (const p of probeValues) {
  const r = await bench(pathFor(cMid), { probe: p })
  const rec = recallAgainst(ref.tops, r.tops)
  console.log(`${p.toFixed(2).padStart(7)} ${r.ms.toFixed(1).padStart(7)} ${r.fetches.toFixed(0).padStart(8)} ${r.mb.toFixed(2).padStart(9)} ${(rec * 100).toFixed(1).padStart(7)}%`)
}
