import { promises as fs } from 'node:fs'
import { asyncBufferFromFile, cachedAsyncBuffer, parquetMetadataAsync } from 'hyparquet'
import { fileWriter } from 'hyparquet-writer'
import { readVectors } from './src/readVectors.js'
import { searchVectors } from './src/searchVectors.js'
import { parseKvMetadata } from './src/utils.js'
import { writeVectors } from './src/writeVectors.js'

/**
 * @import { AsyncBuffer } from 'hyparquet'
 */

const REAL_FILE = process.argv[2] ?? 'data/wiki_en.vectors.parquet'
const SYNTHETIC_FILE = 'bench.parquet'
const SYNTHETIC_COUNT = 50000
const SYNTHETIC_DIMENSION = 384

/**
 * Deterministic pseudo-random vector generator (LCG).
 *
 * @param {number} count
 * @param {number} dimension
 * @returns {Generator<{id: string, vector: Float32Array}>}
 */
function* makeSyntheticVectors(count, dimension) {
  let s = 1
  function rand() {
    const stepped = Math.imul(s, 1664525) + 1013904223
    s = stepped >>> 0
    return s / 0x100000000
  }
  for (let i = 0; i < count; i += 1) {
    const v = new Float32Array(dimension)
    for (let j = 0; j < dimension; j += 1) v[j] = rand() * 2 - 1
    yield { id: `vec-${i}`, vector: v }
  }
}

// Pick the dataset: prefer the real wiki vectors, fall back to synthetic.
const realStat = await fs.stat(REAL_FILE).catch(() => undefined)
const filename = realStat ? REAL_FILE : SYNTHETIC_FILE
const label = realStat ? 'real (wiki)' : 'synthetic (uniform random)'

if (!realStat) {
  let synthStat = await fs.stat(SYNTHETIC_FILE).catch(() => undefined)
  if (!synthStat) {
    console.log(`=== Writing ${SYNTHETIC_COUNT.toLocaleString()} synthetic vectors of dim ${SYNTHETIC_DIMENSION} ===`)
    const writeStart = performance.now()
    const writer = fileWriter(SYNTHETIC_FILE)
    await writeVectors({
      writer,
      dimension: SYNTHETIC_DIMENSION,
      vectors: makeSyntheticVectors(SYNTHETIC_COUNT, SYNTHETIC_DIMENSION),
    })
    synthStat = await fs.stat(SYNTHETIC_FILE)
    console.log(`Wrote ${SYNTHETIC_FILE} in ${(performance.now() - writeStart).toFixed(0)} ms`)
  }
  console.log(`(no ${REAL_FILE}; run \`npm run data:download && npm run data:embed\` for real vectors)`)
}

const stat = await fs.stat(filename)
const sourceFile = await asyncBufferFromFile(filename)
const metadata = await parquetMetadataAsync(sourceFile)
const meta = parseKvMetadata(metadata)
const rawSize = meta.count * meta.dimension * 4

console.log(`\n=== Dataset (${label}) ===`)
console.log(`File: ${filename}`)
console.log(`Vectors: ${meta.count.toLocaleString()} × ${meta.dimension}-dim`)
console.log(`File size: ${stat.size.toLocaleString()} bytes (${(stat.size / rawSize * 100).toFixed(1)}% of raw float32)`)
console.log(`Metric: ${meta.metric}${meta.normalized ? ' (normalized)' : ''}`)
console.log(`Binary column: ${meta.hasBinary}`)

// Sample some stored vectors to use as query vectors.
// (For real data this is more representative than uniform-random queries.)
const QUERY_COUNT = 5
/** @type {{ id: string | number, vector: Float32Array }[]} */
const queries = []
const step = Math.max(1, Math.floor(meta.count / (QUERY_COUNT + 1)))
let nextPick = step
let i = 0
for await (const record of readVectors({ file: sourceFile, metadata })) {
  if (i === nextPick) {
    queries.push({ id: record.id, vector: record.vector })
    nextPick += step
    if (queries.length >= QUERY_COUNT) break
  }
  i += 1
}

/**
 * Wrap an AsyncBuffer with byte / fetch counters.
 *
 * @param {AsyncBuffer} buf
 * @returns {AsyncBuffer & { bytes: number, fetches: number, reset: () => void }}
 */
function instrument(buf) {
  const origSlice = buf.slice.bind(buf)
  const wrapped = {
    byteLength: buf.byteLength,
    bytes: 0,
    fetches: 0,
    slice(start, end) {
      wrapped.bytes += (end ?? buf.byteLength) - start
      wrapped.fetches += 1
      return origSlice(start, end)
    },
    reset() { wrapped.bytes = 0; wrapped.fetches = 0 },
  }
  return wrapped
}

/**
 * Run a configured search across all queries and return aggregated stats + per-query ids.
 *
 * @param {string} label
 * @param {{ rerankFactor?: number }} opts
 * @returns {Promise<{ label: string, avgMs: number, avgBytes: number, avgFetches: number, tops: string[][] }>}
 */
async function runSearchSuite(label, opts) {
  // Per-query: build a fresh cached buffer atop a counter-instrumented raw buffer.
  // (Cold cache each query, so reported bytes/fetches reflect a single from-scratch query.)
  const times = []
  const bytesPer = []
  const fetchesPer = []
  const tops = []
  for (const q of queries) {
    const raw = instrument(await asyncBufferFromFile(filename))
    const cached = cachedAsyncBuffer(raw)
    const start = performance.now()
    const results = await searchVectors({
      url: filename, query: q.vector, topK: 10, sourceFile: cached, sourceMetadata: metadata, ...opts,
    })
    times.push(performance.now() - start)
    bytesPer.push(raw.bytes)
    fetchesPer.push(raw.fetches)
    tops.push(results.map(r => String(r.id)))
  }
  let sMs = 0; let sBytes = 0; let sFetches = 0
  for (let i = 0; i < times.length; i += 1) {
    sMs += times[i]; sBytes += bytesPer[i]; sFetches += fetchesPer[i]
  }
  return {
    label,
    avgMs: sMs / times.length,
    avgBytes: sBytes / times.length,
    avgFetches: sFetches / times.length,
    tops,
  }
}

console.log('\n=== Search ===')
const exact = await runSearchSuite('Exact full scan', { rerankFactor: 0 })
const rerank = meta.hasBinary ? await runSearchSuite('Binary + rerank', { rerankFactor: 10 }) : null

/**
 * Recall@10 = |intersection| / |reference|, averaged across queries.
 *
 * @param {string[][]} reference
 * @param {string[][]} candidate
 * @returns {number}
 */
function recallAt10(reference, candidate) {
  let sum = 0
  for (let q = 0; q < reference.length; q += 1) {
    const ref = new Set(reference[q])
    let hits = 0
    for (const id of candidate[q]) if (ref.has(id)) hits += 1
    sum += hits / ref.size
  }
  return sum / reference.length
}

/**
 * @param {{ label: string, avgMs: number, avgBytes: number, avgFetches: number }} r
 * @returns {string}
 */
function fmt(r) {
  const throughput = meta.count / (r.avgMs / 1000)
  return `${r.label.padEnd(22)} ${r.avgMs.toFixed(1).padStart(7)} ms  ${r.avgFetches.toFixed(1).padStart(5)} fetches  ${(r.avgBytes / 1e6).toFixed(2).padStart(7)} MB read  (${throughput.toLocaleString(undefined, { maximumFractionDigits: 0 })} vec/s)`
}

console.log(fmt(exact))
if (rerank) {
  console.log(fmt(rerank))
  const recall = recallAt10(exact.tops, rerank.tops)
  console.log(`\nRecall@10 (rerank vs exact): ${(recall * 100).toFixed(1)}%`)
  console.log(`Speedup:                     ${(exact.avgMs / rerank.avgMs).toFixed(2)}× faster`)
  console.log(`Bytes read:                  ${(rerank.avgBytes / exact.avgBytes * 100).toFixed(1)}% of exact`)
}
