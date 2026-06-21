/**
 * Validation harness for the still-open experiments in PLAN_AUTO.md.
 *
 * Subcommands:
 *
 *   recall <src> [N]   — clusters sweep reporting BOTH recall@10 and recall@100,
 *                        to discriminate the ~94% recall@10 ceiling on LLM logs.
 *   smalln <src>       — binary-column crossover: for small N, compare file size
 *                        and search latency/recall of binary-rerank vs exact scan.
 *   scale <src> <Ns>   — clusters sweep at one or more N subsets (comma-separated),
 *                        to confirm the sqrt(N)/2 latency optimum holds across sizes.
 *
 * All files are written under data/_vp_*.parquet and reused if present.
 *
 * Usage:
 *   node scripts/validate-params.js recall data/llmlog.vectors.parquet
 *   node scripts/validate-params.js smalln data/llmlog.vectors.parquet
 *   node scripts/validate-params.js scale  data/tpuf-bench-1000k.parquet 250000,500000,1000000
 */
import { promises as fs } from 'node:fs'
import { asyncBufferFromFile, cachedAsyncBuffer, parquetMetadataAsync } from 'hyparquet'
import { fileWriter } from 'hyparquet-writer'
import { readVectors } from '../src/readVectors.js'
import { searchVectors } from '../src/searchVectors.js'
import { parseKvMetadata } from '../src/utils.js'
import { writeVectors } from '../src/writeVectors.js'

/** @import { AsyncBuffer } from 'hyparquet' */

const MODE = process.argv[2]
const SRC = process.argv[3]
const ARG = process.argv[4]
const QUERY_COUNT = 20

if (!MODE || !SRC) {
  console.error('Usage: node scripts/validate-params.js <recall|smalln|scale> <src> [arg]')
  process.exit(1)
}

/**
 * Read up to `limit` records from a vectors parquet into memory.
 * @param {string} src
 * @param {number} [limit]
 * @returns {Promise<{ records: { id: string, vector: Float32Array }[], meta: any }>}
 */
async function loadRecords(src, limit) {
  const file = await asyncBufferFromFile(src)
  const metadata = await parquetMetadataAsync(file)
  const meta = parseKvMetadata(metadata)
  const records = []
  for await (const record of readVectors({ file, metadata, includeMetadata: false })) {
    records.push(record)
    if (limit && records.length >= limit) break
  }
  return { records, meta }
}

/**
 * Pick evenly spaced query vectors from the corpus.
 * @param {{ vector: Float32Array }[]} records
 * @param {number} count
 * @returns {Float32Array[]}
 */
function pickQueries(records, count) {
  const queries = []
  const step = Math.max(1, Math.floor(records.length / (count + 1)))
  for (let i = 0, pick = step; i < records.length && queries.length < count; i += 1) {
    if (i === pick) { queries.push(records[i].vector); pick += step }
  }
  return queries
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

function avg(a) { let s = 0; for (const x of a) s += x; return s / a.length }

/**
 * Run a search over every query and collect timing + the returned id lists.
 * @param {string} path
 * @param {Float32Array[]} queries
 * @param {number} topK
 * @param {object} extra
 * @returns {Promise<{ ms: number, mb: number, fetches: number, tops: string[][] }>}
 */
async function bench(path, queries, topK, extra) {
  const times = [], bytesA = [], fetchesA = [], tops = []
  for (const q of queries) {
    const raw = instrument(await asyncBufferFromFile(path))
    const cached = cachedAsyncBuffer(raw)
    const start = performance.now()
    const r = await searchVectors({ source: cached, query: q, topK, ...extra })
    times.push(performance.now() - start)
    bytesA.push(raw.bytes); fetchesA.push(raw.fetches); tops.push(r.map(x => String(x.id)))
  }
  return { ms: avg(times), mb: avg(bytesA) / 1e6, fetches: avg(fetchesA), tops }
}

/**
 * Recall of `tops` against reference `refTops`, truncating both to `k`.
 * @param {string[][]} refTops
 * @param {string[][]} tops
 * @param {number} k
 * @returns {number}
 */
function recallAt(refTops, tops, k) {
  let hits = 0, total = 0
  for (let i = 0; i < refTops.length; i += 1) {
    const refSet = new Set(refTops[i].slice(0, k))
    for (const id of tops[i].slice(0, k)) if (refSet.has(id)) hits += 1
    total += refSet.size
  }
  return hits / total
}

/**
 * Write a clustered+binary file for a given cluster count (idempotent).
 * @param {string} tag
 * @param {{ id: string, vector: Float32Array }[]} records
 * @param {any} meta
 * @param {number} clusters
 * @param {boolean} binary
 * @returns {Promise<string>}
 */
async function writeVariant(tag, records, meta, clusters, binary) {
  const path = `data/_vp_${tag}.parquet`
  if (await fs.stat(path).catch(() => undefined)) return path
  const start = performance.now()
  await writeVectors({
    writer: fileWriter(path),
    dimension: meta.dimension,
    metric: meta.metric,
    normalize: meta.normalized,
    vectors: records,
    binary,
    clusters,
  })
  console.log(`  wrote ${path} (clusters=${clusters}, binary=${binary}) in ${((performance.now() - start) / 1000).toFixed(1)}s`)
  return path
}

// --- recall@10 + recall@100 sweep ---------------------------------------
async function runRecall() {
  const limit = ARG ? Number(ARG) : undefined
  const { records, meta } = await loadRecords(SRC, limit)
  const N = records.length
  const sqrtN = Math.round(Math.sqrt(N))
  const queries = pickQueries(records, QUERY_COUNT)
  console.log(`recall: ${SRC} N=${N.toLocaleString()} dim=${meta.dimension} sqrtN=${sqrtN}`)

  const clusterValues = [0, Math.round(sqrtN / 2), sqrtN, 2 * sqrtN]
  const base = `${SRC.replace(/\.parquet$/, '').split('/').pop()}_N${N}`
  const paths = {}
  for (const c of clusterValues) paths[c] = await writeVariant(`${base}_c${c}`, records, meta, c, true)

  // Reference: exact full scan, top-100.
  console.log('Reference: exact top-100 full scan...')
  const ref = await bench(paths[0], queries, 100, { rerankFactor: 0 })

  console.log('\n=== clusters sweep, recall@10 vs recall@100 (probe/rerank default) ===')
  console.log(`${'clusters'.padStart(10)} ${'ms'.padStart(7)} ${'fetches'.padStart(8)} ${'MB read'.padStart(9)} ${'r@10'.padStart(7)} ${'r@100'.padStart(7)}`)
  console.log('-'.repeat(58))
  for (const c of clusterValues) {
    const opts = c === 0 ? { rerankFactor: 10 } : {}
    const r = await bench(paths[c], queries, 100, opts)
    const r10 = recallAt(ref.tops, r.tops, 10)
    const r100 = recallAt(ref.tops, r.tops, 100)
    console.log(`${String(c).padStart(10)} ${r.ms.toFixed(1).padStart(7)} ${r.fetches.toFixed(0).padStart(8)} ${r.mb.toFixed(2).padStart(9)} ${(r10 * 100).toFixed(1).padStart(6)}% ${(r100 * 100).toFixed(1).padStart(6)}%`)
  }
}

// --- small-N binary crossover -------------------------------------------
async function runSmallN() {
  const sizes = (ARG ?? '500,1000,2000,5000,10000,20000').split(',').map(Number)
  const maxN = Math.max(...sizes)
  const { records: all, meta } = await loadRecords(SRC, maxN)
  console.log(`smalln: ${SRC} dim=${meta.dimension}, sizes=${sizes.join(',')}`)
  console.log(`\n${'N'.padStart(7)} ${'noBin MB'.padStart(9)} ${'bin MB'.padStart(8)} ${'+%'.padStart(6)} ${'exact ms'.padStart(9)} ${'rerank ms'.padStart(10)} ${'speedup'.padStart(8)} ${'recall'.padStart(7)}`)
  console.log('-'.repeat(74))
  for (const N of sizes) {
    const records = all.slice(0, N)
    const queries = pickQueries(records, Math.min(QUERY_COUNT, N))
    const tag = `${SRC.replace(/\.parquet$/, '').split('/').pop()}_sn${N}`
    // No-binary file (exact scan only) and binary file (no clusters, rerank path).
    const exactPath = await writeVariant(`${tag}_nobin`, records, meta, 0, false)
    const binPath = await writeVariant(`${tag}_bin`, records, meta, 0, true)
    const exactSize = (await fs.stat(exactPath)).size
    const binSize = (await fs.stat(binPath)).size
    // Reference = exact top-10 on the no-binary file.
    const ref = await bench(exactPath, queries, 10, { rerankFactor: 0 })
    const rerank = await bench(binPath, queries, 10, {})
    const recall = recallAt(ref.tops, rerank.tops, 10)
    const pct = (binSize - exactSize) / exactSize * 100
    const speedup = ref.ms / rerank.ms
    console.log(`${String(N).padStart(7)} ${(exactSize / 1e6).toFixed(2).padStart(9)} ${(binSize / 1e6).toFixed(2).padStart(8)} ${pct.toFixed(1).padStart(5)}% ${ref.ms.toFixed(2).padStart(9)} ${rerank.ms.toFixed(2).padStart(10)} ${speedup.toFixed(2).padStart(7)}x ${(recall * 100).toFixed(1).padStart(6)}%`)
  }
}

// --- clusters sweep at scale --------------------------------------------
async function runScale() {
  const Ns = (ARG ?? '').split(',').filter(Boolean).map(Number)
  if (!Ns.length) { console.error('scale needs comma-separated N list'); process.exit(1) }
  const maxN = Math.max(...Ns)
  console.log(`scale: loading up to ${maxN.toLocaleString()} from ${SRC}...`)
  const { records: all, meta } = await loadRecords(SRC, maxN)
  console.log(`  loaded ${all.length.toLocaleString()} × ${meta.dimension}-dim`)
  for (const N of Ns) {
    const records = all.slice(0, N)
    const sqrtN = Math.round(Math.sqrt(N))
    const queries = pickQueries(records, QUERY_COUNT)
    const clusterValues = [Math.round(sqrtN / 2), sqrtN, 2 * sqrtN]
    const base = `${SRC.replace(/\.parquet$/, '').split('/').pop()}_sc${N}`
    console.log(`\n=== N=${N.toLocaleString()} (sqrtN=${sqrtN}) ===`)
    const paths = {}
    // c=0 reference file (binary, no clusters) for exact top-10.
    const refPath = await writeVariant(`${base}_c0`, records, meta, 0, true)
    for (const c of clusterValues) paths[c] = await writeVariant(`${base}_c${c}`, records, meta, c, true)
    const ref = await bench(refPath, queries, 10, { rerankFactor: 0 })
    console.log(`${'clusters'.padStart(10)} ${'ms'.padStart(7)} ${'fetches'.padStart(8)} ${'MB read'.padStart(9)} ${'recall'.padStart(8)}`)
    console.log('-'.repeat(50))
    for (const c of clusterValues) {
      const r = await bench(paths[c], queries, 10, {})
      const rec = recallAt(ref.tops, r.tops, 10)
      const label = c === Math.round(sqrtN / 2) ? `${c} (√N/2)` : c === sqrtN ? `${c} (√N)` : `${c} (2√N)`
      console.log(`${label.padStart(10)} ${r.ms.toFixed(1).padStart(7)} ${r.fetches.toFixed(0).padStart(8)} ${r.mb.toFixed(2).padStart(9)} ${(rec * 100).toFixed(1).padStart(7)}%`)
    }
  }
}

if (MODE === 'recall') await runRecall()
else if (MODE === 'smalln') await runSmallN()
else if (MODE === 'scale') await runScale()
else { console.error(`unknown mode: ${MODE}`); process.exit(1) }
