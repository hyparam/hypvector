/**
 * Apples-to-apples benchmark against the turbopuffer `vector-10m` workload
 * spec, scaled to 1M Cohere Wikipedia embeddings.
 *
 *   Dataset:  Cohere/wikipedia-2023-11-embed-multilingual-v3 (1M × 1024-dim)
 *   Metric:   cosine
 *   topK:     10
 *   Queries:  random unit vectors (NOT from dataset, matches tpuf)
 *
 * Cold mode: fresh AsyncBuffer per query (no metadata cache, no binary
 *   prefetch). Mirrors tpuf's `disable_cache: true`.
 * Warm mode: single client; metadata parsed once and binary prefetched once
 *   before any query is scored. Mirrors tpuf's warm-cache namespace.
 *
 * Recall@10 is also reported (tpuf reports latency only) so the config we
 * pick can't cheat by trading recall for speed.
 *
 *   node --max-old-space-size=8192 scripts/bench-tpuf.js [url] [local-file] [n-cold] [n-warm]
 *
 * Defaults:
 *   url        https://s3.hyperparam.app/tpuf-bench/tpuf-bench-1m.parquet
 *   local      data/tpuf-bench-1000k.parquet
 *   n-cold     30   (each cold query is a fresh CloudFront fetch — keep small)
 *   n-warm     200  (per-config; sampled over the entire sweep)
 */
// Force IPv4: home network advertises IPv6 to CloudFront but drops packets.
import dns from 'node:dns'
const origLookup = dns.lookup
// @ts-expect-error overload signature
dns.lookup = (hostname, opts, cb) => {
  if (typeof opts === 'function') { cb = opts; opts = {} }
  return origLookup.call(dns, hostname, { ...opts ?? {}, family: 4 }, cb)
}

import {
  asyncBufferFromFile, asyncBufferFromUrl, cachedAsyncBuffer, parquetMetadataAsync, parquetRead,
} from 'hyparquet'
import { defaultVectorColumn } from '../src/constants.js'
import { prefetchBinary } from '../src/prefetch.js'
import { searchVectors } from '../src/searchVectors.js'
import { dotProduct, l2Normalize, parseKvMetadata, unpackFloat32 } from '../src/utils.js'

const URL = process.argv[2] ?? 'https://s3.hyperparam.app/tpuf-bench/tpuf-bench-1m.parquet'
const LOCAL = process.argv[3] ?? 'data/tpuf-bench-1000k.parquet'
const N_COLD = parseInt(process.argv[4] ?? '30', 10)
const N_WARM = parseInt(process.argv[5] ?? '200', 10)
const TOP_K = 10

// Configurations to sweep. Each is run in both cold and warm modes so we can
// see the recall/latency frontier and pick a fair "apples-to-apples" pick.
const CONFIGS = [
  { name: 'rerank=10  probe=0.10', rerankFactor: 10, probe: 0.10 },
  { name: 'rerank=10  probe=0.25', rerankFactor: 10, probe: 0.25 },
  { name: 'rerank=100 probe=0.10', rerankFactor: 100, probe: 0.10 },
  { name: 'rerank=100 probe=0.25', rerankFactor: 100, probe: 0.25 },
]

console.log(`URL:    ${URL}`)
console.log(`Local:  ${LOCAL}`)
console.log(`Queries: cold=${N_COLD}/config, warm=${N_WARM}/config, topK=${TOP_K}`)

const local = await asyncBufferFromFile(LOCAL)
const localMeta = await parquetMetadataAsync(local)
const hv = parseKvMetadata(localMeta)
console.log(`File:    ${hv.count.toLocaleString()} × ${hv.dimension}-dim, metric=${hv.metric}, clusters=${hv.clusters}`)

// Random unit query vectors — matches tpuf's pseudorandom template.
let lcg = 1
function rand() {
  lcg = Math.imul(lcg, 1664525) + 1013904223 >>> 0
  return lcg / 0x100000000
}
function randomUnit(dim) {
  const v = new Float32Array(dim)
  for (let i = 0; i < dim; i += 1) v[i] = rand() * 2 - 1
  return l2Normalize(v)
}
const nQueries = Math.max(N_COLD, N_WARM)
console.log(`\nGenerating ${nQueries} random ${hv.dimension}-dim unit queries...`)
const queries = Array.from({ length: nQueries }, () => randomUnit(hv.dimension))

// Brute-force ground truth — heavy at this scale, so we limit to the cold
// sample size. Warm reuses ground truth for queries < N_COLD; queries beyond
// that count only toward latency, not recall (still statistically valid).
const N_GT = Math.min(N_COLD, nQueries)
console.log(`Computing brute-force ground truth for ${N_GT} queries (~50s)...`)
const tGT = performance.now()
/** @type {Array<{score: number, idx: number}[]>} */
const groundTruthHeaps = Array.from({ length: N_GT }, () => [])
let rowsScored = 0
await parquetRead({
  file: local,
  metadata: localMeta,
  columns: [defaultVectorColumn],
  onChunk: ({ columnData, rowStart }) => {
    for (let r = 0; r < columnData.length; r += 1) {
      const vec = unpackFloat32(columnData[r])
      const rowIdx = rowStart + r
      for (let q = 0; q < N_GT; q += 1) {
        const score = dotProduct(queries[q], vec)
        const heap = groundTruthHeaps[q]
        if (heap.length < TOP_K) {
          heap.push({ score, idx: rowIdx })
          if (heap.length === TOP_K) heap.sort((a, b) => a.score - b.score)
        } else if (score > heap[0].score) {
          heap[0] = { score, idx: rowIdx }
          heap.sort((a, b) => a.score - b.score)
        }
      }
    }
    rowsScored += columnData.length
    if (rowsScored % 50000 < columnData.length) {
      process.stdout.write(`\r  ${rowsScored.toLocaleString()} rows scored `)
    }
  },
})
process.stdout.write('\n')
const groundTruth = groundTruthHeaps.map(h => new Set(h.map(e => e.idx)))
console.log(`  ground truth in ${((performance.now() - tGT) / 1000).toFixed(1)}s`)

/**
 * @param {{rowIndex: number}[]} results
 * @param {Set<number>} truth
 * @returns {number}
 */
function recallAt(results, truth) {
  let matches = 0
  for (const r of results) if (truth.has(r.rowIndex)) matches += 1
  return matches / truth.size
}

/**
 * @param {number[]} arr sorted ascending
 * @param {number} q quantile in [0,1]
 * @returns {number}
 */
function pct(arr, q) { return arr[Math.min(arr.length - 1, Math.floor(q * arr.length))] }

/**
 * @param {string} label
 * @param {number[]} times
 * @param {number[]} recalls
 */
function printRow(label, times, recalls) {
  times.sort((a, b) => a - b)
  const avgRecall = recalls.length ? recalls.reduce((a, b) => a + b, 0) / recalls.length : NaN
  const recallStr = Number.isFinite(avgRecall) ? `${(avgRecall * 100).toFixed(1).padStart(5)}%` : ' n/a'
  console.log(
    `  ${label.padEnd(28)} ` +
    `p50=${pct(times, 0.5).toFixed(0).padStart(5)}ms  ` +
    `p90=${pct(times, 0.9).toFixed(0).padStart(5)}ms  ` +
    `p99=${pct(times, 0.99).toFixed(0).padStart(5)}ms  ` +
    `recall(n=${String(recalls.length).padStart(3)})=${recallStr}`
  )
}

// ── COLD: fresh AsyncBuffer per query, per config ────────────────────────
console.log('\n=== COLD (fresh CloudFront client per query) ===')
for (const cfg of CONFIGS) {
  const times = []
  const recalls = []
  for (let q = 0; q < N_COLD; q += 1) {
    const raw = await asyncBufferFromUrl({ url: URL })
    const cached = cachedAsyncBuffer(raw)
    const t0 = performance.now()
    const hits = await searchVectors({
      source: cached, query: queries[q], topK: TOP_K,
      rerankFactor: cfg.rerankFactor, probe: cfg.probe,
    })
    times.push(performance.now() - t0)
    recalls.push(recallAt(hits, groundTruth[q]))
    process.stdout.write(`\r  ${cfg.name}: ${q + 1}/${N_COLD} `)
  }
  process.stdout.write('\n')
  printRow(cfg.name, times, recalls)
}

// ── WARM: single client + prefetched binary, sweep configs ──────────────
console.log('\n=== WARM (single client + prefetched binary) ===')
const warmRaw = await asyncBufferFromUrl({ url: URL })
const warmCached = cachedAsyncBuffer(warmRaw)
const warmMeta = await parquetMetadataAsync(warmCached)
const tPrefetch = performance.now()
const warmBin = await prefetchBinary({ source: warmCached, metadata: warmMeta })
console.log(`  setup: metadata + ${(warmBin.byteLength / 1e6).toFixed(0)} MB binary in ${((performance.now() - tPrefetch) / 1000).toFixed(2)}s`)
for (const cfg of CONFIGS) {
  const times = []
  const recalls = []
  for (let q = 0; q < N_WARM; q += 1) {
    const t0 = performance.now()
    const hits = await searchVectors({
      source: warmCached, metadata: warmMeta, binary: warmBin,
      query: queries[q], topK: TOP_K,
      rerankFactor: cfg.rerankFactor, probe: cfg.probe,
    })
    times.push(performance.now() - t0)
    if (q < N_GT) recalls.push(recallAt(hits, groundTruth[q]))
    process.stdout.write(`\r  ${cfg.name}: ${q + 1}/${N_WARM} `)
  }
  process.stdout.write('\n')
  printRow(cfg.name, times, recalls)
}

console.log('\nReference (tpuf vector-1m, 768-dim, c2-standard-30 in GCP us-central1):')
console.log('  warm:  p50=  8ms  p90= 10ms  p99= 35ms')
console.log('  cold:  p50=343ms  p90=444ms  p99=554ms')
console.log('Reference (tpuf vector-10m, 1024-dim — for scale, 10× our N):')
console.log('  warm:  p50= 14ms  p90= 17ms  p99= 27ms')
console.log('  cold:  p50=874ms  p90=1214ms  p99=1686ms')
