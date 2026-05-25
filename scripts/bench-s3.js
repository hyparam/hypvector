/**
 * Benchmark search against the file hosted on s3.hyperparam.app (CloudFront in
 * front of the hyperparam-public bucket). Each query starts with a fresh
 * AsyncBuffer so the per-query result is a true cold-client number — what a
 * first-time visitor of a static site would see.
 *
 *   node scripts/bench-s3.js [url] [queries]
 *
 * Defaults: https://s3.hyperparam.app/hypvector/wiki_en.vectors.parquet, 10 queries
 */
// Force IPv4: this network's path to CloudFront's IPv6 announces but drops packets.
import dns from 'node:dns'
const origLookup = dns.lookup
// @ts-expect-error overload signature
dns.lookup = (hostname, opts, cb) => {
  if (typeof opts === 'function') { cb = opts; opts = {} }
  return origLookup.call(dns, hostname, { ...(opts ?? {}), family: 4 }, cb)
}

import { asyncBufferFromFile, asyncBufferFromUrl, cachedAsyncBuffer, parquetMetadataAsync } from 'hyparquet'
import { prefetchBinary } from '../src/prefetch.js'
import { readVectors } from '../src/readVectors.js'
import { searchVectors } from '../src/searchVectors.js'
import { parseKvMetadata } from '../src/utils.js'

const URL = process.argv[2] ?? 'https://s3.hyperparam.app/hypvector/wiki_en.vectors.parquet'
const N_QUERIES = parseInt(process.argv[3] ?? '10', 10)
const LOCAL_SEED = 'data/wiki_en.vectors.clustered.parquet'

console.log(`URL:     ${URL}`)
console.log(`Queries: ${N_QUERIES} (cold client each)`)

// Seed query vectors from the local file (don't pay HTTP cost for setup).
const seed = await asyncBufferFromFile(LOCAL_SEED)
const seedMeta = await parquetMetadataAsync(seed)
const meta = parseKvMetadata(seedMeta)
console.log(`File:    ${meta.count.toLocaleString()} × ${meta.dimension}-dim, clusters=${meta.clusters}`)

const queries = []
const step = Math.max(1, Math.floor(meta.count / (N_QUERIES + 1)))
let nextPick = step; let i = 0
for await (const r of readVectors({ file: seed, metadata: seedMeta })) {
  if (i === nextPick) { queries.push(r.vector); nextPick += step; if (queries.length >= N_QUERIES) break }
  i += 1
}

/**
 * @param {import('hyparquet').AsyncBuffer} buf
 */
function instrument(buf) {
  const origSlice = buf.slice.bind(buf)
  const w = {
    byteLength: buf.byteLength, bytes: 0, fetches: 0,
    slice(start, end) {
      w.bytes += (end ?? buf.byteLength) - start
      w.fetches += 1
      return origSlice(start, end)
    },
  }
  return w
}

/**
 * @param {string} label
 * @param {object} opts
 */
async function suite(label, opts) {
  const times = []; const bytesPer = []; const fetchesPer = []
  // Fetch fresh metadata from the URL once, since hosted services usually
  // count this in their cold number too. Then re-parse per query is cheap.
  for (const q of queries) {
    const raw = instrument(await asyncBufferFromUrl({ url: URL }))
    const cached = cachedAsyncBuffer(raw)
    const start = performance.now()
    await searchVectors({ source: cached, query: q, topK: 10, ...opts })
    times.push(performance.now() - start)
    bytesPer.push(raw.bytes)
    fetchesPer.push(raw.fetches)
  }
  times.sort((a, b) => a - b)
  const p = (q) => times[Math.min(times.length - 1, Math.floor(q * times.length))]
  const avg = times.reduce((a, b) => a + b, 0) / times.length
  const avgBytes = bytesPer.reduce((a, b) => a + b, 0) / bytesPer.length
  const avgFetches = fetchesPer.reduce((a, b) => a + b, 0) / fetchesPer.length
  console.log(
    `${label.padEnd(22)} ` +
    `p50=${p(0.5).toFixed(0).padStart(5)}ms  ` +
    `p90=${p(0.9).toFixed(0).padStart(5)}ms  ` +
    `avg=${avg.toFixed(0).padStart(5)}ms  ` +
    `${avgFetches.toFixed(0).padStart(4)} fetches  ` +
    `${(avgBytes / 1e6).toFixed(2).padStart(6)} MB`
  )
}

console.log()
await suite('Exact full scan', { rerankFactor: 0 })
await suite('Binary + rerank', { rerankFactor: 10 })

// Prefetch binary once over the real URL, then run the rerank suite reusing
// it. Simulates a long-lived client (browser session) where the binary
// column is loaded at app start.
const prefetchStart = performance.now()
const prefetchRaw = await asyncBufferFromUrl({ url: URL })
const prefetchCached = cachedAsyncBuffer(prefetchRaw)
const binary = await prefetchBinary({ source: prefetchCached })
const prefetchMs = performance.now() - prefetchStart
console.log(`\nprefetched binary in ${prefetchMs.toFixed(0)} ms (${(binary.byteLength / 1e6).toFixed(2)} MB)`)
await suite('+ prefetched binary', { rerankFactor: 10, binary })
