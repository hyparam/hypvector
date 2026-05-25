/**
 * Benchmark search against a localhost HTTP server with an artificial
 * per-request delay, simulating cloud RTT. The point is to compare
 * "many tiny fetches" vs "fewer larger fetches" under realistic
 * round-trip costs.
 *
 *   node scripts/bench-http.js [file] [rttMs]
 *
 * Defaults: data/wiki_en.vectors.clustered.parquet  rtt=20ms
 */
import { promises as fs } from 'node:fs'
import { createReadStream } from 'node:fs'
import http from 'node:http'
import { asyncBufferFromUrl, cachedAsyncBuffer, parquetMetadataAsync } from 'hyparquet'
import { readVectors } from '../src/readVectors.js'
import { searchVectors } from '../src/searchVectors.js'
import { parseKvMetadata } from '../src/utils.js'

const FILE = process.argv[2] ?? 'data/wiki_en.vectors.clustered.parquet'
const RTT_MS = parseInt(process.argv[3] ?? '20', 10)
const PORT = 8765

const stat = await fs.stat(FILE)
console.log(`Serving ${FILE} (${(stat.size / 1e6).toFixed(1)} MB) with +${RTT_MS} ms per request`)

// Tiny range-supporting HTTP server.
const server = http.createServer((req, res) => {
  setTimeout(() => {
    if (req.method === 'HEAD') {
      res.writeHead(200, { 'content-length': stat.size, 'accept-ranges': 'bytes' })
      res.end()
      return
    }
    const range = req.headers.range
    if (range) {
      const m = /bytes=(\d+)-(\d+)?/.exec(range)
      if (!m) { res.writeHead(416); res.end(); return }
      const start = parseInt(m[1], 10)
      const end = m[2] ? parseInt(m[2], 10) : stat.size - 1
      res.writeHead(206, {
        'content-type': 'application/octet-stream',
        'content-range': `bytes ${start}-${end}/${stat.size}`,
        'content-length': end - start + 1,
        'accept-ranges': 'bytes',
      })
      createReadStream(FILE, { start, end }).pipe(res)
    } else {
      res.writeHead(200, { 'content-length': stat.size })
      createReadStream(FILE).pipe(res)
    }
  }, RTT_MS)
})
await new Promise(r => server.listen(PORT, r))
const url = `http://127.0.0.1:${PORT}/file.parquet`

try {
  // Use the local file for getting queries (don't pay HTTP cost just to seed).
  const seedFile = await asyncBufferFromUrl({ url })
  const metadata = await parquetMetadataAsync(seedFile)
  const meta = parseKvMetadata(metadata)
  console.log(`Vectors: ${meta.count} × ${meta.dimension}; clusters: ${meta.clusters}`)

  const queries = []
  const step = Math.max(1, Math.floor(meta.count / 6))
  let i = 0
  let nextPick = step
  for await (const r of readVectors({ file: seedFile, metadata })) {
    if (i === nextPick) { queries.push(r.vector); nextPick += step; if (queries.length >= 5) break }
    i += 1
  }

  async function bench(opts) {
    const times = []
    for (const q of queries) {
      const buf = await asyncBufferFromUrl({ url })
      const cached = cachedAsyncBuffer(buf)
      const start = performance.now()
      await searchVectors({ source: cached, metadata, query: q, topK: 10, ...opts })
      times.push(performance.now() - start)
    }
    let sum = 0
    for (let i = 0; i < times.length; i += 1) sum += times[i]
    return sum / times.length
  }

  const exact = await bench({ rerankFactor: 0 })
  const probe05 = await bench({ probe: 0.5 })
  const probe25 = await bench({ probe: 0.25 })
  const probe1 = await bench({ probe: 0.1 })

  console.log(`\nRTT=${RTT_MS}ms per request`)
  console.log(`Exact full scan:      ${exact.toFixed(0).padStart(6)} ms/query`)
  console.log(`Rerank probe=0.5:     ${probe05.toFixed(0).padStart(6)} ms/query`)
  console.log(`Rerank probe=0.25:    ${probe25.toFixed(0).padStart(6)} ms/query  (default)`)
  console.log(`Rerank probe=0.1:     ${probe1.toFixed(0).padStart(6)} ms/query`)
} finally {
  server.close()
}
