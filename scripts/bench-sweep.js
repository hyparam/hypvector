/**
 * Sweep probe values + cluster counts against a clustered hypvector file
 * and report bytes / fetches / recall for each. Tests how aggressively we
 * can prune phase-1 scanning before recall collapses.
 */
import { promises as fs } from 'node:fs'
import { asyncBufferFromFile, cachedAsyncBuffer, parquetMetadataAsync } from 'hyparquet'
import { readVectors } from '../src/readVectors.js'
import { searchVectors } from '../src/searchVectors.js'
import { parseKvMetadata } from '../src/utils.js'

/**
 * @import { AsyncBuffer } from 'hyparquet'
 */

const filename = process.argv[2] ?? 'data/wiki_en.vectors.clustered.parquet'

const stat = await fs.stat(filename)
const sourceFile = await asyncBufferFromFile(filename)
const metadata = await parquetMetadataAsync(sourceFile)
const meta = parseKvMetadata(metadata)
console.log(`File: ${filename}`)
console.log(`Vectors: ${meta.count.toLocaleString()} × ${meta.dimension}-dim`)
console.log(`Size: ${(stat.size / 1e6).toFixed(2)} MB; clusters: ${meta.clusters}`)

const QUERY_COUNT = 10
const queries = []
const step = Math.max(1, Math.floor(meta.count / (QUERY_COUNT + 1)))
let i = 0
let nextPick = step
for await (const record of readVectors({ file: sourceFile, metadata })) {
  if (i === nextPick) {
    queries.push(record.vector)
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
  }
  return wrapped
}

/**
 * @param {{ probe?: number, rerankFactor?: number, label: string }} opts
 * @returns {Promise<{ label: string, avgMs: number, avgBytes: number, avgFetches: number, tops: string[][] }>}
 */
async function run({ label, ...opts }) {
  const times = []
  const bytesPer = []
  const fetchesPer = []
  const tops = []
  for (const q of queries) {
    const raw = instrument(await asyncBufferFromFile(filename))
    const cached = cachedAsyncBuffer(raw)
    const start = performance.now()
    const results = await searchVectors({
      source: cached, metadata, query: q, topK: 10, ...opts,
    })
    times.push(performance.now() - start)
    bytesPer.push(raw.bytes)
    fetchesPer.push(raw.fetches)
    tops.push(results.map(r => String(r.id)))
  }
  return {
    label,
    avgMs: avg(times),
    avgBytes: avg(bytesPer),
    avgFetches: avg(fetchesPer),
    tops,
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

const exact = await run({ label: 'Exact', rerankFactor: 0 })
console.log(`\n${'config'.padEnd(28)} ${'ms'.padStart(7)} ${'fetches'.padStart(8)} ${'MB read'.padStart(10)} ${'% bytes'.padStart(9)} ${'recall'.padStart(8)}`)
console.log('-'.repeat(75))
console.log(`${exact.label.padEnd(28)} ${exact.avgMs.toFixed(1).padStart(7)} ${exact.avgFetches.toFixed(0).padStart(8)} ${(exact.avgBytes / 1e6).toFixed(2).padStart(10)} ${'100.0'.padStart(8)}% ${'—'.padStart(8)}`)

const configs = [
  { label: 'probe=1.0  rrx10', rerankFactor: 10, probe: 1.0 },
  { label: 'probe=0.5  rrx10', rerankFactor: 10, probe: 0.5 },
  { label: 'probe=0.25 rrx10', rerankFactor: 10, probe: 0.25 },
  { label: 'probe=0.1  rrx10', rerankFactor: 10, probe: 0.1 },
  { label: 'probe=0.05 rrx10', rerankFactor: 10, probe: 0.05 },
  { label: 'probe=0.25 rrx30', rerankFactor: 30, probe: 0.25 },
  { label: 'probe=0.1  rrx30', rerankFactor: 30, probe: 0.1 },
  { label: 'probe=0.1  rrx50', rerankFactor: 50, probe: 0.1 },
  { label: 'probe=0.05 rrx50', rerankFactor: 50, probe: 0.05 },
  { label: 'probe=0.05 rrx100', rerankFactor: 100, probe: 0.05 },
]

for (const cfg of configs) {
  const r = await run(cfg)
  let hits = 0; let total = 0
  for (let q = 0; q < exact.tops.length; q += 1) {
    const ref = new Set(exact.tops[q])
    for (const id of r.tops[q]) if (ref.has(id)) hits += 1
    total += ref.size
  }
  const recall = hits / total
  console.log(`${r.label.padEnd(28)} ${r.avgMs.toFixed(1).padStart(7)} ${r.avgFetches.toFixed(0).padStart(8)} ${(r.avgBytes / 1e6).toFixed(2).padStart(10)} ${(r.avgBytes / exact.avgBytes * 100).toFixed(1).padStart(8)}% ${(recall * 100).toFixed(1).padStart(7)}%`)
}
