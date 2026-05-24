import { promises as fs } from 'node:fs'
import { asyncBufferFromFile, parquetMetadataAsync } from 'hyparquet'
import { fileWriter } from 'hyparquet-writer'
import { readVectors } from './src/readVectors.js'
import { searchVectors } from './src/searchVectors.js'
import { parseKvMetadata } from './src/utils.js'
import { writeVectors } from './src/writeVectors.js'

const REAL_FILE = 'data/wiki_en.vectors.parquet'
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

// Instrument the source buffer to count bytes/fetches per query.
const fresh = await asyncBufferFromFile(filename)
let bytesRead = 0
let fetches = 0
const origSlice = fresh.slice.bind(fresh)
fresh.slice = function (start, end) {
  bytesRead += (end ?? fresh.byteLength) - start
  fetches += 1
  return origSlice(start, end)
}

console.log('\n=== Search (linear scan) ===')
const times = []
for (let q = 0; q < queries.length; q += 1) {
  bytesRead = 0
  fetches = 0
  const start = performance.now()
  const results = await searchVectors({
    url: filename,
    query: queries[q].vector,
    topK: 10,
    sourceFile: fresh,
    sourceMetadata: metadata,
  })
  const ms = performance.now() - start
  times.push(ms)
  const top = results[0]
  const note = String(top.id) === String(queries[q].id) ? ' (= query)' : ''
  console.log(`Query ${q + 1} (id=${queries[q].id}): ${ms.toFixed(0)} ms, ${fetches} fetches, ${bytesRead.toLocaleString()} bytes, top id=${top.id} score=${top.score.toFixed(4)}${note}`)
}

const avg = times.reduce((s, t) => s + t, 0) / times.length
const throughput = meta.count / (avg / 1000)
console.log(`\nAverage query: ${avg.toFixed(0)} ms (${throughput.toLocaleString(undefined, { maximumFractionDigits: 0 })} vectors/s)`)
