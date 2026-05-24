import { promises as fs } from 'node:fs'
import { asyncBufferFromFile } from 'hyparquet'
import { fileWriter } from 'hyparquet-writer'
import { searchVectors } from './src/searchVectors.js'
import { writeVectors } from './src/writeVectors.js'

const FILENAME = 'bench.parquet'
const COUNT = 50000
const DIMENSION = 384

/**
 * Deterministic pseudo-random vector generator.
 *
 * @param {number} count
 * @param {number} dimension
 * @returns {Generator<{id: string, vector: Float32Array}>}
 */
function* makeVectors(count, dimension) {
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

// Write benchmark
let stat = await fs.stat(FILENAME).catch(() => undefined)
if (!stat) {
  console.log(`=== Writing ${COUNT.toLocaleString()} vectors of dim ${DIMENSION} ===`)
  const writeStart = performance.now()
  const writer = fileWriter(FILENAME)
  await writeVectors({
    writer,
    dimension: DIMENSION,
    vectors: makeVectors(COUNT, DIMENSION),
  })
  stat = await fs.stat(FILENAME)
  const writeMs = performance.now() - writeStart
  console.log(`Wrote ${FILENAME} in ${writeMs.toFixed(0)} ms`)
}

const rawSize = COUNT * DIMENSION * 4
console.log(`File size: ${stat.size.toLocaleString()} bytes`)
console.log(`Raw float32 size: ${rawSize.toLocaleString()} bytes`)
console.log(`Overhead: ${(stat.size / rawSize * 100).toFixed(1)}% of raw`)

// Search benchmark
console.log('\n=== Search (linear scan) ===')
const sourceFile = await asyncBufferFromFile(FILENAME)
sourceFile.slice = (orig => function (start, end) {
  sourceFile.bytesRead += (end ?? sourceFile.byteLength) - start
  sourceFile.fetches += 1
  return orig.call(sourceFile, start, end)
})(sourceFile.slice.bind(sourceFile))

const queries = 5
const times = []
for (let i = 0; i < queries; i += 1) {
  sourceFile.bytesRead = 0
  sourceFile.fetches = 0
  const query = new Float32Array(DIMENSION)
  for (let j = 0; j < DIMENSION; j += 1) query[j] = Math.random() * 2 - 1
  const start = performance.now()
  const results = await searchVectors({ url: FILENAME, query, topK: 10, sourceFile })
  const ms = performance.now() - start
  times.push(ms)
  console.log(`Query ${i + 1}: ${ms.toFixed(0)} ms, ${sourceFile.fetches} fetches, ${sourceFile.bytesRead.toLocaleString()} bytes read, top score=${results[0].score.toFixed(4)}`)
}

const avg = times.reduce((s, t) => s + t, 0) / times.length
console.log(`\nAverage query: ${avg.toFixed(0)} ms`)
