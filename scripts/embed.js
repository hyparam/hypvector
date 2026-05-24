import { promises as fs } from 'node:fs'
import { asyncBufferFromFile, parquetMetadataAsync, parquetReadObjects } from 'hyparquet'
import { fileWriter } from 'hyparquet-writer'
import { writeVectors } from '../src/writeVectors.js'

const SOURCE = 'data/wiki_en.parquet'
const DEST = 'data/wiki_en.vectors.parquet'
const MODEL = 'Xenova/all-MiniLM-L6-v2'
const DIMENSION = 384
const BATCH = 32
const LIMIT = parseInt(process.argv[2] ?? '10000', 10)
const TEXT_COLUMN = process.argv[3] ?? 'title'

const sourceStat = await fs.stat(SOURCE).catch(() => undefined)
if (!sourceStat) {
  console.error(`Missing ${SOURCE}. Run \`npm run data:download\` first.`)
  process.exit(1)
}

let transformers
try {
  transformers = await import('@huggingface/transformers')
} catch {
  console.error('Missing dependency. Install with:')
  console.error('  npm install --no-save @huggingface/transformers')
  process.exit(1)
}

console.log(`Loading embedding model: ${MODEL}`)
const extractor = await transformers.pipeline('feature-extraction', MODEL)

const file = await asyncBufferFromFile(SOURCE)
const metadata = await parquetMetadataAsync(file)
const totalRows = Number(metadata.num_rows)
const numRows = Math.min(totalRows, LIMIT)
console.log(`Source: ${SOURCE} (${totalRows.toLocaleString()} rows)`)
console.log(`Embedding column "${TEXT_COLUMN}" for ${numRows.toLocaleString()} rows in batches of ${BATCH}`)

/**
 * Stream embedded vectors batch-by-batch from the source parquet.
 *
 * @returns {AsyncGenerator<{ id: string, vector: Float32Array }>}
 */
async function* embedAll() {
  const start = performance.now()
  for (let i = 0; i < numRows; i += BATCH) {
    const end = Math.min(i + BATCH, numRows)
    const rows = await parquetReadObjects({
      file,
      metadata,
      rowStart: i,
      rowEnd: end,
      columns: [TEXT_COLUMN],
    })
    const texts = rows.map(r => String(r?.[TEXT_COLUMN] ?? ''))
    const output = await extractor(texts, { pooling: 'mean', normalize: true })
    // output.data is a flat Float32Array of length rows.length * DIMENSION
    for (let j = 0; j < rows.length; j += 1) {
      const offset = j * DIMENSION
      yield {
        id: String(i + j),
        vector: output.data.slice(offset, offset + DIMENSION),
      }
    }
    const elapsed = (performance.now() - start) / 1000
    const rate = end / elapsed
    const eta = (numRows - end) / rate
    process.stdout.write(`\r  ${end.toLocaleString()} / ${numRows.toLocaleString()} (${rate.toFixed(0)} rows/s, eta ${eta.toFixed(0)}s)   `)
  }
  process.stdout.write('\n')
}

const writeStart = performance.now()
const writer = fileWriter(DEST)
await writeVectors({
  writer,
  dimension: DIMENSION,
  vectors: embedAll(),
  // Extractor already normalizes; setting the flag records that in kv metadata.
  normalize: true,
})

const stat = await fs.stat(DEST)
const seconds = (performance.now() - writeStart) / 1000
console.log(`Wrote ${DEST}: ${stat.size.toLocaleString()} bytes in ${seconds.toFixed(1)}s`)
const rawSize = numRows * DIMENSION * 4
console.log(`Raw float32 size: ${rawSize.toLocaleString()} bytes (${(stat.size / rawSize * 100).toFixed(1)}% of raw)`)
