/**
 * Embed the tool-reasoning-sft jupyter agent dataset (LLM tool/code logs)
 * into a hypvector parquet, one row per non-system message. Each id is
 * `<convIdx>:<msgIdx>:<role>` so we can recover provenance from search results.
 *
 * Uses the same MiniLM model + 384-dim + normalize=true as scripts/embed.js
 * so the resulting file is directly comparable to data/wiki_en.vectors.parquet.
 */
import { promises as fs } from 'node:fs'
import { asyncBufferFromFile, parquetMetadataAsync, parquetReadObjects } from 'hyparquet'
import { fileWriter } from 'hyparquet-writer'
import { writeVectors } from '../src/writeVectors.js'

const SOURCE = 'data/llmlog.parquet'
const DEST = process.argv[3] ?? 'data/llmlog.vectors.parquet'
const MODEL = 'Xenova/all-MiniLM-L6-v2'
const DIMENSION = 384
const BATCH = 32
const LIMIT = parseInt(process.argv[2] ?? '100000', 10)
const CONV_BATCH = 256

const sourceStat = await fs.stat(SOURCE).catch(() => undefined)
if (!sourceStat) {
  console.error(`Missing ${SOURCE}. Download the HF dataset first.`)
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
const totalConvs = Number(metadata.num_rows)
console.log(`Source: ${SOURCE} (${totalConvs.toLocaleString()} conversations)`)
console.log(`Target: up to ${LIMIT.toLocaleString()} message embeddings`)

async function* embedAll() {
  const start = performance.now()
  let emitted = 0
  for (let i = 0; i < totalConvs && emitted < LIMIT; i += CONV_BATCH) {
    const end = Math.min(i + CONV_BATCH, totalConvs)
    const convs = await parquetReadObjects({ file, metadata, rowStart: i, rowEnd: end, columns: ['messages'] })

    // Flatten conv batch → message rows we want to embed (skip empty + system).
    const items = []
    for (let c = 0; c < convs.length; c += 1) {
      const convIdx = i + c
      const msgs = convs[c]?.messages ?? []
      for (let m = 0; m < msgs.length; m += 1) {
        const role = msgs[m]?.role
        const content = msgs[m]?.content
        if (!content || role === 'system') continue
        items.push({ id: `${convIdx}:${m}:${role}`, text: content })
      }
    }

    for (let j = 0; j < items.length && emitted < LIMIT; j += BATCH) {
      const slice = items.slice(j, Math.min(j + BATCH, items.length, j + LIMIT - emitted))
      const out = await extractor(slice.map(it => it.text), { pooling: 'mean', normalize: true })
      for (let k = 0; k < slice.length; k += 1) {
        const offset = k * DIMENSION
        yield { id: slice[k].id, vector: out.data.slice(offset, offset + DIMENSION) }
        emitted += 1
      }
    }

    const elapsed = (performance.now() - start) / 1000
    const rate = emitted / elapsed
    const eta = (LIMIT - emitted) / rate
    process.stdout.write(`\r  ${emitted.toLocaleString()} / ${LIMIT.toLocaleString()} msgs (${rate.toFixed(0)} msg/s, eta ${eta.toFixed(0)}s)   `)
  }
  process.stdout.write('\n')
}

const writeStart = performance.now()
const writer = fileWriter(DEST)
await writeVectors({
  writer,
  dimension: DIMENSION,
  vectors: embedAll(),
  normalize: true,
  binary: true, // gives us a baseline file with the binary column ready
})

const stat = await fs.stat(DEST)
const seconds = (performance.now() - writeStart) / 1000
console.log(`Wrote ${DEST}: ${stat.size.toLocaleString()} bytes in ${seconds.toFixed(1)}s`)
