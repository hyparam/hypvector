/**
 * Embedding-model comparison on LLM tool/code logs.
 *
 * Eval task: for each conversation we have a `user` question and an `answer`
 * message. Build a labeled (query, target_id) set, embed every message in the
 * corpus with model M, then for each query search the corpus and check
 * whether the target ranks in the top-K. Report:
 *   - hits@1   — fraction of queries where the target is the #1 result
 *   - hits@10  — fraction where the target is in the top 10
 *   - MRR@10   — mean reciprocal rank of the target (0 if not in top 10)
 */
import { promises as fs } from 'node:fs'
import { asyncBufferFromFile, cachedAsyncBuffer, parquetMetadataAsync, parquetReadObjects } from 'hyparquet'
import { fileWriter } from 'hyparquet-writer'
import { searchVectors } from '../src/searchVectors.js'
import { writeVectors } from '../src/writeVectors.js'

const SRC = 'data/llmlog.parquet'
const CONV_LIMIT = parseInt(process.argv[2] ?? '500', 10)
const TOP_K = 10

const ALL_MODELS = {
  'MiniLM-L6': { name: 'MiniLM-L6', id: 'Xenova/all-MiniLM-L6-v2', dim: 384 },
  'bge-small': { name: 'bge-small', id: 'Xenova/bge-small-en-v1.5', dim: 384 },
  'bge-base': { name: 'bge-base', id: 'Xenova/bge-base-en-v1.5', dim: 768 },
  'jina-code': { name: 'jina-code', id: 'jinaai/jina-embeddings-v2-base-code', dim: 768 },
  // OpenAI API models. `dim` uses the `dimensions` param (Matryoshka-truncated).
  'oai-3-small-384': { name: 'oai-3-small-384', id: 'text-embedding-3-small', dim: 384, api: 'openai' },
  'oai-3-small': { name: 'oai-3-small', id: 'text-embedding-3-small', dim: 1536, api: 'openai' },
  'oai-3-large': { name: 'oai-3-large', id: 'text-embedding-3-large', dim: 3072, api: 'openai' },
}
// Optional 3rd arg: comma-separated model keys. Defaults to all.
const MODELS = (process.argv[3] ? process.argv[3].split(',') : Object.keys(ALL_MODELS))
  .map(k => ALL_MODELS[k.trim()])
  .filter(Boolean)

const sourceStat = await fs.stat(SRC).catch(() => undefined)
if (!sourceStat) { console.error(`Missing ${SRC}`); process.exit(1) }

// Minimal .env loader for OPENAI_API_KEY (avoids a dotenv dependency).
async function loadEnvKey(name) {
  if (process.env[name]) return process.env[name]
  const text = await fs.readFile('.env', 'utf8').catch(() => '')
  const line = text.split('\n').find(l => l.startsWith(`${name}=`))
  return line ? line.slice(name.length + 1).trim().replace(/^["']|["']$/g, '') : undefined
}
const OPENAI_API_KEY = await loadEnvKey('OPENAI_API_KEY')

// transformers.js is only needed for local models; import lazily.
let transformers
async function getTransformers() {
  if (!transformers) {
    try { transformers = await import('@huggingface/transformers') }
    catch { console.error('Install @huggingface/transformers first'); process.exit(1) }
  }
  return transformers
}

/**
 * Embed a batch of texts with the OpenAI embeddings API.
 * @param {string[]} texts
 * @param {string} model
 * @param {number} dimensions
 * @returns {Promise<Float32Array[]>}
 */
async function openaiEmbed(texts, model, dimensions) {
  const res = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: { authorization: `Bearer ${OPENAI_API_KEY}`, 'content-type': 'application/json' },
    body: JSON.stringify({ model, input: texts, dimensions }),
  })
  if (!res.ok) throw new Error(`OpenAI ${res.status}: ${(await res.text()).slice(0, 200)}`)
  const { data } = await res.json()
  return data.map(d => Float32Array.from(d.embedding))
}

const file = await asyncBufferFromFile(SRC)
const metadata = await parquetMetadataAsync(file)
const totalConvs = Math.min(Number(metadata.num_rows), CONV_LIMIT)
console.log(`Loading ${totalConvs} conversations from ${SRC}...`)

// Extract messages once. Each message gets an id = `${convIdx}:${msgIdx}`.
const messages = []
const pairs = [] // { queryIdx (into messages), targetIdx (into messages) }
const convs = await parquetReadObjects({ file, metadata, rowStart: 0, rowEnd: totalConvs, columns: ['messages'] })
for (let c = 0; c < convs.length; c += 1) {
  const msgs = convs[c]?.messages ?? []
  /** @type {number | null} */
  let lastUser = null
  /** @type {number | null} */
  let lastUserMsgIdx = null
  for (let m = 0; m < msgs.length; m += 1) {
    const role = msgs[m]?.role
    const content = msgs[m]?.content
    if (!content || role === 'system') continue
    const idx = messages.length
    messages.push({ id: `${c}:${m}`, role, content, convIdx: c, msgIdx: m })
    if (role === 'user') { lastUser = idx; lastUserMsgIdx = m }
    else if (role === 'answer' && lastUser !== null && lastUserMsgIdx !== null && m > lastUserMsgIdx) {
      pairs.push({ queryIdx: lastUser, targetIdx: idx })
      lastUser = null
    }
  }
}
console.log(`Corpus: ${messages.length.toLocaleString()} messages; labeled pairs: ${pairs.length}`)

/**
 * Bench wrapper around an open AsyncBuffer that counts bytes/fetches.
 *
 * @param {import('hyparquet').AsyncBuffer} buf
 * @returns {import('hyparquet').AsyncBuffer & { bytes: number, fetches: number }}
 */
function instrument(buf) {
  const slice = buf.slice.bind(buf)
  const w = {
    byteLength: buf.byteLength, bytes: 0, fetches: 0,
    slice(s, e) { w.bytes += (e ?? buf.byteLength) - s; w.fetches += 1; return slice(s, e) },
  }
  return w
}

/**
 * Run the eval for one model.
 *
 * @param {{ name: string, id: string, dim: number, api?: string }} modelCfg
 * @returns {Promise<object | undefined>}
 */
async function evalModel(modelCfg) {
  console.log(`\n=== ${modelCfg.name} (${modelCfg.id}, ${modelCfg.dim}-dim) ===`)
  const isApi = modelCfg.api === 'openai'
  if (isApi && !OPENAI_API_KEY) { console.log('  skipped: no OPENAI_API_KEY'); return undefined }

  let extract
  if (!isApi) {
    console.log('  loading model...')
    const t = await getTransformers()
    extract = await t.pipeline('feature-extraction', modelCfg.id)
  }

  // API models tolerate large batches; local models stay at 16.
  const BATCH = isApi ? 256 : 16
  // OpenAI input cap is 8191 tokens; clip generously by chars (~4 chars/token).
  const clip = isApi ? (s => s.length > 24000 ? s.slice(0, 24000) : s) : (s => s)
  console.log(`  embedding ${messages.length} messages (batch=${BATCH})...`)
  const allEmbeds = new Float32Array(messages.length * modelCfg.dim)
  const t0 = performance.now()
  for (let i = 0; i < messages.length; i += BATCH) {
    const end = Math.min(i + BATCH, messages.length)
    const texts = []
    for (let j = i; j < end; j += 1) texts.push(clip(messages[j].content))
    if (isApi) {
      const vecs = await openaiEmbed(texts, modelCfg.id, modelCfg.dim)
      for (let k = 0; k < vecs.length; k += 1) allEmbeds.set(vecs[k], (i + k) * modelCfg.dim)
    } else {
      const out = await extract(texts, { pooling: 'mean', normalize: true })
      allEmbeds.set(out.data, i * modelCfg.dim)
    }
    if (i % (BATCH * (isApi ? 4 : 50)) === 0 && i > 0) {
      const elapsed = (performance.now() - t0) / 1000
      const rate = i / elapsed
      process.stdout.write(`\r    ${i}/${messages.length} (${rate.toFixed(0)} msg/s, eta ${((messages.length - i) / rate).toFixed(0)}s)   `)
    }
  }
  console.log(`\n  embed time: ${((performance.now() - t0) / 1000).toFixed(1)}s`)

  // Write a parquet with auto defaults (binary+cluster at this N).
  const path = `data/eval_${modelCfg.name}.parquet`
  await fs.rm(path, { force: true })
  function* gen() {
    for (let i = 0; i < messages.length; i += 1) {
      yield { id: messages[i].id, vector: allEmbeds.subarray(i * modelCfg.dim, (i + 1) * modelCfg.dim) }
    }
  }
  await writeVectors({ writer: fileWriter(path), dimension: modelCfg.dim, vectors: gen(), normalize: true })
  console.log(`  wrote ${path}`)

  // Open + bench. For each pair, query with the user vector, check rank of target.
  const raw = instrument(await asyncBufferFromFile(path))
  const cached = cachedAsyncBuffer(raw)
  const meta = await parquetMetadataAsync(cached)
  let hits1 = 0, hits10 = 0
  let mrrSum = 0
  const queryTimes = []
  for (const pair of pairs) {
    const queryVec = allEmbeds.subarray(pair.queryIdx * modelCfg.dim, (pair.queryIdx + 1) * modelCfg.dim)
    const queryId = messages[pair.queryIdx].id
    const targetId = messages[pair.targetIdx].id
    const start = performance.now()
    // Fetch TOP_K + 1 then drop the query's own self-match before ranking.
    const raw10 = await searchVectors({ source: cached, metadata: meta, query: queryVec, topK: TOP_K + 1 })
    queryTimes.push(performance.now() - start)
    const ranked = raw10.filter(r => r.id !== queryId).slice(0, TOP_K)
    const rank = ranked.findIndex(r => r.id === targetId)
    if (rank === 0) hits1 += 1
    if (rank >= 0) {
      hits10 += 1
      mrrSum += 1 / (rank + 1)
    }
  }
  const size = (await fs.stat(path)).size
  const metrics = {
    name: modelCfg.name,
    dim: modelCfg.dim,
    sizeMB: size / 1e6,
    msPerQuery: queryTimes.reduce((s, x) => s + x, 0) / queryTimes.length,
    hits1: hits1 / pairs.length,
    hits10: hits10 / pairs.length,
    mrr10: mrrSum / pairs.length,
  }
  console.log(`  hits@1=${(metrics.hits1 * 100).toFixed(1)}%  hits@10=${(metrics.hits10 * 100).toFixed(1)}%  MRR@10=${metrics.mrr10.toFixed(3)}  ms/query=${metrics.msPerQuery.toFixed(1)}  size=${metrics.sizeMB.toFixed(1)}MB`)
  return metrics
}

const results = []
for (const m of MODELS) {
  try {
    results.push(await evalModel(m))
  } catch (err) {
    console.log(`  FAILED: ${err?.message ?? err}`)
  }
}

console.log('\n=== Summary ===')
console.log(`${'model'.padEnd(12)} ${'dim'.padStart(4)} ${'size MB'.padStart(8)} ${'ms/q'.padStart(6)} ${'hits@1'.padStart(7)} ${'hits@10'.padStart(8)} ${'MRR@10'.padStart(7)}`)
console.log('-'.repeat(60))
for (const m of results) {
  console.log(`${m.name.padEnd(12)} ${String(m.dim).padStart(4)} ${m.sizeMB.toFixed(1).padStart(8)} ${m.msPerQuery.toFixed(1).padStart(6)} ${(m.hits1 * 100).toFixed(1).padStart(6)}% ${(m.hits10 * 100).toFixed(1).padStart(7)}% ${m.mrr10.toFixed(3).padStart(7)}`)
}
