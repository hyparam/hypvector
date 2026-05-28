/**
 * Approximate the Zilliz "filtered recall" experiment against hypvector.
 *
 * Mirror the real workload: vectors are semantically distributed (random
 * unit vectors across the embedding sphere), and tenant_id is a *metadata
 * filter*, independent of vector position. The interesting query is
 * "top-K similar to Q within tenant X", and the question is whether each
 * strategy actually recovers the true within-tenant top-K.
 *
 *   A) unfiltered      — search the whole corpus, ignore tenant. The
 *                        result is mostly cross-tenant noise; recall
 *                        against the within-tenant truth is ~fraction-of-
 *                        corpus-in-target-tenant.
 *   B) post-filter     — search the whole corpus with an inflated topK,
 *                        then reject rows from the wrong tenant. This is
 *                        the failure mode the Zilliz benchmark exposes:
 *                        the global top-K doesn't contain enough target-
 *                        tenant rows for the filter to recover them.
 *   C) shard-as-filter — search just the target tenant's parquet file.
 *                        hypvector's native pre-filter via file sharding.
 *
 * Usage:
 *   node scripts/bench-zilliz.js [vectors] [tenants] [queries] [dim]
 *
 * Defaults: 50000 / 16 / 50 / 384
 */
import { promises as fs } from 'node:fs'
import { asyncBufferFromFile, parquetMetadataAsync } from 'hyparquet'
import { fileWriter } from 'hyparquet-writer'
import { prefetchBinary } from '../src/prefetch.js'
import { searchVectors } from '../src/searchVectors.js'
import { dotProduct, l2Normalize } from '../src/utils.js'
import { writeVectors } from '../src/writeVectors.js'

const TOTAL = parseInt(process.argv[2] ?? '50000', 10)
const TENANTS = parseInt(process.argv[3] ?? '16', 10)
const QUERIES = parseInt(process.argv[4] ?? '50', 10)
const DIM = parseInt(process.argv[5] ?? '384', 10)
const TOP_K = 100
// Sweep post-filter overfetch factors so we can see whether mode B catches up
// once we ask for enough global candidates. 1× = no overfetch, just topK.
const POST_FILTER_OVERFETCHES = [1, 10, 30, 100]
const WHOLE_FILE = 'data/zilliz_whole.parquet'
const TENANT_DIR = 'data/zilliz_tenants'
const PER_TENANT = Math.ceil(TOTAL / TENANTS)

console.log(`Config: ${TOTAL.toLocaleString()} vectors × ${DIM}-dim, ${TENANTS} tenants (~${PER_TENANT.toLocaleString()} each), ${QUERIES} queries, top-${TOP_K}`)

// Deterministic LCG so re-runs match.
let lcg = 1
function rand() {
  lcg = Math.imul(lcg, 1664525) + 1013904223 >>> 0
  return lcg / 0x100000000
}
// Uniform cube → normalized. Not strictly uniform on the sphere, but in dim=384
// the bias is tiny (concentration of measure) and it's ~10× faster than
// Box-Muller, which matters at 1M scale.
function unitVec(dim) {
  const v = new Float32Array(dim)
  for (let i = 0; i < dim; i += 1) v[i] = rand() * 2 - 1
  return l2Normalize(v)
}

console.log(`Generating ${TOTAL.toLocaleString()} random unit vectors with independent tenant labels...`)
// Tenant_id is independent of vector position — matches the real-world
// "metadata filter on semantic embeddings" workload (e.g., agent_id on logs).
const genStart = performance.now()
/** @type {Float32Array[]} */
const allVecs = new Array(TOTAL)
/** @type {Int32Array} */
const tenantOf = new Int32Array(TOTAL)
for (let i = 0; i < TOTAL; i += 1) {
  allVecs[i] = unitVec(DIM)
  tenantOf[i] = Math.floor(rand() * TENANTS)
}
console.log(`  generated in ${((performance.now() - genStart) / 1000).toFixed(1)}s`)

// Indexes within each tenant for ground-truth and per-tenant files.
/** @type {number[][]} */
const tenantRows = Array.from({ length: TENANTS }, () => [])
for (let i = 0; i < TOTAL; i += 1) tenantRows[tenantOf[i]].push(i)

console.log('\nWriting whole-corpus parquet...')
{
  // No clustering on the whole-corpus file: tenant_id is independent of
  // vector position, so k-means partitions would be orthogonal to tenants
  // and probe<1 would silently cap recall. Use plain binary+rerank so the
  // comparison with per-tenant files is on equal footing.
  const writer = fileWriter(WHOLE_FILE)
  await writeVectors({
    writer, dimension: DIM, normalize: false, binary: true,
    vectors: function* () {
      for (let i = 0; i < TOTAL; i += 1) yield { id: `t${tenantOf[i]}-r${i}`, vector: allVecs[i] }
    }(),
  })
  const stat = await fs.stat(WHOLE_FILE)
  console.log(`  ${WHOLE_FILE}: ${(stat.size / 1e6).toFixed(1)} MB`)
}

console.log('Writing per-tenant parquets...')
await fs.mkdir(TENANT_DIR, { recursive: true })
const tenantFiles = []
for (let t = 0; t < TENANTS; t += 1) {
  const path = `${TENANT_DIR}/t${t}.parquet`
  tenantFiles.push(path)
  const rows = tenantRows[t]
  const writer = fileWriter(path)
  await writeVectors({
    writer, dimension: DIM, normalize: false, binary: true,
    vectors: function* () {
      for (const i of rows) yield { id: `t${t}-r${i}`, vector: allVecs[i] }
    }(),
  })
}
console.log(`  ${tenantFiles.length} files in ${TENANT_DIR}/`)

// Brute-force ground truth: for each query, return the true within-tenant top-K.
function trueTopKInTenant(query, t) {
  const rows = tenantRows[t]
  const scored = rows.map(i => ({ id: `t${t}-r${i}`, score: dotProduct(query, allVecs[i]) }))
  scored.sort((a, b) => b.score - a.score)
  return new Set(scored.slice(0, TOP_K).map(s => s.id))
}

// Pick query vectors from random rows, perturbed slightly so the query isn't a perfect self-hit.
/** @type {{ tenant: number, vec: Float32Array }[]} */
const queries = []
for (let q = 0; q < QUERIES; q += 1) {
  const t = Math.floor(rand() * TENANTS)
  const rows = tenantRows[t]
  const pickRow = rows[Math.floor(rand() * rows.length)]
  const base = allVecs[pickRow]
  // Small uniform perturbation so the query isn't a perfect self-hit.
  const v = new Float32Array(DIM)
  for (let d = 0; d < DIM; d += 1) v[d] = base[d] + 0.05 * (rand() * 2 - 1)
  queries.push({ tenant: t, vec: l2Normalize(v) })
}

console.log('\nComputing ground truth (brute-force within-tenant top-K)...')
const groundTruth = queries.map(({ tenant, vec }) => trueTopKInTenant(vec, tenant))

// Open files + parse metadata + prefetch binary, once.
const wholeBuf = await asyncBufferFromFile(WHOLE_FILE)
const wholeMeta = await parquetMetadataAsync(wholeBuf)
const wholeBin = await prefetchBinary({ source: wholeBuf, metadata: wholeMeta })

const tenantBufs = await Promise.all(tenantFiles.map(p => asyncBufferFromFile(p)))
const tenantMetas = await Promise.all(tenantBufs.map(b => parquetMetadataAsync(b)))
const tenantBins = await Promise.all(tenantBufs.map((b, i) => prefetchBinary({ source: b, metadata: tenantMetas[i] })))

/**
 * @param {string} label
 * @param {(query: { tenant: number, vec: Float32Array }) => Promise<Array<{ id: string }>>} queryFn
 * @returns {Promise<{ recall: number, ms: number }>}
 */
async function runMode(label, queryFn) {
  let totalRecall = 0
  const t0 = performance.now()
  for (let q = 0; q < queries.length; q += 1) {
    const hits = await queryFn(queries[q])
    const truth = groundTruth[q]
    let matches = 0
    for (const h of hits) if (truth.has(String(h.id))) matches += 1
    totalRecall += matches / truth.size
  }
  const ms = (performance.now() - t0) / queries.length
  const recall = totalRecall / queries.length
  console.log(`${label.padEnd(28)} recall@${TOP_K}=${(recall * 100).toFixed(1).padStart(5)}%   ${ms.toFixed(1).padStart(6)} ms/query`)
  return { recall, ms }
}

console.log('\n=== Results ===')
await runMode('A) unfiltered whole-corpus', ({ vec }) =>
  searchVectors({ source: wholeBuf, metadata: wholeMeta, binary: wholeBin, query: vec, topK: TOP_K })
)

for (const f of POST_FILTER_OVERFETCHES) {
  await runMode(`B) post-filter (overfetch ${f.toString().padStart(3)}×)`, async ({ tenant, vec }) => {
    const hits = await searchVectors({
      source: wholeBuf, metadata: wholeMeta, binary: wholeBin, query: vec, topK: TOP_K * f,
    })
    const prefix = `t${tenant}-r`
    return hits.filter(h => String(h.id).startsWith(prefix)).slice(0, TOP_K)
  })
}

await runMode('C) shard-as-filter (1 file)', ({ tenant, vec }) =>
  searchVectors({ source: tenantBufs[tenant], metadata: tenantMetas[tenant], binary: tenantBins[tenant], query: vec, topK: TOP_K })
)

await runMode('C\') shard via array (1 file)', ({ tenant, vec }) =>
  searchVectors({ source: [tenantBufs[tenant]], metadata: [tenantMetas[tenant]], binary: [tenantBins[tenant]], query: vec, topK: TOP_K })
)
