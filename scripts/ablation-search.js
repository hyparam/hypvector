/**
 * Search-side ablation: hold the data file constant and toggle individual
 * search-time optimizations to see which still pull their weight.
 *
 * Variants:
 *   E1) baseline                  current code, all optimizations on
 *   E2) phase 2 per-candidate     coalesce maxGap=0 (one parquetRead per candidate)
 *   E3) no phase 3                fetch ids alongside vectors in phase 2
 *
 * Page-size ablation requires re-writing the file, so it's exercised by
 * the writeVectors `pageSize` option directly (see commit history for the
 * sweep that picked 32 KB as the default).
 */
import { promises as fs } from 'node:fs'
import { asyncBufferFromFile, cachedAsyncBuffer, parquetMetadataAsync, parquetRead } from 'hyparquet'
import { hammingDistanceBytes } from '../src/cluster.js'
import { defaultBinaryColumn, defaultClusterProbeFraction, defaultIdColumn, defaultVectorColumn } from '../src/constants.js'
import { readVectors } from '../src/readVectors.js'
import { searchVectors } from '../src/searchVectors.js'
import { cosineSimilarity, dotProduct, l2Normalize, packBinary, parseKvMetadata } from '../src/utils.js'

/**
 * @import { AsyncBuffer } from 'hyparquet'
 */

const FILE = process.argv[2] ?? 'data/abl_C_cluster.parquet'

const stat = await fs.stat(FILE)
const sourceFile = await asyncBufferFromFile(FILE)
const metadata = await parquetMetadataAsync(sourceFile)
const meta = parseKvMetadata(metadata)
console.log(`File: ${FILE} (${(stat.size / 1e6).toFixed(1)} MB; ${meta.count} × ${meta.dimension})`)
/** @type {Float32Array[]} */
const queries = []
const step = Math.max(1, Math.floor(meta.count / 11))
let i = 0
let nextPick = step
for await (const r of readVectors({ file: sourceFile, metadata })) {
  if (i === nextPick) { queries.push(r.vector); nextPick += step; if (queries.length >= 10) break }
  i += 1
}

/**
 * Wrap an AsyncBuffer with byte / fetch counters.
 *
 * @param {AsyncBuffer} buf
 * @returns {AsyncBuffer & { bytes: number, fetches: number }}
 */
function instrument(buf) {
  const orig = buf.slice.bind(buf)
  const w = {
    byteLength: buf.byteLength,
    bytes: 0,
    fetches: 0,
    slice(s, e) { w.bytes += (e ?? buf.byteLength) - s; w.fetches += 1; return orig(s, e) },
  }
  return w
}

/**
 * @param {(q: Float32Array, cached: AsyncBuffer) => Promise<{ id: string, score: number, rowIndex: number }[]>} searchFn
 * @returns {Promise<{ ms: number, mb: number, fetches: number, tops: string[][] }>}
 */
async function bench(searchFn) {
  const times = [], bytes = [], fetches = [], tops = []
  for (const q of queries) {
    const raw = instrument(await asyncBufferFromFile(FILE))
    const cached = cachedAsyncBuffer(raw)
    const start = performance.now()
    const r = await searchFn(q, cached)
    times.push(performance.now() - start)
    bytes.push(raw.bytes); fetches.push(raw.fetches); tops.push(r.map(x => String(x.id)))
  }
  return { ms: avg(times), mb: avg(bytes) / 1e6, fetches: avg(fetches), tops }
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

// E1: baseline using the current searchVectors
const E1 = await bench((q, cached) => searchVectors({
  source: cached, metadata, query: q, topK: 10,
}))

// E2: same path but per-candidate (no run coalescing). Implemented inline.
const E2 = await bench((q, cached) => searchAblated(q, cached, { coalesce: false, deferId: true }))

// E3: same as E1 but ids fetched in phase 2 alongside vectors.
const E3 = await bench((q, cached) => searchAblated(q, cached, { coalesce: true, deferId: false }))

/**
 * @param {string[][]} ref
 * @param {string[][]} cand
 * @returns {number}
 */
function recall(ref, cand) {
  let hits = 0; let total = 0
  for (let q = 0; q < ref.length; q += 1) {
    const r = new Set(ref[q])
    for (const id of cand[q]) if (r.has(id)) hits += 1
    total += r.size
  }
  return hits / total
}

console.log(`\n${'variant'.padEnd(34)} ${'ms'.padStart(7)} ${'fetches'.padStart(8)} ${'MB read'.padStart(9)} ${'recall'.padStart(8)}`)
console.log('-'.repeat(70))
console.log(`${'E1) baseline (all opts on)'.padEnd(34)} ${E1.ms.toFixed(1).padStart(7)} ${E1.fetches.toFixed(0).padStart(8)} ${E1.mb.toFixed(2).padStart(9)} ${'(ref)'.padStart(8)}`)
console.log(`${'E2) -coalesce (per-candidate)'.padEnd(34)} ${E2.ms.toFixed(1).padStart(7)} ${E2.fetches.toFixed(0).padStart(8)} ${E2.mb.toFixed(2).padStart(9)} ${(recall(E1.tops, E2.tops) * 100).toFixed(1).padStart(7)}%`)
console.log(`${'E3) -deferId (id in phase 2)'.padEnd(34)} ${E3.ms.toFixed(1).padStart(7)} ${E3.fetches.toFixed(0).padStart(8)} ${E3.mb.toFixed(2).padStart(9)} ${(recall(E1.tops, E3.tops) * 100).toFixed(1).padStart(7)}%`)

/**
 * Inline simplified search with togglable optimizations. Mirrors searchRerank
 * but with coalesce / deferId knobs.
 *
 * @param {Float32Array} query
 * @param {AsyncBuffer} file
 * @param {{ coalesce: boolean, deferId: boolean }} opts
 * @returns {Promise<{ id: string, score: number, rowIndex: number }[]>}
 */
async function searchAblated(query, file, opts) {
  const dim = meta.dimension
  const binaryBytes = dim + 7 >> 3
  const candidatesK = 100
  const topK = 10

  let queryF32 = query
  let scoring = meta.metric
  if (meta.metric === 'cosine' && meta.normalized) {
    queryF32 = l2Normalize(query)
    scoring = 'dot'
  }

  // Phase 1: cluster-restricted Hamming scan
  const queryBin = packBinary(queryF32, dim)
  const queryBinU32 = new Uint32Array(queryBin.buffer.slice(queryBin.byteOffset, queryBin.byteOffset + queryBin.byteLength))
  const wordsPerRow = binaryBytes >> 2

  // Pick clusters
  const cs = meta.centroids
  const offsets = new Uint32Array(cs.length + 1)
  for (let c = 0; c < cs.length; c += 1) {
    offsets[c + 1] = offsets[c] + meta.clusterCounts[c]
  }
  const ranked = cs.map((c, i) => ({ i, d: hammingDistanceBytes(queryBin, c) })).sort((a, b) => a.d - b.d)
  const probe = Math.max(1, Math.ceil(cs.length * defaultClusterProbeFraction))
  const wantedClusters = ranked.slice(0, probe).map(c => c.i).sort((a, b) => a - b)
  const scanRanges = []
  for (const c of wantedClusters) scanRanges.push({ rowStart: offsets[c], rowEnd: offsets[c + 1] })

  /** @type {{ rowIndex: number, hamming: number }[]} */
  const heap = []
  await Promise.all(scanRanges.map(({ rowStart, rowEnd }) => parquetRead({
    file, metadata, columns: [defaultBinaryColumn], rowStart, rowEnd, useOffsetIndex: true,
    onChunk: ({ columnName, columnData, rowStart: cs0 }) => {
      if (columnName !== defaultBinaryColumn) return
      const rows = columnData
      if (rows.length === 0) return
      const flat = rows[0].byteOffset % 4 === 0 ? new Uint32Array(rows[0].buffer, rows[0].byteOffset, rows.length * wordsPerRow) : null
      for (let r = 0; r < rows.length; r += 1) {
        let d = 0
        if (flat) {
          for (let j = 0; j < wordsPerRow; j += 1) {
            let v = flat[r * wordsPerRow + j] ^ queryBinU32[j]
            v = v - (v >>> 1 & 0x55555555)
            v = (v & 0x33333333) + (v >>> 2 & 0x33333333)
            d += (v + (v >>> 4) & 0x0f0f0f0f) * 0x01010101 >>> 24
          }
        } else {
          d = hammingDistanceBytes(rows[r], queryBin)
        }
        if (heap.length < candidatesK) heap.push({ rowIndex: cs0 + r, hamming: d })
        else {
          let worst = 0
          for (let i = 1; i < heap.length; i += 1) if (heap[i].hamming > heap[worst].hamming) worst = i
          if (d < heap[worst].hamming) heap[worst] = { rowIndex: cs0 + r, hamming: d }
        }
      }
    },
  })))

  const candidateRows = [...new Set(heap.map(c => c.rowIndex))].sort((a, b) => a - b)
  const wanted = new Set(candidateRows)
  const runs = opts.coalesce ? coalesce(candidateRows, 64) : candidateRows.map(r => ({ rowStart: r, rowEnd: r + 1 }))

  // Phase 2
  const cols = opts.deferId ? [defaultVectorColumn] : [defaultIdColumn, defaultVectorColumn]
  /** @type {Map<number, { v?: Float32Array, id?: string }>} */
  const collected = new Map()
  const dec = new TextDecoder()
  await Promise.all(runs.map(({ rowStart, rowEnd }) => parquetRead({
    file, metadata, columns: cols, rowStart, rowEnd, useOffsetIndex: true,
    onChunk: ({ columnName, columnData, rowStart: cs0 }) => {
      for (let i = 0; i < columnData.length; i += 1) {
        const ri = cs0 + i
        if (!wanted.has(ri)) continue
        let e = collected.get(ri); if (!e) { e = {}; collected.set(ri, e) }
        if (columnName === defaultVectorColumn) {
          const b = columnData[i]
          if (b.byteOffset % 4 === 0) e.v = new Float32Array(b.buffer, b.byteOffset, dim)
          else { const c = new Float32Array(dim); new Uint8Array(c.buffer).set(b); e.v = c }
        } else if (columnName === defaultIdColumn) {
          const raw = columnData[i]
          e.id = typeof raw === 'string' ? raw : dec.decode(raw)
        }
      }
    },
  })))

  const scored = []
  for (const [ri, e] of collected) {
    if (!e.v) continue
    const s = scoring === 'dot' ? dotProduct(queryF32, e.v) : cosineSimilarity(queryF32, e.v)
    scored.push({ rowIndex: ri, score: s, id: e.id })
  }
  scored.sort((a, b) => b.score - a.score)
  const winners = scored.slice(0, topK)

  if (opts.deferId) {
    const ids = await fetchIds(file, winners.map(w => w.rowIndex))
    return winners.map((w, i) => ({ id: ids[i], score: w.score, rowIndex: w.rowIndex }))
  }
  return winners.map(w => ({ id: w.id ?? String(w.rowIndex), score: w.score, rowIndex: w.rowIndex }))
}

/**
 * @param {number[]} rows
 * @param {number} gap
 * @returns {{ rowStart: number, rowEnd: number }[]}
 */
function coalesce(rows, gap) {
  const out = []
  let s = rows[0]; let e = rows[0] + 1
  for (let i = 1; i < rows.length; i += 1) {
    if (rows[i] - e <= gap) e = rows[i] + 1
    else { out.push({ rowStart: s, rowEnd: e }); s = rows[i]; e = rows[i] + 1 }
  }
  out.push({ rowStart: s, rowEnd: e })
  return out
}

/**
 * @param {AsyncBuffer} file
 * @param {number[]} rows
 * @returns {Promise<string[]>}
 */
async function fetchIds(file, rows) {
  const sorted = [...new Set(rows)].sort((a, b) => a - b)
  const wanted = new Set(sorted)
  const runs = coalesce(sorted, 64)
  const dec = new TextDecoder()
  const byRow = new Map()
  await Promise.all(runs.map(({ rowStart, rowEnd }) => parquetRead({
    file, metadata, columns: [defaultIdColumn], rowStart, rowEnd, useOffsetIndex: true,
    onChunk: ({ columnName, columnData, rowStart: cs0 }) => {
      if (columnName !== defaultIdColumn) return
      for (let i = 0; i < columnData.length; i += 1) {
        const ri = cs0 + i; if (!wanted.has(ri)) continue
        const raw = columnData[i]
        byRow.set(ri, typeof raw === 'string' ? raw : dec.decode(raw))
      }
    },
  })))
  return rows.map(r => byRow.get(r) ?? String(r))
}
