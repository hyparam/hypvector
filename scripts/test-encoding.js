/**
 * Empirically test encoding/codec knobs that were dismissed without
 * measurement:
 *   - BYTE_STREAM_SPLIT on the float vector column (with and without
 *     ZSTD)
 *   - SNAPPY / ZSTD globally
 *
 * Writes variant files and benches them so we can see file size + query
 * speed tradeoffs side-by-side.
 */
import { promises as fs } from 'node:fs'
import { asyncBufferFromFile, cachedAsyncBuffer, parquetMetadataAsync } from 'hyparquet'
import { compressors as readCompressors } from 'hyparquet-compressors'
import { fileWriter, parquetWrite, schemaFromColumnData } from 'hyparquet-writer'
import { snappyCompress } from 'hyparquet-writer/src/snappy.js'

// Reader compressors (decompress) for hyparquet, writer needs compress fns.
// hyparquet-writer ships snappyCompress; hyparquet-compressors only has
// decompressors. So we can write SNAPPY locally; ZSTD compression would
// need a separate compress lib.
const writeCompressors = { SNAPPY: snappyCompress }
import { binaryKMeans, reorderClustersByHamming } from '../src/cluster.js'
import { defaultBinaryColumn, defaultIdColumn, defaultVectorColumn } from '../src/constants.js'
import { readVectors } from '../src/readVectors.js'
import { searchVectors } from '../src/searchVectors.js'
import { l2Normalize, packBinary, packFloat32, parseKvMetadata } from '../src/utils.js'

const SRC = 'data/wiki_en.vectors.parquet'
const src = await asyncBufferFromFile(SRC)
const srcMd = await parquetMetadataAsync(src)
const srcMeta = parseKvMetadata(srcMd)
console.log(`Loading ${srcMeta.count.toLocaleString()} × ${srcMeta.dimension}-dim from ${SRC}`)

const records = []
for await (const r of readVectors({ file: src, metadata: srcMd, includeMetadata: false })) records.push(r)

const dim = srcMeta.dimension
const binaryBytes = (dim + 7) >> 3
const ids = records.map(r => String(r.id))
const packed = records.map(r => packFloat32(l2Normalize(r.vector)))
const packedBin = records.map(r => packBinary(l2Normalize(r.vector), dim))

const { assignments, centroids } = binaryKMeans(packedBin, binaryBytes, 128, 6, 1)
const remap = reorderClustersByHamming(centroids)
const reordered = new Array(centroids.length)
for (let i = 0; i < centroids.length; i++) reordered[remap[i]] = centroids[i]

const order = new Int32Array(ids.length)
for (let i = 0; i < ids.length; i++) order[i] = i
const sorted = Array.from(order).sort((a, b) => remap[assignments[a]] - remap[assignments[b]])
const idsOut = new Array(ids.length)
const packedOut = new Array(ids.length)
const binOut = new Array(ids.length)
const counts = new Uint32Array(centroids.length)
for (let i = 0; i < sorted.length; i++) {
  const s = sorted[i]
  idsOut[i] = ids[s]; packedOut[i] = packed[s]; binOut[i] = packedBin[s]
  counts[remap[assignments[s]]] += 1
}

const centBuf = new Uint8Array(centroids.length * binaryBytes)
for (let c = 0; c < centroids.length; c++) centBuf.set(reordered[c], c * binaryBytes)
/**
 * @param {Uint8Array} bytes
 * @returns {string}
 */
function b64(bytes) {
  return Buffer.from(bytes).toString('base64')
}

const kvMetadata = [
  { key: 'hypvector.version', value: '0' }, { key: 'hypvector.dimension', value: String(dim) },
  { key: 'hypvector.metric', value: 'cosine' }, { key: 'hypvector.normalized', value: 'true' },
  { key: 'hypvector.binary', value: 'true' }, { key: 'hypvector.count', value: String(ids.length) },
  { key: 'hypvector.clusters', value: String(centroids.length) },
  { key: 'hypvector.centroids', value: b64(centBuf) },
  { key: 'hypvector.clusterCounts', value: b64(new Uint8Array(counts.buffer)) },
]

async function writeVariant(path, { codec, vectorEncoding, vectorCodec, binaryCodec, binaryEncoding }) {
  const columnData = [
    { name: defaultIdColumn, data: idsOut },
    {
      name: defaultVectorColumn,
      data: packedOut,
      ...(vectorEncoding ? { encoding: vectorEncoding } : {}),
      ...(vectorCodec ? { codec: vectorCodec } : {}),
    },
    {
      name: defaultBinaryColumn,
      data: binOut,
      ...(binaryEncoding ? { encoding: binaryEncoding } : {}),
      ...(binaryCodec ? { codec: binaryCodec } : {}),
    },
  ]
  const schemaInput = columnData.map(c => c.name === defaultIdColumn ? { ...c, type: /** @type {const} */ ('STRING') } : c)
  const schema = schemaFromColumnData({
    columnData: schemaInput,
    schemaOverrides: {
      [defaultVectorColumn]: { name: defaultVectorColumn, type: 'FIXED_LEN_BYTE_ARRAY', type_length: dim * 4, repetition_type: 'REQUIRED' },
      [defaultBinaryColumn]: { name: defaultBinaryColumn, type: 'FIXED_LEN_BYTE_ARRAY', type_length: binaryBytes, repetition_type: 'REQUIRED' },
    },
  })
  await parquetWrite({
    writer: fileWriter(path), schema, rowGroupSize: 1000, pageSize: 32768,
    codec, columnData, kvMetadata, compressors: writeCompressors,
  })
  return (await fs.stat(path)).size
}

const queries = []
const step = Math.max(1, Math.floor(records.length / 6))
for (let i = step; queries.length < 5 && i < records.length; i += step) queries.push(records[i].vector)

async function bench(path) {
  const md = await parquetMetadataAsync(await asyncBufferFromFile(path))
  const times = []
  let bytes = 0
  for (const q of queries) {
    const buf = await asyncBufferFromFile(path)
    const wrapped = {
      byteLength: buf.byteLength,
      slice(s, e) { bytes += (e ?? buf.byteLength) - s; return buf.slice(s, e) },
    }
    const cached = cachedAsyncBuffer(wrapped)
    const t = performance.now()
    await searchVectors({
      source: cached, metadata: md, query: q, topK: 10, compressors: readCompressors,
    })
    times.push(performance.now() - t)
  }
  let sum = 0
  for (let i = 0; i < times.length; i += 1) sum += times[i]
  return { ms: sum / times.length, mb: bytes / queries.length / 1e6 }
}

// Compression options are limited to what hyparquet-writer can compress
// (currently SNAPPY only via its internal snappyCompress). ZSTD on write
// would need an external compress lib.
const variants = [
  { name: 'baseline', label: 'UNCOMPRESSED', codec: 'UNCOMPRESSED' },
  { name: 'binsnap', label: 'vector=UNCOMP, binary=SNAPPY', codec: 'UNCOMPRESSED', binaryCodec: 'SNAPPY' },
  { name: 'allsnap', label: 'all=SNAPPY (global)', codec: 'SNAPPY' },
  { name: 'bss', label: 'vector=BSS uncomp', codec: 'UNCOMPRESSED', vectorEncoding: 'BYTE_STREAM_SPLIT' },
  { name: 'bsssnap', label: 'vector=BSS+SNAPPY', codec: 'UNCOMPRESSED', vectorEncoding: 'BYTE_STREAM_SPLIT', vectorCodec: 'SNAPPY' },
]

console.log(`\n${'variant'.padEnd(28)} ${'file MB'.padStart(9)} ${'ms'.padStart(7)} ${'MB read'.padStart(10)}`)
console.log('-'.repeat(60))
for (const v of variants) {
  const path = `data/enc_${v.name}.parquet`
  let size
  try {
    size = await writeVariant(path, v)
  } catch (e) {
    console.log(`${v.label.padEnd(28)} write failed: ${e.message}`)
    continue
  }
  try {
    const r = await bench(path)
    console.log(`${v.label.padEnd(28)} ${(size / 1e6).toFixed(1).padStart(9)} ${r.ms.toFixed(1).padStart(7)} ${r.mb.toFixed(2).padStart(10)}`)
  } catch (e) {
    console.log(`${v.label.padEnd(28)} ${(size / 1e6).toFixed(1).padStart(9)} bench failed: ${e.message}`)
  }
  await fs.unlink(path).catch(() => {})
}
