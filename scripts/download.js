import { createWriteStream, promises as fs } from 'node:fs'
import { pipeline } from 'node:stream/promises'

const url = 'https://s3.hyperparam.app/wiki_en.parquet'
const dest = 'data/wiki_en.parquet'

await fs.mkdir('data', { recursive: true })

const existing = await fs.stat(dest).catch(() => undefined)
if (existing) {
  console.log(`${dest} already exists (${existing.size.toLocaleString()} bytes)`)
  process.exit(0)
}

console.log(`Downloading ${url}...`)
const start = performance.now()
const res = await fetch(url)
if (!res.ok) throw new Error(`fetch failed: ${res.status} ${res.statusText}`)
if (!res.body) throw new Error('response has no body')
await pipeline(res.body, createWriteStream(dest))

const stat = await fs.stat(dest)
const seconds = (performance.now() - start) / 1000
const mb = stat.size / 1e6
console.log(`Saved ${dest}: ${stat.size.toLocaleString()} bytes in ${seconds.toFixed(1)}s (${(mb / seconds).toFixed(1)} MB/s)`)
