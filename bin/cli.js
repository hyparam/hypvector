#!/usr/bin/env node

import { pathToFileURL } from 'node:url'
import { inspect } from './inspect.js'

/**
 * Command line entry point.
 *
 * Usage: npx hypvector <vectors.parquet>
 */
async function main() {
  const path = process.argv[2]
  if (!path) {
    console.error('Usage: npx hypvector <vectors.parquet>')
    process.exit(1)
  }
  await inspect({ path })
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(error => {
    console.error(error.message)
    process.exit(1)
  })
}
