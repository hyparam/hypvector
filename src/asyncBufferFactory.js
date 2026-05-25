import { asyncBufferFromFile, asyncBufferFromUrl, cachedAsyncBuffer } from 'hyparquet'

/**
 * @import { AsyncBuffer } from 'hyparquet'
 */

/**
 * Default AsyncBuffer factory: uses node fs for local paths and HTTP fetch
 * otherwise. Wraps the result in cachedAsyncBuffer so repeated reads of
 * the same byte range (footer, offset indexes, overlapping pages) are
 * served from memory.
 *
 * @param {{ source: string, signal?: AbortSignal }} options
 * @returns {Promise<AsyncBuffer>}
 */
export async function defaultAsyncBufferFactory({ source, signal }) {
  /** @type {AsyncBuffer} */
  let raw
  if (source.startsWith('http://') || source.startsWith('https://')) {
    /** @type {RequestInit} */
    const requestInit = signal ? { signal } : {}
    raw = await asyncBufferFromUrl({ url: source, requestInit })
  } else {
    raw = await asyncBufferFromFile(source)
  }
  return cachedAsyncBuffer(raw)
}
