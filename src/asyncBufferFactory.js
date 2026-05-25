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
 * @param {{ url: string, signal?: AbortSignal }} options
 * @returns {Promise<AsyncBuffer>}
 */
export async function defaultAsyncBufferFactory({ url, signal }) {
  /** @type {AsyncBuffer} */
  let raw
  if (url.startsWith('http://') || url.startsWith('https://')) {
    /** @type {RequestInit} */
    const requestInit = signal ? { signal } : {}
    raw = await asyncBufferFromUrl({ url, requestInit })
  } else {
    raw = await asyncBufferFromFile(url)
  }
  return cachedAsyncBuffer(raw)
}
