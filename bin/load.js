import fs from 'node:fs/promises'
import { asyncBufferFromFile, asyncBufferFromUrl, parquetMetadataAsync } from 'hyparquet'

/**
 * @import {AsyncBuffer, FileMetaData} from 'hyparquet'
 */

/**
 * Load a parquet file from a local path or remote URL.
 *
 * @param {string} path
 * @returns {Promise<{ file: AsyncBuffer, metadata: FileMetaData }>}
 */
export async function loadParquet(path) {
  /** @type {AsyncBuffer | undefined} */
  let file
  if (path.startsWith('http://') || path.startsWith('https://')) {
    try {
      file = await asyncBufferFromUrl({ url: path })
    } catch (error) {
      console.error(`Failed to load Parquet file from URL: ${path}`)
      throw error
    }
  } else {
    try {
      await fs.access(path)
    } catch {
      throw new Error(`Parquet file not found: ${path}`)
    }
    try {
      file = await asyncBufferFromFile(path)
    } catch (error) {
      console.error(`Failed to load Parquet file from path: ${path}`)
      throw error
    }
  }
  /** @type {FileMetaData | undefined} */
  let metadata
  try {
    metadata = await parquetMetadataAsync(file)
  } catch (error) {
    throw new Error(`Failed to read Parquet metadata from file: ${path}`, { cause: error })
  }
  return { file, metadata }
}
