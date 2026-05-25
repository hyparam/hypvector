// Version of the parquet vector format
export const hypVectorVersion = 0

// Default rows per row group when not clustered. Larger row groups mean
// fewer fetches for a full exact scan, which dominates the no-cluster path.
export const defaultRowGroupSize = 10000

// Default rows per row group when clustered. Smaller row groups give the
// search finer parallelism (each cluster's data fits in fewer row groups,
// each row group's pages can be fetched in parallel). On a real HTTP
// benchmark with RTT=20ms, this is ~18% faster than rg=10000 on the
// clustered + rerank path.
export const defaultClusteredRowGroupSize = 1000

// Default name of the vector column
export const defaultVectorColumn = 'vector'

// Default name of the binary (sign-bit) rerank column
export const defaultBinaryColumn = 'vector_bin'

// Default name of the id column
export const defaultIdColumn = 'id'

// Default parquet page size (in bytes) when a binary column is written.
// Smaller pages let useOffsetIndex fetch only the pages containing the
// per-candidate rows in the rerank phase 2 scan. 32 KB empirically
// minimizes bytes-read on a 384-dim wiki benchmark; smaller pages save
// more on phase 2 but cost more on phase-1 page-header overhead.
export const defaultBinaryPageSize = 32 * 1024

// Default number of k-means iterations when clustering.
export const defaultClusterIterations = 6

// Default fraction of clusters scanned in phase 1 at query time when the
// file has cluster metadata. Lower = faster but lower recall.
export const defaultClusterProbeFraction = 0.25
