// Version of the parquet vector format
export const hypVectorVersion = 0

// Default rows per row group when not clustered. Larger row groups mean
// fewer fetches for a full exact scan, which dominates the no-cluster path.
// When clustering is enabled, each cluster becomes its own row group
// (variable sizes), so this default doesn't apply.
export const defaultRowGroupSize = 10000

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

// Upper bound on clusters probed under the *default* fraction. Clusters grow
// as ~sqrt(N)/2, so 0.25 x nlist keeps rising with N; measured recall knees
// well before that at scale (~92% at 80-96 lists on 1M x 1024, vs 93% at the
// uncapped 125). Capping the default trims ~25% of roundtrips and ~30% of
// bytes above ~400k vectors for ~1pp recall. Only applies when `probe` is
// left default; an explicit `probe` is honored literally.
export const defaultClusterProbeCap = 96

// When `binary` is not specified at write time, the column is added once
// the corpus is at least this large. Below the threshold, exact full scan
// is fast enough that the rerank path's overhead isn't worth the column.
export const defaultAutoBinaryThreshold = 10000
