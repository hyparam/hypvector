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
// as ~sqrt(N)/2, so 0.25 x nlist keeps rising with N, but the clusters needed
// to reach the recall ceiling stay roughly flat (~25-45) regardless of N. A
// WildChat 1024-dim sweep found 48, 72, and 96 lists give statistically
// indistinguishable recall@10 at 1M and 3.2M (within ~1pp over 20 exact-scan
// queries, no consistent direction). Their top-10 sets are not bit-identical:
// over 200 queries, cap 48 matches cap 96 on ~93% (1M) to ~97% (3.2M), the
// rest reshuffling near-ties at the list boundary, not losing true neighbors.
// Capping at 48 reads ~42% fewer bytes than 96 at scale with no measurable
// recall loss; structurally, shrinking the cap can only lose recall, never
// gain it, since probed clusters are a subset. Residual misses are a
// rerankFactor limit, not a probe limit. Only applies when `probe` is left
// default; an explicit `probe` is honored literally.
export const defaultClusterProbeCap = 48

// When `binary` is not specified at write time, the column is added once
// the corpus is at least this large. Below the threshold, exact full scan
// is fast enough that the rerank path's overhead isn't worth the column.
export const defaultAutoBinaryThreshold = 10000
