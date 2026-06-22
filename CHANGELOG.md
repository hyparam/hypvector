# Changelog

## [0.2.0]
 - `searchVectors` accepts arrays for `source`, `metadata`, and `binary` to query across multiple parquet files in one call; results are heap-merged into a global top-K and each carries a `sourceIndex`. Single-source callers are unchanged
 - `writeVectors` auto-tunes `binary` and `clusters` from the input size: binary turns on at N >= 10000 and `clusters` defaults to round(sqrt(N)/2) when both are left unset. Passing either flag explicitly opts out of that knob's auto behavior
 - `normalize` now defaults to `true` in `writeVectors`. Cosine on normalized vectors reduces to dot product; pass `normalize: false` to opt out for magnitude-sensitive dot/euclidean metrics
 - `writeVectors` gains a streaming path: when `binary` is set and clustering is off, it flushes one row group at a time, lowering peak memory from O(N) to O(row group)
 - The default cluster `probe` is capped at 96 lists so the probed-list count stops growing with N once recall saturates; an explicit `probe` bypasses the cap
 - Fix clustered binary (Hamming) scan that could throw a RangeError or score the wrong bytes when a cluster's rows span multiple parquet pages
 - Drop the redundant `hypvector.count` KV metadata; readers take the count from the parquet footer's row count

## [0.1.1]
 - Add `prefetchBinary` to load the binary column into memory once at startup; pass the result as `binary` to `searchVectors` to skip phase-1 fetches on every subsequent query
 - When clustering is enabled, write one parquet row group per cluster so phase-1 binary scans and phase-2 candidate fetches stay within a single column chunk per cluster

## [0.1.0]
 - Initial release
