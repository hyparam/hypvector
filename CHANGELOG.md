# Changelog

## [0.1.1]
 - Add `prefetchBinary` to load the binary column into memory once at startup; pass the result as `binary` to `searchVectors` to skip phase-1 fetches on every subsequent query
 - When clustering is enabled, write one parquet row group per cluster so phase-1 binary scans and phase-2 candidate fetches stay within a single column chunk per cluster

## [0.1.0]
 - Initial release
