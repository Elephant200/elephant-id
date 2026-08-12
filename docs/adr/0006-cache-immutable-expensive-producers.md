# Cache immutable expensive producers by content hash

## Decision

One generic cache store preserves outputs from independently expensive, immutably named model operations and final per-ear tear-profile extraction, while inexpensive orchestration remains uncached. Cache keys follow the actual dependency chain from source content hashes and upstream record keys; any output-changing producer change requires a new producer name, and human-readable paths and identifiers remain metadata rather than cache identity.
