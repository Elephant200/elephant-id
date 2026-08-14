# Cache immutable expensive producers

**Status:** amended by [ADR 0008](0008-use-permanent-opaque-photo-identity.md)

## Decision

One generic cache store preserves independently expensive model operations and final per-ear tear profiles while leaving orchestration uncached. Immutable producer names carry model, weights, prompt, preprocessing, thresholds, and every other output-changing configuration choice. Photo-level keys begin with the permanent photo UUID and add only actual dependent inputs such as crop coordinates or side. Keys remain readable, writes are atomic, and loaded records are validated.
