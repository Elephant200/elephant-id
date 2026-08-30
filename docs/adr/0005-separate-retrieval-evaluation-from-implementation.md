# Separate identity-retrieval evaluation from implementation

**Status:** amended by [ADR 0008](0008-use-permanent-opaque-photo-identity.md)

The identity-retrieval evaluator owns ground-truth labels, real same-sighting ear pairs, leakage prevention, evaluation protocol, derived ranks, metrics, and reproducibility. Matching owns the shared `CatalogMatcher` seam; evaluation is its deliberately simple consumer. It receives a matcher already composed with its image-only PhotoStore and other implementation dependencies. The matcher receives neutral SightingEarPair values and a run-scoped opaque candidate-key assignment, then returns an exact candidate-key-to-finite-float mapping. `match` is logically stateless: identity-neutral caches may affect performance, but prior calls never affect scores. The matcher does not receive the identity-aware Dataset, known-elephant labels, paths, dates, masks, landmarks, contours, tear profiles, model configuration, or matching internals.
