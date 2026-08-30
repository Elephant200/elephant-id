# Separate identity-retrieval evaluation from implementation

**Status:** amended by [ADR 0008](0008-use-permanent-opaque-photo-identity.md)

The identity-retrieval evaluator owns ground-truth labels, real same-sighting ear pairs, leakage prevention, evaluation protocol, derived ranks, metrics, and reproducibility. A `CatalogMatcher` under evaluation receives neutral Photo and SightingEarPair values, an image-only PhotoStore, and opaque catalog candidate keys, then returns one finite similarity score per candidate. It does not receive the identity-aware Dataset, known-elephant labels, paths, dates, masks, landmarks, contours, tear profiles, model configuration, or matching internals.
