# Identity-Retrieval Evaluation

This document defines the implementation-independent benchmark for complete elephant catalog matchers. Model-specific localization, segmentation, and landmark evaluation belongs beside model training code.

## Evaluation Seam

The evaluator owns ground truth and the research Dataset. It receives a catalog matcher already composed with its image-only PhotoStore and other implementation dependencies, then hands it only neutral SightingEarPair objects and an opaque candidate-key assignment generated once for the evaluation run. The known-elephant label stays with the Dataset, which the matcher never receives; paths serve solely as PhotoStore locators, never read as strings.

The catalog matcher uses its injected PhotoStore to obtain original encoded bytes and returns a read-only mapping whose keys exactly equal the catalog keys and whose values are finite similarity floats. Larger scores mean stronger matches; no universal score range or ordering is promised. The PhotoStore exposes no identity resolution. Each call is logically independent of earlier calls, although identity-neutral caches may accelerate repeated processing.

## Retrieval Benchmark

The private retrieval benchmark set at `dataset/elephants-alive/benchmark/` contains a `manifest.csv` and the sighting folders. The manifest lists real same-sighting ear pairs - each referencing its two source Photos by `photo_id` - with their true known-elephant identities, and is what the evaluator reads; the folders hold the declared left and right photo references. Dataset resolves each `photo_id` to its canonical Photo and original bytes.

Evaluation code never derives names, dates, labels, or grouping from paths or filenames. The manifest and Dataset metadata supply those facts directly. The private benchmark set is gitignored, so no identity data enters version control.

Benchmark-set validation requires one declared left Photo and one declared right Photo per pair. Both carry the pair's sighting ID; the same Photo may serve both sides.

## Leave-One-Sighting-Out Protocol

For each protocol-eligible example:

1. Hold its SightingEarPair out as the query.
2. Exclude that sighting and its Photos from the catalog.
3. Retain every other eligible sighting, including other sightings of the query elephant.
4. Group catalog examples by private known-elephant label.
5. Replace labels with the run's fresh opaque candidate keys before calling the catalog matcher.
6. Match the query against the complete candidate set.
7. Recover the target key privately and derive its rank from the candidate scores.

A query whose elephant has no remaining catalog sighting is a protocol exclusion. Exclusions are determined before any catalog matcher runs and reported explicitly.

## Failure Accounting

Every catalog matcher uses the same protocol-eligible query denominator.

The benchmark set is curated so every selected image yields a valid two-sided extraction. Any extraction failure - query or catalog side - is therefore unexpected: it fails the run with a clear error naming the curation expectation, rather than being scored or counted. A failure signals a selection error, not a retrieval outcome. Missing photo bytes, weights, or required infrastructure, corrupt required cache state, and invalid catalog-matcher results also fail the run.

## Ranking and Metrics

Each result contains every issued candidate key exactly once, no foreign keys, and finite scores. The evaluator rejects partial results. It derives ranks from these candidate scores; ordering is a view, not part of the catalog-matcher contract.

Target rank is one plus the number of candidates with a strictly higher score. Equal scores share a competition rank.

Initial metrics are:

- top-1, top-3, top-5, top-10, and top-15 retrieval rate;
- mean reciprocal rank;
- median rank;
- protocol-eligible query count.

Uncertainty is estimated by seeded bootstrap resampling over eligible queries. System comparisons use the same resamples for paired intervals. Extraction parameters are set qualitatively from the alpha shapes; matching parameters are tuned on a separate parameter-tuning set. Neither uses the benchmark, so its numbers carry no tuning leakage.

## Reproducibility

Reproduction inputs are code, models, images, and the assigned identity data.

The assigned photo and sighting IDs are preserved artifacts rather than values derived from image content. They may be shared with reviewers under the same controlled data access as the private images.

Each report records the exact git commit and identifies the benchmark set.

`uv run eval` runs the default benchmark set with the standard cached processor composition. Parameter-tuning tools compose an uncached AlphaTear extractor with cached SAM3 feature segmentation and landmark detection. Output writing remains outside deterministic evaluation calculation.

See [ADR 0005](adr/0005-separate-retrieval-evaluation-from-implementation.md), [ADR 0007](adr/0007-pin-evaluation-by-git-commit.md), [ADR 0008](adr/0008-use-permanent-opaque-photo-identity.md), and [ADR 0010](adr/0010-cache-reusable-processing-through-composition.md).

## Future Protocols

Add another protocol only for a concrete research question; candidate directions live in [future.md](future.md).
