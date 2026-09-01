# Identity-Retrieval Evaluation

This document defines the implementation-independent benchmark for complete elephant catalog matchers. Model-specific localization, segmentation, and landmark evaluation belongs beside model training code.

## Evaluation Seam

The evaluator owns ground truth and the research Dataset. It receives a catalog matcher already composed with its image-only PhotoStore and other implementation dependencies, then hands it only neutral SightingEarPair objects and an opaque candidate-key assignment generated once per call to `evaluate`. The known-elephant label stays with the Dataset, which the matcher never receives; paths serve solely as PhotoStore locators, never read as strings.

The catalog matcher uses its injected PhotoStore to obtain original encoded bytes and returns a read-only mapping whose keys exactly equal the catalog keys and whose values are finite similarity floats. Larger scores mean stronger matches; no universal score range or ordering is promised. The PhotoStore exposes no identity resolution. Each call is logically independent of earlier calls, although identity-neutral caches may accelerate repeated processing.

The package exposes `load_benchmark(path) -> RetrievalBenchmark` and `evaluate(benchmark, dataset, matcher) -> EvaluationResult`. `RetrievalBenchmark` is a simple parsed benchmark manifest: it retains the declared known-elephant names and sighting and Photo IDs, but no Dataset, resolved domain objects, paths, or PhotoStore. Evaluation owns the fixed retrieval protocol and returns its deterministic scientific result. The package defines no evaluation-run or report value.

## Retrieval Benchmark

The private retrieval benchmark set at `dataset/elephants-alive/benchmark/` contains a `manifest.csv`. Each row has exactly `known_elephant_name,sighting_id,left_photo_id,right_photo_id`: one real same-sighting ear pair selected by permanent IDs, with its true known-elephant identity used only by evaluation. Dataset resolves each `photo_id` to its canonical Photo and original bytes.

Evaluation code never derives names, dates, labels, or grouping from paths or filenames. The manifest and Dataset metadata supply those facts directly. The private benchmark set is gitignored, so no identity data enters version control.

The benchmark is a sample used to estimate expected closed-set retrieval performance beyond the benchmark itself, not merely a fixed collection whose descriptive statistics are the scientific endpoint. An unseen elephant is unseen during system development but represented by prior catalog evidence when queried. Detailed population and deployment claims belong in scientific reporting rather than this package contract.

Primary metric point estimates treat every eligible query equally for compatibility and clarity. Elephants are the independent sampling unit for uncertainty; sightings and eligible queries are repeated observations nested within elephants.

`load_benchmark` reports all manifest-internal errors together through one `BenchmarkValidationError`. It validates exact columns, canonical UUIDv4 values, required fields, and duplicate declarations without resolving Dataset metadata.

Before its first catalog-matcher call, `evaluate` resolves the complete benchmark against the separately supplied Dataset and reports all cross-reference errors together through `BenchmarkValidationError`. Every sighting and Photo must resolve, the declared identity must agree with Dataset, and the declared left and right Photos must carry the row's sighting ID. The same Photo may serve both sides. Missing bytes, models, corrupt caches, extraction failures, and invalid matcher results raise `EvaluationError`; no partial result is returned.

## Leave-One-Sighting-Out Protocol

For each eligible query:

1. Hold its SightingEarPair out as the query.
2. Exclude that sighting and its Photos from the catalog.
3. Retain every other benchmark sighting, including other sightings of the query elephant and ineligible queries.
4. Group catalog examples by private known-elephant label.
5. Replace labels with `evaluate`'s fresh opaque candidate keys before calling the catalog matcher.
6. Match the query against the complete candidate set.
7. Recover the target key privately and derive its rank from the candidate scores.

A query whose elephant has no remaining catalog sighting is ineligible. Its evidence remains in other queries' catalogs, so ineligibility does not alter valid distractor evidence. Ineligible queries are determined before any catalog matcher runs and reported explicitly.

## Failure Accounting

Every catalog matcher uses the same eligible-query denominator.

The benchmark set is curated so every selected image yields a valid two-sided extraction. Any extraction failure - query or catalog side - is therefore unexpected: the first runtime failure aborts evaluation with query and Photo context and a clear error naming the curation expectation, rather than being scored or counted. A failure signals a selection error, not a retrieval outcome. Missing photo bytes, weights, or required infrastructure, corrupt required cache state, and invalid catalog-matcher results also abort evaluation.

`EvaluationStage` has two values: `CATALOG_MATCHING` when the matcher raises while processing a query fold, and `MATCHER_RESULT_VALIDATION` when returned keys or scores violate the matcher interface. `EvaluationError` records the stage, active query sighting ID, and any recoverable query/catalog role, photo ID, and side. An underlying `SightingAnalysisError` remains the cause and retains its more precise analysis stage. `BenchmarkValidationError` remains separate because it aggregates declaration problems before evaluation starts. Metric or bootstrap failures after validated finite scores are programming defects, not another public evaluation stage.

## Ranking and Metrics

Each result contains every issued candidate key exactly once, no foreign keys, and finite scores. The evaluator rejects partial results. It derives ranks from these candidate scores; ordering is a view, not part of the catalog-matcher contract.

Target rank is one plus the number of candidates with a strictly higher score. Equal scores share a competition rank.

Initial metrics are:

- top-1, top-3, top-5, top-10, and top-15 retrieval rate;
- mean reciprocal rank;
- median rank;
- eligible query count.

Primary metric point estimates are unweighted over eligible queries. Two-sided 95% percentile intervals use 100,000 bootstrap resamples with seed `42`: sample eligible elephants with replacement, retain all eligible queries nested within each sampled elephant, and recompute the ordinary unweighted query metrics over the resampled observations. Bootstrap calculation uses completed query scores and never reruns matching. Paired system comparisons reuse the same sampled elephant indices. Extraction parameters are set qualitatively from the alpha shapes; matching parameters are tuned on a separate parameter-tuning set. Neither uses the benchmark, so its numbers carry no tuning leakage.

## Reproducibility

Reproduction inputs are code, models, images, and the assigned identity data.

The assigned photo and sighting IDs are preserved artifacts rather than values derived from image content. They may be shared with reviewers under the same controlled data access as the private images.

`EvaluationResult` is the deterministic scientific value returned by `evaluate`; evaluation defines no separate run or report value. For every eligible query it retains every candidate similarity score mapped privately back to known-elephant name. It does not retain the input `RetrievalBenchmark`. Target ranks, aggregate metrics, and uncertainty intervals are reconstructed from the canonical scores rather than recorded independently. Candidate keys, model intermediates, and matching provenance are not retained.

Its interface consists of three mappings: `scores`, `metrics`, and `intervals`. `scores` is grouped as true known-elephant name to query sighting ID to candidate known-elephant name to finite similarity float. `metrics` and `intervals` are deterministic derived views of those canonical scores. Evaluation exposes no folds, stored ranks, candidate keys, serialization, or matcher provenance through the result.

The thin CLI composes standard AlphaPhant and prints metrics and intervals. It defines no report or persistence model. A parameter-tuning entry point may reuse `evaluate` with trial-specific matcher composition: it uses a raw AlphaTear extractor while retaining cached SAM3 feature segmentation and landmark detection.

See [ADR 0005](adr/0005-separate-retrieval-evaluation-from-implementation.md), [ADR 0007](adr/0007-pin-evaluation-by-git-commit.md), [ADR 0008](adr/0008-use-permanent-opaque-photo-identity.md), and [ADR 0010](adr/0010-cache-reusable-processing-through-composition.md).

## Future Protocols

Add another protocol only for a concrete research question; candidate directions live in [future.md](future.md).
