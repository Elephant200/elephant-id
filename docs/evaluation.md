# Identity-retrieval evaluation

Evaluation compares complete catalog matchers. It owns known-elephant identity, manifest validation, query/catalog splits, candidate keys, and metric calculation. Matchers receive neutral sighting ear pairs and an image-only PhotoStore, never Dataset. All returned candidate scores must be finite and cover exactly the supplied candidate keys. Ranking is derived from these scores.

## Inputs and folds

`load_benchmark(path)` reads a manifest with columns `known_elephant_name,sighting_id,left_photo_id,right_photo_id`. IDs are permanent UUIDv4 values. Dataset validates identity assignments and photo membership before matching. Names, grouping, and ear side are never inferred from filenames.

`evaluate(benchmark, dataset, matcher)` holds each eligible sighting out once. That sighting and its photos are absent from the catalog. All other sightings remain, grouped under opaque candidate keys. A sighting is eligible only if its elephant retains catalog evidence. Single-sighting elephants remain distractors but do not supply queries. The correct elephant is always among the candidates.

A malformed manifest raises `BenchmarkValidationError`. Missing bytes, model or preparation failures, and incomplete or nonfinite scores raise `EvaluationError`. The curated benchmark requires complete analysis, so errors abort evaluation; failed cases are not silently omitted. This protocol therefore does not estimate field preparation coverage. A field study must report that outcome separately.

## Metrics and uncertainty

Target rank is one plus the number of strictly higher candidate scores. Ties share a competition rank. The result retains every candidate score per query; top-k rates, reciprocal rank, median rank, and eligibility counts are derived views. Point estimates weight eligible queries equally.

Primary comparisons are top-1 and top-5. Paired percentile intervals resample queried elephants with replacement, retain all their queries, and recompute the same query-weighted difference. Use 100,000 resamples and seed 42. Pair queries and candidates by explicit keys rather than iteration order.

The bootstrap holds recorded catalog scores fixed. It accounts for repeated queries within an elephant, but does not rebuild catalogs, repeat selection, or measure curation uncertainty. Shared reference evidence can induce additional dependence. Do not describe these intervals as covering all population uncertainty.

Report metrics on the actual candidate catalog, including its size and target ties. Dataset growth changes the retrieval task; do not normalize results to a hypothetical fixed pool. Compare matchers on identical query and candidate sets using `evaluation.comparison.paired_delta`. `EvaluationResult.ranks` and `.metrics` also support saved-score inspection without constructing a matcher.

## Development and confirmation

Develop on identity-disjoint tuning catalogs A and B, each with 89 elephants and histogram `{1:36, 2:25, 3:20, 4:7, 5:1}`. Fixed image-quality gates and machine visual review precede matching. Analyzability alone is not image quality. Both catalogs guide selection and are development data, not independent final tests.

A new mechanism must avoid losses at both primary endpoints in both catalogs and improve at least one endpoint in each. Choose shared plateaus, not sweep maxima. Exact Shapley covers at most eight binary decisions. Retain positive contributions in each catalog whose mean across catalogs is at least 0.5 points at both primary endpoints. Document bundled and dependent decisions explicitly.

Prefer fewer decisions when observed loss is within one query per catalog and paired intervals do not establish a loss. This is a practical parsimony rule, not a formal noninferiority test. Benchmark attributions cannot guide pruning.

Confirmation requires fixed source, extraction producers, model identities, manifest hashes, comparator settings, and endpoints. Use the same preparation and folds for all matchers. The benchmark is not a historically untouched set; stronger generalization claims require a genuinely independent cohort.

## Reproduction

Run Python from the repository root with `uv`. Install local inference and development dependencies with `uv sync --all-groups`. Private photos and model access are required for end-to-end evaluation.

```bash
uv run eval
uv run python scripts/compare_matchers.py MANIFEST --output comparison.json
uv run python scripts/ablation.py MANIFEST --reference comparison.json --output ablation.json
```

Comparison and ablation commands require explicit paths and refuse to overwrite existing result reports. Matrix reuse checks the script, active source, numerical runtime, settings, producers, and manifest. MiewID uses `conservationxlabs/miewid-msv3` revision `4f1d7f2b521149e5fe34bb85f377248ce9971a7d`. It is a separate zero-shot comparator.

The completed study's private evidence is in `.scratch/alphaphant_optimization/evidence.tar.gz`. Its index describes source snapshots, immutable score tables, tuning manifests, quality records, and hashes. The archive includes the numerical source used for the published confirmation; current score-preserving engineering changes are identified separately.

After extracting the evidence and the locked source into separate directories, verify existing results without model inference or photo access:

```bash
uv run python scripts/audit_publication.py EVIDENCE --source LOCKED_SOURCE --output audit.json
```

The audit checks source hashes, score coverage, finite values, target ties, metrics, paired intervals, and all 128 exact ablation subsets. Saved-score verification is distinct from a clean-environment end-to-end reproduction. Never commit private photos, manifests, canonical identity-bearing scores, or model secrets. Controlled data access is needed for independent reproduction.

## Excluded mechanisms

These mechanisms do not belong in the production configuration. Retest one only with a new mechanism not covered by its existing evidence.

| Family | Excluded alternatives |
|---|---|
| Geometry and representation | Radial depth, negative depth, free per-bin weights, depth deadbanding, whole-profile cosine substitution |
| Alignment | DTW/per-segment elasticity, independent scale alignment, shared depth/change-channel alignment, symmetric comparison in the selected composition |
| Local matching | Window LNBNN and local winning intervals |
| Scale or channel combination | Rank fusion, scale-subset winners, per-scale CSLS, scale/channel z-normalization |
| Catalog correction | Query-side CSLS constant, own-candidate exclusion, incoming background similarity, candidate-balanced neighbors |
| Evidence combination | Ear-decisiveness weighting, paired-sighting evidence, neighbor diffusion across queries |

The candidate-balanced and paired-sighting alternatives have mixed-sign exact Shapley contributions across tuning catalogs. The query-side CSLS term is a candidate-independent constant and adds no ranking information. MiewID fusion is outside the AlphaPhant-alone result and adds no component to this implementation.

## Scientific claim boundaries

Selection on finite data can overfit even without fitting a neural network; see [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html). Exact subset attribution does not remove sampling or selection uncertainty. A prospective replication should fix eligibility, quality criteria, model versions, endpoints, an elephant-level sample-size rationale, and a stopping rule before outcomes are seen. Include independent quality review, near-duplicate screening, inference-training overlap checks where records permit them, temporal splits when relevant, and explicit preparation-failure coverage.
