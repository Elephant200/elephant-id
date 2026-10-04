# Retrieval Evaluation

Retrieval evaluation measures complete catalog matchers. It does not depend on one matcher implementation. Each localization, segmentation, and landmark model has its own evaluation beside its training code.

## Interface

- `load_benchmark(path, dataset) -> Benchmark` reads a benchmark file. It returns the sighting ear pairs grouped by known elephant.
- `evaluate(benchmark, matcher) -> EvaluationResult` runs the protocol.

Only the evaluator sees the known-elephant names. The catalog matcher does not see them.

For each query, the evaluator gives the catalog matcher:

- the query sighting ear pair;
- the catalog, as sighting ear pairs grouped by candidate key.

The catalog matcher returns one finite similarity score for each candidate key. It must not add or omit a key. A larger score is a stronger match.

## Benchmark Files

`dataset/elephants-alive/benchmark/` is private and gitignored. It contains:

| File | Content |
|---|---|
| `all_approved_sighting_ear_pairs.csv` | All approved sighting ear pairs. |
| `tuning_final.csv` | The tuning set. |
| `benchmark_final.csv` | The benchmark set. `uv run eval` uses this file by default. |
| `tuning_v2.csv`, `benchmark_v3.csv` | Earlier versions of the two sets. |

Each file has exactly these columns: `known_elephant_name,sighting_id,left_photo_id,right_photo_id`. Each row is one sighting ear pair. One photo can supply both ears.

The split uses only elephants with three or more approved sightings. Each elephant and all its sightings go into one set. The two sets have equal sighting counts and similar sightings-per-elephant distributions. The sets share no elephant, sighting, or photo. Thus, tuning does not use benchmark elephants.

## Validation

`load_benchmark` checks the columns. It checks each row against the Dataset and rejects a repeated sighting. It stops at the first bad row, and the error gives the file and row number.

## Leave-One-Sighting-Out Protocol

For each eligible query:

1. Use its sighting ear pair as the query.
2. Remove the query sighting from the catalog. Keep all other sightings in the catalog, including ineligible queries.
3. Replace each known elephant in the catalog with a candidate key.
4. Match the query against all matching candidates.
5. Find the rank of the true known elephant from the similarity scores.

A query is ineligible if no other sighting of its elephant is in the set. Ineligible queries stay in the catalog as distractors.

## Failures

A failure stops evaluation. The error gives the query sighting ID. Evaluation does not return a partial result.

Curation makes sure that each selected photo gives a valid extraction for both ears. Thus, an extraction failure is a curation error, not a retrieval result.

## Ranks and Metrics

The rank of the true known elephant is one plus the number of candidates with a higher score. Candidates with equal scores have the same rank.

Metrics:

- top-1, top-3, top-5, top-10, and top-15 retrieval rate;
- mean reciprocal rank;
- median rank;
- eligible query count.

All eligible queries have equal weight in the point estimates.

The intervals are two-sided 95% percentile bootstrap intervals. The bootstrap uses 100,000 resamples and seed `42`. Each resample selects eligible elephants with replacement and keeps all eligible queries of each selected elephant. To compare two systems, use the same resamples for both.

## Evaluation Result

`EvaluationResult` has three mappings:

- `scores`: true known elephant → query sighting ID → matching candidate → similarity score.
- `metrics` and `intervals`: calculated from `scores`.

## Entry Point

`uv run eval [path]` builds standard AlphaPhant and evaluates `benchmark_final.csv` or the given file. It prints the metrics and intervals.

Related decisions: [ADR 0005](adr/0005-separate-retrieval-evaluation-from-implementation.md), [ADR 0007](adr/0007-pin-evaluation-by-git-commit.md), [ADR 0008](adr/0008-use-permanent-opaque-photo-identity.md), [ADR 0010](adr/0010-cache-reusable-processing-through-composition.md).
