# Identity-retrieval evaluation

Evaluation compares complete catalog matchers on the same queries and candidate evidence.
It owns known-elephant labels, manifest validation, query folds, scores, and metrics.
Matchers receive neutral sighting ear pairs and opaque candidate keys.

## Inputs and query folds

A manifest has these columns:

```text
known_elephant_name,sighting_id,left_photo_id,right_photo_id
```

All IDs are permanent UUIDs. Dataset verifies identity assignments and photo membership.
Both selected Photos must belong to the stated sighting.
Names, grouping, and ear side do not come from filenames.

`evaluate` uses each eligible sighting as a query once.
That sighting and its photos are absent from the candidate catalog for its query.
Other sightings remain under their opaque candidate keys.
The correct elephant always remains a candidate.
An elephant with one sighting supplies distractor evidence but no eligible query.

The original tuning and benchmark manifests each contain 100 elephants and 300 pairs, with three sightings per elephant.
The sets have no shared elephant, sighting, or photo.
The requested expansion preserves these manifests and adds separate versions with three to five sightings per elephant.
[Research](research.md) records final counts, manifest locations, and results for each version.
A new result must name its manifest. It must not reuse an older set's denominator or catalog size.

## Image selection and expansion

Image quality determines selection. Retrieval scores must not determine which images enter a set.
Existing model grades determine inspection order, not automatic acceptance or rejection.
A usable pair needs readable ear-shape evidence in both selected images from the same sighting.
A reviewer can accept harmless interior occlusion, mild pose differences, or an incorrect model field.
A hidden outer margin, unreliable endpoints, or insufficient edge detail can make an image unusable.
Review records must identify the visible reason for each decision.

Sol directs Luna visual reviewers for the current expansion.
The records must distinguish agent image review from human review.
New identities must not belong to the other set or a documented model-training group.
Annotation-file overlap alone does not prove training on the current model weights.
Unknown exposure must remain explicit.

Preserve old manifest rows. Select at most five sightings per elephant.
A new elephant needs at least three accepted pairs.
Prefer different dates and screen obvious duplicates.
Freeze both expanded manifests before comparison scores are available.
Report preparation or matching failures before changing a frozen manifest.
Record user-requested exclusions in a new manifest and preserve the previous version.
The current benchmark revision excludes two added pairs at the user's request, before retrieval scores were available.

## Metrics and paired comparisons

Target rank is one plus the number of strictly higher scores.
Equal scores share a competition rank. Report target ties because this rule can affect top-k accuracy.
Store every candidate score for every query.
`EvaluationResult.ranks` and `.metrics` derive ranks and metrics from those scores.

Top-1 and top-5 are the primary measures.
Point estimates weight queries equally. More sightings give an elephant more weight in this average.
For unequal sighting counts, also report the mean of the per-elephant accuracies.
Report the number of elephants, queries, and candidate sightings.

Use paired comparisons on the same queries and candidate catalog.
Report gains, losses, net changes, and the number of affected elephants.
A two-percentage-point change on 300 queries equals six net outcomes; it need not mean only six queries changed.

`paired_delta` uses 100,000 elephant resamples with seed 42.
It retains all queries from each sampled elephant and recomputes the query-weighted difference.
The method holds catalog scores fixed. It does not rebuild catalogs or repeat selection.
Intervals omit selection, image-curation, and new-catalog uncertainty.
An interval that includes zero proves neither equivalence nor a meaningful difference.

Expanded sets change both query evidence and reference evidence.
Compare methods on identical expanded catalogs; do not attribute all changes from an old result to a larger sample.
Report original-member and added-sighting results separately when useful.
Those subgroup results still use the expanded catalog and are not direct repetitions of the old task.

## Method selection

The current seven-scale method remains unchanged during the expansion comparison.
The requested alternatives are alpha scales `1.10/12.00` and alpha `1.10` plus a convex reference.
Fix these methods before expanded-set results are available.
Use the same preparation and evaluation procedure for all three.

Earlier studies used strict two-point-gain and no-loss rules.
Those rules explain historical decisions in the research record; they are not universal practical thresholds.
The new report must show accuracy, uncertainty, execution cost, and representation size together.
No automatic promotion rule is assumed from a small observed difference.
The user will review the trade-off before the active method changes.

The historical benchmark has prior development exposure.
Additional sightings of its elephants do not create a wholly independent identity test.
The report must separate new elephants from new sightings of existing elephants.

## Failures and reproduction

A malformed manifest raises `BenchmarkValidationError`.
Missing bytes, inference failures, preparation failures, and invalid matcher results raise `EvaluationError`.
Errors stop evaluation. They do not silently reduce the denominator.
A failed complete run is not an accuracy result for the full set.
Curated-set success does not estimate field preparation coverage.

Run Python from the repository root with `uv`.
Private photos and model access are required for complete evaluation.

```bash
uv sync --all-groups
uv run eval
uv run python scripts/compare_matchers.py MANIFEST --methods alphaphant curvrank --output comparison.json
```

The comparison command requires explicit manifest and output paths and refuses an existing report.
Historical experimental commands need their saved source versions.
The active comparison command does not expose rejected research options or MiewID.
[Research](research.md#reproduction-and-preserved-evidence) lists preserved source, scores, reviews, and manifests.
