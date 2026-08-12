# Identity-Retrieval Evaluation

This document defines the implementation-independent benchmark for complete AlphaPhant systems. Ear localization, ear segmentation, and ear landmark detection use separate model-specific evaluations beside their future training code.

## Evaluation Seam

The evaluator owns ground truth. A system under evaluation receives:

- opaque photo keys for each query ear pair;
- opaque candidate keys grouping the catalog evidence it may use.

It returns every candidate key exactly once with a finite similarity score, ordered from highest to lowest. The evaluator does not expose elephant names, original file paths, dates, masks, anchors, ear contours, tear profiles, model settings, or pipeline classes.

An evaluation adapter resolves opaque photo keys to images internally and composes the concrete analysis and matching implementation. The exact adapter and internal data representation remain implementation choices.

## Evaluation Suite

The evaluation suite is a directory of real sighting ear pairs authored by the image picker, held in the private dataset at `dataset/elephants-alive/sighting-pairs/`. It is organized as `<known elephant>/<sighting>/<photo_id>_<side>.jpg`, so the directory structure itself declares each pair, its sighting, and its ground-truth identity. A pair is exactly the two side files under one sighting directory; nothing is manufactured by shuffling or independently joining left and right photos.

Because the suite lives under the gitignored dataset it is never committed, so no private identity data enters version control and no separate manifest or key-resolver file is needed. Exact-benchmark reproduction reads this directory directly, consistent with [adr/0007-pin-evaluation-by-git-commit.md](adr/0007-pin-evaluation-by-git-commit.md); a reader without the private data gets the code, the method, and the reported metrics. Image hashes stay authoritative in the dataset hash index.

For leakage prevention the evaluator assigns opaque photo and candidate keys in memory before calling a system and resolves them to images internally. These keys are a per-run device, not a stored artifact. Suite validation requires exactly one left and one right file per sighting directory and rejects a source image reused across pairs; cross-sighting pairs cannot occur by construction.

## Leave-One-Sighting-Out Protocol

For each protocol-eligible example:

1. Hold its sighting ear pair out as the query.
2. Exclude that exact sighting and its photos from the catalog.
3. Retain every other eligible sighting, including other sightings of the query elephant.
4. Group catalog examples by private known-elephant label.
5. Replace labels with fresh opaque candidate keys before calling the system.
6. Rank the complete candidate set.
7. Recover the target key privately and record its rank.

A query whose elephant has no remaining catalog sighting is a protocol exclusion. Exclusions are determined once before any implementation runs and are reported explicitly.

## Failure Accounting

Every implementation uses the same protocol-eligible query denominator.

- A query extraction failure counts as a retrieval miss and is reported by side and reason. A miss receives the sentinel rank of catalog size plus one, so median rank stays well defined; its reciprocal-rank contribution is zero.
- Catalog extraction failure does not remove a candidate. A candidate lacking valid two-sided catalog evidence receives score `0.0` with an insufficient-evidence status so the candidate universe stays complete. When the query's own target is that `0.0` candidate, the query is scored as a miss rather than a competition-rank hit tied with other `0.0` candidates.
- Missing weights, unreadable infrastructure, corrupt required cache state, or an invalid system result fails that implementation run.

Every metric uses the same protocol-eligible denominator, so no metric silently reports over a different population. This prevents a system from improving its metrics by silently dropping difficult examples.

## Ranking and Metrics

The benchmark requires a complete ranking because mean reciprocal rank and median rank cannot be recovered from a top-k-only result.

Each result must contain:

- every issued candidate key exactly once;
- no foreign or duplicate keys;
- finite similarity scores;
- descending score order;
- deterministic candidate-key ordering for equal scores.

The target rank is one plus the number of candidates with a strictly higher score. Equal scores therefore share the same competition rank.

Initial metrics are:

- top-1, top-3, top-5, top-10, and top-15 retrieval rate;
- mean reciprocal rank;
- median rank;
- protocol-eligible query count;
- query extraction-failure rate and reasons;
- catalog insufficient-evidence rate.

Because the protocol is deterministic, uncertainty is reported by bootstrapping over the query set — resample the eligible queries with replacement, recompute the metrics, and take percentile intervals — rather than by averaging random seeds. A seeded bootstrap keeps the intervals reproducible, and two systems are compared on the same resamples (paired bootstrap) so a difference can be judged for significance rather than read off two point estimates.

Extraction and matching hyperparameters, such as the alpha value and the tear-profile bin count, are set independently of the retrieval metric — qualitatively, from the resulting shapes — and are not tuned on this evaluation set. The reported numbers are therefore not optimistic through hyperparameter leakage.

## Reproducibility

Each report is pinned by the git commit that produced it, marked dirty when the working tree has uncommitted changes, and identifies the sighting-pair directory it ran against. There are no separate suite, plan, or system fingerprints.

The reproduction inputs are code, models, and images. Caches only accelerate and never change a result, so a fresh reproduction starts from an empty cache and computes everything from scratch. Because the images are private, exact-number reproduction is internal; an outside reader gets the code, the method, and the reported metrics. Hosted-model outputs (SAM3) are recorded rather than re-called, standing in for a model that cannot be shipped.

The main evaluation is one command, `uv run eval`, runnable from the README against the committed default suite. A `--force` flag recomputes tear profiles so a publication run cannot report stale numbers. Output writing is separate from evaluation so the evaluator itself remains deterministic and side-effect free.

See [adr/0007-pin-evaluation-by-git-commit.md](adr/0007-pin-evaluation-by-git-commit.md).

## Future Protocols

Add another protocol only for a concrete research question. Possible later work includes:

- fixed identity- and time-aware catalog/query partitions;
- uncertainty estimates across multiple real sightings;
- open-set evaluation;
- larger-catalog or approximate-retrieval evaluation;
- statistical comparison of complete systems.
