# AlphaPhant Pipeline

This document defines the research algorithm being locked down. Package names and concrete data classes may change; the behavior and seams described here are the planning baseline.

## Input and Output

The input is a sighting ear pair: one full photo flagged for a high-quality left-ear view and one full photo flagged for a high-quality right-ear view. Both photos come from the same real sighting.

The output is a complete ranked list of known elephants. Each candidate carries:

- its combined similarity score;
- its strongest supporting left-ear catalog evidence;
- its strongest supporting right-ear catalog evidence;
- the two side-level similarity scores and alignments.

AlphaPhant ranks candidates. It does not make an identity decision or add evidence to the catalog.

## Automated Preprocessing

Each reference photo follows the same sequence:

1. **Ear localization and ear segmentation** find the declared ear and produce its mask and ear contour.
2. **Ear landmark detection** finds the two anatomical endpoints that define the relevant contour.
3. **Ear-contour preparation** snaps the landmarks to the segmentation contour, selects the relevant anchor-to-anchor path, and determines the side-aware geometry.
4. **Tear-profile extraction** constructs an alpha shape from the ear contour and produces the one-dimensional tear profile.

The declared side is authoritative for a research ear pair. If the pipeline cannot produce a valid ear of that side, analysis fails explicitly. Research does not add fallback selection heuristics after the pair has been chosen.

SAM3 currently performs ear localization and segmentation. A YOLO keypoint model currently performs ear landmark detection. Their replacements use the semantic inference interfaces in [architecture.md](architecture.md).

## Tear-Profile Matching

Tear-profile matching compares left ears only with left ears and right ears only with right ears.

The underlying alignment is directional because one profile is shifted and stretched against the other. The canonical pair score removes that arbitrary direction:

```
symmetric_similarity(a, b) =
    (similarity(a -> b) + similarity(b -> a)) / 2
```

This is the raw similarity score used for ranking. Cohort normalization and learned calibration are not part of the selected pipeline.

## Catalog Ranking

Catalog sightings are not matching units. Once tear-profile evidence enters the known-elephant catalog, it is grouped only by known elephant and ear side.

For each known elephant:

1. Compare the query left profile with all left catalog profiles and retain the highest similarity score.
2. Compare the query right profile with all right catalog profiles and retain the highest similarity score.
3. Average the retained left and right scores.

The winning left and right evidence may come from different catalog sightings. Their separate provenance remains visible in the result.

Both sides are required. If a known elephant has insufficient valid catalog evidence after extraction, it remains in a complete evaluation ranking with score `0.0` and an insufficient-evidence status. One-sided matching is future work.

## Caching

Expensive model invocations and final per-ear tear profiles are cached. Ear selection, sighting orchestration, contour preparation, catalog grouping, and ranking are not independently cached.

A final tear-profile record is per ear, not per pair or sighting. Its key depends on the segmentation record, landmark record, declared side, and the integer bounding box of the selected ear detection. Keying on the bounding box rather than a positional index is order-independent, so a stale hit cannot occur when the segmentation record holds more than one candidate of that side or when detection ordering shifts. This permits the same evidence to be reused in queries, catalogs, and evaluation folds.

See [architecture.md](architecture.md) for the common cache mechanism and [adr/0006-cache-immutable-expensive-producers.md](adr/0006-cache-immutable-expensive-producers.md) for the decision.

## Verification Boundary

Restructuring must preserve the current numerical core through characterization tests:

- anchor-to-anchor ear-contour construction;
- side inference;
- alpha-shape-derived tear profiles;
- directional tear-profile similarity;
- directional score symmetrization;
- independent per-side catalog maxima;
- arithmetic combination of left and right scores.

Use exact comparisons where deterministic and tight numerical tolerances where numerical libraries require them. End-to-end unit tests use fake inference implementations and synthetic images; they do not require model weights, network access, or private photos.
