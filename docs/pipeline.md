# AlphaPhant Pipeline

This document defines the research algorithm being locked down. Domain, storage, and cache boundaries are defined in [architecture.md](architecture.md).

## Input and Output

The input is a SightingEarPair: one Photo declared for the left ear and one Photo declared for the right ear from the same sighting. The same Photo may serve both sides.

The output is a complete ranked list of known elephants. Each candidate carries:

- its combined similarity score;
- its strongest supporting left-ear catalog evidence;
- its strongest supporting right-ear catalog evidence;
- the two side-level similarity scores and alignments.

AlphaPhant ranks candidates. It does not make an identity decision or add evidence to the catalog.

## Automated Preprocessing

Each reference Photo follows the same sequence:

1. **Photo retrieval and decoding** read original encoded bytes from the PhotoStore and decode them to a BGR image.
2. **Ear localization and ear segmentation** find the declared ear and produce its mask and ear contour.
3. **Ear landmark detection** finds the two anatomical endpoints that define the relevant contour.
4. **Ear-contour preparation** snaps the landmarks to the segmentation contour, selects the relevant anchor-to-anchor path, and determines the side-aware geometry.
5. **Tear-profile extraction** runs AlphaTear, which constructs an alpha shape internally and produces the one-dimensional tear profile.

The declared side is authoritative. If the pipeline cannot produce a valid ear of that side, analysis fails explicitly. Research adds no fallback selection after the ear pair has been chosen.

Candidate reduction uses two intentionally different geometric measures. The legacy preliminary heuristic compares segmentation-mask pixel area before landmark detection; after preparation, declared-side disambiguation compares the filled cleaned-contour area and preserves input order for exact ties.

SAM3 currently performs ear localization and segmentation. A YOLO keypoint model currently performs ear landmark detection. Replacements use the semantic inference interfaces in [architecture.md](architecture.md).

## Tear-Profile Matching

Tear-profile matching compares left ears only with left ears and right ears only with right ears.

The underlying alignment is directional because one profile is shifted and stretched against the other. The canonical pair score removes that arbitrary direction:

```
symmetric_similarity(a, b) =
    (similarity(a -> b) + similarity(b -> a)) / 2
```

This raw similarity score is used for ranking. Cohort normalization and learned calibration are outside the selected pipeline.

## Catalog Ranking

Catalog sightings are not matching units. Once tear-profile evidence enters the known-elephant catalog, it is grouped by known elephant and ear side.

For each known elephant:

1. Compare the query left profile with every left catalog profile and retain the highest similarity score.
2. Compare the query right profile with every right catalog profile and retain the highest similarity score.
3. Average the retained left and right scores.

The winning left and right evidence may come from different catalog sightings. Their separate Photo and sighting provenance remains visible.

Both sides are required: a known elephant is scored only when it has valid left and right catalog evidence. One-sided matching is future work.

## Caching

Expensive model invocations and final per-ear tear profiles are cached. Ear selection, sighting orchestration, contour preparation, catalog grouping, and ranking are not independently cached.

A final tear-profile record is per prepared ear. Its key contains the source photo UUID, integer raster bounding box, inferred side, segmentation producer slug, and landmark producer slug. Bounding-box identity remains stable when candidate ordering changes. Cache identity and producer versioning are defined in [architecture.md](architecture.md).

Caching is selected when processors are composed. Standard runs cache the complete SAM3 multi-feature computation before its ear-only adapter, landmark detection, and AlphaTear extraction. Parameter-tuning runs use a raw unversioned AlphaTear extractor while retaining cached SAM3 features and landmark detection. The analyzer and ranker expose no cache-policy options.
