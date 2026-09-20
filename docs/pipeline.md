# AlphaPhant Pipeline

This document defines the current research algorithm. Domain, storage, and cache boundaries are defined in [architecture.md](architecture.md).

## Input and Output

The input is a SightingEarPair: one Photo declared for the left ear and one Photo declared for the right ear from the same sighting. The same Photo may serve both sides.

`AlphaPhant.match` returns one finite similarity score per candidate. Each side comparison keeps its reference ear and query alignment. The matcher selects the strongest reference for each side and averages the two scores.

`AlphaPhant.analyze` returns the input used for matching. Its immutable `AnalyzedSightingEarPair` contains the sighting ID and the left and right `AnalyzedEar` values. Each ear contains one signed `TearProfile`, its source Photo, anatomical side, and raster box.

A candidate ranking is the descending view of scores; AlphaPhant does not make an identity decision or add evidence to the catalog.

## Automated Preprocessing

Query and reference ears follow the same sequence:

1. **Photo retrieval and decoding** read original encoded bytes from the PhotoStore and decode them to a BGR image.
2. **Ear localization and ear segmentation** find ear locations and produce masks and boxes.
3. **Ear landmark detection** finds the two anatomical endpoints that define the relevant contour.
4. **Ear-contour preparation** extracts the contour from the mask, snaps the landmarks to it, selects the relevant anchor-to-anchor path, and determines the side-aware geometry.
5. **Declared-side resolution** selects the prepared ear for each requested side. Both sides are resolved before extraction starts.
6. **Tear-profile extraction** runs AlphaTear, which constructs an alpha shape internally and produces the one-dimensional tear profile. AlphaPhant attaches source metadata and the original sighting ID.

The declared side is authoritative. If the pipeline cannot produce a valid ear of that side, analysis fails explicitly. Research uses the selected ear pair without fallback selection.

Candidate reduction uses two intentionally different geometric measures. The preliminary heuristic compares segmentation-mask pixel area before landmark detection; after preparation, declared-side disambiguation compares the filled cleaned-contour area and preserves input order for exact ties.

SAM3 currently performs ear localization and segmentation. A YOLO keypoint model currently performs ear landmark detection. Replacements use the semantic inference interfaces in [architecture.md](architecture.md).

## Tear-Profile Extraction

AlphaTear resamples the prepared contour and derives its radius from cleaned ear area.
Morphological opening, controlled by `morph_opening_fraction`, supplies a contour for
building the alpha-shape reference. The resampled, unopened contour remains the
measured contour.

Original detector landmarks define the polar frame. Rays at the retained profile
angles locate origins on the reference boundary; local inward normals determine
which direction to measure from each origin to the measured contour. Signed nearest
crossing distances are divided by the area-derived radius, smoothed, and placed on
the full angular grid. Trimmed bins and missing-origin bins stay zero. Snapped
contour anchors delimit the prepared contour; they do not replace detector landmarks
in the polar frame.

## Tear-Profile Matching

AlphaPhant compares ears from the same side. It moves only the query profile. Each reference profile stays fixed.

The comparator replaces negative depths with zero, compresses the depths, and resamples each profile to a common grid. It then applies each configured stretch and shift to the query.

For each alignment, let `q` be the query depths and `r` be the reference depths. The overlap score is `sum(min(q, r)) / sum(max(q, r))`. The implementation uses the equivalent L1 formula:

```text
total        = sum(q) + sum(r)
distance     = sum(abs(q - r))
intersection = (total - distance) / 2
union        = total - intersection
overlap      = intersection / union
```

If `union` is zero, the overlap score is zero. The comparator multiplies each overlap score by its shift penalty. It returns the highest score and its alignment. A zero score returns stretch 1 and shift 0. Candidate scoring uses this score directly.

SciPy computes L1 distances without a large temporary array of per-bin minima. This reduces memory use and comparison time. Subtraction can lose precision when `total` and `distance` are almost equal. In that case, the comparator calculates `sum(min(q, r))` directly. This keeps zero overlap at zero and preserves small positive overlaps.

AlphaPhant compares all left profiles in one call and all right profiles in another. The comparator builds query alignments once per side and processes references in batches to limit memory use. A comparison with one reference uses the same code.

## Catalog Matching

Catalog sightings are not matching units. Once tear-profile evidence enters the known-elephant catalog, it is grouped by known elephant and ear side.

The matcher first analyzes the query and references, preserving candidate membership and supplied reference order. For each known elephant:

1. Compare the query left profile with every left catalog profile and retain the highest similarity score.
2. Compare the query right profile with every right catalog profile and retain the highest similarity score.
3. Average the retained left and right scores.

The winning left and right evidence may come from different catalog sightings. Each internal comparison keeps its source Photo and sighting ID. For equal scores, the matcher selects the first reference in input order.

Every sighting requires both ears. A candidate is scored only when its supplied reference sightings can be analyzed.

## Caching

Expensive model invocations and final per-ear tear profiles are cached. Ear selection, sighting orchestration, contour preparation, catalog grouping, candidate scoring, and derived rankings have no independent persistent caches. Completed immutable analyses are reused within each AlphaPhant instance.

A final tear-profile record is per prepared ear. Its key contains the source photo UUID, integer raster bounding box, inferred side, segmentation producer slug, and landmark producer slug. Bounding-box identity remains stable when candidate ordering changes. Cache identity and producer versioning are defined in [architecture.md](architecture.md).

Caching is selected when processors are composed. Standard runs cache the complete SAM3 multi-feature computation before its ear-only adapter, landmark detection, and AlphaTear extraction. Parameter-tuning runs use a raw unversioned AlphaTear extractor while retaining cached SAM3 features and landmark detection. Preparation and AlphaPhant expose no cache-policy options.
