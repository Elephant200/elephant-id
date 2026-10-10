# AlphaPhant Pipeline

This document defines the current research algorithm. [architecture.md](architecture.md) defines the domain, storage, and cache boundaries.

## Input and Output

The input is a sighting ear pair: one left-ear photo and one right-ear photo from the same sighting. One photo can supply both ears.

`AlphaPhant.match` returns one finite similarity score for each candidate. Each side comparison keeps its reference ear and its query alignment. For each side, the matcher selects the strongest reference. It then calculates the mean of the two side scores.

`AlphaPhant.analyze` returns the data that matching uses: an immutable `AnalyzedSightingEarPair`. It contains the sighting ID and the left and right `AnalyzedEar` values. Each `AnalyzedEar` contains one signed `TearProfile` for each alpha scale, its source photo, its anatomical side, and its raster box.

A candidate ranking is the scores in descending order. AlphaPhant does not make an identity decision. It does not add evidence to the catalog.

## Automated Preprocessing

Query ears and reference ears go through the same steps:

1. **Photo retrieval and decoding:** Read the original encoded bytes from the photo store. Decode them to a BGR image.
2. **Ear localization and ear segmentation:** Find the ears. Make a mask and a box for each ear.
3. **Ear landmark detection:** Find the two anatomical endpoints of the relevant contour.
4. **Ear-contour preparation:** Extract the contour from the mask. Snap the landmarks to the contour. The snapped landmarks divide the contour into two paths. Keep the ear-flap path: the path that encloses the larger area with the chord between the snapped landmarks. Then calculate the contour geometry.
5. **Declared-side resolution:** Select the prepared ear for each declared side. Resolve both sides before extraction starts.
6. **Tear-profile extraction:** Run AlphaTear. For each alpha scale, AlphaTear builds an alpha shape and makes one one-dimensional tear profile. AlphaPhant then attaches the source data and the original sighting ID.

The pipeline treats the declared side as correct. If the pipeline cannot make a valid ear for that side, analysis fails with an error. Research uses the selected ear pair.

The pipeline detects landmarks and prepares the contour for each segmented ear. It determines every ear's side. For each declared side, it selects the largest prepared ear with that inferred side. Each photo supplies at most one left ear and one right ear.

SAM3 does ear localization and segmentation. A YOLO keypoint model does ear landmark detection. To replace a model, use the semantic inference interfaces in [architecture.md](architecture.md).

## Tear-Profile Extraction

AlphaTear resamples the prepared contour. It calculates a radius from the cleaned ear area.

Morphological opening makes a second contour. AlphaTear builds the alpha-shape reference from this opened contour. The opened contour and the alpha-shape boundary are closed shapes. AlphaTear cuts each one between the anchors with the same ear-flap rule that preparation uses. AlphaTear measures the resampled contour, not the opened contour.

The original detector landmarks define the polar frame. The snapped contour anchors only mark the ends of the prepared contour. They do not replace the detector landmarks in the polar frame.

For each kept profile angle:

1. A ray at that angle finds an origin on the reference boundary.
2. The local inward normal at the origin gives the direction to measure.
3. AlphaTear measures the signed distance from the origin to the nearest crossing of the measured contour.

AlphaTear divides each distance by the radius, smooths the distances, and puts them on the full angular grid. The grid has uniform angles from 0° to 180°, both ends included. Trimmed bins and bins with no origin stay zero.

AlphaTear makes one tear profile for each configured alpha scale. Each alpha scale is a fraction of the radius. The default configuration has one alpha scale, 0.35.

## Tear-Profile Matching

AlphaPhant compares only ears of the same side. It moves only the query profile. Each reference profile stays fixed.

Every compared ear has one tear profile for each alpha scale. All compared tear profiles must have the same length.

The comparator sets negative depths to zero, compresses the depths, and resamples each profile to a common grid. It then applies each configured stretch and shift to the query.

For each alignment and alpha scale, `q` is the query depths and `r` is the reference depths. The overlap score is `sum(min(q, r)) / sum(max(q, r))`. The code uses this equivalent L1 formula:

```text
total        = sum(q) + sum(r)
distance     = sum(abs(q - r))
intersection = (total - distance) / 2
union        = total - intersection
overlap      = intersection / union
```

If `union` is zero, the overlap score is zero. At each alignment, the comparator calculates the mean overlap score across the alpha scales. It multiplies this mean by the shift penalty of the alignment. All alpha scales thus share one stretch and one shift. The comparator returns the highest score and its alignment. If the highest score is zero, it returns stretch 1 and shift 0. Candidate scoring uses this score without change.

SciPy calculates the L1 distances without a large temporary array of per-bin minimums. This uses less memory and time. Subtraction can lose precision when `total` and `distance` are almost equal. In that case, the comparator calculates `sum(min(q, r))` directly. Thus, zero overlap stays zero, and small positive overlaps stay correct.

AlphaPhant compares all left profiles in one call and all right profiles in a second call. The comparator builds the query alignments once for each side and compares all references in one pass. A comparison with one reference uses the same code.

## Catalog Matching

Catalog sightings are not matching units. In the catalog, tear profiles are grouped by known elephant and ear side.

The matcher first analyzes the query and the references. It keeps each reference in its candidate and in its supplied order. Then, for each known elephant:

1. Compare the query left profile with each left catalog profile. Keep the highest similarity score.
2. Compare the query right profile with each right catalog profile. Keep the highest similarity score.
3. Calculate the mean of the two kept scores.

The best left and right profiles can come from different catalog sightings. Each comparison keeps its source photo and sighting ID. If two scores are equal, the matcher selects the first reference in input order.

Each sighting must have both ears. The matcher scores a candidate only if it can analyze all supplied reference sightings of that candidate.

## Caching

The cache stores expensive model calls and the final tear profiles of each ear. These steps have no persistent cache of their own: ear selection, sighting orchestration, contour preparation, catalog grouping, candidate scoring, and candidate ranking. Each AlphaPhant instance reuses its completed analyses.

The cache has one tear-profile record for each prepared ear. The record key contains:

- the source photo UUID;
- the integer raster bounding box;
- the inferred side;
- the segmentation producer slug;
- the landmark producer slug.

The record stores the alpha scales next to the tear profiles. Loading a record fails when its alpha scales differ from the extractor's alpha scales in value or order.

The bounding box keeps the key stable if the candidate order changes. [architecture.md](architecture.md) defines cache identity and producer versions.

Composition selects which processors use the cache. Standard runs cache the full SAM3 multi-feature output (before its ear-only adapter), landmark detection, and AlphaTear extraction. Parameter-tuning runs use a raw, unversioned AlphaTear extractor, but keep the cached SAM3 features and landmark detection. Preparation and AlphaPhant have no cache-policy options.
