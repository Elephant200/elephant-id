# AlphaPhant Pipeline

This document defines the current research algorithm. [architecture.md](architecture.md) defines the domain, storage, and cache boundaries.

## Input and Output

The input is a sighting ear pair: one left-ear photo and one right-ear photo from the same sighting. One photo can supply both ears.

`AlphaPhant.match` returns one finite candidate score for each candidate. The candidate score is the mean of the candidate's left score and right score.

`AlphaPhant.analyze` returns the data that matching uses: an immutable `AnalyzedSightingEarPair`. It contains the sighting ID and the left and right `AnalyzedEar` values. Each `AnalyzedEar` contains one signed `TearProfile` for each alpha scale, its source photo, its anatomical side, and its raster box.

A candidate ranking is the list of candidate scores in descending order. AlphaPhant does not make an identity decision. It does not add evidence to the catalog.

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

AlphaTear makes one tear profile for each configured alpha scale. Each alpha scale is a fraction of the ear radius. The default configuration has one alpha scale, 0.35.

AlphaTear makes the shared geometry one time for each ear:

1. Resample the prepared contour to equal arc-length spacing. AlphaTear measures the depths to this resampled contour.
2. Calculate the ear radius from the cleaned ear area.
3. Open the resampled contour morphologically. Cut the opened contour between the anchors with the ear-flap rule of preparation.
4. Make one polar ray for each angle in the measured interval, 5° to 175°. Start each ray at the midpoint of the original detector landmarks.

The polar frame uses the detector landmarks. The snapped contour anchors only mark the ends of the prepared contour.

AlphaTear then does these steps for each alpha scale:

5. Build the alpha shape of the opened contour.
6. Find the point where each ray last leaves the alpha shape. This point is the crossing.
7. Calculate the inward normal of the alpha shape at each crossing:
   1. Cut the alpha-shape boundary with the ear-flap rule.
   2. Find the boundary point that is nearest to the crossing.
   3. Calculate the boundary tangent at that point.
   4. Turn the tangent by 90° toward the inside of the alpha shape.
8. Measure the signed distance from each crossing to the resampled contour, along the normal. Use the contour intersection that is nearest to the crossing.
9. Divide each distance by the ear radius. The result is the depth.
10. Smooth the depths along the angles.

The tear profile contains the depths at 720 uniform angles from 0° to 180°, both ends included. The tear profile also contains its measured interval. These depths are zero:

- the depths outside the measured interval;
- the depths of rays that do not cross the alpha shape.

## Tear-Profile Matching

AlphaPhant calculates the ear similarity between a query ear and each reference ear for each side, left and right.

Every compared ear has one tear profile for each alpha scale. All compared tear profiles must have the same length.

The comparator calculates the ear similarity in these steps:

1. **Depth rows.** Set negative depths to zero, take the square root, and resample each tear profile to the comparison grid.
2. **Alignment.** Apply each configured stretch and shift to the query's depth rows. All alpha scales share one stretch and one shift.
3. **Real-valued Jaccard.** For each alpha scale, compare the aligned query row `q` with the reference row `r`. For nonnegative rows, the real-valued Jaccard is `sum(min(q, r)) / sum(max(q, r))`. The code uses the equivalent L1 form `(|q| + |r| - |q - r|) / (|q| + |r| + |q - r|)`. When the denominator is zero, the real-valued Jaccard is zero.
4. **Alignment score.** Calculate the mean real-valued Jaccard across the alpha scales. Multiply the mean by the shift penalty of the alignment.
5. **Best alignment.** Select the highest alignment score and its alignment. This score is the ear similarity.

When the highest alignment score is at or below 1e-12, the comparator sets the ear similarity to zero. A zero ear similarity has no useful alignment. Thus, the comparator reports stretch 1 and shift 0.

## Catalog Matching

The matcher compares the query with known elephants. The catalog groups tear profiles by known elephant and ear side.

The matcher first analyzes the query and the references. Then, for each known elephant, the matcher:

1. calculates the ear similarity of the query left ear and each left reference ear of the candidate. The left score is the highest of these ear similarities;
2. calculates the right score in the same way;
3. calculates the candidate score as the mean of the left score and the right score.

The left score and the right score can come from different catalog sightings.

Each sighting must have both ears. If the matcher cannot analyze a reference sighting, `match` raises `MatchingError`.

## Caching

The cache stores expensive model calls and the final tear profiles of each ear. These steps have no persistent cache of their own: ear selection, sighting orchestration, contour preparation, catalog grouping, candidate scoring, and candidate ranking. Each AlphaPhant instance reuses its completed analyses.

The cache has one tear-profile record for each prepared ear. The record key contains:

- the source photo UUID;
- the integer raster bounding box;
- the inferred side;
- the segmentation producer slug;
- the landmark producer slug.

The record contains the alpha scales and the tear profiles. Each stored tear profile contains its measured interval and its depths. Loading a record fails when its alpha scales differ from the extractor's alpha scales in value or order.

The bounding box keeps the key stable if the candidate order changes. [architecture.md](architecture.md) defines cache identity and producer versions.

Composition selects which processors use the cache. Standard runs cache the full SAM3 multi-feature output (before its ear-only adapter), landmark detection, and AlphaTear extraction. Parameter-tuning runs use a raw, unversioned AlphaTear extractor, but keep the cached SAM3 features and landmark detection. Preparation and AlphaPhant have no cache-policy options.
