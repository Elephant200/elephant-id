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
3. **Ear landmark detection:** Find the upper landmark and the lower landmark of each ear.
4. **Ear-contour preparation:** Extract the boundary from the mask. Move each landmark to the nearest boundary vertex. The results are the contour anchors. The contour anchors divide the boundary into two paths. Select the ear-flap path: the path that encloses the larger area with the chord between the contour anchors. This path is the ear contour. Then calculate the contour geometry.
5. **Declared-side resolution:** Select the prepared ear for each declared side. Resolve both sides before extraction starts.
6. **Tear-profile extraction:** Run AlphaTear. For each alpha scale, AlphaTear builds an alpha shape and makes one one-dimensional tear profile. AlphaPhant then attaches the source data and the original sighting ID.

The pipeline treats the declared side as correct. If the pipeline cannot make a valid ear for that side, analysis fails with an error. Research uses the selected ear pair.

The pipeline detects landmarks and prepares the contour for each segmented ear. It determines every ear's side. For each declared side, it selects the largest prepared ear with that inferred side. Each photo supplies at most one left ear and one right ear.

SAM3 does ear localization and segmentation. A YOLO keypoint model does ear landmark detection. To replace a model, use the semantic inference interfaces in [architecture.md](architecture.md).

## Tear-Profile Extraction

AlphaTear makes one tear profile for each configured alpha scale. Each alpha scale is a fraction of the ear radius. The default configuration has three alpha scales: 0.35, 2.0, and 10.0.

AlphaTear makes the shared geometry one time for each ear:

1. Resample the prepared contour to equal arc-length spacing. AlphaTear measures the depths to this resampled contour.
2. Calculate the ear radius from the cleaned ear area.
3. Open the resampled contour morphologically. Cut the opened contour between the contour anchors with the ear-flap rule of preparation.
4. Make one polar ray for each angle in the measured interval, 5° to 175°. Start each ray at the midpoint of the landmarks. The ray at 0° points to the upper landmark, the ray at 90° points into the ear flap, and the ray at 180° points to the lower landmark.

The polar frame uses the landmarks. The contour anchors only mark the ends of the ear contour.

AlphaTear then does these steps for each alpha scale:

5. Build the alpha shape of the opened contour.
6. Find the intersection of each ray with the alpha-shape boundary that is farthest from the ray origin. This point is the crossing.
7. Turn the alpha-shape edge that each ray crosses by 90°, into the alpha shape. The result is the inward normal at the crossing.
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
2. **Gradient rows.** Set negative depths to zero. Calculate the gradient per sample inside the measured interval, with one-sided differences at its ends. Take the square root of each magnitude and keep its sign. Resample to the comparison grid. The gradient row is zero outside the measured interval.
3. **Alignment.** Apply each configured stretch and shift to all of the query's rows. All rows share one stretch and one shift. The default stretches are `1.2 ** (k / 8)` for `k = -8…8`, uniform in `ln t` from 1/1.2 to 1.2. The default shifts are the whole bins from -24 to 24. A stretch by `t` divides the gradient by `t`, and the square root turns this into a division by `t ** 0.5`. Thus, multiply each aligned query gradient row by `t ** -0.5`.
4. **Weighted real-valued Jaccard.** For each row, compare the aligned query row `q` with the reference row `r`:
   - Bin `i` has the angle weight `w_i = exp(-0.5 * ((θ_i - 120°) / 35°) ** 2)`, where `θ_i` is the `i`-th angle of the comparison grid. The angle weight applies after alignment, so it stays at the same angles.
   - The code uses the L1 form `(|q| + |r| - |q - r|) / (|q| + |r| + |q - r|)`, with the weighted L1 norm `|x| = sum(w_i * abs(x_i))`. For nonnegative rows, this equals `sum(w * min(q, r)) / sum(w * max(q, r))`.
   - A bin where `q` and `r` have opposite signs adds nothing to the numerator, and adds both magnitudes to the denominator.
   - When the denominator is zero, the weighted real-valued Jaccard is zero.
5. **Alignment score.** For each channel, calculate the mean weighted real-valued Jaccard across the alpha scales. Add the two channel means, each multiplied by its channel weight. The default channel weights are 0.5 and 0.5. Multiply the sum by the shift penalty `exp(-(|b / 240| / 0.16) ** 4)` and the stretch penalty `exp(-(|ln t| / 0.32) ** 4)`, where `b` is the shift in bins and `t` is the stretch. The shift penalty costs about 14% at a shift of 24 bins. The stretch penalty costs about 10% at a stretch of 1/1.2 or 1.2.
6. **Best alignment.** Select the highest alignment score and its alignment. This score is the ear similarity.

When the highest alignment score is at or below 1e-12, the comparator sets the ear similarity to zero. A zero ear similarity has no useful alignment. Thus, the comparator reports stretch 1 and shift 0.

## Catalog Matching

The matcher compares the query with known elephants. The catalog groups tear profiles by known elephant and ear side.

The matcher first analyzes the query and the references. Then, for each known elephant, the matcher:

1. gets the left score from the comparator. The comparator calculates the ear similarity of the query left ear and each left reference ear of the candidate. The left score is the log-mean-exp of these `n` ear similarities `s`: `τ * (logsumexp(s / τ) - log n)`, with the temperature `τ = 0.05`;
2. gets the right score in the same way;
3. calculates the candidate score as the mean of the left score and the right score.

The log-mean-exp is a smooth maximum. When `k` of the `n` references match about equally well and the others score far lower, the left score is about the best ear similarity minus `τ * log(n / k)`. Thus, a candidate scores its best match, minus a small cost when its other sightings do not agree.

The left score and the right score each use all the candidate's reference sightings on that side. The two sides can draw on different catalog sightings. [ADR 0011](adr/0011-score-each-side-by-log-mean-exp.md) gives the reasons for the log-mean-exp.

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

The bounding box gives the same key when the candidate order changes. [architecture.md](architecture.md) defines cache identity and producer versions.

Composition selects which processors use the cache. Standard runs cache the full SAM3 multi-feature output (before its ear-only adapter), landmark detection, and AlphaTear extraction. Parameter-tuning runs use a raw, unversioned AlphaTear extractor, but use the cached SAM3 features and the cached landmark detection. Preparation and AlphaPhant have no cache-policy options.
