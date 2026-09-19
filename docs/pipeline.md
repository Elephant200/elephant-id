# AlphaPhant pipeline

AlphaPhant receives a sighting ear pair and a candidate catalog.
It returns one finite similarity score per candidate. Higher scores indicate stronger matches.
The caller sorts these scores to obtain a ranking. AlphaPhant does not assign an identity.

## Changes from main

This comparison uses `main` at `6b522d5` and the current uncommitted source on `optimized`.
It describes the active seven-scale method. The simpler two-reference methods remain experiments.

| Stage | `main` | Current method |
|---|---|---|
| Preparation interface | `SightingAnalyzer` owns preparation and profile extraction. | `SightingPreparer` owns preparation. Each matcher owns its representation. |
| Extractor input | One prepared ear. | Same. |
| Reference shapes | One alpha scale: `0.35`. | Seven scales: `0.11, 0.22, 0.50, 1.10, 2.50, 5.00, 12.00`. |
| Extractor output | One immutable tear profile. | One ordered tuple of immutable tear profiles. |
| Shared work | One reference requires one geometry calculation. | All references share contour resampling, opening, radius, and polar rays. |
| Comparison channels | Nonnegative depth. | Depth plus separate rising and falling depth changes. |
| Magnitude exponent | `0.50`. | `0.75`. |
| Angular weights | None. | Gaussian weights: center 120 degrees, width 35 degrees. |
| Comparison grid | 240 bins. | Same. |
| Stretch search | 0.80–1.20, step 0.025. | Same; scales share alignment within each channel. |
| Shift limit | 15% of the profile. | 10%. |
| Shift penalty scale | `0.08`, power 4. | `0.16`, power 4. |
| Row similarity | Ruzicka overlap. | Same measure, with a compiled L1 calculation. |
| Catalog correction | None. | Subtract each catalog ear's background similarity from twice the raw score. |
| Sightings per candidate | Highest score on each side. | Similarity-weighted mean on each side. |
| Ear combination | Equal mean of left and right scores. | Same. |
| Inspection result | Profile matches include shift and stretch. | `AlphaPhant.profiles` exposes source-labelled profiles; matching returns scores only. |
| Persistent profile cache | One profile record per ear. | One complete seven-profile record per ear. |
| Alternative matcher | No CurvRank matcher. | CurvRank uses the same prepared ears through the catalog-matcher contract. |

SAM3 segmentation, YOLO landmark detection, immutable photo IDs, and the PhotoStore contract remain.
The active geometry keeps the original vertex-count contour selection and inferred-side rule.
The research record separates geometry experiments from the active method.

## Call sequence

```text
build_standard_alphaphant(photo_store)
  -> build_preparer(photo_store, cache)
  -> build_profile_extractor(preparer, cache)
  -> AlphaPhant(prepare_ears=preparer.prepare, profile_extractor=extractor)

AlphaPhant.match(query, catalog)
  -> profiles(pair), for the query and each distinct catalog sighting
     -> SightingPreparer.prepare(pair)
     -> extract_sighting_profiles(ears, extractor)
        -> extractor.extract(ear), once per ear
  -> _score_side(query, catalog, "left")
     -> _side_matrix -> TearMatcher.match_many
     -> _correct_catalog
     -> _aggregate_evidence
  -> _score_side(query, catalog, "right")
  -> average the two candidate scores
```

The caller supplies one extractor. It does not assemble a sequence of extractors or select comparison channels.
`match` extracts the profiles before scoring. Both side-scoring calls receive those same profiles.
Numerical settings stay with the algorithms that use them.

## Ear preparation

`SightingPreparer` retrieves encoded bytes through PhotoStore and decodes them into BGR images.
Cached complete SAM3 features supply ear detections and masks.
Cached YOLO records supply landmarks in full-image coordinates.
Preparation creates an immutable `PreparedEar` for each side.
A shared source photo requires two separate ears; preparation reads that photo once.

The current side rule selects an ear by inferred image direction, then by cleaned area.
This rule can fail on profile views.
Snapped landmarks divide the mask boundary into two paths.
Preparation selects the path with more vertices and orders it from upper to lower anchor.
Unequal raster sampling can cause this rule to select the inner path.

Original detector landmarks define the polar coordinate frame.
Snapped anchors delimit the contour. These values have different purposes.
Failures retain the photo, side, stage, and original cause in `MatchingError`.

## AlphaTear extraction

The extractor resamples the contour to 1,024 points.
The ear radius is `R = sqrt(2A / pi)`, where `A` is the cleaned contour area.
A morphological opening with radius `0.020R` removes narrow outward mask bumps.
Each alpha radius is its scale fraction multiplied by `R`.
Small radii follow finer contour structure; large radii provide broader references.

Each reference produces 720 angular depth samples over the relevant semicircle.
For each ray, extraction finds the furthest forward reference-boundary crossing.
The nearest sampled reference-path point supplies a tangent.
A three-pixel probe selects the inward normal.
The nearest original-contour crossing along that normal supplies signed depth, divided by `R`.
Positive depth represents an inward tear.

The first and last five degrees are zero.
Missing reference intersections are zero.
Gaussian smoothing uses sigma two bins; missing intersections return to zero after smoothing.
The ordered result uses the immutable producer `alpha-tear-v3-multiscale`.
Historical cache records remain available.

The coordinate system assumes a mostly star-convex ear relative to the landmark midpoint.
Pose, occlusion, segmentation errors, and landmark errors can change the profiles.
Radius normalization does not correct these errors.

## Channels and alignment

The depth channel clips negative depths to zero.
The signed-change channel differentiates the original depth profile, then separates positive and negative changes.
These rising and falling rows cannot substitute for each other.
This channel does not represent negative tear depth.

Each nonnegative row receives magnitude compression with exponent `0.75` and resampling to 240 bins.
Gaussian angular weights favor the lower-ear region.
Differentiation occurs before compression and resampling.

Each channel searches 17 centered stretches and 49 integer shifts: 833 alignments.
Values outside the shifted profile are zero.
All reference rows share one alignment within a channel.
The depth and signed-change channels can select different alignments.
Only the query moves; catalog profiles stay fixed.

For each aligned row, Ruzicka similarity is `sum(min(query, catalog)) / sum(max(query, catalog))`.
The equivalent compiled calculation is `(T - D) / (T + D)`.
Here, `T` is total mass and `D` is L1 distance. Empty rows score zero.
The channel averages row scores and applies `exp(-(abs(shift_fraction) / 0.16)^4)`.
It then selects the highest score.
Depth contributes `0.40`; signed change contributes `0.60`.

## Catalog correction and candidate scores

For each catalog ear, background strength is the mean of its five strongest outgoing same-side similarities.
The ear itself is excluded. Other sightings of the same elephant remain eligible neighbors.
A smaller catalog uses all available neighbors. No neighbor means zero background.
The held-out query never enters this calculation.
A corrected ear score is `2 * raw_similarity - background`.

Each candidate's sighting scores receive weights proportional to `exp((score - maximum) / temperature)`.
Temperature is twice the standard deviation across corrected catalog-ear scores for that query side.
The result is a weighted mean. Zero spread uses the maximum.
The final candidate score is the equal mean of its two side scores.

These scores are not probabilities and need not lie between zero and one.
Cached comparisons cannot add evidence from earlier catalog calls.
The current catalog determines every background and candidate score.

See [research](research.md) for measured results and alternatives.
See [evaluation](evaluation.md) for the comparison procedure.
