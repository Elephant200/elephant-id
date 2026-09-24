# AlphaPhant pipeline

AlphaPhant receives a sighting ear pair and a candidate catalog.
It returns one finite similarity score per candidate. Higher scores indicate stronger matches.
The caller sorts these scores to obtain a ranking. AlphaPhant does not assign an identity.

## Changes from main

This comparison uses `main` at `6b522d5` and the current uncommitted source on `optimized`.
It describes the active three-scale method. Other reference sets remain experiments.

| Stage | `main` | Current method |
|---|---|---|
| Preparation interface | `SightingAnalyzer` owns preparation and profile extraction. | `SightingPreparer` owns preparation. Each matcher owns its representation. |
| Extractor input | One prepared ear. | Same. |
| Reference shapes | One alpha scale: `0.35`. | Three scales: `0.35, 2, 10`. |
| Extractor output | One immutable tear profile. | One ordered tuple of three immutable tear profiles. |
| Shared work | One reference requires one geometry calculation. | The three references share contour resampling, radius, and polar rays. |
| Comparison channels | Nonnegative depth. | Depth plus separate rising and falling depth changes. |
| Magnitude exponent | `0.50`. | `0.50`; restored after the benchmark audit. |
| Angular weights | None. | Gaussian weights: center 120 degrees, width 35 degrees. |
| Comparison grid | 240 bins. | Same. |
| Stretch search | 0.80–1.20, step 0.025. | Full search; every scale and channel shares one alignment. |
| Shift limit | 15% of the profile. | 10%. |
| Shift penalty scale | `0.08`, power 4. | `0.16`, power 4. |
| Row similarity | Ruzicka overlap. | Same measure, with a compiled L1 calculation. |
| Catalog correction | None. | Subtract each catalog ear's background similarity from twice the raw score. |
| Sightings per candidate | Highest score on each side. | Similarity-weighted mean on each side. |
| Ear combination | Equal mean of left and right scores. | Same. |
| Inspection result | Profile matches include shift and stretch. | `AlphaPhant.profiles` exposes source-labelled profiles; matching returns scores only. |
| Persistent profile cache | One profile record per ear. | One complete three-profile record per ear. |
| Alternative matcher | No CurvRank matcher. | CurvRank uses the same prepared ears through the catalog-matcher contract. |

SAM3 segmentation, YOLO landmark detection, immutable photo IDs, and the PhotoStore contract remain.
The active geometry selects the ear contour by enclosed area and keeps the inferred-side rule, with a single-side fallback.
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

Snapped landmarks divide the mask boundary into two paths.
The landmark chord joins the snapped landmarks.
Preparation keeps the path that encloses more area with that chord and orders it from upper to lower anchor.
The ear flap bulges away from the head, so its path encloses more area; the head-side path runs close to the chord.
Vertex count is not used: a ragged head-side edge can have more raster vertices than the flap.

The side rule selects an ear by inferred image direction, then by cleaned area.
Inferred direction assumes a lateral view: the landmarks lie on the flap's head side.
Rear and frontal views reverse that direction.
When a Photo supplies only one declared side and no prepared ear has that inferred side, preparation uses its largest ear.
A Photo that supplies both sides still requires one ear of each inferred side.

Original detector landmarks define the polar coordinate frame.
Snapped anchors delimit the contour. These values have different purposes.
Failures retain the photo, side, stage, and original cause in `MatchingError`.

## AlphaTear extraction

The extractor resamples the contour to 1,024 points.
The ear radius is `R = sqrt(2A / pi)`, where `A` is the cleaned contour area.
It rounds small contour irregularities with an opening radius of `0.02R` and keeps the original contour endpoints.
The opened contour defines the reference shapes; depth is still measured against the original contour.
Each alpha radius is its scale fraction multiplied by `R`.
Small radii follow finer contour structure; large radii provide broader references.

Each reference produces 720 angular depth samples over the relevant semicircle.
For each ray, extraction finds the furthest forward reference-boundary crossing.
The intersected reference edge, turned a quarter turn toward the reference interior, gives the inward direction.
The reference polygon's winding decides which quarter turn points inward, so no nearest-point search or inside probe is needed.
The nearest original-contour crossing along that normal supplies signed depth, divided by `R`.
Positive depth represents an inward tear.

The first and last five degrees are zero.
Missing reference intersections are zero.
Gaussian smoothing uses sigma two bins; missing intersections return to zero after smoothing.
The ordered result uses the immutable producer `alpha-tear-v6-area-contour`.
Historical cache records remain available.

The coordinate system assumes a mostly star-convex ear relative to the landmark midpoint.
Pose, occlusion, segmentation errors, and landmark errors can change the profiles.
Radius normalization does not correct these errors.

## Channels and alignment

The depth channel clips negative depths to zero.
The signed-change channel differentiates the original depth profile, then separates positive and negative changes.
These rising and falling rows cannot substitute for each other.
This channel does not represent negative tear depth.

Each nonnegative row receives square-root magnitude compression and resampling to 240 bins.
Gaussian angular weights favor the lower-ear region.
Differentiation occurs before compression and resampling.

Each query-candidate ear pair checks all 17 centered stretches and 49 integer shifts:
833 alignments. The search selects the exact best combined alignment on this grid.
Values outside the shifted profile are zero.
All depth, rising-change, and falling-change rows at every scale share one alignment.
This is a required property of the method, independent of measured accuracy or speed.
Only the query moves; catalog profiles stay fixed.

For each aligned row, Ruzicka similarity is `sum(min(query, catalog)) / sum(max(query, catalog))`.
The equivalent compiled calculation is `(T - D) / (T + D)`.
Here, `T` is total mass and `D` is L1 distance. Empty rows score zero.
At each alignment, average depth row scores across scales and average signed-change row scores across scales and directions.
Depth contributes `0.40`; signed change contributes `0.60`.
Combine these contributions, then apply `exp(-(abs(shift_fraction) / 0.16)^4)`.
Select the highest combined score only after this calculation. No channel or scale selects a separate alignment.

## Catalog correction and candidate scores

For each catalog ear, background strength is the mean of its five strongest outgoing same-side similarities.
The ear itself is excluded. Other sightings of the same elephant remain eligible neighbors.
A smaller catalog uses all available neighbors. No neighbor means zero background.
The held-out query never enters this calculation.
Comparison batches target catalog evidence only. A query outside the catalog does not cause reverse catalog-to-query or query-to-itself comparisons.
A corrected ear score is `2 * raw_similarity - background`.

Each candidate's sighting scores receive weights proportional to `exp((score - maximum) / temperature)`.
Temperature is twice the standard deviation across corrected catalog-ear scores for that query side.
The result is a weighted mean. Zero spread uses the maximum.
The final candidate score is the equal mean of its two side scores.

These scores are not probabilities and need not lie between zero and one.
Cached comparisons cannot add evidence from earlier catalog calls.
The current catalog determines every background and candidate score.
Within one matcher, each complete sighting ear pair receives a stable local slot for cached score lookup. The permanent domain identifiers remain unchanged.

See [evaluation](evaluation.md) for the comparison procedure.
