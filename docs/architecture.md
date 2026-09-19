# Architecture

AlphaPhant converts one sighting ear pair into scores against a candidate catalog.
The code separates shared image preparation from each matching algorithm's representation.
One extractor owns the complete AlphaTear representation.

## Modules and interfaces

| Module | Responsibility | Main interface or value |
|---|---|---|
| `domain` | Represent photos and sightings without identity labels or storage paths. | Immutable `Photo`, `Sighting`, `SightingEarPair`. |
| `dataset` | Resolve private metadata and known elephants. Retrieve original photo bytes. | `Dataset`, image-only `PhotoStore.read(photo)`. |
| `image` | Decode bytes into BGR arrays. Supply common image and geometry operations. | `decode_image`, `BoundingBox`, `resample_polyline`. |
| `inference` | Segment ears and detect landmarks. | `EarSegmenter`, `EarLandmarkDetector`. |
| `preparation` | Resolve requested ears and prepare immutable geometry. | `SightingPreparer.prepare(pair) -> (left, right)`. |
| `matching` | Define the algorithm-independent catalog interface. | `CatalogMatcher.match(query, catalog)`. |
| `matching.alphaphant` | Extract AlphaTear profiles, compare ears, and combine catalog evidence. | `AlphaPhant`, `AlphaTearExtractor`. |
| `matching.curvrank` | Extract curvature descriptors and score candidates independently. | `CurvRankMatcher`. |
| `evaluation` | Own identities, manifests, query folds, scores, and metrics. | `load_benchmark`, `evaluate`, `EvaluationResult`. |
| `composition` | Construct processing objects and select cache decorators. | `build_standard_alphaphant`. |

A matcher receives neutral sighting ear pairs and an image-only PhotoStore through preparation.
It never receives the identity-aware Dataset.
Evaluation supplies opaque candidate keys instead of elephant names.
The evaluator retains private labels outside the matcher.

## Source layout

```text
src/elephant_id/
  domain/
  dataset.py
  image/
  inference/
  preparation/
    sighting.py             # Photo access, inference, and ear selection.
    ear.py                  # Immutable geometry and contour preparation.
  matching/
    protocol.py             # Catalog contract and diagnostic errors.
    alphaphant/
      matcher.py            # Profile orchestration and catalog scores.
      profile.py            # Immutable profiles and extraction contract.
      extraction.py         # Complete AlphaTear representation.
      similarity.py         # Fixed channels and affine alignment.
      cached.py             # Complete profile-record persistence.
    curvrank.py              # Independent curvature matcher.
  evaluation/
  composition.py            # Object construction.
```

The matching package exports the shared contract.
Callers import `AlphaPhant` from `matching.alphaphant`.
Model implementations load heavy dependencies only when needed.
Saved-score analysis does not require concrete inference implementations.

## Runtime flow

```text
SightingEarPair
  -> PhotoStore: original encoded bytes
  -> shared preparation: decoded images, segmentation, landmarks, ear geometry
  -> one extractor: ordered immutable profiles for each ear
  -> same-side profile comparisons
  -> catalog background correction
  -> weighted mean of each candidate's sighting scores
  -> equal mean of left and right scores
  -> one score per candidate
```

[Pipeline](pipeline.md#call-sequence) maps this flow to the actual functions.
That document also compares the current source with `main`.

## Values and catalog rules

A Photo has a permanent UUID and belongs to one sighting. It contains no encoded bytes, date, path, or elephant label.
A Sighting has a permanent UUID, observation date, and distinct Photos.
A SightingEarPair names one selected Photo per anatomical side from the same sighting.
One source Photo can supply both ears.
Replacing or re-encoding an original asset requires a new photo identity.
The application assigns IDs at import; research preserves them.

The catalog is a read-only mapping from candidate keys to tuples of sighting ear pairs.
There is no catalog wrapper. Mapping order does not affect meaning.
Each result has exactly the supplied candidate keys and a finite score for each key.
Higher scores indicate stronger matches. The interface promises no universal numerical range.
Errors stop evaluation instead of producing partial scores.

Results depend on the supplied query, catalog, source images, and composed method.
Earlier calls must not change the result.
Identity-neutral caches can change time, but not evidence.
Evaluation creates a fresh candidate-key mapping for each evaluation and can reuse it across its query folds.
This trusted interface is not an adversarial identity-hiding system.

## Preparation and extraction

`PreparedEar` is the shared immutable value between inference and each algorithm.
It contains source Photo, raster box, detector landmarks, snapped anchors, contour, image direction, and cleaned area.
Landmarks and geometry use full-image coordinates.
Raster boxes use half-open integer bounds. Conversion floors lower edges, rounds upper edges upward, and clips to the image.

SAM3 supplies complete feature records before a thin adapter selects ear features.
Landmark detectors supply the strongest upper/lower result, or `None` if no ordinary detection exists.
Model variation stays behind these two semantic interfaces.
CurvRank imports preparation, not AlphaPhant internals.

`TearProfile` contains a read-only one-dimensional depth array.
`TearProfileExtractor.extract(ear)` returns the complete ordered tuple of profiles.
`extract_sighting_profiles` adds source labels and pairs left/right outputs by reference order.
`AlphaPhant.profiles(pair)` reuses the same memoized extraction as matching.
There is no list of extractor objects at the caller interface.

The default extractor owns the fixed settings and producer slug.
An explicit `AlphaTearConfig` creates an uncached research extractor.
Comparison channels, alignment, background correction, and evidence aggregation stay inside AlphaPhant.
Rejected research alternatives add no production flags or strategy classes.
[ADR 0011](adr/0011-keep-selected-scoring-inside-alphaphant.md) records the design alternatives.

## Cache and construction

One `CacheManager` stores JSON records with atomic writes.
It owns namespace paths and basic record loading, not algorithm settings or identity resolution.
Each stage decorator owns its keys, serialization, and payload validation.
Composition selects the decorator or raw processor. Matchers have no cache-policy flags.

A producer slug identifies all fixed output-changing settings and model details.
An output-changing implementation needs a new immutable slug.
Runtime keys use the permanent photo UUID and actual dependent inputs.
They do not hash the photo UUID or derived contour.

Examples:

```text
sam3-features/<photo UUID>
yolo26n-keypoints-v1/<photo UUID>__crop_<x1>_<y1>_<x2>_<y2>
```

Final AlphaTear keys also contain the crop, inferred image side, segmentation producer, and landmark producer.
A material preparation change requires a new AlphaTear slug because no separate preparation slug exists in that key.
The profile record stores the complete ordered tuple for one ear.
SAM3 records retain all feature classes. Landmark records retain full-image coordinates.
Historical records remain untouched.

`build_preparer` constructs cached inference.
`build_profile_extractor` adds the complete representation cache with the preparer's inference slugs.
`build_standard_alphaphant` passes preparation and one extractor directly to AlphaPhant.
Builders do not read manifests, prepare datasets, choose folds, or evaluate methods.
The comparison command can share one memoized preparation function across matchers.

Preparation processes the left role first.
If a shared-photo failure occurs before role resolution, the error identifies that first requested role.
`MatchingError` retains photo, side, stage, and original cause.
Source paths remain private metadata, not cache identity.

The future application uses the same neutral values and pipeline after ear selection.
See [application reference](reference/application.md) and [context](context.md) for the domain terms.
