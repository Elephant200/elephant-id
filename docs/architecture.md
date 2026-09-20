# Architecture

AlphaPhant accepts a sighting ear pair and a candidate catalog. It returns one finite similarity score per candidate. Ranking sorts these scores in descending order.

## Ownership

| Package | Responsibility |
| --- | --- |
| `domain` | Immutable `Photo`, `Sighting`, and `SightingEarPair` values with permanent UUID identity |
| `dataset` | Private metadata, known-elephant identity, and the image-only `PhotoStore` |
| `image` | Encoded-byte decoding to BGR images and shared image geometry |
| `inference` | Ear segmentation and full-image ear landmarks |
| `preparation` | Resolved left/right ears and immutable prepared geometry |
| `matching` | Shared catalog contract; AlphaPhant analysis, extraction, comparison, and scoring |
| `evaluation` | Benchmark validation, splits, ground truth, failure records, ranks, and metrics |

`PhotoStore.read(photo)` returns original encoded bytes. Composition passes the PhotoStore to preparation. AlphaPhant receives the preparation callable, neutral sighting pairs, and a candidate catalog. It does not receive the identity-aware Dataset.

## Module Boundaries

```text
preparation/
  ear.py             PreparedEar and contour geometry
  sighting.py        SightingPreparer
matching/
  protocol.py        CatalogMatcher and MatchingError
  alphaphant/
    extraction.py    TearProfile, analyzed values, protocol, and AlphaTear
    similarity.py    EarProfileSimilarity and EarComparison
    cached.py        Profile persistence
    matcher.py       AlphaPhant.analyze and AlphaPhant.match
composition.py       Construct preparation and AlphaPhant
```

Composition constructs AlphaPhant with a preparation callable, one extractor, and one comparator. Preparation uses storage, domain, image, and inference interfaces. Preparation and evaluation use `MatchingError` from `matching.protocol`. This protocol imports no concrete matcher.

The `matching` package exports the shared contract. Import AlphaPhant from `matching.alphaphant`.

## Analysis and Scoring

`SightingPreparer.prepare` resolves both ears and returns them in left, right order. When both sides use one Photo, preparation processes it once. Left-side resolution precedes preparation of a distinct right Photo. Both sides must resolve before extraction starts.

`PreparedEar` contains the source Photo, raster box, full-image contour, original detector landmarks, snapped contour anchors, inferred side, and cleaned area. Its contour is finite and read-only. Its area is positive. Original landmarks define the extraction frame; snapped anchors delimit the contour.

`AlphaPhant.analyze` returns an immutable `AnalyzedSightingEarPair`. It retains the input sighting ID and two `AnalyzedEar` values. Each ear contains its Photo, side, raster box, and one signed `TearProfile`. Profile depths are finite, one-dimensional, and read-only.

`AlphaPhant.match` analyzes the query and references before comparison. It preserves candidate membership and reference order through batching. It compares corresponding sides, selects each candidate's strongest reference for each side, and averages the two scores. Each comparison retains its reference source and query alignment. See [pipeline.md](pipeline.md).

The catalog is a mapping from opaque `CandidateKey` values to tuples of `SightingEarPair` evidence. Each candidate requires at least one pair. `CandidateScores` has exactly the catalog's keys and finite values. Larger scores indicate stronger matches. Errors stop scoring; no partial result is returned.

Preparation and extraction failures use `MatchingError`. It records the Photo ID, declared side, and stage, and retains the original exception as its cause. A failure while processing a shared Photo before side resolution belongs to the left side. Evaluation adds the query and evidence role. See [evaluation.md](evaluation.md).

## Identity and Image Coordinates

A Photo ID identifies one immutable original photo asset. New original bytes require a new Photo ID. A sighting ID identifies one observed event. Photos in a sighting ear pair belong to that sighting. Dataset metadata owns paths, dates, and known-elephant names.

Inference results and prepared geometry use full-image coordinates. Float boxes use half-open `xyxy` bounds. Raster conversion floors lower edges, rounds upper edges up, and clips to the image. YOLO crop coordinates are translated before they cross the landmark interface or enter a cache.

## Caches and Configuration

`CacheManager` owns JSON storage, safe paths, and atomic writes. Cache decorators own keys, serialization, and payload validation.

A producer slug identifies all settings that affect output. An output change requires a new slug. A settled `AlphaTearVersion` contains a slug and immutable `AlphaTearConfig`. A raw experimental configuration has no persistent producer slug. The extractor resolves either input at construction.

Inference keys use the Photo UUID and any dependent crop coordinates. SAM3 caches its complete multi-feature output before the ear-only adapter. Landmark records use full-image coordinates. Profile keys contain the Photo UUID, raster box, inferred side, segmentation slug, and landmark slug. Geometry changes that affect profiles require a new extraction slug.

Standard composition caches inference and settled profiles. Experimental extraction uses the inference caches but does not read or write profile records. Preparation and matching expose no cache-policy flags.

AlphaPhant reuses completed analyses within one instance. Geometry and pair comparisons have no separate cache. Results depend on the supplied evidence and composed processors, not on earlier match calls.
