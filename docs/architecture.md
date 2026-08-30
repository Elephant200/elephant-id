# Architecture

AlphaPhant is a research implementation of a fully automated elephant re-identification algorithm. Its current scope begins with a sighting ear pair and ends with one similarity score per known elephant. A candidate ranking is a derived view of those scores. The future application shares its neutral photo and sighting model; see [reference/application.md](reference/application.md).

## Design Priorities

- Keep the research pipeline small, explicit, and testable.
- Use permanent opaque photo and sighting identity throughout the system.
- Separate neutral domain objects, image storage, and identity-aware research metadata.
- Preserve tear-profile extraction and matching behavior during restructuring.
- Put model variation behind semantic inference interfaces.
- Keep identity-retrieval evaluation independent of catalog-matcher implementation.
- Cache expensive computation and final reusable tear profiles, not orchestration.

## Responsibilities

**Domain** owns the neutral immutable values shared across the system:

- `Photo` carries a UUIDv4 photo ID and its parent sighting ID;
- `Sighting` carries a UUIDv4 sighting ID, a required sighting date, and distinct Photos;
- `SightingEarPair` carries a sighting ID and one Photo per declared side. Both Photos belong to that sighting, and one Photo may serve both sides.

**Dataset** is the private research dataset object. It constructs and resolves domain objects, owns known-elephant metadata, and owns a filesystem-backed PhotoStore. Catalog matchers never receive the Dataset.

**PhotoStore** owns retrieval of original encoded bytes through `read(photo: Photo) -> bytes`. It uses `photo.photo_id` for lookup and does not expose a known-elephant resolver method.

**Image** owns `decode_image(encoded: bytes) -> BgrImage` and BGR image and basic/universal geometry utilities.

**Analysis** owns sighting analysis, immutable prepared-ear geometry, and AlphaTear profile extraction. A prepared ear retains both the original floating-point landmarks used by AlphaTear and the contour-snapped anchors used to delimit its contour.

**Inference** owns implementations of ear localization and segmentation and ear landmark detection. Analysis depends on these semantic capabilities rather than particular model architectures.

**Matching** owns tear-profile matching and catalog matching. Its public `CatalogMatcher` interface accepts a sighting ear pair and candidate catalog and returns candidate scores. AlphaPhant is the concrete catalog matcher; candidate ranking is derived from its scores.

**Evaluation** owns private ground truth, benchmark examples, candidate keys, splits, failure accounting, derived ranks, metrics, and reproducibility. Its catalog-matcher seam is defined in [evaluation.md](evaluation.md#evaluation-seam).

**CacheManager** persists JSON records under stable processor slugs and caller-supplied input keys. Thin stage decorators own keys and typed serialization while preserving processor behavior and identity.

## Processing Module Shape

Processing code is organized by capability rather than generic implementation
buckets:

```text
analysis/
  analyzer.py
  ear_preparation.py
  tear_profile.py
  profile_extraction/
    protocol.py
    alpha_tear.py
    cached.py

inference/
  detection.py
  segmentation/
    protocol.py
    sam3/
      features.py
      cached.py
      ear_segmenter.py
  landmarks/
    protocol.py
    yolo.py
    cached.py
```

`PreparedEar` is the single semantic intermediate between inference and
profile extraction. It is immutable and contains the source Photo and raster
box, original detector landmarks, snapped contour anchors, a finite full-image
contour running between those anchors, inferred side, and positive cleaned
area. `TearProfile` contains only immutable normalized one-dimensional depths.
AlphaTear configurations expose intentional research parameters; numerical
implementation constants remain private.

A settled AlphaTear configuration and its producer slug travel together as one
colocated `AlphaTearVersion`. Experimental tuning passes a raw
`AlphaTearConfig` and therefore has no persistent producer slug.

## Data and Identity

`photo_id` permanently identifies one immutable original photo asset. Replacing or re-encoding its original bytes creates a new photo ID. Photo IDs are only generated upon import in the application context; they are never generated in the research context.

`sighting_id` permanently identifies one observed event independently of elephant identity, date, or photo identity. A Sighting contains unique Photos whose `sighting_id` matches its own.

Both IDs are UUIDv4 values represented as UUIDs in Python and standard UUID strings in metadata. They encode no names, dates, paths, or parent-child structure.

The research dataset uses one metadata file with this shape:

```
photo_id,sighting_id,date,name,image_path
```

Dataset owns the complete metadata. Its PhotoStore receives only the `photo_id -> image_path` mapping. Paths remain private storage metadata; code resolves known-elephant identity from Dataset metadata rather than parsing paths or filenames.

The assigned photo and sighting IDs are permanent data artifacts and are preserved with the dataset.

## Runtime Data Flow

```
SightingEarPair + PhotoStore
  -> encoded bytes for each Photo
  -> decode_image to BgrImage
  -> ear localization and segmentation
  -> ear landmark detection
  -> immutable prepared ear
  -> AlphaTear profile for each side
  -> same-side matching against catalog evidence
  -> strongest left and right evidence per candidate
  -> one similarity score per candidate
```

## Inference Seams

Sighting analysis depends on three semantic processing capabilities:

- an `EarSegmenter` produces only ear masks and locations from a full BGR image;
- an `EarLandmarkDetector` returns the strongest upper/lower landmark detection, or `None` when a crop contains no ordinary detection;
- a `TearProfileExtractor` transforms one immutable prepared ear into a reusable tear profile.

SAM3 currently supplies ear localization and segmentation. Its expensive reusable computation returns every requested feature class, so that full multi-feature computation is cached before a thin semantic adapter filters it to ears exactly once. The current YOLO keypoint model supplies ear landmarks. Future implementations remain behind the same semantic interfaces.

Technical writing uses ear landmark detection for model output. Prepared-ear code distinguishes the original detector `landmarks`, which define AlphaTear's polar frame, from contour-snapped `anchors`, which delimit the prepared contour.

Model detections retain floating-point full-image geometry. Raster crops use an immutable integer `BoundingBox` with half-open coordinates, produced by flooring lower edges, ceiling upper edges, and clipping to the image. Public inference results, landmark cache records, and prepared-ear geometry all use full-image coordinates; crop-relative YOLO output is translated before it crosses the landmark processor interface or is persisted.

## Cache Architecture

One generic CacheManager persists records for every producer. It owns safe paths, JSON loading, atomic replacement, and obvious-corruption handling; it does not select whether a processing stage is cached or understand processor payloads.

Each settled deterministic processor exposes a stable human-readable `producer_slug` identifying the model, weights, prompt, preprocessing, thresholds, and every other output-changing setting. An output-changing processor change gets a new slug; ordinary refactoring does not. An experimental AlphaTear extractor may remain unversioned because parameter-tuning composition never persists its output. Cached decorators delegate the same slug as their wrapped processors and add only persistence behavior.

Keys contain only runtime input identity and actual dependent inputs. They remain readable:

```
sam3-features/<photo UUID>
yolo26n-keypoints-v1/<photo UUID>__crop_<x1>_<y1>_<x2>_<y2>
```

The SAM3 record contains the complete multi-feature result on both hits and misses; only its downstream ear adapter filters classes. Landmark records contain full-image-relative output. Final AlphaTear keys contain the source photo UUID, integer raster box, prepared ear's inferred side, segmentation producer slug, and landmark producer slug. They do not hash the derived contour or carry separate ear-preparation provenance. A material preparation or extraction change bumps the settled AlphaTear slug.

CacheManager namespaces and safely stores caller-supplied keys; it does not hash keys, read photos, resolve Dataset metadata, or understand producer payloads. Cached decorators construct keys, serialize typed outputs to JSON, parse JSON back to typed outputs, and validate their own records.

Standard construction caches SAM3's full feature computation, landmark detection, and settled AlphaTear extraction. Parameter-tuning construction injects a raw experimental AlphaTear extractor, bypassing profile reads and writes while retaining cached SAM3 features and landmark detection. CacheManager has no permission or tuning modes. Sighting analysis and catalog matching are unaware of cache policy.

When one source Photo supplies both declared sides, it is prepared on the first
side encountered and then reused. A retrieval, decoding, inference, or
preparation failure before side resolution is attributed to that first declared
side; analysis processes left before right.

Source paths and legacy identifiers are metadata, not cache identity. Existing cache records are migrated from legacy identifiers to photo UUIDs by joining the preserved original CSV and assigned CSV through unchanged image paths.

## Unsupported Prototypes

The existing `apps/` prototypes are historical and receive no compatibility guarantees. The API prototype is preserved on the `desktop-prototype` branch and removed from the active branch. Active modules do not retain legacy interfaces solely to keep prototypes runnable.
