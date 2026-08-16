# Architecture

AlphaPhant is a research implementation of a fully automated elephant re-identification algorithm. Its current scope begins with a sighting ear pair and ends with a ranked known-elephant catalog. The future application shares its neutral photo and sighting model; see [reference/application.md](reference/application.md).

## Design Priorities

- Keep the research pipeline small, explicit, and testable.
- Use permanent opaque photo and sighting identity throughout the system.
- Separate neutral domain objects, image storage, and identity-aware research metadata.
- Preserve tear-profile extraction and matching behavior during restructuring.
- Put model variation behind semantic inference interfaces.
- Keep identity-retrieval evaluation independent of ranker implementation.
- Cache expensive computation and final reusable tear profiles, not orchestration.

## Responsibilities

**Domain** owns the neutral immutable values shared across the system:

- `Photo` carries a UUIDv4 photo ID and its parent sighting ID;
- `Sighting` carries a UUIDv4 sighting ID, a required sighting date, and distinct Photos;
- `SightingEarPair` carries a sighting ID and one Photo per declared side. Both Photos belong to that sighting, and one Photo may serve both sides.

**Dataset** is the private research dataset object. It constructs and resolves domain objects, owns known-elephant metadata, and owns a filesystem-backed PhotoStore. Rankers never receive the Dataset.

**PhotoStore** owns retrieval of original encoded bytes through `read(photo: Photo) -> bytes`. It uses `photo.photo_id` for lookup and does not expose a known-elephant resolver method.

**Image** owns `decode_image(encoded: bytes) -> BgrImage` and BGR image and basic/universal geometry utilities.

**Analysis** owns sighting analysis, ear-contour preparation, alpha-shape construction, and tear-profile extraction.

**Inference** owns implementations of ear localization and segmentation and ear landmark detection. Analysis depends on these semantic capabilities rather than particular model architectures.

**Matching** owns tear-profile matching and known-elephant ranking.

**Evaluation** owns private ground truth, benchmark examples, candidate keys, splits, failure accounting, metrics, and reproducibility. Its ranker boundary is defined in [evaluation.md](evaluation.md#evaluation-seam).

**CacheManager** persists records under stable processor identities and caller-supplied input keys. Thin stage decorators add caching without changing processor behavior or identity.

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
  -> ear contour
  -> tear profile for each side
  -> same-side matching against catalog evidence
  -> strongest left and right evidence per candidate
  -> complete ranked candidate list
```

## Inference Seams

Analysis requires two semantic capabilities:

- produce ear masks and locations from a full BGR image;
- locate the upper and lower anatomical landmarks that define the relevant ear contour.

SAM3 currently supplies ear localization and segmentation. The current YOLO keypoint model supplies ear landmarks. Future implementations remain behind the same semantic interfaces.

Technical writing uses the term ear landmark detection. Code uses `anchor` for detected endpoints.

## Cache Architecture

One generic CacheManager persists records for every producer. It owns safe paths, JSON loading, atomic replacement, and obvious-corruption handling; it does not select whether a processing stage is cached.

Each deterministic processor exposes a stable `producer_id` identifying the model, weights, prompt, preprocessing, thresholds, and every other output-changing setting. An output-changing processor change gets a new identity; ordinary refactoring does not. Cached decorators delegate the same identity as their wrapped processors and add only persistence behavior.

Keys contain only runtime input identity and actual dependent inputs. They remain readable:

```
sam3-features/<photo UUID>
yolo26n-keypoints-v1/<photo UUID>__crop_<x1>_<y1>_<x2>_<y2>
```

Dependent records use upstream processor identities and semantic inputs such as side or crop coordinates. CacheManager namespaces and safely stores caller-supplied keys; it does not hash keys, read photos, resolve Dataset metadata, or understand producer payloads. Writes are atomic and loaded records are validated.

Standard construction decorates ear segmentation, ear landmark detection, and tear-profile extraction with cached adapters. Parameter-tuning construction leaves the tear-profile extractor undecorated, bypassing profile reads and writes while retaining cached segmentation and landmark detection. Sighting analysis and ranking are unaware of cache policy.

Source paths and legacy identifiers are metadata, not cache identity. Existing cache records are migrated from legacy identifiers to photo UUIDs by joining the preserved original CSV and assigned CSV through unchanged image paths.

## Unsupported Prototypes

The existing `apps/` prototypes are historical and receive no compatibility guarantees. The API prototype is preserved on the `desktop-prototype` branch and removed from the active branch. Active modules do not retain legacy interfaces solely to keep prototypes runnable.
