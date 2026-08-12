# Architecture

AlphaPhant is a research implementation of a fully automated elephant re-identification algorithm. Its current scope begins with a sighting ear pair and ends with a ranked known-elephant catalog. The future application workflow surrounds this algorithm; see [workflow.md](workflow.md).

## Design Priorities

- Keep the research pipeline small, explicit, and testable.
- Preserve tear-profile extraction and tear-profile matching behavior during restructuring.
- Put model variation behind semantic inference interfaces.
- Keep identity-retrieval evaluation independent of pipeline implementation.
- Cache expensive computation and final reusable tear profiles, not inexpensive orchestration.
- Treat package names and data representations as provisional.

## Responsibilities

The responsibilities are firmer than their eventual filenames.

**Analysis** owns sighting analysis and the geometric path from model output to one tear profile per ear. It contains ear-contour preparation, alpha-shape construction, and tear-profile extraction.

**Inference** owns implementations of ear localization, ear segmentation, and ear landmark detection. Analysis depends on those capabilities rather than on SAM3, YOLO, U-Net, BiRefNet, or another architecture.

**Matching** owns tear-profile matching and known-elephant ranking. It compares same-side ears, symmetrizes the directional similarity score, selects the strongest catalog evidence independently per side, and averages the two side scores.

**Evaluation** owns benchmark examples, opaque evaluation keys, leakage-free splits, failure accounting, metrics, and reproducibility. It observes ranked candidates rather than masks, anchors, contours, tear profiles, or model settings.

**Cache storage** is one generic mechanism for records from immutable producers. Each model invocation and final per-ear tear-profile extraction supplies its producer name, key inputs, metadata, serialization, and validation.

**Historical data access** resolves labeled photos and sightings from the research dataset. Identity-bearing historical filenames are provenance, not requirements imposed on AlphaPhant.

## Data Flow

```
sighting ear pair
  -> ear localization and ear segmentation
  -> ear landmark detection
  -> ear contour
  -> alpha-shape-derived tear profile for each side
  -> same-side tear-profile matching against catalog evidence
  -> strongest left and right similarity score per known elephant
  -> mean two-side similarity score
  -> ranked candidate list
```

Research supplies the ear pair directly. A future application performs ear selection first and enters the same flow.

## Inference Seams

Analysis requires two semantic capabilities:

- produce ear masks and locations from a full photo;
- locate the upper and lower anatomical landmarks that define the relevant ear contour.

SAM3 currently supplies ear localization and segmentation. The current YOLO keypoint model supplies ear landmarks. A future detector-plus-segmentation composition remains hidden behind the same segmentation interface. Downstream geometry remains unchanged when an inference implementation changes.

Technical writing uses ear landmark detection. Code continues to use `anchor` as the noun, including names such as `upper_anchor` and `lower_anchor`.

## Cache Architecture

One cache store serves every producer. Its global mode comes from `ELEPHANT_ID_CACHE_MODE`:

- `read_write` uses valid records and computes and saves misses;
- `read_only` uses valid records and fails clearly on a miss;
- `disabled` computes without reading or writing cache records.

Durable producer categories are:

- `sam3-features`;
- `sam3-body`, retained for possible later research;
- `yolo26n-keypoints-v1`;
- future immutable model names;
- `tear-profile-v1`.

Any output-changing model, configuration, prompt, preprocessing rule, or algorithm change gets a new producer name. Ordinary refactoring does not. Keys follow actual inputs: source content hashes for photo-level models, upstream record keys and crop coordinates for dependent models, and segmentation plus landmark record keys and declared side for a final tear profile.

Source paths, photo identifiers, timestamps, and configuration summaries are metadata. `dataset/elephants-alive/image_hashes.csv` maps `image_path` to `content_sha256`. The coded dataset is treated as immutable until an explicit rebuild or verification.

## Unsupported Prototypes

The existing `apps/` prototypes are historical and receive no compatibility guarantees. The API prototype will be preserved on the `desktop-prototype` branch and removed from the active branch. Active modules should not retain legacy interfaces merely to keep prototypes runnable.

The reasons behind these boundaries live in [adr/](adr/).
