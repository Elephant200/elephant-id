# Repository File Tree

Local checkout layout; huge trees (`dataset/`, `node_modules/`, `.venv/`, `.next/`, `.git/`) are only sketched.

## Root

```text
elephant-id/
├── .env
│   └── Local secrets — not committed; used by scripts
├── .env.example
│   └── Expected variables template.
├── .gitattributes / .gitignore
│   └── Git ignore + attributes (including LFS).
├── .pre-commit-config.yaml
│   └── Pre-commit hooks (`ruff --fix`).
├── .python-version
│   └── Interpreter pin (~matches `pyproject.toml` `>=3.12,<3.14`).
├── AGENTS.md
│   └── Agent and contributor guidelines (commands, boundaries, code style, logging).
├── CLAUDE.md
│   └── Claude Code pointer to `AGENTS.md`.
├── LICENSE · README.md
│   └── License and project intro.
├── pyproject.toml · uv.lock
│   └── Python deps and lockfile for `uv`.
```

## Local And Tooling Directories

```text
├── .cursor/
│   └── rules/ [summarized]
│       └── Agent rules (dataset safety, `ruff`, `uv run python`).
├── .vscode/
│   └── settings.json
├── .git/ [summarized]
│   └── Git internals.
├── .venv/ [summarized]
│   └── Virtualenv (`uv sync`).
```

## Caches And Data

```text
├── .cache/ [summarized]
│   ├── sam3/features/*.json — cached SAM3 feature JSON (see `Sam3Service`, `CacheManager`).
│   └── anchor/*.json — cached anchor keypoint detections (see `AnchorService`).
├── dataset/ [summarized]
│   ├── ear_features.jpg — reference beside `docs/seek.md`.
│   ├── elephants-alive/
│   │   └── `images.csv` + `coded/` / `raw/` / `uncoded/` photo trees (`Dataset`).
│   ├── ELPephants/ [summarized]
│   │   └── Legacy corpus (e.g. `all/`, `batches/`, `cropped/`, size splits, `tough/`).
│   ├── elephant-voices/ (female/, male/)
│   └── samples/ (sightings/, starred/)
```

## Application Code

```text
├── apps/
│   ├── visualization/
│   │   ├── README.md
│   │   │   └── Usage notes for the local Flask visualization app that browses dataset sightings and SEEK metadata.
│   │   ├── __init__.py · __main__.py
│   │   │   └── Package marker and module entry for `uv run python -m apps.visualization`.
│   │   ├── actions.py
│   │   │   └── Mutating visualization actions such as saving review state or interacting with dataset-derived records.
│   │   ├── analyzer.py
│   │   │   └── Session-only `AnalyzerWorkbench`: runs optional `PhotoAnalyzer`, retains results, and exports JSON/dashboard/profile artifacts.
│   │   ├── analyzer_render.py
│   │   │   └── Matplotlib dashboard renderer for full-photo analyzer diagnostics (`dashboard_png`).
│   │   ├── app.py
│   │   │   └── Flask app factory wiring routes, `ReviewerState`, and `AnalyzerWorkbench` for the local viewer.
│   │   ├── config.py
│   │   │   └── Visualization configuration values such as dataset paths and server defaults used by `app.py`.
│   │   ├── filters.py
│   │   │   └── Filtering helpers for limiting visible sightings/photos in the local dataset browser.
│   │   ├── paths.py
│   │   │   └── Filesystem path helpers shared by visualization routes, thumbnail generation, and dataset access.
│   │   ├── routes.py
│   │   │   └── Flask routes for the index, images, thumbnails, saved sightings, and analyzer API (`/api/analyzer`, dashboard PNG, JSON export).
│   │   ├── samples.py
│   │   │   └── Utilities for finding and presenting curated sample sightings from `dataset/samples`.
│   │   ├── sam3.py
│   │   │   └── Session-only SAM3 overlay runs for the visualization app (body/features presets, in-memory job state).
│   │   ├── seek_codes.py
│   │   │   └── SEEK-code presentation and parsing helpers for the visualization UI.
│   │   ├── state.py
│   │   │   └── Thread-safe in-memory reviewer state; loads sightings/photos by consuming `Dataset.iter_sightings()` and pulls seek codes from `Dataset.metadata`.
│   │   ├── thumbs.py
│   │   │   └── On-the-fly in-memory thumbnail generation (JPEG via PIL) used to avoid loading full-resolution dataset photos. No on-disk cache.
│   │   ├── static/app.js
│   │   │   └── Browser-side behavior for the Flask visualization UI served by `templates/index.html`.
│   │   ├── static/styles.css
│   │   │   └── CSS for the local visualization interface, paired with `templates/index.html`.
│   │   └── templates/index.html
│   │       └── Main Flask template that renders the visualization app shell and connects to static assets.
│   └── web/ [summarized]
│       └── Next.js (`apps/web`, marketing shell).
```

## Documentation

```text
├── docs/
│   ├── filetree.md
│   │   └── This repository map, generated from the local checkout and intended to guide contributors through source, data, caches, and generated files.
│   ├── field-context.md
│   │   └── Refined field workflow, deployment, SEEK-use, adaptive-record, and v1-scope context from field conversations.
│   ├── goals.md
│   │   └── Product and research goals for the Elephant ID system, referenced by architecture and implementation planning.
│   ├── papers/ [summarized]
│   │   └── Curvrank and SEEK reference PDFs plus `curvrank 1.tex` (including `SEEK_ System for Elephant Ear-pattern Knowledge ...pdf`).
│   ├── pipeline.md
│   │   └── Intended production pipeline: import, per-photo analysis, human-in-the-loop review, SEEK coding, and matching.
│   ├── sam3_sample_response.json
│   │   └── Example SAM3 response payload used to understand prediction schema and support visualization work.
│   ├── seek.md
│   │   └── SEEK coding reference documentation that informs `SeekCode` parsing and dataset ground-truth interpretation.
│   ├── tear-embedding.md
│   │   └── Design record for the v1 tear-depth profile (`tear_profile.py`); superseded in implementation by v2.
│   ├── tear-embedding-v2.md
│   │   └── Current angular tear-profile coordinate system and pipeline parameters.
│   └── technical-architecture.md
│       └── Architecture overview connecting datasets, AI services, coding models, and future application surfaces.
```

## Python Package

```text
├── src/
│   └── elephant_id/
│       ├── __init__.py
│       │   └── Package marker for `elephant_id`, enabling imports from code, scripts, and tests.
│       ├── cache.py
│       │   └── JSON cache manager used by AI services to persist expensive model responses under `.cache`.
│       ├── constants.py
│       │   └── Shared constants for SAM3 query presets, cache paths, thresholds, and Roboflow workflow identifiers.
│       ├── dataset.py
│       │   └── Dataset abstraction that loads `images.csv`, resolves `Photo` paths, groups `Sighting`s, and reads images with caching.
│       ├── log.py
│       │   └── Entry-point loguru configuration (`configure_logging()`); library code logs via `logger` only.
│       ├── visualize.py
│       │   └── Mask decoding, prediction overlays, and tear-profile diagnostic plotting helpers (OpenCV + Matplotlib).
│       ├── ai/
│       │   ├── __init__.py
│       │   │   └── Public AI subpackage exports: `AgeService`, `AnchorService`, `Detection`, `GenderService`, `Sam3Service`.
│       │   ├── age.py
│       │   │   └── Age regression CNN runner and cached `AgeService` (local PyTorch; stub runner).
│       │   ├── anchor.py
│       │   │   └── Anchor keypoint YOLO26 runner and cached `AnchorService` (local ultralytics).
│       │   ├── detection.py
│       │   │   └── Typed `Detection` dataclass shared across AI services (box, mask, keypoints, serialization).
│       │   ├── gender.py
│       │   │   └── Gender classification CNN runner and cached `GenderService` (local PyTorch; stub runner).
│       │   └── sam3.py
│       │       └── Roboflow SAM3 workflow wrapper and cached `Sam3Service` that operates on `Dataset` and `Photo` objects.
│       ├── coding/
│       │   ├── __init__.py
│       │   │   └── Public coding exports (`SeekCoder`).
│       │   ├── coder.py
│       │   │   └── `SeekCoder` orchestrates per-photo analysis into a sighting-level result dict (preview SEEK aggregation is still a stub).
│       │   ├── photo_analyzer.py
│       │   │   └── Runs SAM3, anchor, gender, and age models on a photo and delegates field evidence to per-field analyzers.
│       │   ├── age.py
│       │   │   └── `AgeFieldAnalyzer`; runs `AgeService` on the masked body crop.
│       │   ├── gender.py
│       │   │   └── `GenderFieldAnalyzer`; runs `GenderService` on the masked body crop.
│       │   ├── tusks.py
│       │   │   └── `TuskFieldAnalyzer`; infers tusk presence and side from tusk, trunk, and view evidence.
│       │   └── ears/
│       │       ├── __init__.py
│       │       │   └── Public ear-field exports: `AnchoredEar`, `EarFieldAnalyzer`.
│       │       ├── analyzer.py
│       │       │   └── `EarFieldAnalyzer`: per-ear diagnostics, tear profiles, and tear/hole placeholders.
│       │       ├── anchored_ear.py
│       │       │   └── `AnchoredEar` and mask-to-anchor-contour preparation helpers.
│       │       ├── geometry.py
│       │       │   └── Ear-margin geometry primitives: densify, ring side paths, alpha shape, inward normals, nearest-crossing ray scans.
│       │       └── tear_profile.py
│       │           └── Angular tear-depth profile: `tear_profile()` / `embed()` -> `TearProfile` or 1-D array.
│       ├── matching/
│       │   ├── __init__.py
│       │   │   └── Public tear-matching exports: `TearMatcher`, `TearMatcherConfig`, `TearMatch`, `TearMatchGallery`.
│       │   └── tear_matcher.py
│       │       └── Sparse tear-depth profile matcher: penalized shift, stretch search, pair scoring, and gallery ranking.
│       ├── domain/
│       │   ├── __init__.py
│       │   │   └── Public data-model exports for `SeekCode`, `Photo`, and `Sighting`.
│       │   ├── photo.py
│       │   │   └── Immutable `Photo` dataclass validating identifier, path, elephant name, and sighting relationships.
│       │   ├── seek_code.py
│       │   │   └── Immutable SEEK code parser/formatter and validator used by datasets and tests.
│       │   └── sighting.py
│       │       └── Immutable `Sighting` dataclass that groups photos for one elephant/date and validates consistency.
│       └── image/
│           ├── __init__.py
│           │   └── Re-exports canonical `BgrImage` type alias.
│           ├── bgr.py
│           │   └── `BgrImage` type alias (OpenCV HWC BGR uint8 convention).
│           ├── boxes.py
│           │   └── Bounding-box coordinate utilities (center/xyxy conversion, clipping; half-open float boxes).
│           ├── masks.py
│           │   └── COCO RLE mask decoding and mask geometry helpers.
│           └── transforms.py
│               └── Pixel-space crop/mask transforms producing `BgrImage` outputs.
```

## Scripts, Legacy Code, Models, And Tests

```text
├── legacy/
│   ├── background.py
│   │   └── Roboflow SAM3 background-removal workflow (reference).
│   ├── batch_to_coco.py
│   │   └── Batch conversion to COCO format.
│   ├── crop.py
│   │   └── Legacy cropping utilities.
│   ├── distill.py
│   │   └── Model distillation tooling.
│   ├── hole_experiments.py
│   │   └── Ear-notch/hole detection method comparison on anchored ear crops.
│   ├── model.py
│   │   └── Legacy Roboflow inference wrapper.
│   └── tear_doc_figures.py
│       └── Regenerates decision-record diagrams for `docs/tear-embedding.md`.
├── model_weights/ [summarized]
│   ├── anchor_extraction_yolo26/ · anchor_extraction_yolo26_v2/ · anchor_extraction_yolov11/
│   ├── ear_detection_yolo26/ · ear_segmentation_yolo26/
│   ├── face_detection_yolov11/
│   ├── sam3/
│   ├── tail_detection_yolo26/ · tusk_detection_yolo26/
│   └── Each folder: `weights.pt` (or `sam3.pt`) + `info.txt`.
├── outputs/   (gitignored)
│   └── Regenerable figures and metrics, one subdirectory per script (`outputs/<script>/`).
├── legacy/tear_algorithm/
│   └── Frozen exploration snapshots (tear_*.py, arPLS.py) with a README of what each tried and concluded; not maintained.
├── scripts/
│   ├── anchor_training_data.py
│   │   └── Generate, augment, and export anchor keypoint training data from dataset photos.
│   ├── ear_embedding.py
│   │   └── Visual QA: per-photo ear image beside its 1-D tear profile with detected events.
│   ├── evaluation.py
│   │   └── Evaluate tear-profile retrieval on the filtered good-ear set: metrics, case figures, and gallery ranking.
│   ├── holes.py
│   │   └── Ear-notch/hole contour exploration on masked ear crops via Laplacian-of-Gaussian filtering.
│   ├── matching.py
│   │   └── Database-vs-query tear-matching experiment: enroll/query split, retrieval metrics, and alignment-audit case figures.
│   ├── quality.py
│   │   └── Exploratory ear-image quality heuristic on `outputs/ear_segmentation_filtered` crops.
│   ├── view.py
│   │   └── Local script that runs SAM3, anchor, and ear analyzers across named preset photos with OpenCV display.
│   ├── visualize_analyzer.py
│   │   └── Matplotlib dashboard for one photo's `PhotoAnalyzer` output; saves under `outputs/analyzer/`.
│   └── warm_sam3_cache.py
│       └── Warm the SAM3 cache for every photo in the dataset.
└── tests/
    ├── conftest.py
    │   └── Shared pytest fixtures (RLE helpers, sample photos/sightings).
    ├── test_anchor.py
    │   └── Unit tests for `AnchorService` caching, crop translation, and detection serialization.
    ├── test_boxes.py
    │   └── Unit tests for box coordinate conversion and clipping.
    ├── test_cache.py
    │   └── Unit tests for `CacheManager` get-or-compute behavior.
    ├── test_coder.py
    │   └── Unit tests for `SeekCoder` and coding pipeline integration.
    ├── test_dataset.py
    │   └── Unit tests for `Dataset` loading, lookup, sighting grouping, ground-truth SEEK codes, and image caching.
    ├── test_detection.py
    │   └── Unit tests for `Detection` geometry, serialization, and factory methods.
    ├── test_masks.py
    │   └── Unit tests for RLE mask decoding and bounds.
    ├── test_photo.py
    │   └── Unit tests for `Photo` validation rules around identifiers, relative paths, and sighting naming.
    ├── test_sam3.py
    │   └── Unit tests for `Sam3Service` preset resolution and caching.
    ├── test_seek_code.py
    │   └── Unit tests for `SeekCode` parsing, formatting, equality, hashing, and invalid-code rejection.
    ├── test_sighting.py
    │   └── Unit tests for `Sighting` validation around date/name consistency, duplicate photos, and empty sightings.
    ├── test_transforms.py
    │   └── Unit tests for crop and mask image transforms.
    └── test_visualize.py
        └── Unit tests for prediction overlay drawing.
```
