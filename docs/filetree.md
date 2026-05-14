# Repository File Tree

Local checkout layout; huge trees (`dataset/`, `node_modules/`, `.venv/`, `.next/`, `.git/`) are only sketched.

## Root

```text
elephant-id/
├── .env
│   └── Local secrets — not committed; used by scripts (e.g. `scripts/visualize_sam3.py`).
├── .env.example
│   └── Expected variables template.
├── .gitattributes / .gitignore
│   └── Git ignore + attributes (including LFS).
├── .python-version
│   └── Interpreter pin (~matches `pyproject.toml` `>=3.11,<3.14`).
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
├── .git/ [summarized]
│   └── Git internals.
├── .venv/ [summarized]
│   └── Virtualenv (`uv sync`).
```

## Caches And Data

```text
├── .cache/ [summarized]
│   └── sam3/*.json — cached SAM3 feature JSON (see `src/elephant_id/ai/sam3.py`, `CacheManager`).
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
│   │   ├── __main__.py
│   │   │   └── Module entry point for `uv run python -m apps.visualization`, delegating to the Flask app.
│   │   ├── actions.py
│   │   │   └── Mutating visualization actions such as saving review state or interacting with dataset-derived records.
│   │   ├── app.py
│   │   │   └── Flask app factory/runtime wiring that connects routes, configuration, and state for the local viewer.
│   │   ├── config.py
│   │   │   └── Visualization configuration values such as dataset paths and server defaults used by `app.py`.
│   │   ├── filters.py
│   │   │   └── Filtering helpers for limiting visible sightings/photos in the local dataset browser.
│   │   ├── paths.py
│   │   │   └── Filesystem path helpers shared by visualization routes, thumbnail generation, and dataset access.
│   │   ├── routes.py
│   │   │   └── Flask route definitions that serve the index, images, thumbnails, and related viewer endpoints.
│   │   ├── samples.py
│   │   │   └── Utilities for finding and presenting curated sample sightings from `dataset/samples`.
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
│   ├── goals.md
│   │   └── Product and research goals for the Elephant ID system, referenced by architecture and implementation planning.
│   ├── sam3_sample_response.json
│   │   └── Example SAM3 response payload used to understand prediction schema and support visualization work.
│   ├── seek.md
│   │   └── SEEK coding reference documentation that informs `SeekCode` parsing and dataset ground-truth interpretation.
│   └── technical-architecture.md
│       └── Architecture overview connecting datasets, AI services, coding models, and future application surfaces.
```

## Python Package

```text
├── src/
│   ├── elephant_id/
│   │   ├── __init__.py
│   │   │   └── Package marker for `elephant_id`, enabling imports from code, scripts, and tests.
│   │   ├── cache.py
│   │   │   └── JSON cache manager used by `Sam3Service` to persist expensive model responses under `.cache`.
│   │   ├── coding.py
│   │   │   └── Placeholder module for future SEEK-code generation logic that will likely combine model outputs and dataset models.
│   │   ├── constants.py
│   │   │   └── Shared constants for SAM3 query presets, cache paths, thresholds, and Roboflow workflow identifiers.
│   │   ├── dataset.py
│   │   │   └── Dataset abstraction that loads `images.csv`, resolves `Photo` paths, groups `Sighting`s, and reads images with caching.
│   │   ├── visualize.py
│   │   │   └── Mask decoding and drawing utilities that convert SAM3 predictions into annotated PIL images.
│   │   ├── ai/
│   │   │   ├── __init__.py
│   │   │   │   └── Public AI subpackage export surface, currently exposing `Sam3Service`.
│   │   │   ├── anchor.py
│   │   │   │   └── Placeholder module for future anchor extraction logic tied to elephant feature localization.
│   │   │   └── sam3.py
│   │   │       └── Roboflow SAM3 workflow wrapper and cached service layer that operates on `Dataset` and `Photo` objects.
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   │   └── Public data-model exports for `SeekCode`, `Photo`, and `Sighting`.
│   │   │   ├── photo.py
│   │   │   │   └── Immutable `Photo` dataclass validating identifier, path, elephant name, and sighting relationships.
│   │   │   ├── seek_code.py
│   │   │   │   └── Immutable SEEK code parser/formatter and validator used by datasets and tests.
│   │   │   └── sighting.py
│   │   │       └── Immutable `Sighting` dataclass that groups photos for one elephant/date and validates consistency.
│   └── elephant_id.egg-info/
│       ├── PKG-INFO
│       │   └── Installed package metadata generated from `pyproject.toml` during an editable/local build.
│       ├── SOURCES.txt
│       │   └── Generated list of package source files included in the Python distribution metadata.
│       ├── dependency_links.txt
│       │   └── Setuptools metadata file for dependency links, currently part of generated egg-info.
│       ├── requires.txt
│       │   └── Generated dependency list corresponding to `pyproject.toml` requirements.
│       └── top_level.txt
│           └── Generated top-level package name metadata for the `elephant_id` distribution.
```

## Scripts, Legacy Code, Models, And Tests

```text
├── legacy/ [summarized]
│   └── Old Roboflow/crop/batch/COCO tooling — kept for reference.
├── model_weights/ [summarized]
│   └── Trained checkpoints (`weights.pt` + `info.txt` per folder): anchors (YOLO26 / v11), ear/face/tail/tusk heads.
├── scripts/
│   └── visualize_sam3.py
│       └── Local script that loads a dataset sighting, runs cached SAM3 predictions, and displays annotated images via `visualize_predictions`.
└── tests/
    ├── test_dataset.py
    │   └── Unit tests for `Dataset` loading, lookup, sighting grouping, ground-truth SEEK codes, and image caching.
    ├── test_photo.py
    │   └── Unit tests for `Photo` validation rules around identifiers, relative paths, and sighting naming.
    ├── test_seek_code.py
    │   └── Unit tests for `SeekCode` parsing, formatting, equality, hashing, and invalid-code rejection.
    └── test_sighting.py
        └── Unit tests for `Sighting` validation around date/name consistency, duplicate photos, and empty sightings.
```
