# Agent Guidelines

## Project Description

Elephant ID is a human-in-the-loop system for identifying individual African
elephants from sighting photo folders. It uses AI, structured SEEK coding, and
expert review to produce draft identification records, support matching against
known elephants, and efficiently utilize human oversight.

## Commands

Run Python commands from the repo root with `uv`; do not activate `.venv` or
call `python` directly.

```bash
uv run pytest -v
uv run pytest tests/test_boxes.py -v
uv run ruff check .
uv run python -m apps.visualization
uv sync
uv sync --group visualization # dependencies for the visualization app
uv sync --group local # dependencies for local AI; should always be installed
```

After Python code changes, run `uv run ruff check .` and fix reported issues.
`legacy/` is excluded from Ruff; avoid incidental cleanup there.

## Stack

- Python package: Python `>=3.12,<3.14`, `uv`, `pytest`, `ruff`, `numpy`,
  `opencv-python`, `pandas`, `pycocotools`.
- Local AI extras: `inference-sdk`, `roboflow`, `torch`, `torchvision`,
  `ultralytics`.
- Visualization app: Flask, development-only, local dataset reviewer.

## Project Structure

- `src/elephant_id/`: Python package.
- `tests/`: Python unit tests.
- `apps/visualization/`: development-only Flask reviewer for local `dataset/`.
- `apps/web/`: a next.js landing page; do not touch.
- `scripts/`: local exploration and model/demo scripts.
- `legacy/` and `.curvrank_ref/`: historical/reference material; do not
  modernize unless asked.
- `docs/`: SEEK, architecture, file tree, and reference papers.

## Boundaries

- Never commit secrets, `.env` contents, API keys, model credentials, or private
  dataset contents.
- `dataset/` is user data. Do not delete, move, overwrite, or bulk-clean
  anything inside it unless explicitly instructed.
- Undo behavior must only remove artifacts created in the current session.
- Ignore `apps/web/` unless explicitly asked to work on it.
- Do not touch generated/vendor/runtime directories such as `.venv/`, `.next/`,
  `node_modules/`, `__pycache__/`, `.ruff_cache/`, or `.pytest_cache/` except
  for explicit maintenance.
- Never revert or overwrite existing user changes unless explicitly asked.

## Code Style

- Type annotations are required on function and method signatures.
- Every module, package, function, and class has a docstring.
- Keep docstrings concise and accurate. Use `Args:`, `Returns:`/`Yields:`, 
  and `Raises:` only when they add non-obvious information.
- Use `Raises:` when validation is part of the contract.
- Prefer clear names and structure over comments; add comments only for
  non-trivial logic.

Good docstring styles:

```python
def clip_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Clip a half-open xyxy box to image bounds.

    Args:
        x1: Left edge, inclusive.
        y1: Top edge, inclusive.
        x2: Right edge, exclusive.
        y2: Bottom edge, exclusive.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        Integer coordinates expanded outward to whole pixels.

    Raises:
        ValueError: If the image or box is invalid, or the box misses the image.
    """
```

```python
def center_to_xyxy(
    x: float,
    y: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    """Convert center-format coordinates to half-open xyxy coordinates."""
```

## Images And Geometry

- The canonical in-memory image type is `BgrImage` from `elephant_id.image`.
- `BgrImage` means HWC, BGR, `uint8`, OpenCV-native.
- Do not introduce PIL/RGB conversions inside the package unless there is a
  clear boundary reason.
- Decode and encode with `cv2.imread`, `cv2.imdecode`, or `cv2.imencode` at call
  sites; do not add thin codec wrapper modules.
- `elephant_id.image` deliberately re-exports only `BgrImage`; import helpers
  from `elephant_id.image.boxes`, `elephant_id.image.masks`, or
  `elephant_id.image.transforms`.
- Geometry is float `xyxy`, half-open: `x2`/`y2` are exclusive.
- Convert to integer pixels only at raster boundaries, preferably through
  `clip_xyxy()` or `mask_bounds()`.
- OpenCV drawing APIs use inclusive endpoints; draw half-open boxes with
  `x2 - 1`, `y2 - 1`.
- Public color/background arguments are human-facing RGB even though buffers are
  BGR; flip to BGR only at the write boundary.
- COCO RLE masks use `size=[height, width]` and `counts: str | bytes`; decoded
  masks are 2D boolean arrays.

## Domain And Dataset

- `Photo`, `Sighting`, and `SeekCode` are immutable validated domain objects.
- `Photo.image_path` must be relative, non-escaping, and match the identifier
  stem.
- `Sighting.sighting_id` is `{elephant_name}_{iso_date}`, and all photos must
  belong to that sighting.
- Preserve the SEEK-code grammar exactly. Unknown is `None`/`_`; ages are two
  digits; right-ear sectors are `0/7/8/9`; left-ear sectors are `0/3/4/5`.
- `Dataset` lazy-loads metadata CSVs. Iteration order matters and must preserve
  CSV row order.
- `Dataset.read_image()` returns fresh image copies and uses an internal LRU
  cache.

## Cache And AI Services

- Cache files are JSON envelopes under `.cache/{namespace}/{key}.json`.
- Cache namespaces and keys must not escape their roots.
- Keep cache writes atomic using temp-file replacement.
- Prefer service wrappers over direct runner use. Services handle caching, key
  construction, coordinate normalization, and serialization.
- Local model services may require optional local dependencies, model weights,
  dataset files, `.env`, and `ROBOFLOW_API_KEY`.
- Avoid broad imports from `elephant_id.ai` in core code when optional local
  model dependencies may be missing.

## Visualization App

- `apps/visualization` is local development tooling, not production UI.
- Flask route JSON field names are an API contract with
  `apps/visualization/static/app.js`.
- The visualization app can create, rename, delete, and mirror files under
  `dataset/samples`; priority state is encoded by the `** ` filename prefix.

## Testing

- Prefer small synthetic arrays/images and fixtures over real dataset
  dependencies.
- Use existing fixtures such as `rle_from_mask`, `make_photo`, and
  `make_sighting`.
- Use fakes/recording stubs for model clients and caches; do not initialize real
  model clients in unit tests.
- Add or update focused tests when changing validation, geometry, cache keys,
  serialization, or dataset ordering.

## Git Workflow

- Check `git status --short` before editing.
- Keep changes scoped to the request.
- Do not commit unless explicitly asked.
- Report noteworthy out-of-scope issues at the end of your response with file
  paths.
