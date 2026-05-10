# Visualization

A development-only Flask app for visualizing the `dataset/` contents (sightings, photos, SEEK codes, etc.). It is intended for local inspection and labeling work, not for production use — the production UI lives in `apps/web/` (Next.js).

## Install

From the repo root:

```bash
uv sync --group visualization
```

## Run

```bash
uv run python -m apps.visualization
```

Then open the URL printed in the terminal (default: http://127.0.0.1:8000).
