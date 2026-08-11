# Matching Image Picker

Local Flask tool for building the paper's high-quality matching dataset: pick
one canonical left-ear and one canonical right-ear image per sighting, per
elephant, from the labeled historical dataset.

Run from the repo root:

```bash
uv run python -m apps.image_picker
```

Then open `http://127.0.0.1:8010`.

## How it works

- Candidate ear crops come from the production `PhotoAnalyzer`, so body/ear
  selection and anchoring are the same code path as the rest of the project.
- Each anchored ear is scored by its `AnchoredEar.quality` prior. A sighting
  qualifies when it has at least one left-side and one right-side candidate
  above the quality threshold; an elephant is listed once it has at least five
  qualifying sightings.
- Analysis runs against cached SAM3/anchor outputs, so reviewing needs no live
  model calls or credentials in the common case. Missing outputs are computed
  lazily when local weights and `ROBOFLOW_API_KEY` are available.
- Clicking a candidate immediately upserts its row in
  `outputs/high_quality/manifest.csv` and exports two images: a full-frame copy
  under `images_full/` and a tight ear crop under `images_crop/`. Changing a
  pick deletes the stale exports. The manifest is the only persisted state.
