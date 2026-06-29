# Image Picker

Tiny local Flask app for selecting side-specific high-quality ear examples.

Run from the repo root:

```bash
uv run python -m apps.image_picker
```

Then open `http://127.0.0.1:8010`.

The app reads dataset metadata from `dataset/elephants-alive/images.csv`, uses cached SAM3/anchor outputs when available, lazily computes missing model outputs when local credentials and weights are available, and exports selected original photos to `outputs/high_quality`.
