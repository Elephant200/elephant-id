# Alphaphant Desktop

Offline desktop app for Elephants Alive field researchers: import a sighting
folder, review the extracted ear evidence, rank likely matches against the
known-elephant catalog, and confirm a match or enroll a new individual.

- Frontend: Electron + React (Vite) in this directory.
- Backend: FastAPI sidecar in `src/elephant_id/api/`, spawned by Electron via
  `uv run python -m elephant_id.api`. Everything runs locally; model inference
  is served from the repo `.cache/` plus local model weights.

## Prerequisites

From the repo root:

```bash
uv sync --group api --group local
```

The sidecar needs `outputs/tear_matching_eval/hq_profiles.npz` (the gallery)
and `outputs/high_quality/manifest.csv` (ear-crop images). On first boot it
computes and caches the gallery pairwise score matrix under
`outputs/alphaphant/` (~1 minute).

## Run

```bash
cd apps/desktop
npm install
npm start          # builds the renderer, launches Electron + sidecar
```

Development:

```bash
npm run dev                          # vite dev server on :5183
uv run python -m elephant_id.api     # sidecar on :8756 (from repo root)
# then open http://localhost:5183 in a browser (folder picker becomes a
# path text input), or:
VITE_DEV_SERVER_URL=http://localhost:5183 ALPHAPHANT_EXTERNAL_API=1 npm run electron
```

Environment knobs: `ALPHAPHANT_API_PORT` (default 8756),
`ALPHAPHANT_DATA_DIR` (sidecar state, default `outputs/alphaphant/`),
`ALPHAPHANT_EXTERNAL_API=1` (don't spawn a sidecar),
`ALPHAPHANT_SCREENSHOT=<path>` (capture the window after boot and exit —
used for automated boot verification).

## Demo mode (held-out sightings)

For an honest live demo, hold a few sightings out of the catalog and import
them fresh:

```bash
uv run python scripts/make_demo_holdout.py   # from the repo root
cd apps/desktop
npm run demo
```

The script writes `outputs/alphaphant_demo/`: a filtered gallery
(`gallery_profiles.npz`, held-out sightings removed), one import-ready folder
per held-out sighting under `sightings/`, and `holdout_summary.txt` naming the
true elephants. Held-out sightings are chosen so the true elephant ranks in
the top 2 — the demo shows the matcher succeeding without self-matching.
`npm run demo` launches Electron against that gallery with its own data dir;
delete `outputs/alphaphant_demo/data/store.json` and `data/sightings/` to
reset the demo between runs.

## Demo data (regular mode)

Any dataset sighting folder works as a sample, e.g.
`dataset/elephants-alive/coded/Alvin/2017-11-30`. Photos named
`{Name}_{YYYY-MM-DD}_{seq}.jpg` hit the warm model cache, so analysis runs
fully offline; photos missing from the cache fall back to precomputed gallery
profiles when available and are skipped otherwise.

## Lab page (development only)

The sidebar's Lab page uploads a single image to `POST /dev/analyze` and
shows the full analyzer output (detections overlay, view/age/gender/tusk
suggestions, anchored ear crops, tear profiles) styled like the Review page —
the app equivalent of `scripts/visualize_analyzer.py`.
