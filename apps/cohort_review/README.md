# Sighting ear-pair picker

A local web app for choosing one left and one right ear crop per sighting to build the tuning and benchmark cohorts.

```bash
uv run --group visualization python -m apps.cohort_review
```

Open http://127.0.0.1:5055. Choose a left crop, then a right crop; the right choice saves the pair and advances. Reject sighting skips it. Back and Forward (or the arrow keys) revisit earlier screens, and a new choice there replaces the old decision. Backup and Restore export and import progress.

## Audit and replace

Open http://127.0.0.1:5055/audit to step through every approved pair in manifest order. Each ear is shown in its full photo with the crop box drawn, under the elephant's name and the declared side. Left and Right move; Space flags or unflags the pair. The position is saved, so the audit resumes where you stopped.

Open http://127.0.0.1:5055/replace to work through the flagged sightings in the picker. Choose a new left and right crop, or reject the sighting. Either decision rewrites `approved-manifest.csv` and resolves the flag. The replaced pair is kept under `superseded` in `state.json`. Space on a re-picked pair flags its new crops.

## Queue

- A sighting is usable when both ears have a crop scoring at least 8.
- Only elephants with at least three usable sightings are queued. An elephant leaves the queue at ten approved pairs, or when rejections leave it unable to reach three.
- Pairs are offered by tier 10, 9, then 8. Within a tier, priority is 0.9 × the weaker side's strength + 0.1 × the stronger side's, where strength weights distinct photos by score (10 = 1, 9 = 0.5, 8 = 0.25, 7 = 0.125).

## Files

Reads crop assessments from `outputs/curation/archive/` and metadata from `dataset/elephants-alive/`. Progress lives in `outputs/curation/sighting-pair-review/state.json`; approved pairs go to `approved-manifest.csv` beside it, and audit flags and positions go to `audit.json`. Existing manifests and photos are never modified.

`session.py` owns queue order, decisions, and persistence; `audit.py` owns flags and the replace queue; `app.py` serves the interface.
