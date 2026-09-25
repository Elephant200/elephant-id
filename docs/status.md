# Status

The active three-scale AlphaPhant method uses scales `0.35, 2, 10`, one shared
alignment, square-root value compression, and the full search. It keeps contour
opening and derives inward directions from reference edges. The code uses one
extractor and one profile cache record per ear.
The ear contour is the landmark-to-landmark path that encloses more area with the landmark chord.

## Current work

- Completed expansion: 444 tuning pairs across 108 elephants; 448 benchmark pairs across 105 elephants.
- Sol directed two Luna image reviewers. Image evidence overrode incorrect model fields.
- Both original 300-pair manifests remain unchanged. Expanded sets have no shared elephant, sighting, or photo.
- The three-scale benchmark is complete. The active method scores 366/448
  first and 403/448 top five on the expanded benchmark.
- Its accuracy limit is not established on independent elephants.

[Research](reference/research.md) contains the earlier experiment history and preserved evidence.
[Pipeline](pipeline.md) compares the current method with `main`.

## Future roadmap

The work below concerns model training and the future application.
It is separate from the current append-only research-set expansion.
Older counts describe their stated audit, not new cohort membership.

### Image selection and quality calibration

The future [image picker](../apps/image_picker/README.md) combines automated suggestions with human approval.
Keep rejected candidates recoverable. Inspect a sample of rejected and low-grade images.
Inventory image quality and annotation needs before a new collection or model study.
Retrieval needs complete sighting ear pairs. Model datasets can contain unpaired images.

The earlier target was 100–200 benchmark elephants with three to five accepted sightings each.
The current expansion has the separate target stated above.
Prefer identities without known training exposure when suitable repeated sightings exist.
Keep useful images for both retrieval and model development.

Human calibration should use 100–200 user-graded ear examples. Roboflow VLM is one possible automated grader.
Reuse per-ear quality grades across selection tasks.
The image grade describes its best ear; a poor opposite ear does not lower that grade.
The sighting-pair grade describes its weaker selected ear.
The third-best complete pair can summarize an elephant's retrieval eligibility.
Retain annotation counts and good-image counts per elephant.

The grading process uses the existing cache. It does not need a new grade-editing framework.
A harmless interior obstruction is acceptable when the required outer edge remains readable.
The selection page shows images without numerical quality scores.
Save incomplete selections, support undo, and export only complete validated pairs.
The private selection specification remains in `.scratch/image-curation/spec.md`.

### Identity assignments and annotation

Sightings and elephant assignments are indivisible across development groups.
Model train, validation, and test splits require separate elephant identities.
Crops, labels, and augmentations retain the source assignments.
An annotated benchmark image cannot train a model, even if its sighting differs from the query sighting.

The earlier approved removal of Lotter reflected only two available source sightings.
Keep the historical manifest for reproduction.
The current expansion preserves every row of both 300-pair manifests.
Any future reassignment requires a new explicit dataset decision.

Future segmentation analysis can use suitable human annotations from benchmark elephants.
Verify source mapping and complete the opposite ear from the same sighting.
Do not change a frozen retrieval manifest while choosing model annotations.

Planned annotation work:

- Localization: use SAM3 suggestions, then human-review held-out labels.
- Segmentation: use the user's external service for human masks; prefer new training-image annotations where possible.
- Landmarks: use model suggestions, then human-correct labels for validation and test.

Validate full-image and crop coordinates and retain annotation provenance.
An earlier export at `/Users/alex/Downloads/30997` contains 500 records across 461 sightings.
The then-current identity audit assigned 132 records to benchmark elephants, 295 to tuning elephants, and 73 to neither.
Exact sighting overlap was 40 benchmark sightings and 90 tuning sightings, with 146 records in total.
Forty-two annotated filenames also occurred in older landmark dataset split directories.
Actual training exposure of the active weights remains unknown.
Crop mapping and duplicate checks remain open tasks.
These counts describe that earlier audit; they are not counts for the expanded manifests.

### Model training and evaluation

Planned model work includes YOLO26 localization, U-Net/BiRefNet segmentation comparisons, and heatmap landmarks.
Choose models on model-validation data and evaluate fixed model-test sets.
Use versioned model producers behind the existing inference interfaces.
Tune AlphaPhant on tuning data, then fix its complete settings before a benchmark comparison.

Future comparisons can test current and improved preparation with the same retrieval sets.
AlphaPhant, CurvRank, and a separately justified MiewID study must use identical queries and candidate evidence.
MiewID is not part of the active production comparison command.
Define the human-mask analysis and limited ablations before results are available.
An association between segmentation quality and retrieval does not establish causation.
Automated retrieval uses predicted masks and landmarks.

Report counts, top-1/top-5, paired uncertainty, ties, and failures under [evaluation](evaluation.md).
Preserve every candidate score. Do not silently remove failed queries or use benchmark outcomes to select settings.
A paper must describe image selection, identity splits, annotation provenance, model training, and the limits of curated high-quality images.
Use fixed experiment records for figures, tables, and reproduction instructions.
