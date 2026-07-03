# Tear-Profile Matching

This document describes the current tear-profile matching pipeline: how an approved ear crop becomes a tear profile, how two profiles become a score, how scores become comparable across elephants and combinable across ears, and what the evaluation metrics mean.

The pipeline covers one signal: ear-margin tears. It ranks known-elephant candidates; it does not make identity decisions. Final identification is always a human decision over reviewed evidence and a ranked shortlist.

## Signal Chain

```text
approved ear evidence
  -> tear_profile()      720-bin angular depth profile per ear
  -> TearMatcher         penalized shift/stretch alignment -> raw pair score
  -> cohort normalization
  -> score calibration
  -> side-level evidence
  -> ranked known elephants
```



## Tear Profile Extraction

Each usable ear becomes a one-dimensional array of `TEAR_PROFILE_BINS` values covering 0-180 degrees along the outer ear margin. The profile is anchored by model-detected keypoints and side-aware ear geometry.

Important properties:

- Values are normalized by the equal-area semicircle radius `R = sqrt(2A / pi)`, so profiles are scale-free across photo resolutions.
- Positive values mean inward tears; healthy margin is near zero.
- The angular coordinate is anchored so the same physical tear lands at approximately the same position across photos.
- Residual misalignment comes from real ear curl, camera pose, and extraction error; the matcher absorbs limited shift/stretch differences.



## Pairwise Scoring

Given two profiles, the matcher:

1. Clips negative depths to zero and compresses positive depths with `depth ** 0.5`.
2. Resamples profiles to 240 bins for scoring.
3. Searches centered stretches from 0.80 to 1.20 and shifts up to 15% of the profile.
4. Penalizes implausibly large alignment changes.
5. Scores aligned profiles as area overlap over area union.

The result is a raw score in `[0, 1]`. The alignment model is intentionally rigid. More flexible local warping, such as DTW, performed worse because willingness to warp is itself useful evidence against a true match.

## Cohort Normalization

Raw scores are biased by profile distinctiveness. A smooth ear with one generic bump can match many elephants moderately well, while a feature-rich ear may match everyone poorly.

`symmetrized_cohort_z()` normalizes each score by row and column statistics in the same-side pairwise score matrix. The effect is an empirical distinctiveness weight: the same raw score is stronger evidence between selective ears than between promiscuous ears.

## Score Calibration

`TearScoreCalibrator` maps normalized scores into a calibrated evidence score using labeled same/different profile pairs and tear-mass features.

Calibration makes left-ear and right-ear evidence more comparable. Raw overlap scores from the two sides should not be averaged directly.

## Combining Sides and Ranking

The current product direction requires reviewed left and right ear evidence before matching.

Aggregation follows this shape:

- Same-side photo/profile pairs are scored only against the same side.
  - A left query profile only compares to left catalog profiles.
  - A right query profile only compares to right catalog profiles.
- Each candidate known elephant gets its strongest clean evidence per side.
- The side evidence is combined into a final candidate rank.

Left and right profiles are never matched to each other.

## Evaluation Protocols

The evaluation script reports two main protocols:

**Photo-level leave-one-out**:
Each profile queries every other same-side profile. This isolates matcher and normalization behavior from sighting-level aggregation.

**Elephant-level leave-one-sighting-out**:
Each sighting queries all other sightings, grouped and ranked by known elephant. This mirrors the product task: a new sighting arrives and the system ranks the known-elephant catalog.

Metrics:

- `top-k`: fraction of queries where the correct known elephant ranks at or above `k`.
- `MRR`: mean reciprocal rank.
- `median_rank`: median rank of the correct known elephant.
- `n`: number of scored queries.



## Known Negative Results

The following approaches have already been tried and rejected for this signal:

- hard-thresholding small profile ripples,
- pure peak/event matching,
- banded DTW alignment,
- a negative-depth channel,
- profile-level median aggregation,
- raw uncalibrated side combination,
- trimmed cohort statistics,
- 120+240-bin score ensembles.

The dominant remaining accuracy limit is extraction repeatability. Better segmentation and future learned embeddings are likely more valuable than repeatedly tuning the same matcher.