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

The main data object through matching is a row-indexed profile table. Each row has:

| Field | Meaning |
| --- | --- |
| `profile` | The 720-bin tear-depth vector. |
| `photo_id` | The dataset photo that produced the profile. |
| `identity` | The known elephant label, used only for evaluation and calibration labels. |
| `side` | `left` or `right`; opposite sides are never compared. |

The high-quality evaluation builds that table from `outputs/high_quality/manifest.csv`, then caches usable rows in `outputs/tear_matching_eval/hq_profiles.npz`. The original `sighting_date` is preserved in the cache for traceability, but the evaluation unit is a high-quality, high-resolution left/right image pair rather than the historical date grouping. The profile cache is strict about the manifest fingerprint, evaluation cache contract, tear-profile width, and skipped-row count: if any of those checks fail, the evaluator refuses to run until the cache is rebuilt.

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

The result is a matcher score in `[0, 1]`. The alignment model is intentionally rigid. More flexible local warping, such as DTW, performed worse because willingness to warp is itself useful evidence against a true match.

Score names in the code:

| Score | Range | Where it comes from | Used for |
| --- | --- | --- | --- |
| `overlap_score` | `[0, 1]` | Area overlap divided by area union after the best shift/stretch. | Diagnostics and alignment display. |
| matcher `score` | `[0, 1]` | `overlap_score * shift_penalty`. | Input to the pairwise score matrix. |
| cohort-normalized score | Unbounded | Symmetric row/column z-score of matcher scores within the same-side cohort. | Default calibration input and same-side evidence normalization. |
| calibrated evidence score | Unbounded logistic decision value | Logistic model using the selected score and tear-mass features. | Image-pair and elephant-level ranking. |

## Cohort Normalization

Raw matcher scores are biased by profile distinctiveness. A smooth ear with one generic bump can match many elephants moderately well, while a feature-rich ear may match everyone poorly.

`symmetrized_cohort_z()` normalizes each score by row and column statistics in the same-side pairwise score matrix. The effect is an empirical distinctiveness weight: the same raw score is stronger evidence between selective ears than between promiscuous ears.

Normalization is not a supervised fit. It uses no identity labels. It is transductive over the evaluation cohort: row and column statistics are computed from the same unlabeled set of high-quality profiles being evaluated. The raw same-side matcher matrix is cached in `outputs/tear_matching_eval/hq_pairwise_scores.npz` and strictly tied to both the profile cache and matcher configuration:

1. Score all left-left and right-right profile pairs; different-side entries stay `NaN`.
2. Average `score[i, j]` and `score[j, i]` because matcher alignment is directional.
3. For each finite pair, add the query-row z-score and candidate-column z-score.

The sum rather than average sets the score scale. The downstream calibrator absorbs that scale, so the ranking logic only depends on relative evidence.

## Score Calibration

`TearScoreCalibrator` maps selected pair scores into a calibrated evidence score using labeled same/different profile pairs and tear-mass features. In the default evaluation, the calibrator is fitted on cohort-normalized scores. In the `--no-normalization` ablation, the same calibrator model is fitted on symmetrized raw matcher scores so the ablation changes only the normalization step.

Tear mass is the total positive tear area in depth-degrees. It is a proxy for how much distinctive tear signal exists in the pair. The same match score is usually stronger evidence when both ears have more tear structure than when both ears are nearly smooth.

Calibration makes left-ear and right-ear evidence more comparable. Raw overlap scores from the two sides should not be averaged directly. The returned calibrated evidence score is a logistic decision value, not a probability.

For elephant-level evaluation, identities are split into calibration folds. Each query identity is scored with a calibrator trained on other identities. Training pairs are same-side pairs from the training identities:

- positives: same identity and same side,
- negatives: different identity and same side, subsampled at up to five negatives per positive.

## Combining Sides and Ranking

The current product direction requires reviewed left and right ear evidence before matching.

Aggregation follows this shape:

- Same-side photo/profile pairs are scored only against the same side.
- A left query profile only compares to left catalog profiles.
- A right query profile only compares to right catalog profiles.
- Each candidate known elephant gets its strongest clean evidence per side.
- The side evidence is combined into a final candidate rank.

Left and right profiles are never matched to each other.

The high-quality evaluation first constructs left/right image pairs. For each known elephant, it shuffles that elephant's left-ear profile rows and right-ear profile rows with the evaluation seed, then pairs them up to the smaller side count. If an elephant has five left profiles and three right profiles, it contributes three left+right image pairs and the two surplus left profiles are not used as query units.

The image-pair evaluation uses this aggregation:

1. For a query image pair and one gallery image pair, score left-to-left and right-to-right.
2. For a query image pair and one gallery elephant, keep the best score per side across all of that elephant's gallery image pairs.
3. For `combined`, average that elephant's best left score and best right score.

Reported side modes:

| Mode | Query requirement | Gallery requirement | Candidate score |
| --- | --- | --- | --- |
| `left` | Query image pair has left profile. | Candidate has paired high-quality image pairs. | Best left score. |
| `right` | Query image pair has right profile. | Candidate has paired high-quality image pairs. | Best right score. |
| `combined` | Query image pair has both profiles. | Candidate has left and right high-quality evidence. | Mean of best left and best right scores. |

## Evaluation Protocol

The evaluation script reports two-ear retrieval from high-quality, high-resolution image pairs. Each constructed left/right image pair queries all other image pairs, grouped and ranked by known elephant. This uses the selected score stack from `scripts/evaluation.py`, then applies the side aggregation described above.

The default score stack is cohort-normalized and calibrated. Ablations are explicit single-run flags:

| Flag | Score matrix | Calibration |
| --- | --- | --- |
| none | Cohort-normalized same-side matcher scores. | Identity-disjoint logistic evidence score. |
| `--no-calibration` | Cohort-normalized same-side matcher scores. | Disabled; rank directly on normalized scores. |
| `--no-normalization` | Symmetrized raw matcher scores. | Identity-disjoint logistic evidence score. |
| `--no-normalization --no-calibration` | Symmetrized raw matcher scores. | Disabled; rank directly on raw scores. |

Metrics:

- `top-k`: fraction of queries where the correct known elephant ranks at or above `k`.
- `MRR`: mean reciprocal rank.
- `median_rank`: median rank of the correct known elephant.
- `n`: number of scored queries.

Ranks count only gallery elephants with strictly higher scores ahead of the true elephant. Ties with the true elephant share its rank.

The evaluation script prints the current profile count, constructed image-pair count, selected score stack, cache fingerprints, top-1/top-3/top-5/top-10/top-15, MRR, and median rank. With multiple seeds, `+/-` is variation from random within-identity left/right pairing, not independent sampling uncertainty.

## Empirical Findings

These are the measured results behind the current configuration, recorded from the July 2026 evaluation so the choices are not re-litigated blindly. Numbers come from the high-quality set (`outputs/high_quality`, 683 profiles, leave-one-out same-side) and the elephant-level protocol (~83 elephants) unless noted.

**Image quality dominates.** The same matcher scores top-1 0.23 on the older filtered set versus 0.432 on the high-quality set — image quality nearly doubles accuracy. Extraction repeatability, not matcher tuning, is the dominant limit.

**The winning stack and its increments.** Starting from a plain area-overlap IoU baseline (top-1 0.432):

- depth compression `depth ** 0.5` before overlap: top-1 0.445 (foreshortening tolerance; the optimum is broad over 0.5–0.6),
- 240 scoring bins (up from 120): suppresses impostor overlap of narrow scallops (31% of peaks are <3° wide; impostor mean drops ~1.7x more than genuine),
- centered stretch search 0.80–1.20 (17 steps) plus score symmetrization `(S + S.T) / 2`,
- cohort z-normalization (query+candidate impostor stats, AS-norm style).

Together these move photo-level top-1 from 0.447 to 0.501 and elephant-level top-1 from 0.493 to 0.533 (top-5 0.730, top-10 0.805). A holdout on the filtered set replicated the gain (0.313 to 0.362), so it is not overfit to the high-quality set.

**Calibrated left+right fusion, not raw summation.** On both-side sightings, a max-rule fusion of calibrated side evidence reaches top-1 0.573–0.584 (top-5 0.848) versus ~0.40 for a single side. Summing raw side scores does not help — calibration is what makes the two sides addable.

**What drives a correct retrieval.**

- Query tear mass: top-1 rises from ~12% in the lowest tear-mass quartile to ~40% in the highest. Smooth ears carry little identity signal.
- Time gap: top-1 drops from 0.445 to 0.266 when the nearest positive is three or more years away.

**Failure taxonomy.** Of the residual failures, roughly half are "peaks align but a distractor outscores the true match" — the alignment is right but the calibrated score ordering is wrong. The rest split across misalignment beyond the shift/stretch budget, one-sided signal (a tear present in one ear but absent in its mate), and both-profiles-low-signal. Same-day photo pairs of one ear can still disagree when one view is oblique, because extraction stays pose-sensitive even on high-quality images.

**Why the normalization works.** Cohort z-normalization behaves as an empirical distinctiveness weight: impostor-mean similarity correlates -0.68 with tear mass, and the chronic over-matchers are all "one generic bump at 110–130°" profiles. Symmetrization fixes an asymmetric-mass tail (median score asymmetry 0.005 but max 0.19).

**Aggregation is already saturated.** The max-over-photo-pairs rule (elephant-level top-1 0.493, top-5 0.684, top-10 0.792) beat every profile-fusion variant tried — cross-sighting median, within-sighting median, median+photos were all neutral or harmful, because 88% of high-quality (sighting, side) groups have a single photo and averaging blurs real tears. Veto-clip (capping a profile's mass by its sighting-mates' aligned max) did fix phantom tear masses and is useful for review-UI evidence and mass features, but not for ranking.

**Holes are not currently extractable.** Ear-hole features would add signal, but existing masks are external contours only (`RETR_EXTERNAL`) and carry no holes; LoG/MSER detection was tried without success. Holes need annotation plus a trained detector, not post-processing of the current masks.

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
