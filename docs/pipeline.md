# Elephant ID Pipeline

## Purpose

This document describes the intended production pipeline for Elephant ID: how a folder of sighting photos becomes an analysis package, how human questions fit into the job, what the reviewer sees, when the final SEEK code is generated, and how the sighting moves into matching and filing.

The pipeline is sighting-centered. The system may analyze photos individually, but the product result is one reviewed identification record for one elephant sighting.

## Pipeline Summary

The end-to-end workflow is:

1. Import one folder as one sighting.
2. Run automated per-photo analysis.
3. Ask targeted human questions only when useful evidence is ambiguous at a workflow decision point.
4. Continue all unblocked work while questions wait for answers.
5. Aggregate photo evidence into one sighting-level analysis package.
6. Build a review package with representative images, photo drill-down, field suggestions, and a preview SEEK code.
7. Let the reviewer correct fields, images, crops, and photo-level evidence.
8. Generate the final SEEK code from reviewed structured fields.
9. Move the reviewed sighting into matching.
10. File the final identity decision.

## 1. Intake

A signed-in user selects a Dropbox folder for import. The platform copies the folder into system-controlled storage, creates a sighting record, and starts a top-level sighting analysis job.

The imported folder becomes the system of record for the sighting. Dropbox is an intake source, not the long-term storage layer.

Core requirements:

- One folder represents one sighting of one elephant.
- Every image in the folder is treated as evidence about the same individual.
- The system should combine information across the folder because no single photo is expected to show every useful feature.

Example: one image may show a clear right ear, another may show tusks, another may show body shape, and another may be unusable. The pipeline should combine the useful parts rather than looking for one perfect image.

## 2. Automated Photo Analysis

The system analyzes every photo in the sighting. Photos should be processed independently where possible so that slow, ambiguous, or blocked photos do not prevent useful work on other photos.

Automated photo analysis has two layers:

1. shared per-photo preparation,
2. parallel per-photo field analyzers.

### 2.1 Shared per-photo preparation

For each photo, the system first runs shared preparation:

- run body detection or segmentation,
- run feature detection or segmentation,
- select the relevant elephant body,
- filter features to the selected body,
- score image quality,
- estimate view and visibility,
- record every decision and intermediate artifact.

Shared preparation may emit a question when the image contains useful but ambiguous evidence.

Example: if multiple elephant bodies are found and the system cannot safely choose the sighting elephant, it should ask which highlighted elephant is the sighting elephant, with an option to skip the image if it is not clear enough.

Counterexample: if two bodies are detected but one is much smaller and the larger-body heuristic is reliable, the system should select the larger body, log the decision for telemetry, and continue without asking the user.

### 2.2 Parallel per-photo field analyzers

After shared preparation, photo analysis splits into parallel field analyzers:

- age,
- gender,
- tusks,
- ears.

These analyzers should have a parallel structure even if their complexity differs. Age and gender may be short model-output analyzers. Tusks and ears may combine model output, geometry, and heuristics.

Each analyzer should output structured field evidence. This evidence should be usable for sighting aggregation, telemetry, and review. Only the subset that helps the reviewer understand or correct the sighting should be displayed to the user.

Field analyzers should generally not ask questions themselves. They should analyze what they can, record uncertainty, and emit evidence. For example, the tusk analyzer should infer whether a single tusk is left or right rather than asking the user, because that local decision is usually not pivotal enough to interrupt analysis and can be corrected later in review.

Ear analyzers should fully analyze every usable ear candidate, including masks, contours, curvature, tears, and holes, even if that ear may never become the final selected evidence for the sighting. The question of which ear photo should be trusted belongs later, after the system has compared all ear candidates at the sighting level.

### 2.3 Analysis questions

Human input during analysis should be targeted. A question should be asked when the system finds potentially useful evidence but cannot safely use it without a human answer.

For v1, questions should mainly come from shared photo preparation or sighting-level aggregation decisions. Individual field analyzers should usually emit evidence and uncertainty rather than asking the user directly.

Good question examples:

- "Which highlighted elephant is the sighting elephant?"
- "Should this image be skipped because the relevant elephant is unclear?"
- "Which candidate right-ear image should be used as the sighting's best right ear evidence?" This should be asked after all candidate ear images have already been analyzed.

Poor question examples:

- asking whether to keep one ear or two when a proven area-ratio heuristic is already accurate enough,
- asking about a feature that is not useful,
- asking whether a local tusk detection is a left tusk or right tusk when the tusk analyzer can infer that and the reviewer can correct the sighting-level result later,
- asking the user to confirm every single low-confidence model output.

Each question should have a stable identity so the system does not ask the same question twice for the same feature. For v1, it is acceptable to ask all high-uncertainty questions rather than clustering or deduplicating similar questions across photos, but the questions should still be reserved for useful workflow choices rather than every uncertain field prediction.

### 2.4 Waiting and resuming

The job infrastructure should not need a special "pause" concept. Work items are either runnable, complete, failed, or blocked on a human answer.

If a question is unanswered, only dependent work is blocked. Other photos and field analyzers continue whenever possible. When answers arrive, the dependent work resumes.

Example: if one image contains multiple elephant bodies and needs a human body selection, that image's analyses may be blocked. Other images can still run. Sighting-level aggregation should wait until unresolved questions that may affect the final review package are answered.

Example: if there are several plausible right-ear photos, every candidate ear should still be fully analyzed first. The sighting-level aggregation or review-package step can then ask which analyzed candidate should be used as the best representative right ear.

### 2.5 Internal evidence from photo analysis

Automated analysis should preserve detailed internal evidence for telemetry, auditability, debugging, and future model improvement. This can be technical and does not all belong in the review UI.

At minimum, the backend should preserve:

- raw model outputs,
- selected and rejected body candidates,
- feature detections and overlap scores,
- body/feature filtering decisions,
- image quality scores,
- view and visibility evidence,
- age and gender raw model outputs,
- tusk observations and inferred sides,
- ear masks, crops, anchors, contours, tears, and holes,
- thresholds and heuristic decisions,
- questions asked and answered.

Example: selecting the larger body when a second detected elephant is much smaller should be saved internally as a body-selection decision with candidate areas and threshold values. It should not be shown to the user unless the decision becomes ambiguous enough to require a question.

## 3. Sighting Aggregation

After per-photo analysis, the system combines photo evidence into one sighting-level analysis package.

The aggregation layer should:

- combine age evidence across suitable images,
- combine gender evidence across suitable images,
- infer tusk presence and side from tusk-evaluable photos,
- choose the best right ear evidence,
- choose the best left ear evidence,
- choose representative views for review,
- produce suggested structured fields,
- produce a clearly labeled preview SEEK code.

The aggregation layer should not erase conflicting evidence. If photos disagree, the review package should preserve that disagreement when it is useful for the reviewer to inspect.

Sighting-level aggregation may ask human questions when the question is about choosing between already-analyzed evidence. For example, if multiple high-quality right-ear candidates have different strengths, the system can ask the user which candidate should become the best right ear image. This is different from interrupting each ear analyzer; each candidate should already have contour, curvature, tear, and hole evidence available before the question is asked.

## 4. Review Package

The review package is the user-facing output of analysis. It should be compact, visual, and directly editable. It should not expose every internal intermediate unless the reviewer needs it to understand or correct the result.

The review package contains:

- representative images,
- photo-level drill-down for every image in the sighting,
- suggested structured fields,
- a preview SEEK code,
- relevant warnings, questions, and answers,
- enough visual evidence for the reviewer to verify the suggestions.

### 4.1 Representative images

Representative images are first-class review artifacts. They are not simply the prettiest photos. They should support the system's current sighting-level interpretation.

The review package should include:

- best cropped right ear image,
- best cropped left ear image,
- representative frontal view when available,
- representative left-side view,
- representative right-side view,
- photo-level drill-down for every image in the sighting.

If no frontal view is available, side views should carry more of the tusk review burden. The left view should show the left tusk when the current interpretation requires it. The right view may be useful even when it does not show a right tusk, because absence or occlusion can itself explain the evidence.

Representative image selection should use image quality, view suitability, feature visibility, and agreement with the current sighting-level prediction.

Example: if the system thinks the elephant has one left tusk, the representative tusk image should visibly support a one-left-tusk interpretation.

### 4.2 What each photo should show

For each image in the sighting, the review UI should show:

- predicted image view,
- predicted age buckets,
- predicted gender,
- tusk prediction (if only one tusk is found, its side should be included, but if two are found, side is unnecessary),
- ear contour.

The user should be able to inspect each photo individually and correct photo-level outputs when needed. These corrections should feed into the reviewed structured record or become explicit overrides, depending on the field.

### 4.3 What representative images should show

Representative images should show the same user-facing information as ordinary photos:

- predicted image view,
- predicted age buckets,
- predicted gender,
- tusk prediction, including side when exactly one tusk is predicted,
- ear contour.

Best-ear representative images should additionally show:

- the predicted ear contour in one color,
- predicted tears highlighted in another color along the contour,
- the inside area of each tear highlighted as part of the tear visualization,
- predicted holes highlighted in another color along the inside of the ear.

The visual goal is to let the reviewer quickly verify whether the system's ear coding evidence is plausible without exposing every technical intermediate.

### 4.4 Preview SEEK code

The preview SEEK code is a convenience for review. It should be visibly labeled as provisional and should be editable through the underlying structured fields.

The preview code should not be treated as the canonical code for the sighting. The final SEEK code is generated only after review.

## 5. Human Review

Review is the step where the analysis package becomes authoritative.

The primary review screen should be a sighting summary. It should lead with the representative images and the suggested structured fields, while still allowing the reviewer to inspect every photo individually.

The reviewer should be able to correct:

- age,
- gender,
- right and left tusk presence,
- tusk side interpretations,
- right and left ear crops,
- ear tears and holes,
- extreme feature flags,
- special feature flags,
- representative image choices,
- individual photo-level evidence,
- preview SEEK code fields.

Corrections on representative images should act as field-level overrides when appropriate.

Example: if the representative tusk image is labeled as showing a left tusk but the reviewer corrects it to a right tusk, that correction should overwrite the system's tusk-side conclusion for the sighting unless the reviewer explicitly chooses a narrower edit.

Age and gender should be separately editable structured fields. They should not be attached to one representative image, even if the underlying model evidence came from multiple images.

## 6. Final SEEK Code

The final SEEK code should be generated only after the reviewer finalizes the reviewed structured record.

The final code should be stored with enough provenance to distinguish:

- raw model outputs,
- system suggestions,
- human answers during analysis,
- reviewer corrections,
- reviewed structured fields,
- preview SEEK code,
- final SEEK code.

Raw model output should never directly become the final SEEK code without the reviewer having the opportunity to correct it.

## 7. Matching and Filing

Matching begins after review and final SEEK generation. Coding and matching are separate workflows.

The matching workflow should compare the reviewed sighting against the elephant database and present ranked candidates. The reviewer then chooses an existing elephant, creates a new elephant identity, or leaves the match unresolved.

Only after this final human decision should the reviewed sighting be filed into the long-term database.

## 8. Auditability and V1 Boundaries

For any final field, it should be possible to answer:

- which photos contributed evidence,
- which model outputs were used,
- which candidates were rejected and why,
- which thresholds or heuristics were applied,
- which questions were asked,
- how the user answered,
- what the reviewer corrected,
- why the final value differs from the preview value.

The first version should remain intentionally constrained:

- one folder represents one sighting of one elephant,
- questions may be asked independently for each high-uncertainty case,
- final filing always requires human review,
- matching remains separate from coding,
- user-facing workflow should stay simple even if backend orchestration is complex.

Future versions may cluster similar questions, support collaborative review, handle multi-elephant sightings, use stronger visual matching models, and learn from reviewer corrections.