# Future Application Workflow

This document describes how a future application may surround AlphaPhant. It is not the current implementation target. The current research algorithm is defined in [pipeline.md](pipeline.md).

The future application is an evidence-review and identity-decision tool, not a SEEK coding tool.

## Shared Seam

Research and application differ only before the sighting ear pair exists:

```
research dataset
  -> preselected high-quality sighting ear pair
  -> AlphaPhant

future application
  -> all photos from one sighting
  -> ear selection
  -> sighting ear pair
  -> AlphaPhant
```

Given the ear pair, AlphaPhant is fully automated: sighting analysis, tear-profile matching, and candidate ranking. Application-specific heuristics, reviewer correction, and photo-sufficiency decisions remain upstream.

## Core User Journey

1. Import one grouped one-elephant sighting into the App Library.
2. Run ear localization and segmentation across its photos to produce ear candidates.
3. Review the analysis package and correct evidence as needed.
4. Approve a sighting ear pair: one usable left-ear reference photo and one usable right-ear reference photo.
5. Run AlphaPhant on the approved ear pair.
6. Compare ranked matching candidates and inspect the independently selected left- and right-ear catalog evidence behind each similarity score.
7. Record an identity decision.

Candidate ranking and identity decision are distinct. AlphaPhant ranks known elephants; it does not decide whether the sighting belongs to an existing individual, should create a new known elephant, or remains unresolved.

## Analysis Package

The analysis package is the intermediate review artifact for one sighting. It contains sighting photos, automated evidence, ear candidates, the selected sighting ear pair once approved, tear profiles, and correction state.

During ear selection, the reviewer asks: did the system extract usable evidence from this sighting?

The analysis package should support:

- viewing source photos;
- viewing segmentation overlays;
- viewing ranked left-ear and right-ear ear candidates;
- selecting one best ear per side;
- editing the crop and ear segmentation region;
- previewing the resulting tear profile beside the crop and overlay;
- recording whether evidence was manually corrected.

Preview tear profiles during review help the reviewer judge ear selection. After approval, AlphaPhant produces the final tear profiles used for candidate ranking.

## Evidence Review Gate

Candidate ranking must not run until ear selection is complete.

For the initial application scope, the reviewer must approve both sides:

- one usable left-ear reference photo and segmentation, and
- one usable right-ear reference photo and segmentation.

If either side cannot be approved, the sighting is saved as an unresolved sighting. One-sided matching is future work; see [future.md](future.md).

After the reviewer changes a selected ear, crop, or segmentation, the application regenerates preview tear profiles from the corrected evidence. AlphaPhant runs only after the sighting ear pair is approved.

## Candidate Comparison

After AlphaPhant returns candidate ranking, the comparison view should show matching candidates with the query left and right ears beside each candidate's strongest supporting left-ear and right-ear catalog evidence. Aligned tear profiles should be visible so the reviewer can judge whether the similarity is meaningful.

The winning left and right catalog evidence for a candidate may come from different historical sightings; the view should preserve that per-side provenance.

The reviewer asks: do these aligned signals support this identity?

Tear-profile matching is the current ranking signal. Additional identity signals are research directions in [future.md](future.md).

## Identity Decision

The reviewer owns the identity decision. The system ranks and explains candidates; it does not silently identify the elephant.

The decision states are:

- link the sighting to an existing known elephant,
- create a new known elephant, or
- leave the sighting unresolved.

An unresolved sighting keeps its intermediate analysis without filing it into the known-elephant catalog.

## Initial Application Scope

The first application version is intentionally narrow:

- input is one already-grouped one-elephant sighting;
- normal use works offline after setup;
- original source locations are not mutated;
- review must be obvious and responsive;
- preprocessing for ear selection may run in the background and may be slower than review;
- both ears are required before candidate ranking;
- the reviewer makes every identity decision.

The design should not prevent later batch import, raw camera-dump grouping, one-sided matching, learned embedding models, remote collaboration, or field-time sufficiency feedback.

## Deferred Application Concerns

The research restructuring does not choose:

- storage schema or App Library layout;
- desktop shell or UI framework;
- background-job mechanism;
- model packaging or update delivery;
- detailed review-interface layout;
- identity-decision log schema or operational telemetry;
- catalog update workflow;
- backup, synchronization, or collaboration;
- one-sided matching behavior.

These choices should be made from application requirements when that work resumes. They must enter the shared pipeline through ear selection rather than forking AlphaPhant.
