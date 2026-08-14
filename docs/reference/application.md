# Future Application

This document describes how a future application may surround AlphaPhant. **It is not the current implementation target; do not optimize for this yet.** The current research algorithm is defined in [../pipeline.md](../pipeline.md).

The future application is an evidence-review and identity-decision tool to aid elephant re-ID efforts for conservationists in the field. It uses the shared domain and storage boundaries in [../architecture.md](../architecture.md) without necessarily adopting the research dataset's physical layout.

## Shared Domain and Seam

Research and application differ only before the sighting ear pair exists:

```
research Dataset + PhotoStore
  -> preselected high-quality SightingEarPair
  -> AlphaPhant

future application + its PhotoStore
  -> all Photos from one Sighting
  -> ear selection
  -> SightingEarPair
  -> AlphaPhant
```

Given the ear pair, AlphaPhant is fully automated: sighting analysis, tear-profile matching, and candidate ranking. Application-specific import, duplicate handling, heuristics, reviewer correction, and photo-sufficiency decisions remain upstream.

## Core User Journey

1. Import one grouped one-elephant sighting into application-controlled storage.
2. Assign or recover permanent opaque photo and sighting IDs and construct canonical domain objects.
3. Run ear localization and segmentation across its Photos to produce ear candidates.
4. Review the analysis package and correct evidence as needed.
5. Approve a SightingEarPair: one usable left-ear reference Photo and one usable right-ear reference Photo. One Photo may serve both sides.
6. Run AlphaPhant on the approved ear pair using the application's PhotoStore.
7. Compare ranked matching candidates and inspect the independently selected left- and right-ear catalog evidence behind each similarity score.
8. Record an identity decision.

Candidate ranking and identity decision are distinct. AlphaPhant ranks known elephants; it does not decide whether the sighting belongs to an existing individual, should create a new known elephant, or remains unresolved.

## Analysis Package

The analysis package is the intermediate review artifact for one Sighting. It contains canonical Photo references, automated evidence, ear candidates, the selected SightingEarPair once approved, tear profiles, and correction state. Original bytes remain in the application's PhotoStore rather than inside Photo objects.

During ear selection, the reviewer asks: did the system extract usable evidence from this sighting?

The analysis package should support:

- viewing source photos through the PhotoStore;
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

- one usable left-ear reference Photo and segmentation, and
- one usable right-ear reference Photo and segmentation.

The same Photo may supply both declared sides. If either side cannot be approved, the Sighting is saved as unresolved. One-sided matching is future work; see [../future.md](../future.md).

After the reviewer changes a selected ear, crop, or segmentation, the application regenerates preview tear profiles from the corrected evidence. AlphaPhant runs only after the SightingEarPair is approved.

## Candidate Comparison

After AlphaPhant returns candidate ranking, the comparison view should show matching candidates with the query left and right ears beside each candidate's strongest supporting left-ear and right-ear catalog evidence. Aligned tear profiles should be visible so the reviewer can judge whether the similarity is meaningful.

The winning left and right catalog evidence for a candidate may come from different historical sightings; the view should preserve that per-side Photo and sighting provenance.

The reviewer asks: do these aligned signals support this identity?

Tear-profile matching is the current ranking signal. Additional identity signals are research directions in [../future.md](../future.md).

## Identity Decision

The reviewer owns the identity decision. The system ranks and explains candidates; it does not silently identify the elephant.

The decision states are:

- link the Sighting to an existing known elephant;
- create a new known elephant; or
- leave the Sighting unresolved.

An unresolved Sighting keeps its intermediate analysis without filing it into the known-elephant catalog.

## Initial Application Scope

The first application version is intentionally narrow:

- input is one already-grouped one-elephant Sighting;
- normal use works offline after setup;
- original source locations are not mutated;
- review must be obvious and responsive;
- preprocessing for ear selection may run in the background and may be slower than review;
- both ears are required before candidate ranking;
- the reviewer makes every identity decision.

The design should not prevent later batch import, raw camera-dump grouping, duplicate detection, one-sided matching, learned embedding models, remote collaboration, or field-time sufficiency feedback.

## Deferred Application Concerns

The research restructuring does not choose:

- physical application storage schema or App Library layout;
- whether or how imports detect byte-identical photos;
- desktop shell or UI framework;
- background-job mechanism;
- model packaging or update delivery;
- detailed review-interface layout;
- identity-decision log schema or operational telemetry;
- catalog update workflow;
- backup, synchronization, or collaboration;
- one-sided matching behavior.

These choices should be made from application requirements when that work resumes. They may implement the settled PhotoStore capability differently, but they must enter the shared pipeline through SightingEarPair rather than forking AlphaPhant.

## Desktop Surface

The reviewer-facing surface for this workflow is the offline desktop app in `apps/desktop`; how to build, run, and demo it lives in [`apps/desktop/README.md`](../../apps/desktop/README.md). It is driven as a live demo for field experts and funders, so interaction clarity matters more than density.

### Visual Theme

A professional savannah theme:

- dark green sidebar and header,
- beige main background,
- elephant-gray accents.

Keep the palette centralized as CSS variables in `apps/desktop/src/styles.css` so the theme stays consistent as pages are added.

### Layout and Density

Prefer more, simpler pages over dense single pages. The workflow is split into separate step pages — Ingest, Photos, Match, Review — rather than stacked panels on one screen. When adding a feature, add a new page or step rather than growing an existing page.

Images, graphics, and text should be scaled generously for a laptop screen; the first versions ran too small. Large visual changes are acceptable as long as they respect the API contract with the sidecar: route JSON field names are a contract with the renderer, the same way `apps/visualization` field names are.
