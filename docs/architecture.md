# Architecture

This document describes broad product architecture. It should stay flexible and avoid over-specifying storage schemas, UI component details, or orchestration frameworks before they are needed.

## Core Constraints

Elephant ID should optimize for:

- normal import, analysis, review, matching, and decision logging without internet after setup,
- a reviewer experience that is obvious and responsive,
- app-controlled import into an App Library,
- no mutation of source camera folders or arbitrary user data,
- generated product identifiers instead of identity-bearing filenames,
- human ownership of the final identity decision,
- room for future matching signals beyond the current tear-profile matcher.

The reviewer UI should hide backend complexity. The reviewer should see source photos, evidence, corrections, candidates, and decisions, not orchestration state.

## Main Components

**App Library**:
An app-controlled workspace folder that may live on an external drive. It contains imported images, derived assets, and SQLite metadata.

**Metadata store**:
SQLite-backed metadata for sightings, photos, analysis runs, approved evidence, profiles, candidates, decisions, and telemetry. It should store metadata and references, not large image buffers when files are more appropriate.

**Analysis services**:
Services that produce segmentations, ear crops, anchored ear geometry, tear profiles, and quality/ranking signals for evidence review.

**Evidence review workflow**:
The workflow that lets the reviewer choose one left and one right ear, correct crop or polygon/segmentation evidence, and approve the profiles used for matching.

**Matching workflow**:
The workflow that compares approved sighting profiles against the known-elephant catalog and returns aligned candidate evidence.

**Decision workflow**:
The workflow that logs whether the sighting belongs to an existing known elephant, creates a new known elephant, or remains unresolved.

## Data Flow

```text
grouped sighting folder
  -> import into App Library
  -> source photos + generated metadata IDs
  -> automated segmentation and ear candidate extraction
  -> reviewer selects/corrects left and right evidence
  -> approved tear profiles
  -> ranked known-elephant candidates
  -> aligned candidate comparison
  -> logged identity decision
```

Matching depends on approved evidence, not raw automated output. If evidence is corrected, profiles are regenerated before matching.

## Current Matching Signal

The current working signal is same-side tear-profile matching: compare approved left-ear profiles to left-ear catalog profiles, right-ear profiles to right-ear catalog profiles, then combine the side evidence for ranking.

Future versions may add learned embeddings or other visual descriptors. The architecture should allow matching to become multi-signal without changing the basic review flow: evidence first, candidates second, identity decision last.

## Open Design Areas

These details should remain open until implementation pressure clarifies them:

- exact SQLite schema,
- App Library folder layout,
- local model packaging and update mechanism,
- detailed telemetry schema,
- background job runner,
- backup and sync strategy,
- remote collaboration model,
- UI framework and desktop shell.

These choices should not weaken the core constraints: offline normal use, fast review, app-controlled import, reviewed evidence before matching, and human identity decisions.
