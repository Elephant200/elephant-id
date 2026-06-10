# Elephant ID Technical Architecture

## Purpose

This document describes the high-level technical shape of Elephant ID. It should guide implementation without locking the project into a specific vendor, framework, desktop shell, hosting platform, or database before those decisions are ready.

The current product direction is desktop-first and local-capable. Local execution should work for teams without usable internet, but the architecture should not assume the backend must always be local.

## Architecture Goals

The system should:

- ingest one already grouped photo folder as one elephant sighting,
- run AI and geometry analysis across the folder,
- produce a reviewable SEEK record for v1,
- preserve the option to add richer evidence later, such as contours, curvature signatures, plots, embeddings, crops, and model outputs,
- support human review before matching or filing,
- rank candidate identities without depending on an existing SEEK matcher,
- remain usable with limited or no internet during normal office workflows,
- leave room for connected teams to use remote storage, inference, sync, or matching services where practical.

## Deployment Posture

### Desktop-first posture

The first serious product should feel like a desktop app for office review of local photo folders. The exact shell is an open decision, but the user experience should not depend on a browser-only cloud workflow.

### Backend flexibility

The first version can assume local folders and local-capable analysis, but component boundaries should not make local-only operation a permanent lock. Some teams may later use remote storage, inference, sync, matching, backup, or collaboration services.

## Core Components

The implementation should keep these responsibilities separate even if they initially run in one local process. The main reason is clarity and testability; future remote or hybrid backing is a useful secondary benefit.

### User interface

The UI lets a reviewer import folders, inspect analysis results, answer targeted questions, correct fields and evidence, compare candidate matches, and file the final decision.

The UI should hide backend complexity. It should present the reviewer with concrete evidence, not raw orchestration state.

### Sighting store

The sighting store tracks imported folders, photos, derived assets, analysis status, questions, review decisions, matching decisions, and provenance.

It should preserve the difference between raw model output, system suggestions, human answers, reviewer corrections, final SEEK codes, and future extended descriptors.

### Analysis services

Analysis services run per-photo and sighting-level work. They may include detection, segmentation, view classification, age and sex estimation, tusk analysis, ear analysis, contour extraction, curvature computation, embedding generation, and feature aggregation.

Services should emit structured evidence and provenance, not only final labels.

Analysis services may eventually be backed by local model runners or remote inference, depending on connectivity and hardware.

### Review workflow

The review workflow turns a draft analysis package into an authoritative reviewed SEEK record. Review is where automated evidence becomes accepted, corrected, or rejected.

The final record should be generated from reviewed fields and artifacts, not directly from raw model output.

### Matching workflow

Matching is separate from coding. It compares a reviewed sighting against known elephants and presents ranked candidates for human decision.

V1 matching should start with classic SEEK fields and reviewed structured evidence. The design should not prevent later use of added structured features, Curvrank descriptors, vector embeddings, visual evidence, and reviewer-approved metadata.

## Data Model Principles

The v1 identification object should center on classic SEEK, but it should not be modeled as only an opaque character code.

For v1, it should contain:

- classic SEEK-compatible fields,
- reviewed special markings and body features already represented by SEEK,
- references to representative photos and crops,
- confidence, source, and reviewer provenance for each field.

Future versions may add:

- added structured fields,
- ear contours and deterministic contour coordinates,
- Curvrank curvature signatures or plots,
- learned visual embeddings,
- additional confidence, source, and reviewer provenance for each new feature.

A classic SEEK code should be generated from reviewed fields for compatibility, communication, and review. The reviewed fields should remain available separately so future descriptors can be added without redesigning storage.

## Processing Shape

The pipeline should remain folder-centered:

1. Import or index a local one-elephant folder.
2. Analyze photos independently where possible.
3. Preserve per-photo evidence and derived artifacts.
4. Aggregate evidence into a sighting-level draft.
5. Ask targeted questions only when a useful workflow decision cannot be made safely.
6. Build a review package.
7. Let the reviewer correct fields, representative images, and evidence.
8. Store the final reviewed SEEK record.
9. Run matching as a separate reviewed workflow.
10. File the identity decision.

The orchestration mechanism can change. The important rule is that blocked or ambiguous work should not stop unrelated analysis from continuing.

## Connectivity Constraints

Architecture decisions should assume:

- internet may be too slow for bulk image upload,
- local storage may hold the main image corpus,
- local compute may be the normal inference path,
- connected teams may prefer hosted inference, shared storage, collaboration, or backup later,
- model size and hardware requirements matter,
- review assets should be compact,
- implementation choices should not unnecessarily rule out remote services.

## Open Decisions

The following should remain open until product and implementation constraints are clearer:

- desktop shell,
- local service packaging,
- database choice,
- cloud provider or hosting stack,
- sync strategy,
- model packaging and update mechanism,
- whether and how to store embeddings or descriptor search indexes,
- exact job orchestration framework,
- authentication and permissions model for multi-user deployments.

## Main Risks

- The product may become too complex for reviewers if all internal evidence is exposed.
- Local model requirements may exceed available hardware.
- Ear geometry may become inconsistent unless contour orientation and anchors are deterministic.
- Future matching quality may suffer if the system treats classic SEEK as the only possible feature representation forever.
- Cloud-first assumptions may fail under real field bandwidth.
