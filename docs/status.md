# Current Status

This file describes where the project is today. The workflow and architecture docs describe the intended product direction.

## Product Direction

Elephant ID is moving toward a folder-to-identity-decision product:

1. Import one grouped sighting folder into an app-controlled App Library.
2. Generate an analysis package from source photos.
3. Let the reviewer select and correct one left-ear and one right-ear evidence set.
4. Generate tear profiles from the approved evidence.
5. Rank known-elephant candidates.
6. Let the reviewer compare aligned matches and log an identity decision.

SEEK coding is no longer the product target. New product work should not treat a SEEK code as the main output, review artifact, matching key, or storage model.

## What Exists

The repository currently contains several eras of implementation:

- Domain models for labeled historical dataset photos and sightings.
- Legacy SEEK parsing and coding code used by older tests, metadata, and visualization paths.
- AI service wrappers for model outputs and cached detections.
- Ear segmentation, anchored-ear preparation, tear-profile extraction, and tear-profile matching.
- A local Flask visualization app for reviewing local dataset material.
- Early API/store/matching code for sighting folders, tear profiles, candidate ranking, and review decisions.

The current tear-profile matcher is the primary working identity signal. Future versions may add learned embedding models and other visual signals.

## Legacy SEEK Paths

`SeekCode`, `SeekCoder`, SEEK metadata, and older docs/tests may still exist while the codebase transitions. Treat those paths as compatibility or cleanup debt, not as product direction.

Do not add new product concepts that require:

- identity encoded in filenames,
- final SEEK code generation,
- manual SEEK review,
- matching by SEEK string,
- storage centered on fixed-width SEEK fields.

## Immediate Documentation Shape

The core documentation is intentionally small:

- `docs/status.md` for current reality.
- `docs/workflow.md` for the target product workflow.
- `docs/architecture.md` for broad constraints and boundaries.
- `docs/context.md` for glossary language.

Technical details and historical notes live under `docs/reference/`.
