# Current Status

AlphaPhant is being reorganized around one research pipeline: automated preprocessing of a sighting ear pair, alpha-shape-derived tear-profile extraction, tear-profile matching, and known-elephant candidate ranking.

Read [pipeline.md](pipeline.md) for target behavior and [evaluation.md](evaluation.md) for the benchmark protocol.

## Working Numerical Core

The repository already contains:

- SAM3 model access and cached segmentation results;
- a YOLO keypoint model for ear landmark detection;
- ear-mask contour preparation and side inference;
- alpha-shape-derived tear-profile extraction;
- directional tear-profile alignment and similarity scoring;
- experimental catalog-ranking logic.

The numerical tear-profile and pairwise-matching behavior should survive restructuring. Characterization tests establish that boundary before code moves.

## Structural Debt

The active implementation is obscured by older directions:

- SEEK parsing, coding, and fixed-field domain objects;
- age and gender inference and analysis;
- body, trunk, tail, and tusk orchestration;
- a general `PhotoAnalyzer` joining unrelated features;
- identity-bearing dataset objects used directly by model wrappers;
- normalization and calibration layers outside the selected matcher;
- an evaluator that manufactures left/right pairs;
- API and application prototypes.

New research code should not depend on these paths.

## Preservation Boundaries

- Historical dataset files remain untouched, including SEEK columns and source lists.
- Existing SAM3 body and multi-feature results are migrated and retained under immutable producer names.
- Existing anchor-model results are migrated.
- Heuristic caches and scripts remain untouched, even though the scripts may temporarily break when `legacy` imports disappear.
- Every file under `apps/` remains byte-for-byte untouched and unsupported.
- The API prototype is preserved on the `desktop-prototype` branch and removed from the active branch.
- Unrelated user work and private `dataset` contents stay outside cleanup.

## Intended Top-Level Shape

Responsibilities will move toward:

```
analysis/    sighting analysis and tear-profile extraction
inference/   swappable model implementations
matching/    similarity scoring and catalog ranking
eval/        implementation-independent identity-retrieval evaluation
image/       BGR image and geometry utilities
```

Exact files, class names, and data representations remain provisional. Prefer the smallest interface justified by current callers.

## Documentation Map

- Current pipeline or matching change: read [pipeline.md](pipeline.md).
- Retrieval benchmark, split, failure, or metric change: read [evaluation.md](evaluation.md).
- Module placement, inference seam, or cache change: read [architecture.md](architecture.md).
- Domain or technical naming: read [context.md](context.md).
- Future application work: read [workflow.md](workflow.md).
- Research beyond the locked pipeline: read [future.md](future.md).
- Surprising durable decisions: read [adr/](adr/).
