# Current Status

AlphaPhant analyzes a sighting ear pair and returns one similarity score per catalog candidate. Research supplies one selected Photo for each ear.

## Current Pipeline

Shared preparation retrieves photos, runs ear segmentation and landmark detection, and resolves both ears. AlphaPhant extracts one signed tear profile per ear, compares corresponding sides, selects each candidate's strongest reference per side, and averages the two scores.

SAM3 supplies segmentation. YOLO supplies landmarks. AlphaTear supplies profiles. Standard composition caches inference outputs and extracted profiles. Experimental extraction retains the inference caches.

## Code and Documentation

The core packages are `domain`, `dataset`, `preparation`, `inference`, `matching`, `evaluation`, and `image`.

- [Architecture](architecture.md): ownership, dependencies, and caches.
- [Pipeline](pipeline.md): extraction and scoring.
- [Evaluation](evaluation.md): identity-retrieval evaluation.
- [Context](context.md): terms and definitions.
