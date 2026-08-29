# Future Research

This document records directions beyond the pipeline being locked down now. None are current implementation requirements.

## Better Preprocessing

The first priority is extraction repeatability. Candidate replacements for current inference include:

- detector plus U-Net ear segmentation;
- detector plus BiRefNet ear segmentation;
- alternative ear-landmark networks;
- improved pose, visibility, or contour-quality estimation.

Each implementation should satisfy the semantic inference interfaces in [architecture.md](architecture.md). **After the current migration, replace shared `Detection` returns on those interfaces with typed ear-segmentation and ear-landmark results.** Model training and model-specific evaluation belong in dedicated training areas rather than the identity-retrieval evaluator.

## Additional Identity Signals

Tear profiles are interpretable but cannot represent every useful identity feature. Later research may investigate:

- learned ear or part embeddings;
- depigmentation, vein, scar, or texture descriptors;
- local feature matching;
- holes when suitable annotations and segmentation exist;
- tusk or body evidence when an experiment justifies reintroducing it.

Additional signals should earn inclusion through separate evaluation. The current restructuring does not preserve speculative multi-signal abstractions.

## Broader Retrieval

Later work may explore:

- one-sided queries;
- open-set rejection for elephants absent from the catalog;
- approximate retrieval for much larger catalogs;
- identity- and time-aware fixed test sets;
- uncertainty estimates across repeated sightings.

These extensions should preserve the distinction between similarity-based candidate ranking and a final identity decision.

## Application Research

A future application must select an ear pair from all available sighting photos. That may combine automated quality heuristics, human review, and evidence correction. Once selected, the application uses the same opaque Photo and Sighting values, PhotoStore capability, and AlphaPhant analysis and matching pipeline as research; see [reference/application.md](reference/application.md).

Application import may later add duplicate detection or content identity across external systems. Permanent opaque IDs and immutable original-photo semantics remain shared with the research system.
