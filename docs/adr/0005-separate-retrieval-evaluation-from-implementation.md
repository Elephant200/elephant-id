# Separate identity-retrieval evaluation from implementation

## Decision

The identity-retrieval evaluator owns ground-truth labels, real same-sighting ear pairs, leakage prevention, evaluation protocol, metrics, and reproducibility. A system under evaluation receives opaque photo keys and catalog candidate keys and returns a complete ranked candidate list, without exposing ear masks, landmarks, contours, tear profiles, model configuration, or matching internals to the evaluator.
