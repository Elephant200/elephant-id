# Evaluation Future Work

The current high-quality evaluator should stay simple until the next research meeting: one strict script, one selected score stack per invocation, and explicit flags for turning normalization or calibration off.

After the presentation, the evaluation machinery should be moved behind a small matching/evaluation module that can serve future matching tasks without script sprawl. The target shape is:

- One generalized matching class that owns profile loading, raw same-side pairwise scoring, score normalization, optional calibration, side fusion, and ranking.
- Strict derived artifacts with fingerprints from manifest to profile cache to raw pairwise score cache to reported run.
- A documented choice about cohort normalization: either keep the current transductive evaluation-cohort normalization and name it clearly, or replace it with a catalog-only normalization protocol.
- A short public interface for selecting the input set, pairing seed(s), and enabled scoring steps.
- Testable helpers whose names describe one job; no helper should hide multiple unrelated stages.
- Reporting that is deterministic, printable, and publication-ready by default.

The design goal is validity first, then efficiency, then readability without ceremonial abstractions.
