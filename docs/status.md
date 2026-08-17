# Current Status

AlphaPhant is mid-restructure: consolidating scattered prototypes into one locked research pipeline - sighting ear pair, AlphaTear profile extraction, tear-profile matching, and known-elephant candidate ranking - before any application work begins.

## Where This Is Going

Research pipeline (now) -> identity-retrieval benchmark -> a future review-and-decision application. The application is not being built yet. The present goal is a small, tested, reproducible research pipeline for use in a rigorous publication.

## What Works Today

The numerical core already exists: SAM3 segmentation, YOLO ear-landmark detection, AlphaTear profile extraction, and directional tear-profile matching with catalog ranking.

## What Is Being Removed

Older directions are being cleared from the active path: SEEK coding and fixed-field domain objects; age, gender, body, trunk, tail, and tusk analysis; the general `PhotoAnalyzer`; identity-bearing identifiers; normalization and calibration layers; the synthetic-pair evaluator; and the API and application prototypes. New research code does not depend on these.

## Target Shape

Seven packages: `domain`, `dataset`, `analysis`, `inference`, `matching`, `eval`, `image`. Each package's responsibility is defined in [architecture.md](architecture.md#responsibilities), algorithm behavior in [pipeline.md](pipeline.md), and the benchmark in [evaluation.md](evaluation.md). Step-by-step implementation lives in the separate spec.
