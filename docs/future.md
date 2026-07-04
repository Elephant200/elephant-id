# Future Direction

This document records where the matching work is heading beyond the current tear-profile signal. It is a research direction, not a committed plan. The current product only ships the tear-profile matcher documented in [reference/matching.md](reference/matching.md); everything here is prior art and intended next steps for turning matching into a calibrated multi-signal system.

## Why Move Beyond Tears

Elephants are low-entropy subjects: gray, no coat pattern. Identity lives in ear-margin shape, tears and holes, ear depigmentation and veins, tusks, and scars. The ELPephants benchmark baseline is only 56% top-1 / 80% top-10, and a part-based method reached just 24.3 mAP — a reminder that this is a hard re-ID problem. The consistent lesson is to compute descriptors on standardized **part crops** (especially the ear), not the whole body.

The tear profile is one such part signal. It stays valuable as an interpretable, offline signal, but it should become one input among several rather than the only way the product reasons about identity. The legacy SEEK code is a lossy quantization of ear structure; if kept at all, it is an interpretable interop layer over richer descriptors, not a matching key.

## Key Prior - ElephantBook

ElephantBook (arXiv 2106.15083, deployed at the Mara Elephant Project) is a web human-in-the-loop system that fuses SEEK-style ear codes, CurvRank ear-contour matching, and CNN embeddings. Its central finding is that SEEK and CurvRank are **complementary** — fusing them beats either alone. That validates this project's structured-plus-visual approach. ElephantBook predates modern descriptors, so the opportunity is to improve on it there.

## Candidate Descriptor Stack

All of these are offline-capable, which matters for the desktop product:

- **Global embedding — MiewID-msv3** (Hugging Face `conservationxlabs/miewid-msv3`, EfficientNetV2, 2152-dim; reported to beat MegaDescriptor by +19.2% top-1). Alternative: MegaDescriptor-L-384 (`BVRA/...`) with the `wildlife-tools` toolkit for ArcFace training and evaluation.
- **SSL backbone / dense features — DINOv3** for part alignment and depigmentation texture.
- **Local matching — ALIKED or SuperPoint + LightGlue** (zero-shot, and it produces explainable keypoint overlays for the review UI), or LoFTR.
- **Contour — CurvRank + LNBNN.** Reference material exists in the repo, but note the field experience below.

Field note: a direct CurvRank trial on this dataset was rejected - it did not perform well, and it is an engineered rather than learned descriptor. Treat it as prior art to improve past, not a drop-in.

## Target Architecture — Calibrated Fusion

The intended shape follows **WildFusion** (arXiv 2408.12934):

1. Calibrate each descriptor's raw similarity into a probability via isotonic regression.
2. Average the calibrated scores across descriptors.
3. Shortlist the top ~300 candidates with a cheap global embedding, then run the expensive local/curvature descriptors only on the shortlist (~30x speedup).

This generalizes ElephantBook's fusion finding and solves the "scores must be hand-tuned to be combinable" problem in a principled way. It is the same lesson the tear matcher already learned at small scale: calibration is what makes independent signals addable (see [reference/matching.md](reference/matching.md)).

## Open-Set Is Mandatory

New individuals must never be force-matched onto a known elephant. Threshold the fused probability to detect a "new individual" decision. Evaluate with **BAKS/BAUS** (WildlifeReID-10k, arXiv 2406.09211), and use **identity- and time-aware splits** to avoid leakage — the same discipline the current evaluation harness already applies.
