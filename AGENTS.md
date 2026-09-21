# Agent Guidelines

## Project

AlphaPhant is a fully automated catalog-matching algorithm for elephant re-identification. Given one high-quality image of each ear from the same sighting, it localizes and segments the ears, detects anatomical landmarks, extracts alpha-shape-derived tear profiles, and returns one similarity score per known elephant.

## Start Here

- Current state or cleanup scope: read [docs/status.md](docs/status.md).
- General pipeline flow: read [docs/pipeline.md](docs/pipeline.md).
- Retrieval evaluation, splits, failures, or metrics: read [docs/evaluation.md](docs/evaluation.md).
- Module placement, domain/storage boundaries, or caching: read [docs/architecture.md](docs/architecture.md).
- Naming, domain or technical terms: read [docs/context.md](docs/context.md).
    - Read [docs/context.md](docs/context.md) before editing any documentation or docstrings.
- Future application work: read [docs/reference/application.md](docs/reference/application.md).
- Later research ideas: read [docs/future.md](docs/future.md).
- Surprising decisions: read [docs/adr/](docs/adr/).

## Commands

Run Python from the repository root with `uv`. After Python changes, run `uv run ruff check .` and relevant tests. 

## Scope and Safety

- Root `dataset/` is private user data. Do not touch it unless explicitly asked.
- Preserve existing user changes. Check `git status --short` before editing.
- You must receive explicit confirmation before writing a commit.
- Never touch `legacy`.

## Active Architecture

- **domain** owns `Photo`, `Sighting`, and `SightingEarPair`.
- **dataset** owns private metadata and known-elephant resolution; its **PhotoStore** resolves a `Photo` to original encoded bytes without exposing identity.
- **preparation** owns sighting-ear preparation.
- **inference** owns ear localization, ear segmentation, and ear landmark detection models.
- **matching** owns the public `CatalogMatcher` contract. Its **alphaphant** package is the core of this project. Other matching algorithm implementations, like CurvRank and MiewID, also live here.
- **evaluation** owns implementation-independent identity-retrieval evaluation.
- **image** owns byte decoding, BGR images, and basic + universal geometry utilities.

Keep interfaces narrow and justified by current variation. Research supplies a sighting ear pair directly; future application ear selection remains upstream of the shared AlphaPhant pipeline.

## Python Style

- Use types for all function / method signatures.
- Give every package, module, class, function, and method a concise accurate Google-style docstring. Code in docstrings uses single backticks. Docstrings should clearly state the contract, not the weeds of implementation.
- Prefer clear names and structure over comments. Comment only non-trivial reasoning.
- Use `loguru` for logging.
- Do not over-engineer by making unnecessary abstractions or defensive guards that you don't need.

## Images and Geometry

- `BgrImage` is the canonical in-memory image: HWC, BGR, `uint8`, OpenCV-native.
- PhotoStore returns encoded bytes. Decode them through the shared image-package OpenCV decoder at the image boundary. 
- Float boxes use half-open `xyxy` coordinates. Convert to integer pixels at raster boundaries through the image geometry helpers.
- Public color and background arguments in code are human-facing RGB; convert to BGR at the write boundary.

## Caching

- One cache stores model outputs and tear profiles.
- Use photo UUIDs for photo-level source identity. Add dependent inputs such as crop coordinates as needed. Do not use hashes.
- A `producer_slug` carries model, weight, prompt, preprocessing, threshold, configuration, and algorithm identity.

## Issue Tracker

Issues are markdown files in `.scratch/<feature>/`. See [docs/agents/issue-tracker.md](docs/agents/issue-tracker.md).
