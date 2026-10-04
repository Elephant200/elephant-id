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

- Root `dataset/` is private user data. Do not edit or delete it unless explicitly asked.
- Preserve existing user changes. Check `git status --short` before editing.
- Do not commit without explicit permission.
- Never touch `legacy`.
- `.scratch/` holds specs, tickets, and private regression tests. It is not a scratchpad. Put throwaway scripts, logs, figures, and data in your session scratchpad or a temporary directory.

## Active Architecture

Keep interfaces narrow and justified by current variation. Research supplies a sighting ear pair directly; future application ear selection remains upstream of the shared AlphaPhant pipeline.

## Python Style

- Use types for all function / method signatures.
- Give every package, module, class, function, and method a concise, accurate Google-style docstring. Code in docstrings uses single backticks. Docstrings should clearly state the contract, not the weeds of implementation.
- Prefer clear names and structure over comments. Comment only non-trivial reasoning.
- Keep code and research tooling as simple as possible. Add an abstraction, defensive guard, hash, or provenance record only when current work strictly requires it; this is rare.

## Documentation

- Write docs and docstrings in ASD-STE100 style: active voice, plain direct verbs, and lists for rules with several parts.
- Use the terms in [docs/context.md](docs/context.md).

- State what the code does. Do not state what it does not do, what it used to do, or what another branch does.
- Write a constraint as a positive rule ("Preparation fails when no ear has the declared side"), not as an absence ("There is no fallback"). Only write negative rules if a careful reader would make the wrong assumption; this is rare.
- Report removals, history, and rejected alternatives to the user in chat, not in permanent files.
- Test behavior the code has. Do not add tests that guard against code that never existed.
- To check, consider deleting the sentence. If a reader wouldn't form a wrong belief, it's not necessary.

## Images and Geometry

- Float boxes use half-open `xyxy` coordinates. Convert to integer pixels at raster boundaries through the image geometry helpers.
- Public color and background arguments in code are human-facing RGB; convert to BGR at the write boundary.

## Caching

- One cache stores model outputs and tear profiles.
- Use photo UUIDs for photo-level source identity. Add dependent inputs such as crop coordinates as needed. No hashes.
- A `producer_slug` carries model, weight, prompt, preprocessing, threshold, configuration, and algorithm identity.

## Issue Tracker

Issues are markdown files in `.scratch/<feature>/`. See [docs/agents/issue-tracker.md](docs/agents/issue-tracker.md).
