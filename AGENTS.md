# Agent Guidelines

## Project

AlphaPhant is a fully automated candidate-ranking algorithm for elephant re-identification. Given one high-quality image of each ear from the same sighting, it localizes and segments the ears, detects anatomical landmarks, extracts alpha-shape-derived tear profiles, computes similarity scores, and ranks the known-elephant catalog.

## Start Here

- Current state or cleanup scope: read [docs/status.md](docs/status.md).
- Analysis or matching behavior: read [docs/pipeline.md](docs/pipeline.md).
- Retrieval evaluation, splits, failures, or metrics: read [docs/evaluation.md](docs/evaluation.md).
- Module placement, domain/storage boundaries, or caching: read [docs/architecture.md](docs/architecture.md).
- Naming, domain or technical terms: read [docs/context.md](docs/context.md).
- Future application work: read [docs/reference/application.md](docs/reference/application.md).
- Later research ideas: read [docs/future.md](docs/future.md).
- Surprising durable decisions: read [docs/adr/](docs/adr/).

## Commands

Run Python from the repository root with `uv`. Do not activate `.venv` or call `python` directly.

```bash
uv run pytest
uv run ruff check .
uv sync --all-groups
```

After Python changes, run `uv run ruff check .` and relevant tests. `legacy` is excluded from Ruff; avoid incidental cleanup there.

## Scope and Safety

- `dataset` is private user data. Preserve it unless a request explicitly names an identity assignment, cache migration, or mutation. Never commit dataset contents, credentials, environment files, API keys, or model secrets.
- Preserve existing user changes. Check `git status --short` before editing and never revert unrelated work.
- Do not modernize `legacy` or `.curvrank_ref` unless explicitly asked.
- Do not commit unless explicitly asked. You must receive explicit confirmation before writing a commit.

## Active Architecture

- **domain** owns neutral immutable `Photo`, `Sighting`, and `SightingEarPair` values with permanent opaque UUID identity.
- **dataset** owns private metadata and known-elephant resolution; its **PhotoStore** resolves a `Photo` to original encoded bytes without exposing identity.
- **analysis** owns sighting analysis, ear preparation, and AlphaTear profile extraction.
- **inference** owns swappable implementations of ear localization, ear segmentation, and ear landmark detection.
- **matching** owns tear-profile similarity and catalog ranking.
- **eval** owns implementation-independent identity-retrieval evaluation.
- **image** owns encoded-byte decoding, BGR images, and basic + universal geometry utilities.

Keep interfaces narrow and justified by current variation. Rankers receive neutral domain objects and an image-only PhotoStore, never the identity-aware Dataset. Research supplies a sighting ear pair directly; future application ear selection remains upstream of the shared AlphaPhant pipeline.

## Python Style

- Type every function and method signature.
- Give every package, module, class, function, and method a concise accurate docstring.
- Add `Args`, `Returns`, `Yields`, or `Raises` sections only when they clarify a non-obvious interface. Document validation errors that are part of the interface.
- Prefer clear names and structure over comments. Comment only non-trivial reasoning.
- Use `loguru` for logging. Library code never configures logging; entry points call `elephant_id.log.configure_logging` once.
- Log identifiers, counts, durations, and cache hits at appropriate levels. Never log credentials or raw image and mask buffers.
- Avoid over-engineering by making unnecessary abstractions or defensive guards that you don't need. Never hesitate to ask when in doubt.

## Images and Geometry

- `BgrImage` is the canonical in-memory image: HWC, BGR, `uint8`, OpenCV-native.
- PhotoStore returns encoded bytes. Decode them through the shared image-package OpenCV decoder at the image boundary. Avoid PIL/RGB conversions inside the package without a boundary reason.
- Float boxes use half-open `xyxy` coordinates. Convert to integer pixels only at raster boundaries through the image geometry helpers.
- OpenCV drawing endpoints are inclusive; draw a half-open box through `x2 - 1` and `y2 - 1`.
- Public color and background arguments are human-facing RGB; convert to BGR at the write boundary.
- COCO RLE uses `size=[height, width]` and string or byte counts. Decoded masks are two-dimensional boolean arrays.

## Caching

- One generic cache store serves immutable named producers and final per-ear tear profiles.
- Cache expensive computation, not orchestration.
- Use the permanent photo UUID directly for photo-level source identity. Add only actual dependent inputs such as crop coordinates; keep keys readable rather than hashing them again.
- A `producer_slug` carries model, weight, prompt, preprocessing, threshold, configuration, and algorithm identity. Any output-changing change gets a new immutable slug; those settings do not enter cache keys.
- Keep writes atomic and validate producer payloads on load.
- Select caching only through composition: inject a cached decorator or the raw processor. Analyzers and rankers expose no cache-policy flags.
- Cache the complete SAM3 multi-feature computation before adapting it to the ear-only segmentation protocol. Cache landmark records in full-image coordinates.
- Preserve SAM3 body and complete multi-feature outputs and leave heuristic records untouched. Landmark records may be replaced when their coordinate contract changes. Age and gender records may be removed.

## Testing

- Test code under `src/elephant_id`; do not add unit tests for scripts or apps.
- Use small synthetic arrays and encoded images, fake PhotoStores and inference implementations, and recording cache/model clients.
- Unit tests never initialize real models, require network access, or depend on private photos.
- Characterize current numerical behavior before moving tear-profile or matching code.
- Add focused tests when changing validation, geometry, cache keys, serialization, dataset ordering, catalog aggregation, or evaluation splits.

## Git

- Keep changes scoped to the request and report noteworthy out-of-scope issues.
- Commit messages use imperative style without conventional prefixes.
- Browser or Playwright verification always runs in a subagent and only when necessary.

## Agent skills

### Issue tracker

Issues live as markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default role strings recorded as a `Status:` line. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: glossary at `docs/context.md`, ADRs at `docs/adr/`. See `docs/agents/domain.md`.
