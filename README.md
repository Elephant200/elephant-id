# AlphaPhant

AlphaPhant is a fully automated candidate-ranking algorithm for elephant re-identification. Given one high-quality image of each ear from the same sighting, it localizes and segments the ears, detects anatomical landmarks, extracts alpha-shape-derived tear profiles, computes similarity scores, and ranks the known-elephant catalog.

## Documentation

- [Current status](docs/status.md) — current project state and next steps.
- [Pipeline](docs/pipeline.md) — research algorithm behavior and verification boundary.
- [Evaluation](docs/evaluation.md) — identity-retrieval benchmark protocol.
- [Architecture](docs/architecture.md) — module responsibilities, data flow, inference seams, and caching.
- [Context](docs/context.md) — canonical domain vocabulary.
- [Application](docs/reference/application.md) — future application surrounding the research pipeline.
- [Future research](docs/future.md) — deferred research directions.
- [ADRs](docs/adr/) — durable architectural decisions.

## Development

Run Python from the repo root with `uv`:

```bash
uv run pytest
uv run ruff check .
uv sync --all-groups
```

See [AGENTS.md](AGENTS.md) for agent and contributor guidelines.

## Evaluation

Run the standard retrieval evaluation with `uv run eval`. Pass a manifest path
to evaluate a different benchmark manifest.
