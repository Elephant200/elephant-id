# AlphaPhant

AlphaPhant is a fully automated candidate-ranking algorithm for elephant re-identification. Given one high-quality image of each ear from the same sighting, it localizes and segments the ears, detects anatomical landmarks, extracts alpha-shape-derived tear profiles, computes similarity scores, and ranks the known-elephant catalog.

## Documentation

- [Current status](docs/status.md) — working numerical core, structural debt, and preservation boundaries.
- [Pipeline](docs/pipeline.md) — research algorithm behavior and verification boundary.
- [Evaluation](docs/evaluation.md) — identity-retrieval benchmark protocol.
- [Architecture](docs/architecture.md) — module responsibilities, data flow, inference seams, and caching.
- [Context](docs/context.md) — canonical domain vocabulary.
- [Workflow](docs/workflow.md) — future application surrounding the research pipeline.
- [Future research](docs/future.md) — directions beyond the locked pipeline.
- [ADRs](docs/adr/) — durable architectural decisions.

## Development

Run Python from the repo root with `uv`:

```bash
uv run pytest
uv run ruff check .
uv sync --all-groups local
```

See [AGENTS.md](AGENTS.md) for agent and contributor guidelines.
