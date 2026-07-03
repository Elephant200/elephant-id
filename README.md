# Elephant ID

Elephant ID is a desktop-first, offline-capable system for identifying individual African elephants from grouped sighting photo folders.

The product is simple: import a sighting folder, review the AI-extracted evidence, compare aligned matches against the known-elephant catalog, and log a human identity decision.

## Documentation

- [Current status](docs/status.md) - what exists now, what is legacy, and what is still target direction.
- [Workflow](docs/workflow.md) - the intended product flow from import to identity decision.
- [Architecture](docs/architecture.md) - broad system boundaries and non-negotiable constraints.
- [Context](docs/context.md) - canonical project vocabulary.
- [Reference](docs/reference/README.md) - technical notes, matching details, papers, and older experiments.

## Development

Run Python commands from the repo root with `uv`:

```bash
uv run pytest
uv run ruff check .
```
