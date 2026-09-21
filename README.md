# AlphaPhant

AlphaPhant is a fully automated candidate-ranking algorithm for elephant re-identification. Given one high-quality image of each ear from the same sighting, it localizes and segments the ears, detects anatomical landmarks, extracts alpha-shape-derived tear profiles, computes similarity scores, and ranks the known-elephant catalog.

![African Elephant](https://africageographic.com/wp-content/uploads/2020/01/Guest-Dr.jpg)

## Development

Run Python from the repo root with `uv`:

```bash
uv run pytest
uv run ruff check .
uv sync --all-groups
```

See [AGENTS.md](AGENTS.md) for agent and contributor guidelines.

## Evaluation

Run the standard retrieval evaluation with `uv run eval`. Pass a manifest path to evaluate a different benchmark manifest.
