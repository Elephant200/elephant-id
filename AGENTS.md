# Agent Guidelines

## Project

Alphaphant matches one sighting ear pair against a known-elephant catalog and returns one similarity score per known elephant.

## Start Here

- Current state: read [docs/status.md](docs/status.md).
- Algorithm flow: read [docs/pipeline.md](docs/pipeline.md).
- Evaluation, splits, failures, or metrics: read [docs/evaluation.md](docs/evaluation.md).
- Module placement, coordinates, caching, or photo identity: read [docs/architecture.md](docs/architecture.md).
- Terms, or before editing docs and docstrings: read [docs/context.md](docs/context.md).
- Future application work: read [docs/reference/application.md](docs/reference/application.md).
- Later research: read [docs/future.md](docs/future.md).
- Surprising decisions: read [docs/adr/](docs/adr/).

## Commands

Run Python from the repository root with `uv`. After Python changes, run `uv run ruff check .` and relevant tests.

Change files with your harness's file tools: Edit/Write in Claude Code, `apply_patch` in Codex. Do not use heredocs, python -, or sed -i to write or edit files, even for batches of small edits. This is critical for transparency. Use the shell to run programs: tests, ruff, git, and quick python scripts. 

## Scope and Safety

- Root `dataset/` is private user data. Do not edit or delete it unless explicitly asked.
- Preserve existing user changes. Check `git status --short` before editing.
- Do not commit without explicit permission.
- Never touch `legacy`.
- `.scratch/` holds specs, tickets, and private regression tests. It is not a scratchpad. Put throwaway scripts, logs, figures, and data in your session scratchpad or a temporary directory.

## Active Architecture

Keep interfaces narrow and justified by current variation. Research supplies a sighting ear pair directly; future application ear selection remains upstream of the shared Alphaphant pipeline.

## Python Style

- Use types for all function and method signatures.
- Give every package, module, class, function, and method a concise Google-style docstring. Code in docstrings uses single backticks. State the contract, not implementation weeds.
- Prefer clear names over comments. Comment only non-trivial reasoning.
- Keep code and tooling as simple as possible. Add an abstraction, defensive guard, hash, or provenance record only when current work requires it; this is rare.

## Documentation

- Write docs and docstrings in ASD-STE100 style: active voice, plain verbs, and lists for rules with several parts.
- State what the code does. Write a constraint as a positive rule, not as an absense. Write a negative rule only when a careful reader would otherwise form the wrong belief.
- Report removals, history, and rejected alternatives to the user in chat.
- Test behavior the code actually implements.
- Deletion test: Delete a sentence when a reader would not form a wrong belief without it.

## Images

Public color and background arguments are human-facing RGB. Convert to BGR at the write boundary.

## Issue Tracker

Issues are markdown files in `.scratch/<feature>/`. See [docs/agents/issue-tracker.md](docs/agents/issue-tracker.md).
