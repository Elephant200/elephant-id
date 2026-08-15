# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

This is a single-context repo. There is no root `CONTEXT.md` or `CONTEXT-MAP.md`. Skills that would read or write `CONTEXT.md` use `docs/context.md` instead.

## Before exploring, read these

- **Naming** — `docs/context.md`. Prefer its terms; do not drift to synonyms the glossary explicitly avoids.
- **Durable decisions** — `docs/adr/` entries that touch the area you're about to work in.
- **Which other top-level doc** — follow the Start Here pointers in `AGENTS.md` / `CLAUDE.md` (`architecture.md`, `pipeline.md`, `evaluation.md`, `status.md`, `future.md`, `reference/`).

If a concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## File structure

```
docs/
├── context.md       ← glossary (this repo's CONTEXT.md)
├── architecture.md
├── pipeline.md
├── evaluation.md
├── status.md
├── future.md
├── adr/
├── reference/       ← historical and forward-looking detail
└── agents/
```

When a skill would create `CONTEXT.md`, write `docs/context.md`. When it would create an ADR, write `docs/adr/`.

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0001 (focus on tear-profile retrieval) — but worth reopening because…_
