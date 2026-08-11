# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`docs/context.md`** — this repo's canonical glossary (already established as such in `AGENTS.md`). Prefer its terms: App Library, analysis package, known-elephant catalog, reviewer, evidence review, tear profile, and identity decision. This repo has no root `CONTEXT.md` or `CONTEXT-MAP.md` — `docs/context.md` fills that role.
- **`docs/architecture.md`** and **`docs/status.md`** — broad technical constraints, and what's current vs. legacy vs. cleanup debt. Useful surrounding context alongside the glossary itself.
- **`docs/adr/`** — read ADRs that touch the area you're about to work in. Doesn't exist yet; this repo hasn't recorded any ADRs so far. Create it lazily (e.g. via `/domain-modeling`) when a decision is actually worth recording, rather than upfront.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront.

## File structure

Single-context repo (this repo):

```
/
├── docs/
│   ├── context.md       ← canonical glossary
│   ├── architecture.md
│   ├── status.md
│   └── adr/              ← not yet in use
└── src/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `docs/context.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders) — but worth reopening because…_
