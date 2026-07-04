# Desktop App (Alphaphant)

Design and workflow preferences for the offline desktop reviewer in `apps/desktop`. This captures the intended look and interaction shape; how to build, run, and demo the app lives in [`apps/desktop/README.md`](../../apps/desktop/README.md).

The desktop app is the reviewer-facing surface for the workflow in [../workflow.md](../workflow.md): import a sighting folder, review and correct the extracted ear evidence, compare ranked matches, and log an identity decision. It is currently driven as a live demo for field experts and funders, so interaction clarity matters more than density.

## Visual Theme

A professional savannah theme:

- dark green sidebar and header,
- beige main background,
- elephant-gray accents.

Keep the palette centralized as CSS variables in `apps/desktop/src/styles.css` so the theme stays consistent as pages are added.

## Layout and Density

Prefer more, simpler pages over dense single pages. The workflow is split into separate step pages — Ingest, Photos, Match, Review — rather than stacked panels on one screen. When adding a feature, add a new page or step rather than growing an existing page.

Images, graphics, and text should be scaled generously for a laptop screen; the first versions ran too small. Vast visual changes are acceptable as long as they mostly respect the API contract with the sidecar. (Route JSON field names are a contract with the renderer, the same way `apps/visualization` field names are.)
