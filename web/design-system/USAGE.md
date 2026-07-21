# CiteClaw Design System — how this repo uses it

This directory is the **canonical, in-repo record of the CiteClaw Design System** —
the refined, unified visual language for the web UI (`web/live` local app +
`web/public` deployed app). It was distilled by Claude Design from the hardened
UI and collapses the earlier sprawl (~18 pipeline-block styles, ~11 colour
palettes) into **one committed identity**: warm-cream surfaces, plum-slate ink,
and a single forest-green accent.

`readme.md` is the full specification. Start there. This file is the short
project-facing contract.

## The rule (conformance)

**Every web-UI change must conform to this design system.** Concretely:

- Use the **tokens** (`tokens/*.css`, the `--cc-*` custom properties) for every
  colour, radius, shadow, type size, and spacing value. Do not hardcode hex
  colours or one-off px radii/shadows in `app.css` or JSX.
- Match the **component contracts** in `components/core/` (Button, Badge,
  ProgressBar, SegmentedToggle, PanelHeader, MetricStat) and `components/app/`
  (PaperCard, PipelineBlock, FilterRow). Each has a `.jsx` reference
  implementation, a `.d.ts` prop contract, and a `.prompt.md`. New UI reuses
  these shapes; it does not invent parallel variants.
- **The single most important rule:** green (`--cc-accent`, `#3a7550`) is
  reserved for *critical moments only* — primary action, active tab, current
  selection, live/running state, seed nodes/dots, links, and the brand mark.
  Everything else is cream + ink.

You may deviate only when **(a)** the user explicitly asks to modify the design
system, or **(b)** a new feature genuinely needs an element the system lacks —
in which case add it here (token + component + a note in `readme.md`) as a
first-class part of the system, don't bolt a one-off onto `app.css`.

## How it's wired into the code

The live app does not `@import` these files at runtime. Instead:

- `web/live/static/app.css` embeds the token set from `tokens/*.css` in its
  `:root` block (single source of the `--cc-*` values the whole UI references),
  and its component rules are aligned to the styles in
  `components/**/*.card.html` / the `.jsx` reference components.
- The `.jsx` here are **reference contracts**, not build inputs — the app's
  `assemble_index` concatenates only `web/live/static/jsx/*.jsx`, never this
  directory. Keep the two in sync by hand: when you change a component's look,
  update both the live rule and the reference here.
- Fonts: Inter + IBM Plex Mono via Google Fonts (see `tokens/fonts.css`).
  Icons: Lucide, stroke style, stroke-width 1.75.

## Contents

- `styles.css` — the system's own global entry (`@import`-only); reference.
- `tokens/` — `colors`, `typography`, `spacing`, `radius`, `shadow`, `fonts`.
- `components/core/`, `components/app/` — the component contracts + catalog HTML.
- `ui_kits/webui/` — full-screen Build / Run / Explore recreations (the target).
- `guidelines/` — foundation specimen cards (colours, type, spacing, brand).
- `assets/logo-claw.svg` — the brand mark (green fill; do not redraw).
- `readme.md`, `SKILL.md` — the full spec + portable skill wrapper.
