# CiteClaw Design System

The design language for **CiteClaw** — a transparent, high-fidelity literature-acquisition tool for autonomous scientific discovery. It powers a local **WebUI** (a FastAPI + browser app) with three modes: **Build** (compose a snowball screening pipeline), **Run** (watch it execute live), and **Explore** (dig through the resulting citation network).

This system is the *refined, unified* version of that UI. The shipped product had grown through fast iteration into ~18 pipeline-block styles and 11 color palettes; this design system collapses that into **one committed identity**: warm-cream surfaces, plum-slate ink, and a single forest-green theme accent.

## Sources
- **Codebase:** `CiteClaw-claude-live` (local import). UI lives in `web/live/static/` (no-build JSX components + `app.css`) and `web/live/backend/` (FastAPI). Branch referenced: `codex/ui-product-hardening`.
- **Project README:** `CiteClaw-claude-live/README.md` (product/architecture), `web/live/README.md` (WebUI walkthrough).
- Reader may not have access to the import; values here are self-contained.

## Committed direction
"**2a Layered cream + green**", chosen from a set of explored directions:
- A warm **tonal elevation ladder**: recessed canvas (`--cc-bg`) → panels (`--cc-panel`) → raised cream cards (`--cc-card`, an ivory-cream, never pure white). This is how hierarchy (层次感) is expressed without heavy shadows.
- **One radius family** (9px cards / 8px controls / 999px pills) and **one tight type scale**, replacing the mixed 2/3/5/999px radii and many small type sizes of the original.
- A **forest-green accent** (`#3a7550`) used ONLY in critical moments.

---

## CONTENT FUNDAMENTALS
- **Voice:** precise, technical, and quietly confident — written for researchers. Product-level copy states capability plainly: *"Transparent, high-fidelity literature acquisition for autonomous scientific discovery."*
- **UI copy** is instructional and terse, usually an imperative or a fragment: *"click a step to configure"*, *"Search Semantic Scholar — press Enter"*, *"star papers (☆) to add them here"*. Second person ("you") appears in guidance; the app rarely refers to itself in the first person.
- **Casing:** panel/section labels are **UPPERCASE mono eyebrows** ("SEEDS", "CONFIGURE STEP", "PIPELINE PROGRESS"). Titles, buttons and names are **Sentence case** ("Forward screener", "Run pipeline", "Diversified rerank"). Filter kinds render as short uppercase pills ("YEAR", "LLM", "KEYWORD").
- **Numbers & IDs** are monospace and tabular ("FWD-02", "β = 30", "1,094 cites", "$0.63", "62%"). Step ids follow `PREFIX-NN` (SED/FWD/BWD/RRK/RSC).
- **Tone markers:** exact, no hype, no exclamation. Explains *why* inline ("An empty criterion makes the model accept every paper"). Uses scientific vocabulary directly (snowball, screener, rerank, SPECTER2, pagerank).
- **Emoji:** none in the UI chrome. The only glyphs used semantically are ✓ (done), ★/☆ (seed), ⇄ (linked copy), · (meta separator), № (specimen number).

## VISUAL FOUNDATIONS
- **Surfaces:** warm cream, 4-step elevation ladder (see tokens). Cards are ivory-cream `#fdfbf6` — deliberately not white — so they sit calmly on the cream canvas.
- **Ink:** cool plum-slate (`#1e2735` → `#a49b90`), four steps. Never pure black.
- **Accent (green) — used sparingly:** primary action button, active tab (as green *text* on the white pill), current selection (green border + soft ring), live/running indicators & progress fills, seed nodes/dots in graphs, the brand mark, and links. Everything else stays cream/ink. This is the single most important rule: green means "action / active / alive".
- **Type:** Inter (sans) + IBM Plex Mono. Mono is reserved for IDs, metrics, meta rows, eyebrow labels, and log lines. One tight scale — titles 14px, body 13px, meta 11.5px mono.
- **Borders:** 1px hairlines (`--cc-rule` / `--cc-rule-strong`). Structure comes from hairlines + tone steps first, shadow second.
- **Shadows:** soft and low-opacity (`0 1px 2–3px rgba(30,39,53,.05–.06)` for cards; deeper only for popovers/overlays). No hard or colored shadows.
- **Corner radii:** unified — 9px cards/blocks/panels, 8px controls, 6px small pills, 999px for status chips / count pills / dots / segmented toggles.
- **Cards:** ivory-cream fill, 1px `--cc-rule` border, `--cc-shadow-2`, 9px radius, ~10–12px padding. The **paper card** is the canonical example (fixed 18px lead column so titles align whether or not a seed dot/star is present).
- **Selection / active:** green 1px border + `--cc-ring` (2px soft green halo). Consistent across pipeline blocks, filter rows, list items, nodes.
- **Hover:** subtle — background lifts to `--cc-hover-soft`, border darkens to `--cc-rule-strong`. **Press:** `--cc-hover-deep`. Icon buttons shift ink-3 → ink-1.
- **Canvas texture:** graph/pipeline areas use a dot-grid (`radial-gradient(var(--cc-grid-dot) 1px, transparent 1px)`, 16px cell).
- **Graph imagery:** nodes sized by citations, colored along the year ramp (cool grey → ink), seeds in accent green with a halo; edges are faint slate at ~16% opacity. Cool, quiet, data-first — no photography, no illustration.
- **Motion:** short (`120ms`) ease-out on hover/border/background; an indeterminate sliding bar for live "working" states. No bounces, no decorative animation.
- **Layout:** fixed 3-column shell — left panel · center workspace · right panel — under a 46px top bar and above a 28px mono status bar. Resizable column widths are tokens.

## ICONOGRAPHY
- **Icon set:** [Lucide](https://lucide.dev) (the codebase calls `lucide.createIcons({ attrs: { "stroke-width": 1.75 } })`). All icons are **stroke** style at **stroke-width 1.75**, sized 11–15px inline. Consumers should load Lucide (CDN: `https://unpkg.com/lucide@latest`) rather than hand-drawing glyphs.
- Representative glyphs: search, sliders-horizontal, arrow-right/left, flag, filter, shapes, play, pause, square, settings, star, git-branch, undo-2, external-link, chevron-right, circle-check, circle-x, refresh-cw.
- **Brand mark:** the CiteClaw "claw" glyph — a single SVG `<path>` (see `assets/logo-claw.svg`), filled with `--cc-accent` (green) in this system. This is the product's own mark, copied from source; do not redraw it.
- **No emoji** as icons. A few unicode glyphs carry meaning (✓ ★ ⇄ № ·) — keep those, don't expand the set.
- **Fonts note:** Inter + IBM Plex Mono are loaded from Google Fonts (matching the source), not vendored as binaries. If you need offline binaries, ask and we'll vendor them.

---

## INDEX (manifest)
- `styles.css` — global entry (import this). @imports everything below.
- `tokens/` — `fonts.css`, `colors.css`, `typography.css`, `spacing.css`, `radius.css`, `shadow.css`.
- `guidelines/` — foundation specimen cards (Colors, Type, Spacing, Brand) shown in the Design System tab.
- `components/core/` — Button, Badge, ProgressBar, SegmentedToggle, PanelHeader, MetricStat.
- `components/app/` — PaperCard, PipelineBlock, FilterRow (CiteClaw-specific primitives).
- `ui_kits/webui/` — full-screen recreations: Build, Run, Explore.
- `assets/` — logo-claw.svg.
- `SKILL.md` — portable skill wrapper.

### Intentional additions
- **PanelHeader**, **MetricStat**: not standalone "components" in the source (they're inline markup), but promoted here because they recur across all three screens and encode the eyebrow + count pattern.

## CAVEATS / open questions
- Fonts are Google-hosted, not vendored (matches source).
- Component card previews are static token-driven HTML (visual catalog); the components are also real importable JSX.
- The graph views in the source are WebGL (sigma.js); UI-kit graphs here are representative static SVG.
