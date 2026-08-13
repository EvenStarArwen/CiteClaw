# Hardening to-do

Work deliberately deferred so the first pass could stay faithful to the demo.
Nothing here is a bug in the rewrite — each item is a place where "do exactly
what the demo does" and "what a shipped product should do" disagree, and
fidelity won for now.

**Rules for this file.** Append, never rewrite. Date every entry. Namespace ids
by lane (`H-SCAF-nn` = scaffold lane) so concurrent lanes cannot collide.

---

## Scaffold lane (`H-SCAF-*`) — 2026-08-13

### H-SCAF-01 — self-host the fonts

**Now.** `web/app/index.html` links four stylesheets from
`fonts.googleapis.com`, which pull `.woff2` files from `fonts.gstatic.com` —
the demo's own refs, kept so glyph rendering is identical for parity.

**Why to change it.** The deployment target for this phase is local only. A
machine that is offline, or on a network that blocks Google Fonts, renders the
whole product in fallback typefaces with no indication (see
`missing-states.md` MS-SCAF-01). It is also a third-party request on every page
load, with the privacy question that carries.

**What it takes.** The two families are Newsreader and Hanken Grotesk. Google
serves both as **variable** `.woff2` per unicode-range subset; Hanken Grotesk is
one file per subset (`ieVn2YZDLWuGJpnzaiwFXS9tYtp*.woff2`), Newsreader is one
roman and one italic file per subset. Vendor those, rewrite the `src:` URLs, and
keep the `unicode-range` blocks byte-identical or text falls back per script.

**Do not** collapse the four stylesheets into one while doing it — see
`escalations.md` E-SCAF-03. The fixed-weight and variable-weight declarations
are not interchangeable and three screens depend on the union.

**Verify with.** The parity harness at the two agreed viewports, before and
after. Also `document.fonts` should list a `Hanken Grotesk normal 400 700` face
once self-hosted, exactly as it does today.

### H-SCAF-02 — KaTeX CSS is not loaded yet

The Explore screen's helmet links
`https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css` (1.4 MB, with
its own `KaTeX_*` `@font-face` set). It is **not** in `web/app/index.html`,
because it is screen-specific and heavy and no screen needs it yet. The Explore
rewrite agent must bring it over, pinned to the same 0.16.11.

`web/parity/package.json` already devDepends on `katex@0.16.11`, so a
self-hosted copy is available in the tree rather than from the CDN.

### H-SCAF-03 — the viewport meta blocks pinch-zoom

`maximum-scale=1, user-scalable=no` is copied verbatim from the demo and is an
accessibility regression on touch devices. Raised as `escalations.md` E-SCAF-05;
listed here so it is not lost if the escalation is answered with "keep it for
now".

### H-SCAF-04 — built CSS is intentionally unminified

`vite.config.ts` sets `build.cssMinify: false` so `dev` and `preview` serve the
same bytes and a parity failure can be read straight out of `dist/`. Costs about
5 kB gzipped. Revisit once the parity baselines are locked — and re-run them
after flipping it, because the CSS foundations are verbatim copies containing
`color-mix()`, `:has()` and custom-property indirection that a minifier is
entitled to reshape.

### H-SCAF-05 — `RunSnapshot` is typed as `unknown`

`web/app/src/data/types.ts` types the run/pipeline surface as `unknown` on
purpose: `run-mock.js` exposes a large ad-hoc object and the engine's event
protocol is still being changed (decisions ledger Q4 / Q17). The Runs rewrite
agent should narrow those fields **in that one file**, so the eventual API
adapter has a real contract to satisfy rather than a cast at every call site.

---

## Build rewrite lane (`H-BLD-*`) — 2026-08-13

### H-BLD-01 — Build's fixtures are inside the transplanted logic, not behind `DataSource`

The Build screen renders no data from `src/design-data`. Its fixtures are
embedded in the transplant and in the markup, and every one of them has to move
behind `data/types.ts` when Build is wired to CiteClaw:

| where | what | source line (`Paper Card.dc.html`) |
|---|---|---|
| `build-logic.js` | `mockPapers()` — the paper corpus the sidebar lists | 2530 |
| `build-logic.js` | `expandMock()` — grows the seeded rows to 91 | 2580 |
| `build-logic.js` | `plInit()` — the 7-step default pipeline | 5199 |
| `build-logic.js` | `plParamDefs()` — per-step-type parameter schema | 5083 |
| `build-logic.js` | `dlLatestRun()` — "Run 6 · yesterday · 214 papers" | 5045 |
| `build-logic.js` | `llmModels()` — the provider/model tree | 4106 |
| `BuildSidebar.tsx` | 14 seeded `.pr` rows with `data-venue/-year/-cites/-url/-abstract` | 1650–1789 |
| `BuildConfigPanel.tsx` | the 5 seeded `.cf-filter` rows incl. the long `data-query` | 2334–2339 |
| `TopBar.tsx` | the six-project switcher list and their counts | 1359–1435 |

Do this by adding Build-shaped methods to the `DataSource` interface and feeding
the transplant, **not** by editing the transplant's fixtures in place — the
parity gate depends on `build-logic.js` staying byte-identical to the demo
(`scripts/verify-transplants.mjs` enforces it).

### H-BLD-02 — the transplant has no teardown

`build-logic.js` is the demo's `componentDidMount` verbatim. It adds `document`
and `window` listeners (Escape, `kl-run-activity`, `kl-project`, `resize`) and a
`ResizeObserver`, and the demo never removed them because its screens never
unmount. `BuildScreen.tsx` guards against double-mounting with a flag on the
root node rather than writing the `componentWillUnmount` the demo never had.
Navigating away from Build repeatedly will therefore leak listeners. Writing
that teardown is a behaviour change and belongs to the interaction gate, not the
static one.

### H-BLD-03 — `style-hover` is reproduced as generated CSS

`style-hover="…"` is a dc-runtime directive, not HTML: support.js strips it and
inserts `.scpN:hover { … !important }` into the CSSOM. The rewrite emits the
equivalent rules into `src/styles/demo-style-hover.css` with hash-derived class
names. Six declarations on Build; Runs and Explore add `style-focus` too, which
is not implemented yet. When those screens land, extend the generator in the
same pass — a missing `style-focus` rule is invisible at rest and only shows up
under keyboard navigation, which no parity screenshot exercises.

### H-BLD-04 — the whole design system ships on every screen

Following E-BLD-01, `src/styles/demo-screens.css` pulls all seven screens'
stylesheets (~300 kB unminified) into any screen that renders demo markup, and
`public/design-fonts/` adds 4.2 MB of base64 `@font-face` CSS. Both are correct
for parity and wrong for delivery. The fix is the same single extracted token +
component sheet that `component-duplication.md` §8 asks for, and it must be
produced by **diffing** the seven blocks, not by picking one. Do not attempt it
before E-BLD-01 is answered.

### H-BLD-05 — `multiple` on the import file input

`.sbi-file` carries `multiple` in the template, but the dc-runtime drops the
attribute, so the demo's own file picker is single-select. The rewrite keeps the
template's attribute (it is `display:none`, zero pixels, and the template is the
design source). If the demo's behaviour is the intended one, drop it; if the
template's is, this is a bug the demo has and the rewrite does not.

### H-BLD-06 — sub-threshold rasterisation delta behind the top-bar popovers

With the project / download / account menu open, the rewrite differs from the
demo by up to 8/255 on a single channel, confined to the two `.sb-scrim`
gradient bands at the top and bottom of the papers list (0 pixels over the
harness's 0.1 threshold; 6 954 raw pixels for the project menu, 7–8 for the
others). The demo is self-deterministic in that state, so it is a real
difference — most likely compositing-layer rounding under the popover's
`element.animate()` layer, made slightly different by the two extra wrapper
`<div>`s the dc-runtime puts around `.pc-root`. Invisible, but it is the only
known non-zero delta and should be re-checked if the gate is ever tightened to
raw byte equality for non-default states.
