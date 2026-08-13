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
