# Escalations

Things found while wiring that would **require changing the UI's appearance or
behaviour**. Per the hard rule, nothing here was changed. Each entry describes
what and why, so the orchestrator can raise it with the product owner.

**Rules for this file.** Append, never rewrite. Date every entry. Namespace ids
by lane (`E-IMP-nn` = import lane) so concurrent lanes cannot collide.

---

## Import lane (`E-IMP-*`) — 2026-08-13

### E-IMP-01 — JSON is requested but is not in the demo's format set

**What.** The import brief asks for a JSON line in the format matrix
(CSL-JSON, S2 JSON, and CiteClaw's own `literature_collection.json`). The
rendered demo does not accept JSON:

- `import-resolver.js`: `var EXT_N = { bib, ris, csv, txt, zip, pdf }` — no
  `json`; a `.json` file falls into the unsupported branch.
- Home dropzone copy: *"BibTeX or RIS exports (Zotero, EndNote, Mendeley), CSV,
  a DOI list, or a folder of PDFs, .zip works too."*
- Build / Runs dropzone copy: *"BibTeX, RIS, CSV, a DOI list, or PDFs, folders
  and .zip work too."*
- The single per-file error string: *"Unsupported format: use .bib, .ris, .csv,
  a DOI list, or PDFs."*

**Why it is an escalation.** Accepting JSON changes visible copy in three
surfaces (two dropzone paragraphs plus the error string) and adds a `JSON`
label to the `EXTC` badge map. That is an appearance change, so it was not made.

**Why it might be worth it.** `literature_collection.json` re-import is the
highest-value JSON case for this product: the records already carry
`paper_id` + `external_ids`, so they resolve with **zero** Semantic Scholar
round-trips — the only format in the matrix that does. It turns "open a
finished run as the seed set for a new one" into a supported action.

**Not changed.** Fixtures exist either way (`import-fixtures/json/`, 8 files)
so the decision can be tested in whichever direction it goes.

---

### E-IMP-02 — folder drop is promised in copy but cannot be delivered by the current control

**What.** Both dropzone copy blocks promise folders ("a folder of PDFs";
"folders and .zip work too"). The demo's control is a single
`<input type="file" multiple>` with no `webkitdirectory`, and the drop handler
reads `e.dataTransfer.files` only. A dropped folder therefore produces nothing
at all — no rows, no error (registered as `missing-states.md` MS-IMP-10).

**Why it is an escalation.** The two halves cannot both be fixed without
touching the UI:

- *Drag-and-drop* of a folder can be made to work invisibly, via
  `DataTransferItem.webkitGetAsEntry()` recursion. No appearance change. This
  half is safe.
- The **Browse files** button cannot. `webkitdirectory` turns the native picker
  into a **folder-only** chooser — the same input can no longer multi-select
  individual files. Supporting both from a button means either a second control
  ("Browse files" / "Choose a folder") or a menu on the existing one. Either is
  a visible change to a signed-off screen.

**Options for the product owner:** (a) implement drag-only folder support and
leave the button file-only; (b) add a second affordance; (c) soften the copy to
promise only what the button can do. All three are product calls.

**Not changed.**

---

### E-IMP-03 — no way to paste a list of identifiers

**What.** The demo's import is file-only: dropzone, **Browse files**, and a
*"Or try a sample refs.bib"* demo link. There is no text area.

**Why it matters.** The most common real-world shape of "a DOI list" is a
clipboard, not a file — DOIs copied out of a review's appendix, a spreadsheet
column, or an email. Today the user has to open a text editor, paste, save as
`.txt`, and then upload it. The parser for the pasted case already exists (it
is the same `.txt` branch), so this is purely an input-affordance gap.

**Why it is an escalation.** Adding a paste target means adding a control to
the dropzone — an appearance change on the sole source of visual truth.

**Not changed.** `import-fixtures/lists/` covers the pasted shapes anyway
(`edge-word-paste.txt` with smart quotes, bullets and en-dashes;
`edge-single-line-commas.txt`), so the decision is testable either way.

---

### E-IMP-04 — one error string cannot carry a real parser's failure modes

**What.** The demo has exactly one per-file error message:
*"Unsupported format: use .bib, .ris, .csv, a DOI list, or PDFs."* It fires for
every non-accepted extension and for nothing else.

A real import backend produces at minimum these distinct, actionable failures
(each one is a fixture in `import-fixtures/`):

| Failure | Fixture |
| --- | --- |
| empty file | `bibtex/bad-empty.bib` |
| parsed fine, no records in it | `bibtex/bad-no-entries.bib` |
| malformed / unparseable | `bibtex/bad-unbalanced-braces.bib` |
| unreadable text encoding | `bibtex/edge-utf16le.bib` |
| a web page saved under a citation extension | `ris/bad-html-page.ris` |
| no identifier or title column found | `csv/bad-no-recognisable-columns.csv` |
| valid JSON, but not a bibliography | `json/bad-unrelated-schema.json` |
| not actually a PDF | `pdf/bad-not-a-pdf.pdf` |
| archive too large / unsafe member | `archives/edge-high-compression.zip`, `archives/bad-path-traversal.zip` |
| wrong tool's export — *"export as RIS instead"* | `vendor/bad-endnote-native.enw` |
| over a size / count ceiling | `lists/edge-huge-20k-dois.txt` |

**Why it is an escalation.** All of these are new user-visible copy on a
signed-off screen, and § *Copy tone* in `design-system.md` governs how they
must read. Writing them is a copy/design job, not a wiring job.

**Not changed.** `import-fixtures/README.md` lists a suggested message per
fixture — those are **proposals for review**, not approved copy.

---

### E-IMP-05 — unmatched entries are unaddable in the UI, but the engine supports title-only seeds

**What.** design-system.md § *Import papers* is explicit that a **No match**
row is *"not addable"* — reported by row and counted in the footer, never
added. `_impAdd()` enforces this (`filter(e => e.checked && e.state === 'ok')`).

Meanwhile the backend already supports exactly this case:
`web/live/backend/config_translate.py::_seed_entries` emits `{title: …}` seed
entries for rows without a real id and prepends a `ResolveSeeds` step, and
`src/citeclaw/steps/resolve_seeds.py` resolves them at run time — with a
sibling walk that catches preprint↔published pairs the import-time single match
would miss.

**The tension.** A user importing their own 40-paper reading list, where 3
entries are a workshop paper, a thesis and a very recent preprint that S2's
match endpoint misses, currently loses those 3 with no recourse — even though
the pipeline could resolve them later, or run without them and pick them up
through expansion.

**Why it is an escalation.** Allowing an unmatched row to be added would add a
control to the *Couldn't match* group and change what that group means. That is
a behaviour change to a signed-off screen.

**Not changed.** Flagging it because the capability gap runs the *opposite*
way from usual — the UI is stricter than the backend, and that is easy to
mistake for a bug during wiring.

## Scaffold lane (`E-SCAF-*`) — 2026-08-13

Found while standing up `web/app/` (Vite + React 18 + TS). Nothing below was
changed.

---

### E-SCAF-01 — `explorations-tokens.css` is not what the iPad Demo renders

**What.** `explorations-tokens.css` (56 kB, the design workspace's token/component
export) is **not loaded by the demo at all**. The demo carries a `<style>` block
inside each screen's `<helmet>`, and support.js's helmet manager
(`createHelmetManager` → `compile`) appends all seven of them into the one shared
document `<head>`. Measured line-level overlap: 411 of the tokens file's 450
non-blank lines also appear in at least one demo screen — so it is a good
baseline, but the two are **not identical**:

| | demo screens (all 7) | `explorations-tokens.css` |
| --- | --- | --- |
| root box | `html,body{ height:100%; margin:0; background:#ece5d8; }` | `body{ margin:0; background:#ece5d8; }` — no `height:100%` |
| `.pc-root` opening | `--pin:16px;--pin-sc:calc(var(--pin) - var(--kl-sbw, 8px));…` | neither variable exists |
| form controls | `.pc-root button, .pc-root input, .pc-root textarea, .pc-root select{ font-family:inherit; }` | absent |
| `Fern & ink` / `Fern & ochre` | 4 declarations | ~20 declarations (full `--v-*` set) |

**Not changed.** Both files are shipped verbatim and ordered so the demo wins:
`main.tsx` imports `explorations-tokens.css` then `ipad-demo-shell.css` before any
route, and per-screen CSS (which rewrite agents bring over) is imported from the
screen component, i.e. after both. Every demo screen's `<style>` contains a
*complete* `.pc-root` token block, so a screen that brings its own CSS is
self-sufficient and the divergences resolve in the demo's favour automatically.

**Question for the product owner / design.** Which file is canonical going
forward? If `explorations-tokens.css` is the design system's shared artefact,
it needs to be regenerated from the demo, or it will keep drifting and every
future agent will have to rediscover this table.

---

### E-SCAF-02 — three colour variants can never apply (latent, HTML-entity bug in the demo)

**What.** `<style>` is an HTML *raw text* element, so entities inside it are not
decoded. The demo's screen styles contain
`.pc-root[data-color="Oat &amp; ink"]{…}` and the same for `Fern &amp; ink` and
`Fern &amp; ochre`. Verified in a browser: the CSSOM selector keeps the literal
`&amp;`, while `data-color="Oat &amp; ink"` in *attribute* position decodes
normally to `Oat & ink`. The selector therefore never matches the attribute it
was written for. (The four variants without an ampersand — `Terracotta dusk`,
`Sage library`, `Slate archive`, `Fern library`, `Previous` — are unaffected.)

**Why it is latent, not live.** The locked configuration uses the base
`.pc-root` palette (`scheme: 'Warm paper'`, no `data-color`), so nothing visible
changes today.

**Why it is an escalation and not a bug fix.** Correcting the entity would make
three palettes start applying that currently do not. That is a visible change to
signed-off screens, and it is also a decision about whether those palettes are
part of the product at all.

---

### E-SCAF-03 — the four font stylesheets are not consolidatable

**What.** The demo's screens link four *different* Google Fonts stylesheets.
Two declare Hanken Grotesk as a **variable** face (`font-weight: 400 700`); two
declare **fixed** faces (`400`, `500`, `600`). Those latter screens nevertheless
ask for `font-weight:700` — Runs 38×, Explore 42×, Build 13×. They only render a
true 700 because the demo hoists every screen's `<link>` into one head, so the
variable face from a *sibling* screen's stylesheet is what satisfies the request.

**Consequence.** A screen served on its own with only its own font link renders
those 700s clamped to 600. Any later "let's just load one font stylesheet"
cleanup silently changes rendered weights on three screens.

**Not changed.** `web/app/index.html` reproduces the union, in the demo's mount
order; the per-screen transcript is in
`web/app/src/design-fonts/manifest.ts`.

**Question.** Is 700-on-Hanken intended, or is it a demo accident that happens to
look right? It changes what a self-hosted subset must contain.

---

### E-SCAF-04 — user-visible copy still calls the product a demo

**What.** The shell's rotate invite reads: *"This demo is laid out for the iPad
held wide. Turn the device and it picks up where you left off."* It is real
user-facing copy on a state users will hit (any frame under 1160 px).

**Not changed.** Transcribed verbatim into
`web/app/src/components/DemoViewport.tsx`.

**Question.** Reword for the product? § *Copy tone* in `design-system.md`
governs.

---

### E-SCAF-05 — the viewport meta disables pinch-zoom

**What.** The demo shell ships
`<meta name="viewport" content="… maximum-scale=1, user-scalable=no …">`, which
blocks pinch-zoom on iPad and iPhone. Copied verbatim into
`web/app/index.html` because it changes touch behaviour and is therefore part of
the parity baseline.

**Not changed.** Flagged because it is an accessibility regression that a
reviewer would otherwise attribute to the rewrite rather than to the demo.
