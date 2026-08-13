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

---

## Dissection lane (`E-DIS-*`) — 2026-08-13

Found while producing `demo-architecture.md`, `build-page-spec.md`,
`component-duplication.md` and `state-inventory.md`. Nothing below was changed.

`component-duplication.md` uses short local ids for the duplication findings.
Mapping:

| duplication doc | here |
| --- | --- |
| `D-01` project switcher | `E-DIS-01` |
| `D-02` download menu | `E-DIS-02` |
| `D-06` `.tb-dl-run` / `rid()` | `E-DIS-03` |
| `D-07` `sb-*` vs `rp-*` | `E-DIS-04` |
| `D-08` abstract drill typography | `E-DIS-05` |
| — (found in `build-page-spec.md` §9) | `E-DIS-06` `Run pipeline` |
| §6 second sidebar | `E-DIS-07` |
| §5 canvas chrome | `E-DIS-08` |
| `D-03` / `D-04` / `D-05` | cosmetic — listed at the end, not escalated |

### E-DIS-01 — the project switcher lists a different set of projects on Explore

**What.** The six-row project menu is copy-pasted into three screens, and the
copies disagree.

| Slot | Build / Runs | Explore |
| --- | --- | --- |
| 1 | `AI agents for scientific discovery` **(current)** | `Bounded gaps between primes` |
| 2 | `Evolutionary multi-objective optimization` | `AI agents for scientific discovery` **(current)** |
| 3–6 | Hyper-parameter optimization / Combinatorial optimization / Search-based software engineering / Meta science | identical |

Sources: `Paper Card.dc.html:1359–1435`, `Runs.dc.html:857–933`,
`Explore.dc.html:306–382`.

`Bounded gaps between primes` appears nowhere else in the app.
`Evolutionary multi-objective optimization` — the project whose corpus Explore
actually renders (`topic-data.js`, the 500-paper MOEA/D sample) — is missing
from Explore's own list.

**Why it is an escalation.** The rewrite has one project list; whichever we
pick, one screen visibly changes. Note also that `CLAUDE.md` / `design-system.md`
state *"Explore's top-bar project reads 'Evolutionary multi-objective
optimization'"*, but the file statically renders `AI agents for scientific
discovery` in `.tb-proj-name` (`Explore.dc.html:303`) with no runtime override —
the written spec and the rendered truth already disagree.

**Not changed.** Decision needed: canonical project list, and which project
Explore is "in".

---

### E-DIS-02 — the Download menu is a different component on Explore

**What.** `Paper Card.dc.html:1448–1474` and `Runs.dc.html:946–972` are
byte-identical. `Explore.dc.html:395–424` is a different component.

| Aspect | Build / Runs | Explore |
| --- | --- | --- |
| header | `Download · Run 6 · yesterday`, classed `.tb-dl-head` / `.tb-dl-sep` / `.tb-dl-run` / `.tb-dl-date` | `Download  Run 37  v3  what Explore reads`, **unclassed** (nothing can update it) |
| items | Accepted papers (`CSV · 214 papers`), Full bundle (`CSVs · BibTeX · graph · stats`) | Corpus papers (`CSV  354 papers`), **Literature review** (`Markdown  the current draft`), Full bundle (`CSVs  BibTeX  graph  topics`) |
| separators | literal `·`-style separators | `&nbsp;&nbsp;` |
| empty state | `.tb-dl-none` — *"No runs yet / Results download per run. Start one with Run pipeline."* | **none** |

**Why it is an escalation.** The item list differing by scope is defensible, but
two things are not: the separator style contradicts § *Forbidden typography*
(no `·` separators) — one of the two copies is wrong — and only Build/Runs have
an empty state, so Explore's menu has no defined appearance before a corpus
exists.

**Not changed.** Decision needed: one separator convention; whether Explore
gets an empty state.

---

### E-DIS-03 — `.tb-dl-run` is built two different ways, and Build's copy has a bug

**What.**

- `Paper Card.dc.html:5060` — `String(run.id).replace(/^RUN-0*/i, 'Run ')`
- `Runs.dc.html:2705` — `this.rid(run.id)`, where
  `Runs.dc.html:3849` is `rid(x) { return String(x ?? '').replace(/^RUN[-‑]0*/i, 'Run '); }`

Build hand-rolled a copy of `rid()` that omits the non-breaking hyphen `U+2011`.
A run id spelled `RUN‑0006` renders as `Run 6` on Runs and as the raw
`RUN‑0006` on Build.

**Why it is an escalation rather than a silent fix.** It changes what Build's
download menu renders, and it contradicts the documented rule *"run ids display
as `Run 37` via `rid()`"*.

**Not changed.** Recommendation: one shared `rid()`.

---

### E-DIS-04 — the papers panel exists under two class prefixes

**What.** The same component is implemented twice with disjoint class names and
disjoint JS:

- Build, `sb-*`: header `Paper Card.dc.html:1553–1585`, filter drawer
  `1514–1551`, list `1587–1791`, wiring `wireSidebar` (2684–3081).
- Runs / Explore, `rp-*`: `Runs.dc.html:1019–1122`, `Explore.dc.html:462–585`.

They have diverged in capability: `sb-*` carries the `Search | Import` seg and
the first-run invite; `rp-*` carries the selection bar, pager, drill and change
markers.

**Why it is an escalation.** The rewrite builds one component. Unifying means
one screen gains or loses affordances unless we deliberately preserve both
feature sets — which changes both.

**Not changed.** Decision needed: unify (and accept the union of affordances),
or keep two visually-identical but separate components.

---

### E-DIS-05 — the abstract drill uses different typography on Runs and Explore

**What.** `Runs.dc.html:1123–1150` vs `Explore.dc.html:586–620`, plus a third,
JS-generated copy on Build (`Paper Card.dc.html:3200–3220`).

| Element | Runs (and Build) | Explore |
| --- | --- | --- |
| meta line (year / cites) | `font-size:11.5px` | `font-size:13px` |
| `Abstract` section label | `10.5px; 600; letter-spacing .06em; uppercase; var(--muted2)` — a caps label | `13px; 600; var(--fg)` — a sentence-case heading |
| provenance row | `margin-top:10px` | `margin-top:14px` |

The same drill, opened from the same kind of row, looks different depending on
which screen you opened it from.

**Why it is an escalation.** One component in the rewrite ⇒ one typography.
Either Runs/Build change, or Explore changes.

**Not changed.** Today the demo renders 13 px + sentence-case on Explore, and
11.5 px + caps label on Runs and Build.

---

### E-DIS-06 — `Run pipeline`, the Build screen's primary CTA, has no handler

**What.** `.tb-run` is declared at `Paper Card.dc.html:1475–1478`. Grepping
`tb-run` across the whole file returns **only that declaration** — no listener,
no `querySelector`. The button is wired only on Runs (`Runs.dc.html:1852`,
`2012`, `2042`). Explore carries the same dead button at `Explore.dc.html:425`.

Build's own download empty state points at it: *"Start one with Run pipeline."*

**Why it is an escalation.** Wiring has to give the most prominent button on the
pilot screen a behaviour, and the demo never designed one. Candidates — start a
run and navigate to Runs; start it in place with a status strip; open a
pre-flight confirm — are each a new interaction needing its own busy, disabled
and error states.

**Not changed.** Decision needed: what `Run pipeline` does on Build, and whether
it should exist on Explore at all.

---

### E-DIS-07 — the Build sidebar ships a second variant that never renders

**What.** `Paper Card.dc.html` contains two `.sidebar` blocks inside one
`.pc-drawer`: `[data-style="list"]` at 1513–1792 and `[data-style="cards"]` at
1793–2075. A full diff is **7 changed lines** across 280 — the entire filter
drawer, panel header, search field and filter/sort row (1514–1585 vs 1794–1865)
are byte-identical.

The shell pins `layout="List"`, so the `cards` copy is `display:none` forever —
yet `setupSidebar` iterates every `.sidebar` (2528), so all listeners,
ResizeObservers and scrims are installed twice.

**Why it is a question rather than an obvious delete.** `layout` is a declared
prop with a `Cards` option, and the locked-variant list does not mention it.
Dropping the variant is a scope decision.

**Not changed.** Recommendation: do not port `cards`.

---

### E-DIS-08 — two mutually-exclusive buttons share one absolute position on Explore

**What.** `Explore.dc.html:850` (`.nv-hood`) and `Explore.dc.html:856`
(`.cm-sub`, *View community subgraph*) are both
`position:absolute; left:14px; bottom:12px; z-index:20`.

They never overlap only because the JS guarantees at most one is visible.

**Why it is here.** Flagged so nobody "fixes" the apparent collision by moving
one of them during the rewrite — that would change the layout. The exclusivity
is the contract.

**Not changed.**

---

### Cosmetic copy-paste drift — recorded, not escalated

No product decision needed; listed so a reviewer does not mistake them for
rewrite defects.

- `<span class="tt-label" style="display:none">` exists on Build
  (`Paper Card.dc.html:1482`) and Runs (`Runs.dc.html:984`), is written by
  `applyTheme` (3416 / 1935), is never displayed, and is absent from
  Explore and Home.
- `klFsRow` is duplicated four times; Explore's copy differs only by renaming a
  local `const on` to `const on2` (`Explore.dc.html:3790–3812`).
- `rp-pick-label` is present on Runs' filter drawer and absent from Explore's,
  so Explore's JS cannot target that label.
- `.nv-hood` carries `z-index:20` on Explore and none on Runs.
- `.tb-download` sits one indentation level shallower on Explore.

---

## Parity lane (`E-PAR-*`) — 2026-08-13

Found while building the pixel-parity harness (`web/parity`) and capturing the
Build screen across the responsive sweep. Nothing was changed.

### E-PAR-01 — the rotate gate is 1024 px wide inside a 900 px viewport

**What.** At the `w900-900x1200` sweep viewport the `.rot` gate's box measures
**1024 × 1200** in a 900-wide viewport (measured via `getBoundingClientRect`;
the same element measures 1024 × 1024 at the 768-wide viewport). The gate's
content is centred on the 1024 px box, not the viewport, so the icon, the
"Rotate to landscape" heading and the body copy all sit visibly **right of
centre**, and the document overflows horizontally. Visible in the committed
baseline `web/parity/baseline/design-demo/build/w900-900x1200.png`.

**Why it is here.** Correcting it means changing what the gate looks like at
sub-1024 widths, which the hard rule forbids me from doing unilaterally. It also
cannot be resolved by "just reproduce the demo" without a decision, because the
rewrite has to choose deliberately between:

1. reproducing the off-centre, overflowing gate byte-for-byte (parity gate
   passes, the artefact ships), or
2. centring the gate on the viewport (better looking, but a deliberate,
   recorded divergence from the source of visual truth).

**Needs from the product owner.** Which of the two. If (2), the baseline PNGs for
`w900-900x1200` and `w768-768x1024` must be re-approved, since they are the
reference the rewrite is gated against.

**Not changed.**

### E-PAR-02 — the demo's landscape/portrait switch is the only responsive rule below 1194 px

**What.** Related to E-PAR-01 but distinct: the sweep shows the app renders at
1600×900, 1366×1024 and 1194×834, and is entirely replaced by the rotate gate at
1024×1366, 900×1200 and 768×1024. There is no intermediate folded layout
anywhere. A narrow *landscape* window (e.g. 900×600) is undesigned territory —
it is not portrait, so the gate's rule may or may not catch it depending on how
the rewrite implements the condition.

**Why it is here.** "Responsive layout behaviour of the demo MUST be preserved"
is an explicit product requirement, but preserving it requires knowing whether
the gate's trigger is orientation, aspect ratio, or a width threshold — the
three agree on every viewport captured so far and disagree on narrow landscape.
Picking one is a behaviour decision, not an implementation detail.

**Needs from the product owner.** The intended rule (orientation vs aspect ratio
vs min-width), and what a narrow landscape window should show.

**Not changed.** The harness captures the observed behaviour as-is; see
`web/wiring/missing-states.md` **MS-PAR-03**.

---

## Build rewrite lane (`E-BLD-*`) — 2026-08-13

Raised while building the static Build rewrite (`web/app/src/screens/build`).
Every one of these was found by measurement against the demo, not by reading.
Nothing in the UI was changed to work around any of them.

### E-BLD-01 — Build's appearance depends on the **Runs** stylesheet

**What.** The demo mounts all seven screens into one document, and support.js's
helmet manager hoists each screen's `<style>` into the single `<head>` in mount
order: `Login, Home, Build, Runs, Explore, Settings, System Banners`. Every one
of those blocks is a complete, unscoped stylesheet. So on the **Build** screen,
rules from **Runs** (and Explore, Settings, Banners) load *after* Build's own
and win at equal specificity.

Measured, at 1600×900, on the Build screen's five filter rows:

| selector | Build's own rule (03) | Runs' rule (04) | what the demo actually renders |
|---|---|---|---|
| `.cf-filter` | radius 10px, no bottom border | adds `border-bottom: 1px solid …` | **Runs'** — every row is 57px, not 56px |
| `.cf-rail` | `top:7px; bottom:7px` (42px rail) | `top:0; bottom:0` (56px rail) | **Runs'** — 56px |

With only Build's stylesheet loaded, the rewrite was 2 693 px (0.19% of the
frame) away from the baseline. With all seven loaded in the demo's order it is
byte-identical.

**Why this needs the product owner, not us.** Two things follow, and both are
product decisions:

1. **No screen's CSS can be code-split.** Loading Build alone gives a different
   Build than loading Build after having visited Runs. Any lazy per-route CSS —
   the default in every bundler — introduces a visual difference that depends on
   navigation history. `web/app` therefore loads all seven verbatim blocks on
   every screen (`src/styles/demo-screens.css`), which is faithful but ships
   ~300 kB of CSS to render one screen.
2. **Which `.cf-filter` is canonical?** The demo shows Runs' version everywhere
   because of load order, not because anyone chose it. `component-duplication.md`
   D-07/`E-DIS-04` already asks this question for the papers panel's `sb-*` vs
   `rp-*` prefixes; this is the same question one layer down, for rules that
   silently override each other rather than living in separate components.

**Needs from the product owner.** Confirmation that the *rendered* Build screen
(i.e. with Runs' overrides applied) is the intended design — the product owner's
"what I see in THIS file is what I want" reading says yes, and that is what has
been built. Then a decision on whether the eventual single token/component sheet
should bake Runs' values in, or Build's.

**Not changed.** The rewrite reproduces the demo's whole CSS environment, in the
demo's order, verbatim.

### E-BLD-02 — `explorations-tokens.css` invents rules the demo does not have

**What.** `web/app/src/styles/explorations-tokens.css` is a verbatim copy of a
file in the design workspace that **no demo screen loads** (already noted in
`E-SCAF-01`, left undecided). The scaffold imported it from `main.tsx` as a
baseline. Measured consequence on the Build pilot:

```
tokens file : .cfg-pipe[data-number="Column"] .cf-idx-col { font-family: ui-monospace, monospace; … }
              .cf-num { font-family: ui-monospace, monospace; … }
demo (03/04): the same two rules, WITHOUT font-family
```

Because no demo stylesheet declares `font-family` on those elements, nothing
overrode the tokens file and the config panel's five row numbers rendered in a
monospace face the design never used — 51 wrong pixels, invisible to review,
caught only by the harness. There will be more of these on the screens still to
be rewritten; the file is 450 lines of near-miss.

**Action taken.** `main.tsx` no longer imports it. The file is left in place
because it may yet be regenerated *from* the demo, but nothing references it.

**Needs from the product owner / design lane.** A ruling on the file's status:
regenerate it from the demo and make it the one token sheet, or retire it. Until
then no screen may import it. This supersedes the open question in `E-SCAF-01`.

### E-BLD-03 — `.sidebar` overflows itself by 14px, and the overflow is reachable

**What.** In the demo *and* in the rewrite, the papers sidebar has
`scrollWidth 372` against `clientWidth 358` — a 14px horizontal overflow inside
`overflow:hidden`, caused by `.sb-sort-menu` (`position:absolute; right:0`)
poking past the panel edge. Nothing shows it at rest, but anything that calls
`scrollIntoView` on an element inside the menu — keyboard focus, a screen
reader, browser autoscroll, test automation — scrolls the panel 14px left and
there is no scrollbar or gesture to bring it back. Reproduced with Playwright's
own actionability scroll; the panel then renders 14px off for the rest of the
session.

**Not changed.** Fixing it means changing the sort menu's geometry.

**Needs from the product owner.** Whether the menu is allowed to extend past
the panel (and the panel should therefore not be `overflow:hidden`), or the menu
should be clamped inside it.

### E-BLD-04 — the dead `cards` sidebar was transplanted anyway

`component-duplication.md` §6 recommends **not** porting `.sidebar[data-style="cards"]`
(283 lines that never render because `layout="List"` is pinned), and it is
already open as `E-DIS-07`. The rewrite ported it, because this pass's rule is
"markup is transplanted, not authored" and dropping it is an editorial decision
about the design, not a mechanical conversion. It costs a second copy of every
listener, ResizeObserver and scrim — exactly as in the demo — and zero pixels.

**Needs from the product owner.** The `E-DIS-07` answer. If the `cards` variant
is dead, deleting it is a one-line change to `BuildSidebar.tsx` plus a re-run of
the parity gate (which should stay green).
