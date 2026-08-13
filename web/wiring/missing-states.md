# Missing states register

Places where the demo designed the happy path only — no error, loading, empty,
partial or offline state. Silent failure is a known product concern, so these
are recorded rather than invented.

**Rules for this file.** Append, never rewrite. Date every entry. Namespace ids
by lane (`MS-IMP-nn` = import lane) so concurrent lanes cannot collide. Nothing
here has been implemented or designed — each entry is a question for the design
agent / product owner.

---

## Import lane (`MS-IMP-*`) — 2026-08-13

Surfaces audited: Home wizard step 2 (`.hw-idrop` / `.hw-iparse` / `.hw-irev`),
Build seed sidebar (`.sbi-*`), Runs Add-papers panel (`.ra-*`), and the shared
engine `import-resolver.js` (`fileRowHtml`, `groups`, `rowHtml`, `groupHtml`).
Fixtures that hit each state are named; see `import-fixtures/README.md`.

### MS-IMP-01 — zero results (empty file, empty archive, header-only CSV)

The flow always advances: `_impGo` settles every file row, then unconditionally
calls `_impRev()` + `_impState('review')`. `KLImport.groups()` filters out
groups with no rows, so a zero-entry parse renders the review card as an
**empty box** with a disabled `Add 0 to seed set` footer and no explanation.

- No "this file had nothing in it" state.
- No distinction between *empty file* (0 bytes), *no records* (preamble only),
  and *all files rejected* (see MS-IMP-11).
- **Hit by:** `bibtex/bad-empty.bib`, `bibtex/bad-no-entries.bib`,
  `csv/bad-header-only.csv`, `csv/bad-empty.csv`, `ris/bad-empty.ris`,
  `lists/bad-empty.txt`, `lists/bad-whitespace-only.txt`, `json/bad-empty.json`,
  `pdf/bad-empty.pdf`, `archives/bad-no-members.zip`.

### MS-IMP-02 — no determinate progress for a large import

`fileRowHtml` emits one indeterminate spinner reading `Parsing…`, and `_impGo`
settles files sequentially on a random 340–620 ms timer. That reads correctly
for three files. For a 3 000-entry `.bib` or a 20 000-line DOI list the real
work is thousands of S2 round-trips, and there is no design for:

- determinate progress (`resolved N of M`),
- a per-file byte/entry counter while a single large file is still parsing,
- the fact that resolution, not parsing, dominates the wait.

- **Hit by:** `bibtex/edge-huge-3000.bib`, `lists/edge-huge-20k-dois.txt`,
  `archives/ok-mixed-formats.zip`.

### MS-IMP-03 — no Semantic Scholar failure state inside import

Every entry is resolved on Semantic Scholar, and the existing backend already
raises three distinct user-facing failures (`web/live/backend/s2_seeds.py`):
rate-limited-with-key, rate-limited-keyless ("Add a Semantic Scholar key in
Settings"), and key-rejected (401/403). Plus plain network loss.

The import flow has **nowhere to show any of them**. The mock resolves locally
and deterministically, so the states never had to exist. Today a resolution
failure would either hang the spinner forever or, worse, mark every entry
`No match` — which the UI presents as *"Title not found on Semantic Scholar"*,
a factually wrong message for a 429.

- Note the design system already has a home for connection state — the
  status-strip slot under the top bar (`.bn-sb`, § *Resilience & status
  system*). Whether import should use it or show something local is a design
  question, not an implementation one.
- **Hit by:** every fixture, once the backend is real.

### MS-IMP-04 — no ceilings and no ceiling messages

No design exists for max file size, max files per import, max entries per
import, or max uncompressed archive size — nor for what the UI says when one is
hit. The only per-file error string in the whole demo is
`Unsupported format: use .bib, .ris, .csv, a DOI list, or PDFs`.

- **Hit by:** `bibtex/edge-huge-3000.bib`, `lists/edge-huge-20k-dois.txt`,
  `archives/edge-high-compression.zip`, `adversarial/edge-single-long-line.txt`.

### MS-IMP-05 — duplicates *within* the upload have no state

The `dupe` state means one thing only: *already in the corpus / in your seed
set*. There is no state for the same paper appearing twice **inside the import**
— two files exporting the same library, one file with a repeated citekey, or a
DOI written three different ways in one list.

Options are all design decisions: collapse silently, collapse with a count on
the row, or show a fourth group.

- **Hit by:** `bibtex/edge-duplicate-keys.bib`,
  `lists/edge-duplicate-identifiers.txt`, `archives/ok-mixed-formats.zip`
  (the `.bib` and `.ris` members overlap).

### MS-IMP-06 — no text-encoding failure state

Real exports arrive as cp1252, UTF-16, or UTF-8 with stray legacy bytes. The
demo never reads bytes, so there is no state for "we could not decode this
file", and no state for "we decoded it lossily — some characters may be wrong",
which is the more dangerous case because author names silently corrupt.

- **Hit by:** `bibtex/edge-latin1.bib`, `bibtex/edge-utf16le.bib`,
  `ris/edge-mixed-encoding.ris`, `bibtex/bad-binary.bib`.

### MS-IMP-07 — no unreadable-PDF state

Three distinct PDF failures, one missing state each: password-protected,
no text layer (a scan), and structurally corrupt. The demo's only PDF-specific
copy is the per-entry reason `No DOI in the PDF: title match below threshold`,
which is a *resolution* outcome, not a *readability* one.

- **Hit by:** `pdf/edge-no-text-layer.pdf`, `pdf/bad-truncated.pdf`,
  `pdf/bad-not-a-pdf.pdf`. Password-protected is not in the corpus (needs a
  crypto-capable writer) but belongs to the same gap.

### MS-IMP-08 — no cancel, and no abandon-mid-parse state

`_impGo` guards re-entry with a generation counter (`this._impGen`), so a
second drop supersedes the first — but there is no user-facing **Cancel**, no
confirmation when navigating away mid-import, and no state for "you left; we
stopped". With a real backend a cancelled import also has to clean up its
staging directory and any in-flight S2 work.

### MS-IMP-09 — pressing Add discards unresolved *Ambiguous* rows silently

`_impAdd()` takes `entries.filter(e => e.checked && e.state === 'ok')`, then
sets `this._impRes = null`. The success note reads:

> **N papers imported**, counted in your seed set. *M* unmatched entr(y|ies)
> (was|were) skipped.

`M` counts only `state === 'none'`. Rows still sitting in **Needs a decision**
(`state === 'multi'`) are neither added nor mentioned — they simply vanish with
the discarded result. The design's own promise is that nothing is dropped
silently; this is the one place the demo does it.

Design question: block Add while ambiguous rows remain, count them in the note,
or confirm before discarding.

- **Hit by:** `bibtex/edge-no-doi.bib`, `ris/partial-missing-doi.ris`.

### MS-IMP-10 — folder drop is promised but has no implementation or state

Both dropzone copy blocks promise folders ("a folder of PDFs", "folders and
.zip work too"). In the demo:

- the file input is `<input type="file" multiple>` — **no `webkitdirectory`**;
- the drop handler reads `e.dataTransfer.files` only — no
  `webkitGetAsEntry()` recursion, so a dropped folder yields nothing.

So a user who follows the copy gets **silence**: no file rows, no error, the
dropzone just stays as it is. See also `escalations.md` **E-IMP-02** — fixing
this may need a second control, which is a UI change.

### MS-IMP-11 — "every file failed" opens an empty review card

`_impGo` calls `_impState('review')` after the last file settles regardless of
whether any file succeeded. Drop three `.docx` files and the flow leaves the
per-file error rows behind and lands on a blank review card with
`Add 0 to seed set`. The errors the user needs are on the screen they just
left.

- **Hit by:** `adversarial/bad-unsupported.docx`, `.pages`, `.md`,
  `vendor/bad-endnote-native.enw`, `vendor/bad-endnote-xml.xml`,
  `vendor/bad-zotero-rdf.rdf`.

### MS-IMP-12 — the Add step has a success state and no failure state

`_impAdd()` disables the button, shows `Adding N…`, waits 700 ms on a
`setTimeout`, and unconditionally shows the green success note. There is no
error branch: with a real backend, a failed or partially-failed add has no
design. Nor is there a state for a partial add (12 of 15 accepted).

---

## Scaffold lane (`MS-SCAF-*`) — 2026-08-13

Noticed while standing up `web/app/`.

### MS-SCAF-01 — fonts come from a remote CDN and there is no state for them not arriving

Every screen loads its typefaces from `fonts.googleapis.com` /
`fonts.gstatic.com` with `display=swap`. The demo has no offline, blocked or
slow-CDN state.

This one is sharper than it looks, because the agreed deployment for this phase
is **local only** (decisions ledger Q16). On a machine without internet — or
behind a network that blocks Google Fonts, which is the normal case in parts of
the world this will run in — the entire UI silently falls back to
`-apple-system` and a default serif. Nothing is broken enough to notice at a
glance; the layout just quietly stops matching the design, because Newsreader
and Hanken Grotesk have different metrics from the fallbacks.

There is no design for: fonts still loading, fonts failed, running with
fallbacks. Related hardening item: self-hosting the two families, in
`hardening-todo.md` (H-SCAF-01).

### MS-SCAF-02 — no state for "the screen itself is still arriving"

The demo preloads all seven screens into one document, so switching screens is
instant and no loading state was ever needed. The rewrite necessarily
code-splits per screen (and, once wired, waits on the backend for the data the
screen renders). There is no designed state for the gap between "user tapped
Runs" and "Runs can paint" — neither a skeleton, nor a spinner, nor a rule that
says the old screen stays until the new one is ready.

The scaffold currently renders `null` during that gap, which is a placeholder,
not a decision.

### MS-SCAF-03 — nothing is designed below the shell's minimum width

The shell root is `min-width:1024px`, and the rotate invite appears when the
root's own width drops under 1160 px. Between 1024 and 1160 the invite covers
the screen, which is the designed behaviour. **Below** 1024 px the root stops
shrinking and the page scrolls horizontally instead — the invite is still
displayed, but it is now inside a box wider than the window, so it is
off-centre and can be scrolled away from.

There is no phone layout and no designed narrow state; the demo simply assumes
the frame is at least iPad-wide. Worth a decision before anyone opens this on a
phone.

## Dissection lane (`MS-DIS-*`) — 2026-08-13

Surfaces audited: the Build screen end to end (`Paper Card.dc.html`), plus the
cross-screen chrome (top bar, page shell, Runs replay driver, Login). Companion
documents: `demo-architecture.md`, `build-page-spec.md`, `state-inventory.md`.

### MS-DIS-01 — the search error state exists but is unreachable

`.sb-error` ("Couldn't load papers" / *"Something went wrong on our end. Check
your connection and try again."* / **Retry**) is fully designed and present in
the markup twice (`Paper Card.dc.html:1641–1648` and `1921–1928`).

`apply()` renders it only when `sidebar.dataset.state === 'error'`
(`Paper Card.dc.html:2799–2806`). The **only** writer of that value is a
`.sb-demo` state switcher (`3065–3078`) — and `.sb-demo` does not exist in the
shipped template (`grep sb-demo` returns one hit, in the JS). The demo therefore
cannot show a failed Semantic Scholar search at all.

Once the search is wired to a real backend this state becomes reachable, but its
**trigger set is undefined**: no distinction between offline, timeout, 4xx
(bad query / rate-limited) and 5xx, and `Retry` currently just re-applies the
local filter (`3064`) rather than re-issuing the query.

### MS-DIS-02 — `Run pipeline` has no busy, disabled, or failure state on Build

`.tb-run` (`Paper Card.dc.html:1475`) has no handler at all (escalated as
`E-DIS-06`). Beyond the missing behaviour, none of the states a real run-start
needs are designed **on Build**:

- pressed / busy (Runs has `btnBusy('Starting…')`, Build has nothing);
- disabled because the pipeline is invalid, has no seed papers, or has an
  unapplied filter draft;
- start failed (engine unreachable, budget exceeded, key invalid);
- a run is already running for this project.

Runs designed *Starting…* and *Stopping…* hero phase rows; Build has no
equivalent surface.

### MS-DIS-03 — the pipeline canvas and config panel have no loading state

The search list is the only Build surface with a loading treatment
(`.sb-searching`, three skeleton cards, `Paper Card.dc.html:1612–1632`).

The pipeline canvas renders from an in-memory model built synchronously in
`plInit()`, and the config panel renders from the same model, so neither ever
waits. Once both read a real project, there is a designed gap between "project
selected" and "pipeline painted" — no skeleton, no spinner, and no rule about
whether the previous project's pipeline stays on screen.

Same gap for the project switcher: `applyProject` plays a 420 ms fade-through on
`<main>` and then expects the new content to already be there.

### MS-DIS-04 — no failure or partial state for the filter-config editor

`validateEditor` (`Paper Card.dc.html:4550`) covers **field-level** validation
only (`N issues to fix` / `Applied ✓`). There is no state for:

- an LLM filter whose model list cannot be loaded (`llmModels`, 4106, is a
  hard-coded array);
- a cost estimate that cannot be computed (the re-screen flow on Runs
  cost-confirms LLM/keyword edits — Build has no equivalent);
- an apply that is rejected by the backend after passing client validation;
- a config that was valid when opened and became invalid because the pipeline
  changed underneath it.

### MS-DIS-05 — the download menu has only an empty state

`.tb-dl-menu` (`Paper Card.dc.html:1448–1474`) designs exactly two situations:
items available, and `.tb-dl-none` ("No runs yet"). Missing:

- export in progress (these are potentially large CSV/BibTeX/GraphML files);
- export failed;
- a run whose artefacts are incomplete because it was stopped;
- per-item disabled with a reason (e.g. no graph because the run never expanded).

On Explore the empty state does not exist at all (see `E-DIS-02`).

### MS-DIS-06 — no offline or degraded state on Build

The connection set (offline / server-unreachable-with-countdown / restored)
lives entirely in `System Banners.dc.html` and is driven by the shell's
`connection` prop. Build itself has **no** degraded behaviour: the search box
stays enabled, the import dropzone stays enabled, `Run pipeline` looks
identical. Nothing on the page tells the user which actions will fail while
offline, and nothing is disabled.

Related: the `connection` prop is deliberately **excluded** from the hidden
demo-state switcher (a locked decision in `ipad-demo-audit.md`), so these states
were never exercised on the demo device either.

### MS-DIS-07 — the theme has no single source of truth

`syncTheme` (`KnowledgeLab iPad Demo.dc.html:212–233`) mirrors `data-theme`
between the seven mounted `.pc-root` elements with a `MutationObserver` and a
`_themeLock` re-entrancy guard, re-attaching at 400 / 1200 / 2500 ms to catch
late mounts. There is no stored preference, no `prefers-color-scheme` read, and
no defined state for "a page mounted while the theme was dark".

Not an error state, but an undefined one: the rewrite needs a decision on where
theme lives and whether it persists (there is currently no storage key for it —
the only `localStorage` key in the whole app is `kl-filter-groups`).

### MS-DIS-08 — no state for a slow or failed module load

Five of Build's dependencies load asynchronously: `paper-row.js` and
`import-resolver.js` through the helmet `__resources` shim, and the viz engines
through dynamic `import()` on other screens.

`setupSidebar` (`Paper Card.dc.html:2526`) handles *late* by deferring on
`kl-paper-row-ready`, but there is no timeout and no failure path — if
`paper-row.js` never arrives, the sidebar simply stays as the 14 static template
rows with no star lane and no wiring, silently. This is the exact failure the
2026-08-11 bundle regression produced on device (recorded in
`ipad-demo-audit.md` § Demo artifacts), and it produced no visible error.

### MS-DIS-09 — the import flow has no state for "the resolver itself is down"

`import-resolver.js` is a local deterministic mock, so the only failures it can
express are per-entry (`Couldn't match`) and per-file (`Unsupported format`).
Once resolution is a network call there is no designed state for the service
being unavailable, timing out mid-batch, or partially resolving — i.e. the
review screen has no "N of M resolved, retry the rest" shape.

(The import lane's `MS-IMP-*` entries cover the per-file and per-entry gaps in
detail; this entry is specifically about the transport.)

### MS-DIS-10 — leaving a screen mid-flow is undefined

Pages are never unmounted (`state-inventory.md` §1), so every partial state
survives navigation with no designed treatment:

- an open filter-config draft with unsaved edits, left by switching to Runs;
- an import stuck in `review` with unresolved rows, left by switching projects;
- a search in flight (the 850 ms window) when the project changes underneath it;
- a run replaying on Runs while the user edits the pipeline on Build.

There is no "you have unsaved changes" affordance anywhere in the demo, and no
rule for whether a project switch should discard drafts.

---

## Parity lane (`MS-PAR-*`) — 2026-08-13

Surfaces audited: the frozen demo's runtime asset loading and its responsive
folding behaviour, observed while building the pixel-parity harness
(`web/parity`). Found by network tracing and by capturing the Build screen at
six viewports.

### MS-PAR-01 — six CDN scripts have no offline or load-failure state

The bundle is almost self-contained — sibling `.js`/`.dc.html`, the Google Fonts
CSS (woff2 payloads inlined as `data:` URIs), the touch icon and React 18.3.1
UMD are all embedded. But `Paper Card.dc.html`, `Runs.dc.html` and
`Explore.dc.html` still fetch six scripts from public CDNs at runtime:

- `cdnjs …/smooth-scrollbar/8.8.4/smooth-scrollbar.min.js`
- `cdnjs …/smooth-scrollbar/8.8.4/plugins/overscroll.min.js`
- `jsDelivr …/marked@12.0.2/marked.min.js`
- `jsDelivr …/katex@0.16.11/dist/katex.min.js`
- `jsDelivr …/turndown@7.2.0/dist/turndown.js`
- `jsDelivr …/turndown-plugin-gfm@1.0.2/dist/turndown-plugin-gfm.js`

There is no designed state for any of them failing. On a plane, behind a
locked-down network, or during a CDN outage, smooth scrolling, markdown
rendering, LaTeX rendering and markdown export degrade or break, and the only
feedback is a bundler toast that the demo's own guard script deliberately hides
(see MS-PAR-02). Note the design already has a home for connection state — the
status-strip slot under the top bar (`.bn-sb`) — and a `connection` variant for
*backend* reachability, but nothing covers *asset* reachability.

- **Consequence for the rewrite:** `web/app` should bundle these locally rather
  than inherit the CDN dependency. The harness refuses to depend on the live
  internet: it mirrors these six URLs from version-pinned npm copies and aborts
  (and records) any other external request — `web/parity/src/cdn-mirror.mjs`.

### MS-PAR-02 — the `smooth-scrollbar` load race is silently swallowed

`smooth-scrollbar.min.js` and its `overscroll` plugin are two independent
`<script src>` tags with no ordering guarantee. The plugin frequently wins the
race and throws `TypeError: Cannot read properties of undefined (reading
'ScrollbarPlugin')`. The bundle's error hook turns that into a
`[bundle] Script error.` toast, which a `MutationObserver` guard in the demo
template then hides on purpose (documented in `ipad-demo-audit.md` as P2-14, an
"opaque-origin inlining artifact, zero impact" — but this instance is a real
race, not the masked artifact).

Consequence: whether overscroll bounce is present becomes a coin flip per page
load, for a behaviour the demo exposes as a prop (`overscroll-bounce`), and
nothing in the UI says so. Reproduced on every unmirrored load during harness
bring-up; with load order pinned, the demo loads with **zero console errors**.

- **Consequence for the rewrite:** load order must come from a bundler import
  graph, not two parallel script tags. The harness pins the order for capture
  determinism only — that is not a product fix.

### MS-PAR-03 — below landscape, the whole app is replaced by a rotate gate

At 1024×1366, 900×1200 and 768×1024 the demo renders only the full-screen
*"Rotate to landscape / This demo is laid out for the iPad held wide…"* panel
(`.rot`). It is present-but-`display:none` at 1600×900, 1366×1024 and 1194×834.

There is no designed narrow-width or portrait layout at all — not folded, not
degraded — for any screen or state. There is also no designed state for a
landscape window that is merely *narrow* (e.g. 900×600), as distinct from
portrait.

- This is likely a deliberate product decision to reproduce rather than a gap to
  fill in silently, but it should be a conscious one.
- The gate is part of the committed baseline:
  `web/parity/baseline/design-demo/build/{ipad-portrait-1024x1366,w900-900x1200,w768-768x1024}.png`.
- Related appearance question raised in `escalations.md` (**E-PAR-01**).
