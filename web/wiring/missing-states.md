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
