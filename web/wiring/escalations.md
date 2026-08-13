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
