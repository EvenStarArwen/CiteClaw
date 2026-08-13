# Import format matrix — proposal

**Status:** proposal for the orchestrator / product owner. Nothing in the UI
was changed to produce this document.
**Date:** 2026-08-13
**Companion:** `import-fixtures/README.md` (the test corpus), `missing-states.md`,
`escalations.md`, `design-lane-worklist.md`.

---

## 0. What the new UI promises

Sources of truth read for this document:

| Source | Path |
| --- | --- |
| Design spec | `ui_design/design-system.md` § *Import papers (a resolver, not a file manager)* (lines 503–512), § *Compact paper cards*, § *PDF availability* |
| Shared import engine | `ui_design/import-resolver.js` |
| Rendered demo | `ui_design/KnowledgeLab iPad Demo.html` |

### 0.1 `import-resolver.js` is paper import — confirmed, not assumed

The filename could plausibly have meant a design-tool module resolver. It does
not. Evidence:

1. Its own header: *"KnowledgeLab's shared bulk-import mock (see
   design-system.md § Import papers) … One flow, many parsers: extract
   references → resolve on Semantic Scholar → match-review → add."*
2. Its public surface is `window.KLImport = { parse, sample, groups, rowHtml,
   groupHtml, candPopHtml, fileRowHtml, extOf }` — all bibliographic.
3. `var EXT_N = { bib:[24,12], ris:[16,10], csv:[10,6], txt:[5,4], zip:[5,4],
   pdf:[1,0] }` — the accepted extension set and the mock's per-file entry
   yield.
4. The iPad demo **bundles this exact file**. The demo is a self-extracting
   bundle (base64 + gzip assets behind a `__bundler/manifest` script tag); its
   `__bundler/ext_resources` list registers `{"id":"importresolver", …}`, and
   the unpacked asset is **byte-identical** to `ui_design/import-resolver.js`.
   Home, Build and Runs each load it with
   `[['paperrow','./paper-row.js'],['importresolver','./import-resolver.js']]`.

So it is both the spec's reference implementation and the demo's live engine.

### 0.2 The three import surfaces in the rendered demo

| Surface | Toggle | Dropzone class | Headline | Footer button |
| --- | --- | --- | --- | --- |
| Home wizard, step 2 *Anchor with seed papers* | `.hm-fseg.hw-iseg` — `Search \| Import` | `.hw-idrop` | "Drop the collection you already keep" | `Add N to seed set` |
| Build, seed search sidebar | `.fc-seg.sb-imode` — `Search \| Import` | `.sbi-*` | "Drop your reference files" | `Add N to seed set` |
| Runs, Add-papers panel | inside `.rf-stash` | `.ra-drop` | "Drop your reference files" | `Add N papers` |

All three share `KLImport` for parsing and for the review-row / group markup,
so the format contract must be **one** contract.

### 0.3 The promise, in the demo's own words

Home dropzone body copy (verbatim):

> BibTeX or RIS exports (Zotero, EndNote, Mendeley), CSV, a DOI list, or a
> folder of PDFs, .zip works too. Every entry is matched on Semantic Scholar;
> anything unmatched is reported, never added silently.

Build / Runs dropzone body copy (verbatim):

> BibTeX, RIS, CSV, a DOI list, or PDFs, folders and .zip work too. Every entry
> is matched on Semantic Scholar before it joins the seed set.

The only per-file error string that exists today:

> Unsupported format: use .bib, .ris, .csv, a DOI list, or PDFs

Three commitments fall out of that copy and are non-negotiable for any backend:

1. **Resolver, not a file manager.** Files are transient input. Nothing is
   stored as a library. The output of an import is S2-resolved records.
2. **Nothing disappears.** Every extracted reference lands in exactly one of
   four review states — Matched / Ambiguous (`N matches`) / Already in corpus /
   No match — and the No-match count is repeated in the footer
   (`N entries couldn't be matched: listed above, not added.`).
3. **One flow, many parsers.** Format differences live in the extract step
   only; resolve → review → add is identical for every format.

### 0.4 Per-format visuals in the demo

There are **no per-format icons**. The file row (`fileRowHtml`) renders a
34 px chip containing an uppercase **text** label from
`var EXTC = { bib:'BIB', ris:'RIS', csv:'CSV', txt:'TXT', pdf:'PDF', zip:'ZIP' }`,
falling back to `f.ext.toUpperCase()` or `'?'`. The dropzone itself uses one
generic upload-arrow glyph for every format. See `design-lane-worklist.md`
item **DL-01** — this is raised as a design-lane proposal, not changed here.

---

## 1. How the OLD backend handles seeds today

Read-only survey of `web/live/backend/` and the engine it drives.

| Piece | Path | What it does |
| --- | --- | --- |
| Seed search | `web/live/backend/s2_seeds.py` | The *only* seed-acquisition path that exists. A thin `GET /graph/v1/paper/search` (relevance search) returning `{id,title,abstract,authors,year,venue,cites,externalIds}`. Paging capped at `offset+limit ≤ 1000` (`_S2_MAX_REACH`), ≤ 100 per page. Reads `S2_API_KEY` / `SEMANTIC_SCHOLAR_API_KEY` / `CITECLAW_S2_API_KEY`, works keyless at a low rate. Retries once on 429; raises `S2SearchError` with user-facing copy for 429 / 401 / 403. |
| HTTP surface | `web/live/backend/server.py` | `GET /api/seeds/search`, `POST /api/seeds/abstract`. **There is no upload endpoint, no multipart handler, and no file parsing anywhere in the web backend.** |
| Seed → config | `web/live/backend/config_translate.py::_seed_entries` | Turns the starred rows into CiteClaw `seed_papers`: `{paper_id: …}` when a real S2 id is present, else `{title: …}` — and if any title-only entry exists, prepends a `ResolveSeeds` step to the pipeline. Demo ids `s1`–`s8` are deliberately treated as title-only. Honours the seed node's `maxSeeds` cap. Raises `TranslationError("No seed papers selected…")` on an empty set. |
| Title → id | `src/citeclaw/steps/resolve_seeds.py` | `ResolveSeeds` resolves `{title}` entries through `ctx.s2.search_match` (`GET /paper/search/match`, **uncached** by design). `include_siblings=True` additionally walks `external_ids` (`DOI:` / `ARXIV:` prefixes) and does a title round-trip to pick up preprint↔published pairs. |
| Id → record | `src/citeclaw/clients/s2/api.py` | `fetch_metadata(paper_id)` accepts prefixed ids, so `DOI:10.x/y` and `ARXIV:2301.12345` already resolve today. `_batch_fetch` POSTs `/paper/batch` in chunks of `_MAX_BATCH = 500`, with a per-paper GET fallback at ~1 rps when a batch fails. |
| Throughput | `src/s2mirror/` | The self-hosted mirror serves `POST /graph/v1/paper/batch` and `/paper/{id}` un-throttled with 1000-row pages. **`/paper/search/*` is proxied to the real S2 API at 1 rps.** This asymmetry is the single most important fact for import planning — see §3. |

**Bottom line:** the old backend can *search* for seeds and can *resolve ids and
titles* through the engine, but it has **zero** file-import capability. Every
row in §2 needs new backend work; the difference between phases is how much.

### 1.1 What already exists in the engine and can be reused

| Need | Already built |
| --- | --- |
| DOI / arXiv id → S2 record | `SemanticScholarClient.fetch_metadata` with `DOI:` / `ARXIV:` prefixes; `_batch_fetch` at 500/POST |
| Title → S2 record (the "match" in match-review) | `search_match` (`/paper/search/match`) — returns one best match, which is exactly the demo's *Matched* state |
| Ambiguity (the `N matches` pill) | **Not** available from `search_match` (it returns a single best match). Needs `search_relevance` with a score threshold, or the mirror's search proxy. New work. |
| Dedupe against the existing corpus | `MergeDuplicates` + `dedup.py` (DOI/arXiv + title similarity + SPECTER2 cosine) — pipeline-level, would need lifting into the import path |
| PDF → text | `src/pdfclaw/` (PyMuPDF / pypdf / Modal GROBID at `PDFCLAW_GROBID_URL`) |
| PDF → DOI → record | `src/pdfclaw/s2_enrich.py` (`/v1/paper/batch` DOI enrichment) + `title_search.py` (arXiv title fallback) |
| PDF → references (the "extract references" half) | `src/citeclaw/steps/_pdf_reference_extractor.py` (LLM-based, used by `ExpandByPDF`) |
| BibTeX **writing** | `src/citeclaw/output/` — a writer exists; there is **no reader** |
| BibTeX / RIS / CSV **reading** | Nothing. No `bibtexparser`, `rispy`, or `pybtex` in `pyproject.toml`. |

---

## 2. The format matrix

**Difficulty** is parse difficulty only (1 = a weekend, 5 = a project).
**Phase** — `pilot` = must ship with the first wired import; `phase 2` = the
release after; `later` = backlog with a written reason.

### 2.1 BibTeX (`.bib`) — **pilot**

| | |
| --- | --- |
| **User expectation** | "This is the file my reference manager gives me." Zotero, JabRef, Overleaf, Mendeley, and Google Scholar's per-result *BibTeX* link all emit it. It is the single highest-volume case and the demo's own sample (`sample()` parses `refs.bib`; the dropzone link reads *"Or try a sample refs.bib"*). |
| **Difficulty** | **4/5.** BibTeX has no standard. `@string` macros, `#` concatenation, `crossref` inheritance, brace-vs-quote delimiters, nested `{{…}}` case protection, LaTeX accent escapes (`\"{u}`), maths in titles, comment forms, and per-tool extensions (Better BibTeX's `file = {name:path:mime}`). Do **not** hand-roll a regex parser. |
| **Backend needed** | A real BibTeX grammar (recommend `bibtexparser` v2 or `pybtex`), a LaTeX→Unicode de-escaper, an entry-type→"is this a paper" table, and a field-priority rule (`doi` → `eprint`/`archivePrefix` → `url` → title match). |
| **Resolve path** | `doi` present (majority) → `/paper/batch` with `DOI:` prefixes, 500/POST, mirror-backed, fast. No DOI → `search_match` at 1 rps. |
| **Fixtures** | `bibtex/*` (18 files) + `vendor/ok-zotero-betterbibtex.bib` |

### 2.2 RIS (`.ris`) — **pilot**

| | |
| --- | --- |
| **User expectation** | "This is the Export button on Web of Science / Scopus / EndNote / ProQuest." Named explicitly in the dropzone copy. Second-highest volume after BibTeX, and the *only* clean export path out of EndNote. |
| **Difficulty** | **2/5.** A flat `TAG  - value` line format — far simpler than BibTeX. The traps are mechanical, not grammatical: folded continuation lines (the wrapped-abstract case), CRLF, BOM, missing blank-line separators, missing `ER`, and per-vendor tag aliases (Mendeley puts the DOI in `M3`, not `DO`; `T1`/`A1`/`JF`/`Y1` instead of `TI`/`AU`/`JO`/`PY`). |
| **Backend needed** | A folding-aware line reader, a tag-alias table (at minimum EndNote / Web of Science / Scopus / Mendeley flavours), and a `TY` → paper/not-a-paper table. `rispy` covers ~80 % of this; the alias table is ours. |
| **Resolve path** | Same as BibTeX. |
| **Fixtures** | `ris/*` (11 files) + `vendor/ok-mendeley.ris` |

### 2.3 Plain DOI / arXiv / URL lists (`.txt`) — **pilot**

| | |
| --- | --- |
| **User expectation** | "I have a list of DOIs from a colleague / a review's appendix / a spreadsheet column." Zero-friction; users expect to paste anything vaguely identifier-shaped and have it work. |
| **Difficulty** | **1/5** for extraction, **2/5** for normalisation. The work is a normaliser, not a parser: strip `doi:` / `DOI: ` / `https://doi.org/` / `http://dx.doi.org/` prefixes and trailing punctuation; recognise arXiv new-style (`2301.12345v2`) and old-style (`cs.LG/0601001`); map DataCite `10.48550/arXiv.X` to `ARXIV:X` (the engine already special-cases this); decide whether a non-identifier line is a **title** or an error. |
| **Backend needed** | An identifier normaliser + a "line is a title" fallback into `search_match`. Nothing else. |
| **Resolve path** | The cheapest of all formats — pure `/paper/batch`, mirror-backed, un-throttled. A 20 000-DOI list is 40 POSTs. |
| **Phase note** | This is the format with the best effort-to-value ratio in the whole matrix; ship it first even if BibTeX slips. |
| **Fixtures** | `lists/*` (13 files) |

### 2.4 CSV with DOI columns (`.csv`) — **pilot**

| | |
| --- | --- |
| **User expectation** | "I keep my reading list in a spreadsheet", plus Zotero's *Export → CSV*. The demo's copy names CSV explicitly, so it cannot be deferred. |
| **Difficulty** | **2/5** for parsing (`csv` stdlib), **3/5** for column detection. Real traps: BOM + `;` delimiter (Excel in a European locale), tab-delimited files named `.csv`, quoted fields with embedded newlines, ragged rows, and header-name variance (`DOI` / `doi_url` / `DI` / `Article DOI`). |
| **Backend needed** | Delimiter + encoding sniffing, a header-alias table (DOI / arXiv / PMID / title / author / year), and a *documented* precedence when several identifier columns are present. |
| **Resolve path** | Same as §2.3 when a DOI column exists; falls back to title match otherwise. |
| **Fixtures** | `csv/*` (11 files) |

### 2.5 PDF — single (**pilot**) and batch (**phase 2**)

| | |
| --- | --- |
| **User expectation** | "I have the papers themselves." Two distinct expectations: (a) *this PDF is a paper, add it*; (b) *this PDF is a review, add everything it cites*. **The demo only implements (a)** — `EXT_N.pdf = [1,0]`, one PDF ⇒ one entry. Expectation (b) is not in the design at all. |
| **Difficulty** | **3/5** single, **5/5** batch. Single: extract text, find a DOI in the first page or the XMP metadata, fall back to a title match on the first text block. Batch adds concurrency, per-file progress, memory ceilings, scanned PDFs with no text layer, and encrypted files. |
| **Backend needed** | An upload endpoint with a size ceiling + `pdfclaw`'s parser stack (PyMuPDF in-process; GROBID at `PDFCLAW_GROBID_URL` when structure matters) + `s2_enrich.py` for DOI→record. All of this exists; the missing piece is the HTTP surface and the staging/cleanup discipline (§4). |
| **Bonus the design already specifies** | A matched PDF attaches its full text to the record — `pdfSrc:'user'`, and the row's PDF mark tooltip reads *"from your import"* (design-system.md § PDF availability). That is a real product advantage: imported PDFs raise the corpus's `full text for N` count without any OA fetch. |
| **Explicitly out of scope** | OCR for scanned PDFs. `pdf/edge-no-text-layer.pdf` exists to pin the failure message, not to justify an OCR pipeline. |
| **Fixtures** | `pdf/*` (9 files), `archives/ok-pdfs.zip` |

### 2.6 `.zip` archives — **phase 2**

| | |
| --- | --- |
| **User expectation** | "I zipped my folder because the browser wouldn't take a folder." The spec is explicit: *".zip is accepted as a convenience, never required."* |
| **Difficulty** | **3/5** — mostly safety, not parsing. Path traversal (`../../../x`), uncompressed-size bombs, nested archives, macOS `__MACOSX/._*` resource forks and `.DS_Store` noise, and mixed contents (the demo's mock assumes a zip is all PDFs; a real one is not). |
| **Backend needed** | A hardened extractor: reject absolute and `..` member paths, cap total uncompressed bytes **before** extracting, cap member count, refuse nested archives, extract into a per-import staging dir that is always cleaned up. Then fan each member out to the parser for its own extension. |
| **Phase note** | Phase 2 rather than pilot **only** because it multiplies the pilot's blast radius; the parsing itself is free once §2.1–2.5 exist. |
| **Fixtures** | `archives/*` (7 files) |

### 2.7 Zotero / EndNote / Mendeley exports — **split**

| Export | Verdict |
| --- | --- |
| Zotero → BibTeX / Better BibTeX | **pilot**, covered by §2.1. Needs the `file`-field and unbraced-month tolerances. |
| Zotero → RIS | **pilot**, covered by §2.2. |
| Zotero → CSV | **pilot**, covered by §2.4. |
| Zotero → CSL JSON | **phase 2**, see §2.8. |
| Mendeley → RIS / BibTeX | **pilot**, needs the `M3`/`T1`/`A1`/`JF`/`Y1` alias table. |
| EndNote → RIS | **pilot**. This is the supported EndNote path and matches the demo's copy ("RIS exports (… EndNote …)"). |
| EndNote → `.enw` (native tagged) | **later** — reject with guidance. `%0 %A %T %J %D %R` is a distinct format; the right answer is a message telling the user to export as RIS instead. |
| EndNote → XML | **later** — same. |
| Zotero → RDF | **later** — same. |
| Zotero API / live sync | **later.** This crosses the design's hard line: *"a resolver, not a file manager … never a stored library."* A live Zotero connection would mean stored credentials and a synced library. It should not be built without an explicit product decision that reverses that line. |
| **Fixtures** | `vendor/*` (5 files) |

### 2.8 JSON — **phase 2, and it needs sign-off first**

| | |
| --- | --- |
| **User expectation** | Three different files all called "JSON": CSL-JSON (Zotero / Pandoc / Better BibTeX), an S2 API dump, and — most valuable here — a previous CiteClaw run's `literature_collection.json`. |
| **Difficulty** | **2/5** per schema, **3/5** for schema detection. Also NDJSON under a `.json` name, and records nested under an envelope (`data.items`). |
| **Backend needed** | A schema sniffer over three known shapes + an explicit "unknown JSON shape" rejection. `ok-citeclaw-collection.json` needs no S2 round-trip at all — the records already carry `paper_id` and `external_ids`. |
| **Blocked on** | **JSON is not in the demo's accepted extension set** (`EXT_N` has no `json`) and is absent from both dropzone copy blocks and from the unsupported-format error string. Wiring it would change visible copy in three surfaces. Raised as `escalations.md` **E-IMP-01**; not implemented. |
| **Fixtures** | `json/*` (8 files) |

### 2.9 Summary

| Format | Difficulty | Phase | Blocking dependency |
| --- | --- | --- | --- |
| Plain DOI / arXiv list (`.txt`) | 1–2 | **pilot** | identifier normaliser |
| RIS (`.ris`) | 2 | **pilot** | `rispy` + tag-alias table |
| CSV (`.csv`) | 2–3 | **pilot** | delimiter sniffing + header aliases |
| BibTeX (`.bib`) | 4 | **pilot** | real grammar + LaTeX de-escaper |
| PDF, single | 3 | **pilot** | upload endpoint + `pdfclaw` |
| PDF, batch | 5 | phase 2 | concurrency + progress + ceilings |
| `.zip` | 3 | phase 2 | hardened extractor |
| CSL-JSON / S2 JSON / CiteClaw JSON | 2–3 | phase 2 | **E-IMP-01 sign-off** |
| `.enw` / EndNote XML / Zotero RDF | 3 | later | reject-with-guidance copy (**E-IMP-04**) |
| Zotero live sync | 5 | later | contradicts "not a file manager" |
| OCR for scanned PDFs | 5 | **no** | out of product scope |

---

## 3. The resolve step is the real cost, not the parse step

Every format above converges on the same second half: **resolve on Semantic
Scholar**. Two paths with wildly different economics:

| Path | Route | Rate |
| --- | --- | --- |
| Has DOI / arXiv id | `POST /paper/batch`, 500 ids per request | Mirror-backed and un-throttled (`src/s2mirror` serves `/paper/batch` from local shards). A 3 000-entry `.bib` = 6 requests. |
| Title only | `GET /paper/search/match` per entry | The mirror **proxies search upstream at 1 rps**. A 3 000-entry title-only file = ~50 minutes. |

Design consequences the backend must respect:

1. **Partition before resolving.** Split extracted entries into id-bearing and
   title-only, batch the first, queue the second. Never resolve serially in
   file order — one title-only entry at position 3 must not stall 2 997 batched
   ones behind it.
2. **The demo's parse animation is per-file and indeterminate.** `fileRowHtml`
   emits one spinner and `_impGo` settles files one at a time with a random
   340–620 ms delay. That is fine for 3 files; for a 20 000-DOI list the user
   needs determinate progress. Registered as `missing-states.md` **MS-IMP-02**, not
   changed here.
3. **`search_match` is deliberately uncached** (`resolve_seeds.py`: *"title
   resolutions are rare and freshness matters more than savings"*). For import
   that assumption inverts — title resolutions become the bulk case. An import
   cache keyed on the normalised title is worth having, but note it changes an
   engine-level design decision.
4. **The `N matches` / Ambiguous state has no backend today.**
   `search_match` returns one best match. Producing candidates (the demo's
   popover shows *Journal version* vs *Preprint* with venue/year/citations)
   needs `search_relevance` plus a scoring threshold, or reuse of the
   preprint↔published sibling logic already in `ResolveSeeds(include_siblings=True)`.
5. **Rate-limit errors already have copy** — `s2_seeds.py` raises three
   user-facing `S2SearchError` messages (429 keyed, 429 keyless, 401/403). The
   import flow has nowhere to show them (`missing-states.md` **MS-IMP-03**).

---

## 4. Non-negotiables for whoever builds the backend

1. **Never silently drop an entry.** `extracted == matched + ambiguous +
   duplicate + no_match`, asserted per file. This is the design's stated
   promise and the review UI's whole reason to exist.
2. **Files are transient.** Stage under a per-import temp dir, delete on
   completion *and* on abandonment. No user file becomes a stored library
   object. (§ *Import papers*: "never a stored library, no Drive-style manager".)
3. **Extension is a hint, content is the truth** — or the exact opposite,
   consistently. The demo dispatches purely on extension (`extOf()`), so a
   BibTeX file named `.ris` is currently a parse failure. Pick a rule and pin
   it with `adversarial/wrong-ext-*`.
4. **Escape everything into the review row.** `rowHtml` writes titles and
   authors into `innerHTML`. `import-resolver.js::esc()` escapes `&`, `<`, `"`
   — enough, but the real implementation must not regress it.
   `adversarial/edge-html-injection.bib` is the regression test.
5. **Cap the import.** Bytes per file, files per import, entries per import,
   uncompressed bytes per archive. Every ceiling needs a message; none of those
   messages exist in the demo (`missing-states.md` **MS-IMP-04**).
6. **Dedupe twice.** Against the target corpus / seed set (the demo's `dupe`
   state) *and* within the upload itself (two files, same paper — no state for
   this today, `missing-states.md` **MS-IMP-05**).
