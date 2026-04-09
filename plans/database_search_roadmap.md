# CiteClaw — Expansion Family + Web UI Roadmap

> **What this file is.** A flat checklist of implementation tasks for adding the `ExpandBy*` family of pipeline steps, an iterative meta-LLM search agent, a graph-reinforcement step, a human-in-the-loop checkpoint, and a beautiful web UI to CiteClaw. Designed to be executed by Claude Code on an hourly cron, one task per invocation.
>
> **Design spec.** The full architectural discussion lives at `~/.claude/plans/quiet-hugging-lecun.md` on Mingyu's machine — that document explains the *why* behind every decision. This roadmap is the execution view: just the *what*, *files*, and *verify*. If something here is ambiguous, check the design spec.

---

## RESUME PROTOCOL — read this every invocation

You are a fresh Claude Code instance woken on a cron schedule. Your job is to make ONE unit of progress on this roadmap and exit. Follow this protocol exactly:

**Step 1 — Sanity-check the previous run.** Scroll to the "Last run feedback" section below and read the most recent entry. If it ends with `❌` (failure), read the error note carefully and decide whether the previous task is now actually complete or whether you need to re-attempt it. If it ends with `✅` (success), trust it and move on.

**Step 2 — Find your task.** Find the first unchecked `- [ ]` item. Read its `What`, `Why`, `Files touched`, and `Verify done` subsections completely before doing anything.

**Step 3 — Implement.** Make the changes described in `Files touched`. Stay in scope — do only what the task asks. If you discover the task as written is unworkable (e.g., a referenced file doesn't exist, a dependency is missing), STOP, append a `- ❌` feedback entry explaining the blocker, do NOT tick the box, do NOT commit. The next invocation (or the user) will resolve it.

**Step 4 — Verify.** Run the exact command from `Verify done`. If it passes, proceed to Step 5. If it fails:
- Read the error and try a focused fix (do not flail).
- If fixed, re-run the verification.
- If still failing after one focused attempt, STOP, append a `- ❌` feedback entry with the error, do NOT tick, do NOT commit.

**Step 5 — Tick the box and log feedback.** Change `- [ ]` to `- [x]`. Then append a feedback entry under the task using this format:
```
  - ✅ 2026-04-09 — <2-3 sentence summary of what was done, any assumptions, and any followups the next run should know about>
```
Use the actual current date (`date +%Y-%m-%d`). If you discovered something the next task should be aware of (e.g., "PA-02 introduced a `_TTL_DAYS` constant — PA-05 will reuse it"), say so.

**Step 6 — Update the "Last run feedback" section.** At the top of this file, prepend a one-line entry to the "Last run feedback" section:
```
- 2026-04-09 14:32 — completed PA-01 ✅ (added search_bulk/search_match/search_relevance to SemanticScholarClient; tests green)
```
Keep the section to the most recent 10 entries; trim older ones. This is the at-a-glance status the user reads in the morning.

**Step 7 — Commit and push (per CLAUDE.md auto-commit rule).** This step is mandatory; CLAUDE.md says "after ANY change to this project you MUST `git add`, `git commit`, `git push origin main`". Do exactly this:
```bash
git status                                          # confirm what changed
git add <list specific files explicitly>            # NEVER `git add -A` or `git add .`
git add plans/database_search_roadmap.md            # always include this file
git commit -m "<task-id>: <one-line summary>"       # e.g. "PA-01: add S2 search endpoints to client"
git push origin main
```
**Files you must NEVER stage** (per CLAUDE.md): `CLAUDE.md`, `BRAINSTORM.md`, `.claude/`, `data_bio/`, `test_data/`, `scratch/`, `*.db`, `*.log`, `.DS_Store`. They are gitignored but `git add -A` would force-include them.

If `git push` fails, do NOT force-push. Surface the error in the feedback log and stop.

**Step 8 — STOP.** One task per invocation. Do not start the next task. Exit cleanly.

### Special phase rules

- **Phase F (Meta-review agent) is HUMAN-GATED.** If you reach a Phase F task, STOP immediately and append a feedback entry saying "reached Phase F human gate, awaiting user approval". Do not implement.
- **Phase E (Web UI) runs in parallel with Phases C/D.** On odd-numbered cron runs, prefer the next unchecked Phase C/D task; on even-numbered runs, prefer the next unchecked Phase E task. If one phase is fully done, work the other. Skip this rule if you're inside a strict dependency chain (the task list will say so).
- **Skipping is allowed if a task is blocked.** If task X cannot be completed (missing dependency, design ambiguity), append `- ⏭️` feedback explaining why, do NOT tick X, and move on to find the NEXT task that is unblocked. The user will return to X manually.

---

## Last run feedback (most recent first; keep ≤ 10 entries)

- 2026-04-09 01:08 — completed PB-02 ✅ (added agent_decision branch to stub_respond with three-state lifecycle initial→refine→satisfied driven by `"query":` count; minor PB-01 tweak: changed `"query":` → `"query" —` in template legend so the initial branch is reachable; 10 new tests in TestStubAgentDecisionBranch; tests/test_llm.py 71/71 green; full suite 616/6, zero regressions)
- 2026-04-09 00:34 — completed PB-01 ✅ (new src/citeclaw/prompts/search_refine.py with SYSTEM + USER_TEMPLATE + RESPONSE_SCHEMA; thinking field is FIRST in the schema's properties dict; literal `"agent_decision"` quoted in the user template for PB-02's stub; verify command + format() round-trip both green — **Phase B started**)
- 2026-04-09 00:26 — completed PA-10 ✅ (extended FakeS2Client with search_bulk/search_match/fetch_recommendations/fetch_author_papers as canned-response surfaces + register_* helpers; recs keyed on sorted tuple; deepcopy on the way out; 25 new tests in test_search_phase_a_e2e.py; 4-file verify 126/126 green; full suite 606/6 — **Phase A DONE**)
- 2026-04-09 00:19 — completed PA-09 ✅ (new src/citeclaw/search/ package with apply_local_query — pure AND-ed predicate filter over PaperRecord lists; strict on missing metadata except abstract_regex which is lenient; 26 new tests in test_search_query_engine.py; full suite 581 passed/6 skipped)
- 2026-04-09 00:13 — completed PA-08 ✅ (added rejection_ledger + searched_signals + reinforcement_log fields to Context; record_rejections now also appends to ledger using same category key as rejection_counts; 5 new tests in test_filters_runner.py; full suite 555 passed/6 skipped)
- 2026-04-09 00:08 — completed PA-07 ✅ (replaced PaperSource str enum with constants namespace class adding SEARCH/SEMANTIC/AUTHOR/REINFORCED; PaperRecord.source is now plain str; production sites already used string literals so zero call-site changes; updated 3 test sites in test_models.py; full suite 550 passed/6 skipped)
- 2026-04-09 00:03 — completed PA-06 ✅ (added fields_of_study + publication_types to PaperRecord, extended PAPER_FIELDS with fieldsOfStudy/s2FieldsOfStudy/publicationTypes, paper_to_record merges legacy + s2 lists with dedup; 9 new tests in test_models.py; full suite 550 passed/6 skipped)
- 2026-04-08 23:58 — completed PA-05 ✅ (wired search_bulk through S2CacheLayer with sha256(q,filters,sort,token) hash; added get/put_search_results to cache layer; 10 new tests covering hit/miss + negative coverage for uncached search_match/recommendations; 48 total green)
- 2026-04-08 23:54 — completed PA-03 ✅ (added fetch_author_papers to SemanticScholarClient with cache-first pagination capped at limit; added get/put_author_papers to S2CacheLayer; 11 new tests, 38 total green)
- 2026-04-08 23:47 — completed PA-04 ✅ (added search_queries + author_papers cache tables, 5 new methods, _SEARCH_TTL_DAYS_DEFAULT=30 constant, 14 new test_cache.py tests; PA-03 skipped ⏭️ since it depends on PA-04's methods which now exist — PA-03 unblocked for next run)

---

## Architectural decisions (reference)

These were settled in the design conversation. Do NOT relitigate them; if you disagree with one, note it in feedback and stop, do not unilaterally change direction.

1. **`ExpandBy*` family, not a monolithic `DatabaseSearch` step.** Each retrieval paradigm (LLM-driven search, semantic kNN, author-graph traversal) is its own step, composable at the same level as `ExpandForward` / `ExpandBackward`. Users compose them freely in YAML.
2. **Meta-LLM agent is iterative by default** (`max_iterations=4`) with two-level thinking: (a) outer loop across LLM calls sees prior transcripts; (b) inner `thinking` field placed first in the JSON response schema forces per-call chain-of-thought before any structured decision. Native reasoning tokens (`reasoning_effort="high"`) stack on top for capable models.
3. **`ExpandBySemantics` uses S2 Recommendations API** (`POST /recommendations/v1/papers`), not local kNN. Zero new embedding infrastructure.
4. **Signal-driven grounding.** Each `ExpandBy*` step uses its input signal as anchor context. Users insert a `Rerank` (with diversity) before the step to control what's fed in.
5. **No new `SearchEngine` Protocol.** Each step calls S2 methods directly. Rejected as over-engineering for the actual use case.
6. **Per-paper rejection ledger** (`ctx.rejection_ledger: dict[str, list[str]]`) is populated by `record_rejections` and consumed by `HumanInTheLoop` for balanced sampling.
7. **`source: str`** instead of frozen enum on `PaperRecord`, so new sources (`search`, `semantic`, `author`, `reinforced`, etc.) can be added without schema migration.
8. **Filter atoms must tolerate `fctx.source=None`.** Verified once in PC-05 and enforced via build-time errors thereafter.
9. **Web UI lives in `web/`** as a subdirectory. Stack: React 18 + Vite + TypeScript + Tailwind v4 + shadcn/ui + sigma.js (ForceAtlas 2) + React Flow + FastAPI + WebSockets.
10. **Phase F (meta-review agent) is human-gated.** Cron-Claude stops if it reaches it.

---

## Phase A — S2 surface + pure utilities

Goal: every Phase A module is unit-testable with zero pipeline touch.

- [x] **PA-01. `search_bulk` / `search_match` / `search_relevance` on `SemanticScholarClient`**
  - **What.** Extend `src/citeclaw/clients/s2/api.py` with three methods:
    - `search_bulk(query, *, filters=None, sort=None, token=None, limit=1000) -> dict` → `GET /paper/search/bulk`. Forwards `year, venue, fieldsOfStudy, minCitationCount, publicationTypes, publicationDateOrYear, openAccessPdf` from `filters`. `fields="paperId,title"` only. `req_type="search"`.
    - `search_match(title) -> dict | None` → `GET /paper/search/match`. `req_type="search_match"`.
    - `search_relevance(query, *, limit=100, offset=0) -> dict` → `GET /paper/search`. `req_type="search"`.
  - All three reuse `_throttle`, `_http.get`, existing backoff. Cache wiring is PA-05.
  - **Why.** Minimum S2 surface Phase B's agent depends on.
  - **Files touched.** `src/citeclaw/clients/s2/api.py`. New: `tests/test_s2_search_api.py`.
  - **Verify done.** `pytest tests/test_s2_search_api.py -x` (uses monkey-patched `S2Http.get`; no network).
  - ✅ 2026-04-08 — Added a "Search" section to `api.py` with all three methods, an `httpx` import for the `search_match` 404→None catch, and a `_SEARCH_BULK_FILTER_KEYS` whitelist tuple so PA-05 can reuse the same allowlist when wiring caches. New `tests/test_s2_search_api.py` has 14 tests using a `_Recorder` helper that monkey-patches `client._http.get`. Note for next runs: stale `__pycache__` from the old `CitNet2` repo path broke pytest collection — had to wipe it once; if a future task sees `ModuleNotFoundError: citeclaw`, run `find . -name __pycache__ -exec rm -rf {} +` and use `PYTHONPATH=src python -m pytest …` since the package isn't pip-installed.

- [x] **PA-02. `fetch_recommendations` on `SemanticScholarClient`**
  - **What.** Add to `src/citeclaw/clients/s2/api.py`:
    - `fetch_recommendations(positive_ids, *, negative_ids=None, limit=100, fields="paperId,title") -> list[dict]` → `POST /recommendations/v1/papers` with body `{"positivePaperIds": [...], "negativePaperIds": [...]}`. `req_type="recommendations"`.
    - `fetch_recommendations_for_paper(paper_id, *, limit=100, fields=...) -> list[dict]` → `GET /recommendations/v1/papers/forpaper/{paper_id}`.
  - **Why.** Powers `ExpandBySemantics`. S2 does the SPECTER2 kNN over its full corpus for us.
  - **Files touched.** `src/citeclaw/clients/s2/api.py`. Append to `tests/test_s2_search_api.py`.
  - **Verify done.** `pytest tests/test_s2_search_api.py -x`.
  - ✅ 2026-04-08 — Both methods unwrap S2's `recommendedPapers` envelope so callers always get a flat list. Recommendations live outside `/graph/v1`, so I added a small `S2Http.get_url(full_url, ...)` helper (mirrors `get` but skips BASE_URL prepend) — that lightweight http.py addition is the one file outside the task's listed "Files touched" but it's the cleanest way to keep retry/throttle/budget shared. New constants `RECOMMENDATIONS_BATCH_URL` / `RECOMMENDATIONS_FORPAPER_URL` in api.py. Also extended `_Recorder` in tests with `install_post` and `install_get_url` siblings — PA-03 will need install_get_url too when pagination tests are added.

- [x] **PA-03. `fetch_author_papers` on `SemanticScholarClient`**
  - **What.** Add `fetch_author_papers(author_id, *, limit=100, fields="paperId,title,year,venue,citationCount") -> list[dict]` → `GET /graph/v1/author/{author_id}/papers` with pagination. `req_type="author_papers"`. Caches per-author under the new `author_papers` cache table (PA-04).
  - **Why.** Powers `ExpandByAuthor`. Today's `fetch_authors_batch` only returns author metadata, not paper lists.
  - **Files touched.** `src/citeclaw/clients/s2/api.py`, `src/citeclaw/cache.py` (depends on PA-04). Append to `tests/test_s2_search_api.py`.
  - **Verify done.** `pytest tests/test_s2_search_api.py -x`.
  - ⏭️ 2026-04-08 — Skipped this run because PA-03's caching arm depends on `Cache.get_author_papers`/`put_author_papers`, which only land in PA-04. Did PA-04 first; now unblocked. Next run should pick this up — will need to call `cache.get_author_papers(author_id)` / `cache.put_author_papers(author_id, papers)` after the paginated S2 GET, and use the new `S2Http.get_url` helper from PA-02 if pagination logic ends up there. Pagination follow-up: S2's author/papers endpoint paginates via `offset`/`next` token — model the loop on `S2Http.paginate` rather than reinventing it (consider adding an `author_papers` branch or building a small in-`api.py` paginator).
  - ✅ 2026-04-08 — Added `fetch_author_papers` to api.py with an inline cache-first paginator (mirrors `S2Http.paginate`'s offset/limit shape but lives in api.py since the URL is `/author/{id}/papers` not `/paper/{id}/{edge}`). Module-level constant `_AUTHOR_PAPERS_PAGE_SIZE = 100`. The `limit` arg caps both pagination *and* the returned slice — so the cached entry reflects exactly what was fetched, not the author's full corpus (deliberate trade-off documented in the docstring; if a downstream user later asks for a bigger limit, they get the cached short list). Also added `get_author_papers`/`put_author_papers` to `S2CacheLayer` (wraps Cache and bumps `_s2_cache["author_papers"]` on hit) — that cache_layer.py file is outside PA-03's listed "Files touched" but it's the only way to wire the new cache table through the budget tracker. 11 new tests including a `_install_paginated_get` helper for multi-page scenarios; full file at 38 tests.

- [x] **PA-04. Cache tables: `search_queries` + `author_papers`**
  - **What.** Append to `_SCHEMA` in `src/citeclaw/cache.py`:
    ```sql
    CREATE TABLE IF NOT EXISTS search_queries (
      query_hash TEXT PRIMARY KEY,
      query_json TEXT NOT NULL,
      result_json TEXT NOT NULL,
      fetched_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS author_papers (
      author_id TEXT PRIMARY KEY,
      data TEXT NOT NULL,
      fetched_at TEXT NOT NULL
    );
    ```
    Add `Cache.get_search_results/put_search_results/has_search_results(query_hash, ttl_days=30)` and `Cache.get_author_papers/put_author_papers(author_id)`.
  - **Files touched.** `src/citeclaw/cache.py`. Append to `tests/test_cache.py`.
  - **Verify done.** `pytest tests/test_cache.py -x`.
  - ✅ 2026-04-08 — Added both tables to `_SCHEMA` plus a `_SEARCH_TTL_DAYS_DEFAULT = 30` module constant so PA-05 can reference the same default. New helper `Cache._is_fresh(fetched_at_iso, ttl_days)` parses ISO timestamps and tolerates naive datetimes (older rows). 14 new test_cache.py tests covering put/get/has roundtrip, TTL expiration via direct SQL backdating, persistence across instances, query_json round-trip, and the empty-list-vs-missing distinction for author_papers. The TTL test pattern (UPDATE … fetched_at to a backdated ISO string then read) is reusable — PA-05's cache-hit tests can copy it.

- [x] **PA-05. Wire caches into `search_bulk`, `fetch_recommendations`, `fetch_author_papers`**
  - **What.** Add `query_hash = sha256(json.dumps({"q": q, "filters": f, "sort": s, "token": t}, sort_keys=True)).hexdigest()` to `search_bulk`. Cache via new `S2CacheLayer.get_search_results/put_search_results` (records hits in `BudgetTracker.record_s2("search", cached=True)`). Cache `fetch_author_papers` per-author. Do NOT cache `search_match` or `fetch_recommendations` (freshness matters).
  - **Files touched.** `src/citeclaw/clients/s2/cache_layer.py`, `src/citeclaw/clients/s2/api.py`.
  - **Verify done.** Extend `tests/test_s2_search_api.py`: call each cached method twice with identical args; second call must serve from cache.
  - ✅ 2026-04-08 — `search_bulk` now hashes `{"q","filters","sort","token"}` with `sort_keys=True` (deliberately omitting `limit` so a wider pre-fetch can serve narrower followers from cache). Hit returns the cached payload before any HTTP work; miss persists the full response (including `total`, `token`, etc.). Added `get_search_results`/`put_search_results` to `S2CacheLayer` — hits bump `_s2_cache["search"]` via `record_s2(..., cached=True)`. `fetch_author_papers` was already cached in PA-03, so PA-05 didn't need to touch it again. 10 new tests: 7 in `TestSearchBulkCacheWiring` (hit, miss-on-different-q/filters/sort/token, dict-key-order independence, full-payload round-trip) and 3 in `TestUncachedSurfaces` proving `search_match`/`fetch_recommendations`/`fetch_recommendations_for_paper` still always reach the network. `_query_hash_for` test helper duplicates the SUT's hash recipe so cache-key inspection is possible without monkeying with internals; PA-09's local query engine can ignore it.

- [x] **PA-06. Extend `PaperRecord` with `fields_of_study` + `publication_types`**
  - **What.** Add `fields_of_study: list[str] = Field(default_factory=list)` and `publication_types: list[str] = Field(default_factory=list)` to `src/citeclaw/models.py::PaperRecord`. Extend `PAPER_FIELDS` in `api.py` with `fieldsOfStudy,publicationTypes,s2FieldsOfStudy`. Extend `paper_to_record` in `converters.py` to populate them (merge `s2FieldsOfStudy` into `fields_of_study`).
  - **Files touched.** `src/citeclaw/models.py`, `src/citeclaw/clients/s2/api.py`, `src/citeclaw/clients/s2/converters.py`. New test in `tests/test_models.py`.
  - **Verify done.** `pytest tests/ -x`.
  - ✅ 2026-04-09 — Added both fields with `Field(default_factory=list)` defaults so existing PaperRecord constructions stay backward-compatible. `paper_to_record` merges `fieldsOfStudy` (legacy flat strings) and `s2FieldsOfStudy` (`{category, source}` dicts) into a single deduplicated list, preserving legacy-first ordering. Robust to None / non-list / non-string entries — important because S2's response shape is inconsistent across paper records. 9 new tests in `test_models.py` (2 in `TestPaperRecord` for defaults + direct construction, 7 in `TestPaperToRecordSubjectFields` for converter merge logic). Full `pytest tests/ -x` green: 550 passed, 6 skipped (topic_model extras + live_s2 markers; pre-existing). PA-09's local query engine will consume these fields directly.

- [x] **PA-07. `PaperRecord.source: str` instead of frozen enum**
  - **What.** Replace the `PaperSource` enum field on `PaperRecord` with `source: str = "backward"`. Keep `PaperSource` as a constants namespace: `class PaperSource: SEED="seed"; FORWARD="forward"; BACKWARD="backward"; SEARCH="search"; SEMANTIC="semantic"; AUTHOR="author"; REINFORCED="reinforced"`. Audit all call sites that compare `p.source == PaperSource.X` — they continue working because `use_enum_values=True`.
  - **Files touched.** `src/citeclaw/models.py`. Possibly a few call sites in steps/.
  - **Verify done.** `pytest tests/ -x`.
  - ✅ 2026-04-09 — Audit found ALL production assignments (`load_seeds.py`, `expand_forward.py`, `expand_backward.py`) and comparisons (`network.py`, `checkpoint.py`, `graphml_writer.py`) already used string literals — the enum was a vestigial type annotation. Replaced `class PaperSource(str, enum.Enum)` with a plain `class PaperSource` namespace adding the four new sources, changed `source: PaperSource = PaperSource.BACKWARD` to `source: str = "backward"`, and updated 3 test sites in `test_models.py` (line 93 dropped the `.value`, lines 158-160 became direct string compares, and added asserts for the new SEARCH/SEMANTIC/AUTHOR/REINFORCED constants). Zero `src/` files outside `models.py` needed touching. Full `pytest tests/ -x` green: 550 passed, 6 skipped. PaperRecord docstring on `source` now points readers to `PaperSource` for canonical values without forcing them to use it.

- [x] **PA-08. `Context` additions: rejection ledger + idempotency sets + reinforcement log**
  - **What.** In `src/citeclaw/context.py`, add three fields:
    ```python
    rejection_ledger: dict[str, list[str]] = field(default_factory=dict)
    searched_signals: set[str] = field(default_factory=set)
    reinforcement_log: list[dict] = field(default_factory=list)
    ```
    Update `record_rejections` in `src/citeclaw/filters/runner.py` to also append to `rejection_ledger[paper.paper_id]`.
  - **Why.** `HumanInTheLoop` needs per-paper rejection reasons; `ExpandBy*` steps need per-signal idempotency; `ReinforceGraph` needs a place to log decisions.
  - **Files touched.** `src/citeclaw/context.py`, `src/citeclaw/filters/runner.py`. New test asserting the ledger is populated on rejection.
  - **Verify done.** `pytest tests/ -x`.
  - ✅ 2026-04-09 — Added all three fields with `field(default_factory=...)` defaults so existing Context constructions stay backward-compatible. `record_rejections` now appends to `rejection_ledger.setdefault(paper.paper_id, []).append(key)` using the SAME key as `rejection_counts` — this guarantees the per-paper ledger and the global counts can never disagree, which `HumanInTheLoop` will rely on for balanced sampling. 5 new tests in `TestRecordRejections`: single-rejection, multi-rejection accumulation across calls, separation by paper_id, blank-category falls through as "unknown", and a baseline assertion that the new fields start empty on a fresh Context. Full `pytest tests/ -x` green: 555 passed (5 more than last run), 6 skipped. Note for PC-01: `searched_signals` is the key the ExpandBy* family will hash into; the docstring on the field describes the expected fingerprint shape (step name + signal ids + agent config).

- [x] **PA-09. `src/citeclaw/search/query_engine.py` — pure `apply_local_query`**
  - **What.** New package `src/citeclaw/search/__init__.py` + `src/citeclaw/search/query_engine.py`. Exports one pure function:
    ```python
    def apply_local_query(
        papers: list[PaperRecord], *,
        venue_regex: str | None = None,
        year_min: int | None = None, year_max: int | None = None,
        min_citations: int | None = None,
        fields_of_study_any: list[str] | None = None,
        publication_types_any: list[str] | None = None,
        abstract_regex: str | None = None,
        title_regex: str | None = None,
    ) -> list[PaperRecord]
    ```
    AND-ed predicates; strict on missing metadata except `abstract_regex` (lenient — S2 often lacks abstracts). Regexes with `re.IGNORECASE`.
  - **Why.** S2 API can't express regex, abstract text search, or arbitrary unions. Used optionally by expand steps for post-fetch trim.
  - **Files touched.** New: `src/citeclaw/search/__init__.py`, `src/citeclaw/search/query_engine.py`, `tests/test_search_query_engine.py`.
  - **Verify done.** `pytest tests/test_search_query_engine.py -x` with ~10 cases.
  - ✅ 2026-04-09 — Created the new `search/` package with `__init__.py` re-exporting `apply_local_query` (so callers can `from citeclaw.search import apply_local_query`). Pure function — no Context, no S2, no LLM dependency. Each predicate is skipped when None and AND-ed when set; missing-metadata behavior matches the spec exactly (strict everywhere except `abstract_regex`, which is lenient because S2 often returns no abstract). Regexes pre-compile once with `re.IGNORECASE` and use `re.search` semantics so callers don't need to anchor. 26 new tests in `test_search_query_engine.py` organized in 8 classes (TestNoPredicates / TestYearRange / TestMinCitations / TestVenueRegex / TestTitleRegex / TestAbstractRegex / TestFieldsOfStudyAny / TestPublicationTypesAny / TestCombinedPredicates) — well above the spec's "~10 cases". Full suite 581 passed/6 skipped. PC-01's `ExpandBySearch` will pipe its hydrated candidates through this before calling `apply_block` so callers can use both approaches together.

- [x] **PA-10. `FakeS2Client` extensions + Phase A e2e test**
  - **What.** Extend `tests/fakes.py::FakeS2Client` with `search_bulk`, `search_match`, `fetch_recommendations`, `fetch_author_papers` — query-keyed canned responses suitable for downstream Phase B and C tests. Then write `tests/test_search_phase_a_e2e.py` exercising each new API method against the fake.
  - **Files touched.** `tests/fakes.py`, new `tests/test_search_phase_a_e2e.py`.
  - **Verify done.** `pytest tests/test_s2_search_api.py tests/test_cache.py tests/test_search_query_engine.py tests/test_search_phase_a_e2e.py -x`. **Phase A DONE** when all green.
  - ✅ 2026-04-09 — Added 4 canned-response surfaces to FakeS2Client (`search_bulk`/`search_match`/`fetch_recommendations`/`fetch_author_papers`) plus matching `register_*` helpers; init seeds the four backing dicts. Each surface is order-independent where it matters (`fetch_recommendations` keys on the *sorted* tuple of positive ids, mirroring the cache hash recipe), accepts ignored-but-signature-compatible kwargs (`filters`/`sort`/`token` for search_bulk; `negative_ids`/`fields` for recs; `fields` for author_papers), and deepcopies returned dicts so test mutation can't poison the canned table. New `tests/test_search_phase_a_e2e.py` has 25 tests in 5 classes (TestFakeSearchBulk/TestFakeSearchMatch/TestFakeFetchRecommendations/TestFakeFetchAuthorPapers + a TestFakeSurfaceIntegration cross-method test that proves one client can serve all four surfaces and the per-method call counters stay isolated). Verification command (4-file run) green at 126/126; full suite 606 passed/6 skipped — zero regressions. **Phase A is now DONE** — next run starts Phase B.

---

## Phase B — Iterative meta-LLM search agent

- [x] **PB-01. Prompt module `src/citeclaw/prompts/search_refine.py`**
  - **What.** New file with:
    - `SYSTEM` — role: "You design targeted literature-database queries given a topic and a sample of papers already in the collection. Before committing to a query, think out loud in the `thinking` field. Inspect results, refine, decide satisfied/abort."
    - `USER_TEMPLATE` — takes `{topic_description}`, `{anchor_papers_block}`, `{transcript}` (prior turns including prior `thinking`), `{iteration}`, `{max_iterations}`, `{target_count}`. Output JSON matching `RESPONSE_SCHEMA`.
    - `RESPONSE_SCHEMA` — JSON Schema enforcing fields IN ORDER: `thinking` (string, FIRST), `query` (object with `text`, optional `filters`, optional `sort`), `agent_decision` (enum: initial|refine|satisfied|abort), `reasoning` (string).
    - The literal string `"agent_decision"` MUST appear in `USER_TEMPLATE` for stub recognition.
  - **Files touched.** New: `src/citeclaw/prompts/search_refine.py`.
  - **Verify done.** `python -c "from citeclaw.prompts.search_refine import SYSTEM, USER_TEMPLATE, RESPONSE_SCHEMA; assert 'agent_decision' in USER_TEMPLATE and RESPONSE_SCHEMA['properties']['thinking']['type'] == 'string'"`.
  - ✅ 2026-04-09 — Created the new prompt module with all three exports. SYSTEM emphasizes the "think before deciding" pattern and lists the four lifecycle states. USER_TEMPLATE renders all six placeholders (topic_description / anchor_papers_block / transcript / iteration / max_iterations / target_count) and contains the literal `"agent_decision"` (quoted exactly as it would appear in JSON) inside a numbered field-order legend — that's what PB-02's stub will key on via `if '"agent_decision"' in user:`. RESPONSE_SCHEMA is a `dict[str, Any]` with `properties` insertion-ordered as `thinking → query → agent_decision → reasoning`, all four required, `additionalProperties: False`, and the four-element enum on agent_decision. Verified the format() round-trip works with realistic placeholder values and the quoted token survives formatting. PB-02 can now monkey-patch the stub against this schema; PB-03's AgentTurn dataclass mirrors the same field names so JSON parsing in PB-04 will be straightforward.

- [x] **PB-02. Stub client extension for agent prompts**
  - **What.** Add a branch to `stub_respond` in `src/citeclaw/clients/llm/stub.py`: `if '"agent_decision"' in user:`. Count `"query":` occurrences in `user` (transcript grows per iteration). Return deterministic JSON with ALL four fields (thinking first):
    - 0 → `{"thinking": "stub: initial exploration", "query": {"text": "test topic"}, "agent_decision": "initial", "reasoning": "stub initial"}`
    - 1 → `{"thinking": "stub: prior was too broad, narrowing", "query": {"text": "test topic narrowed"}, "agent_decision": "refine", "reasoning": "stub refine"}`
    - ≥2 → `{"thinking": "stub: results saturated", "query": {"text": "test topic narrowed"}, "agent_decision": "satisfied", "reasoning": "stub satisfied"}`
  - **Files touched.** `src/citeclaw/clients/llm/stub.py`. Append to `tests/test_llm.py`.
  - **Verify done.** `pytest tests/test_llm.py -x`. Tests assert `thinking` field non-empty.
  - ✅ 2026-04-09 — Added the agent_decision branch to `stub_respond` (placed right after the topic_label branch so it short-circuits all screening branches). All three responses use Python dict literals so json.dumps preserves the schema's `thinking → query → agent_decision → reasoning` insertion order. **One follow-up tweak to PB-01 was required:** PB-01's USER_TEMPLATE legend originally contained the literal substring `"query":` (in the field-order numbered list), which would have made the iteration counter start at 1 — meaning the `initial` branch was unreachable. Fix was minimal: changed `2. "query": object with...` to `2. "query" — an object with...` (em-dash instead of colon). PB-01's verify command (`'agent_decision' in USER_TEMPLATE` + thinking type check) still passes after the tweak. 10 new tests in `TestStubAgentDecisionBranch` (test_llm.py) covering: (a) all three lifecycle states triggered by 0/1/2 prior `"query":` keys, (b) ≥2 stays satisfied for higher counts, (c) every state has non-empty thinking and all four fields, (d) JSON serialization preserves thinking-first order, (e) end-to-end through StubClient with category="meta_search_agent" bumps `budget._llm_tokens["meta_search_agent"]`, (f) bare template has zero `"query":` so the initial branch is reachable, (g) branch detector doesn't steal unrelated prompts. `pytest tests/test_llm.py -x` 71/71 green; full suite 616 passed/6 skipped (+10 from this task) with zero regressions. **Note for PB-04**: the transcript-rendering for prior turns MUST embed the literal `"query":` JSON key once per turn (e.g., serialize each prior AgentTurn as JSON inside the transcript block) so the iteration counter advances naturally as the agent loops.

- [ ] **PB-03. Agent module + dataclasses**
  - **What.** New `src/citeclaw/agents/__init__.py` + `src/citeclaw/agents/iterative_search.py`:
    ```python
    @dataclass
    class AgentConfig:
        max_iterations: int = 4
        max_llm_tokens: int = 200_000
        target_count: int = 200
        search_limit_per_iter: int = 500
        summarize_sample: int = 20
        model: str | None = None
        reasoning_effort: str | None = "high"

    @dataclass
    class AgentTurn:
        iteration: int
        thinking: str
        query: dict
        n_results: int
        unique_venues: list[str]
        year_range: tuple[int | None, int | None]
        sample_titles: list[str]
        decision: str
        reasoning: str

    @dataclass
    class SearchAgentResult:
        hits: list[dict]
        transcript: list[AgentTurn]
        final_decision: str
        tokens_used: int
        s2_requests_used: int
    ```
  - **Files touched.** New: `src/citeclaw/agents/__init__.py`, `src/citeclaw/agents/iterative_search.py`.
  - **Verify done.** `python -c "from citeclaw.agents.iterative_search import AgentConfig, AgentTurn; c = AgentConfig(); assert c.max_iterations == 4 and c.reasoning_effort == 'high'"`.

- [ ] **PB-04. `run_iterative_search` loop**
  - **What.** Implement:
    ```python
    def run_iterative_search(
        topic_description: str,
        anchor_papers: list[PaperRecord],
        llm_client: LLMClient,
        ctx: Context,
        config: AgentConfig,
    ) -> SearchAgentResult: ...
    ```
    Loop body per iteration: format `USER_TEMPLATE` with topic + anchor block + transcript-so-far → `llm_client.call(SYSTEM, user, category="meta_search_agent", response_schema=RESPONSE_SCHEMA)` → parse JSON (extract `thinking`, `query`, `agent_decision`, `reasoning`) → `ctx.s2.search_bulk(...)` → dedup cumulative hits → summarize via 20-sample `enrich_batch` (unique venues, year range, sample titles) → append `AgentTurn` → break on `satisfied`/`abort`/`max_iterations`/`max_llm_tokens`.
    Transcript formatting for the next iteration's user prompt MUST include each prior turn's `Thinking:`, `Query:`, `Observed:`, `Sample titles:`, `Decision:` lines so the agent's earlier reasoning is visible to its later self.
    LLM client built once at start with `build_llm_client(ctx.config, ctx.budget, model=config.model or ctx.config.search_model or ctx.config.screening_model, reasoning_effort=config.reasoning_effort)`.
    When `anchor_papers` is empty, render the block as `"(No anchor papers — bootstrap from topic description alone.)"`.
  - **Files touched.** `src/citeclaw/agents/iterative_search.py`.
  - **Verify done.** Next task tests it.

- [ ] **PB-05. Unit tests for the agent**
  - **What.** New `tests/test_iterative_search_agent.py`. Drives `run_iterative_search` with `StubLLMClient` + `FakeS2Client.search_bulk`. Asserts:
    - `max_iterations=3` with 5 anchor papers → transcript has 3 turns, `final_decision == "satisfied"`.
    - `max_iterations=1` → transcript has 1 turn (single-shot mode works).
    - Default `AgentConfig` has `max_iterations == 4`.
    - Empty `anchor_papers` → agent still runs (topic-only fallback).
    - **Every `AgentTurn.thinking` is non-empty** (proves scratchpad round-trips).
    - Iteration N+1's user prompt CONTAINS iteration N's thinking text (proves Level-1 transcript accumulation).
    - `budget._llm_tokens.get("meta_search_agent", 0) > 0`.
    - `budget._s2_api.get("search", 0) == iterations`.
  - **Files touched.** New: `tests/test_iterative_search_agent.py`.
  - **Verify done.** `pytest tests/test_iterative_search_agent.py -x`. **Phase B DONE** when green.

- [ ] **PB-06. Manual validation script**
  - **What.** New `scratch/try_iterative_search.py`. ~50 lines argparse: loads `config_bio.yaml`, builds real S2 + LLM clients, runs the agent with `--topic` and optional `--anchor-papers` DOI list, prints transcript.
  - **Files touched.** New: `scratch/try_iterative_search.py`. (NOT committed — scratch/ is gitignored.)
  - **Verify done.** `python -c "import ast; ast.parse(open('scratch/try_iterative_search.py').read())"`.

---

## Phase C — `ExpandBy*` family (integration)

- [ ] **PC-01. `ExpandBySearch` step (FLAGSHIP — ship this first)**
  - **What.** New `src/citeclaw/steps/expand_by_search.py`. Class:
    ```python
    class ExpandBySearch:
        name = "ExpandBySearch"
        def __init__(self, *, topic_description=None, max_anchor_papers=20,
                     agent: AgentConfig, screener, apply_local_query_args=None): ...

        def run(self, signal, ctx) -> StepResult:
            # 1. Fingerprint (step, signal_ids, agent_config); skip if in ctx.searched_signals.
            # 2. anchor_papers = signal[:max_anchor_papers] (rerank upstream for diversity).
            # 3. topic = self.topic_description or ctx.config.topic_description.
            # 4. result = run_iterative_search(topic, anchor_papers, llm, ctx, self.agent)
            #    where llm is built via build_llm_client at the top of this method.
            # 5. hydrated = ctx.s2.enrich_batch([{"paper_id": h["paperId"]} for h in result.hits])
            # 6. ctx.s2.enrich_with_abstracts(hydrated)
            # 7. (Optional) hydrated = apply_local_query(hydrated, **self.apply_local_query_args)
            # 8. Dedup against ctx.seen; stamp source="search" on novel; add to ctx.seen.
            # 9. fctx = FilterContext(ctx=ctx, source=None, source_refs=None, source_citers=None)
            # 10. passed, rejected = apply_block(new, self.screener, fctx); record_rejections.
            # 11. Add passed to ctx.collection.
            # 12. Mark fingerprint in ctx.searched_signals.
            # 13. Return StepResult(signal=passed, in_count=len(hydrated), stats={...}).
    ```
  - **Why.** The flagship feature. Demonstrates the full agent loop end-to-end.
  - **Files touched.** New: `src/citeclaw/steps/expand_by_search.py`. Register in `src/citeclaw/steps/__init__.py` with `_build_expand_by_search(d, blocks)`.
  - **Verify done.** PC-08 e2e test covers it.

- [ ] **PC-02. `ExpandBySemantics` step**
  - **What.** New `src/citeclaw/steps/expand_by_semantics.py`:
    ```python
    class ExpandBySemantics:
        name = "ExpandBySemantics"
        def __init__(self, *, max_anchor_papers=10, limit=100,
                     use_rejected_as_negatives=False, screener): ...

        def run(self, signal, ctx):
            # Same fingerprint-and-skip pattern.
            # anchor_ids = [p.paper_id for p in signal[:max_anchor_papers]]
            # negative_ids = list(ctx.rejected)[:50] if use_rejected_as_negatives else None
            # raw = ctx.s2.fetch_recommendations(anchor_ids, negative_ids=negative_ids, limit=self.limit)
            # Hydrate, enrich abstracts, dedup, stamp source="semantic", apply screener, add survivors.
    ```
    No LLM, no agent. S2 API does the SPECTER2 kNN.
  - **Files touched.** New: `src/citeclaw/steps/expand_by_semantics.py`. Register in `steps/__init__.py`.
  - **Verify done.** PC-08 e2e test.

- [ ] **PC-03. `ExpandByAuthor` step**
  - **What.** New `src/citeclaw/steps/expand_by_author.py`:
    ```python
    class ExpandByAuthor:
        name = "ExpandByAuthor"
        def __init__(self, *, top_k_authors=10, author_metric="h_index",
                     papers_per_author=50, screener): ...

        def run(self, signal, ctx):
            # 1. Collect distinct author_ids from p.authors across signal.
            # 2. ctx.s2.fetch_authors_batch(author_ids) → metadata.
            # 3. Rank by author_metric: "h_index" / "citation_count" / "degree_in_collab_graph".
            #    For "degree_in_collab_graph", build the graph inline via author_graph.export_author_graphml's helper.
            # 4. Select top_k_authors.
            # 5. For each: ctx.s2.fetch_author_papers(author_id, limit=papers_per_author).
            # 6. Flatten, dedup against ctx.seen, hydrate + enrich abstracts.
            # 7. Stamp source="author"; apply screener; add survivors to collection.
    ```
  - **Files touched.** New: `src/citeclaw/steps/expand_by_author.py`. Register in `steps/__init__.py`. May need to refactor `src/citeclaw/author_graph.py` to expose graph-building logic separately from the GraphML writer.
  - **Verify done.** PC-08 e2e test.

- [ ] **PC-04. `ResolveSeeds` step (preprint + published pairs)**
  - **What.** New `src/citeclaw/steps/resolve_seeds.py`. Reads `ctx.config.seed_papers` which now allows entries of either `{paper_id: ...}` or `{title: ...}`. For each:
    - `{paper_id: ...}` → keep as-is.
    - `{title: ...}` → `ctx.s2.search_match(title)` → resolved paperId.
    - For each resolved paper: fetch metadata to get `external_ids`. If `include_siblings=True`, attempt to fetch each external ID (DOI, ArXiv) as a separate S2 paper. If they resolve to DIFFERENT paper IDs, add ALL to the result set.
    - Write result to `ctx.resolved_seed_ids: list[str]`.
    Then update `src/citeclaw/steps/load_seeds.py` to read `ctx.resolved_seed_ids` if present, else fall back to `ctx.config.seed_papers`.
  - **Why.** S2 sometimes has citation/reference data on only one of preprint/published — loading both maximizes graph coverage before `MergeDuplicates`.
  - **Files touched.** New: `src/citeclaw/steps/resolve_seeds.py`. Modified: `src/citeclaw/steps/load_seeds.py`. Register in `steps/__init__.py`. Update `src/citeclaw/config.py` seed schema to accept `{title: ...}` entries.
  - **Verify done.** New test in `tests/test_resolve_seeds.py` using `FakeS2Client.search_match` with a title that resolves to two distinct paper_ids (preprint + published).

- [ ] **PC-05. Verify all filter atoms tolerate `fctx.source=None`**
  - **What.** Audit `src/citeclaw/filters/atoms/*.py` and `src/citeclaw/filters/measures/*.py` for any access to `fctx.source` / `fctx.source_refs` / `fctx.source_citers` without a None check. Existing similarity measures should already handle it (CLAUDE.md claim — verify). For any filter that strictly requires a source, raise a clear error in its constructor / build-time check, not at runtime, with message: `"Filter X requires a source paper but was used in a source-less context (likely ExpandBySearch / ExpandBySemantics / ExpandByAuthor)"`.
  - **Files touched.** Possibly `src/citeclaw/filters/atoms/*.py`, `src/citeclaw/filters/measures/*.py`. New test asserting each atom + measure handles `source=None` without crashing.
  - **Verify done.** `pytest tests/ -x`.

- [ ] **PC-06. `search_model` global in `Settings` + seed schema update**
  - **What.** Add `search_model: str = ""` to `Settings` in `src/citeclaw/config.py` (empty → fall back to `screening_model`). Update seed config parsing to accept `{paper_id: ...}` OR `{title: ...}` (or both).
  - **Files touched.** `src/citeclaw/config.py`. Test in `tests/test_config.py`.
  - **Verify done.** `pytest tests/test_config.py -x`.

- [ ] **PC-07. Example YAML `config_bio_with_expansion.yaml`**
  - **What.** New file at project root demonstrating the full family. Do NOT modify `config_bio.yaml`. Shape:
    ```yaml
    seed_papers:
      - title: "Highly accurate protein structure prediction with AlphaFold"
      - title: "HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution"

    search_model: gemini-3-pro
    reasoning_effort: high

    pipeline:
      - step: ResolveSeeds
        include_siblings: true
      - step: LoadSeeds
      - step: ExpandForward
        max_citations: 100
        screener: forward_screener
      - step: ExpandBackward
        screener: backward_strict
      - step: Rerank
        metric: pagerank
        k: 10
        diversity: {type: walktrap, n_communities: 3}
      - step: ExpandBySearch
        agent: {max_iterations: 4, target_count: 150, reasoning_effort: high}
        screener: forward_screener
      - step: ExpandBySemantics
        max_anchor_papers: 10
        limit: 100
        screener: forward_screener
      - step: ExpandByAuthor
        top_k_authors: 10
        author_metric: h_index
        papers_per_author: 30
        screener: backward_loose
      - step: ReinforceGraph
        metric: pagerank
        top_n: 30
        screener: backward_loose
      - step: Rerank
        metric: citation
        k: 200
      - step: Finalize
    ```
    The `forward_screener` / `backward_strict` / `backward_loose` block definitions are copied from `config_bio.yaml`'s `blocks:` section verbatim. **The thesis: every new expand step reuses an existing screener — no new screening rules invented.**
  - **Files touched.** New: `config_bio_with_expansion.yaml`.
  - **Verify done.** `python -c "from citeclaw.config import load_settings; s = load_settings('config_bio_with_expansion.yaml'); assert len(s.pipeline_built) > 10"`.

- [ ] **PC-08. End-to-end stub-mode pipeline test**
  - **What.** New `tests/test_expand_family_e2e.py`. Uses stub LLM + `FakeS2Client` (with `search_bulk`, `fetch_recommendations`, `fetch_author_papers` extensions from PA-10). Minimal pipeline exercising the full chain: `ResolveSeeds → LoadSeeds → ExpandForward → Rerank → ExpandBySearch → ExpandBySemantics → ExpandByAuthor → Finalize`. Asserts:
    - Each step appears in the shape table output.
    - `ctx.collection` grows across each expand step.
    - `budget._s2_api` has entries for `search`, `recommendations`, `author_papers`.
    - `budget._llm_tokens.get("meta_search_agent", 0) > 0`.
    - Re-running the whole pipeline is a no-op (idempotency via `ctx.searched_signals`).
  - **Files touched.** New: `tests/test_expand_family_e2e.py`.
  - **Verify done.** `pytest tests/test_expand_family_e2e.py -x`.

- [ ] **PC-09. Docs update + full-suite smoke**
  - **What.** Update `CLAUDE.md` (note: gitignored, won't be pushed):
    - Add rows for `ResolveSeeds`, `ExpandBySearch`, `ExpandBySemantics`, `ExpandByAuthor`, `ReinforceGraph`, `HumanInTheLoop` to "Pipeline steps reference" table.
    - Add "Expansion family" section explaining the composable model with a pointer to `config_bio_with_expansion.yaml`.
    - Add `search_model` to the YAML schema example.
    - Add the fuzzy-title seed schema example.
  - Also update `README.md` (NOT gitignored, will be pushed) with brief mention of the new steps.
  - Then `pytest tests/ -x`.
  - **Files touched.** `CLAUDE.md`, `README.md`.
  - **Verify done.** `grep -q "ExpandBySearch" README.md && pytest tests/ -x`. **Phase C DONE** when green.

---

## Phase D — ReinforceGraph + Human-in-the-Loop (CLI)

- [ ] **PD-01. `ReinforceGraph` step v1**
  - **What.** New `src/citeclaw/steps/reinforce_graph.py`:
    ```python
    class ReinforceGraph:
        name = "ReinforceGraph"
        def __init__(self, *, metric="pagerank", top_n=30,
                     percentile_floor=0.9, screener): ...

        def run(self, signal, ctx):
            # 1. Build combined graph over ctx.collection ∪ ctx.seen via network.build_citation_graph.
            # 2. compute_pagerank(graph).
            # 3. For each rejected paper (in ctx.seen but not ctx.collection): compute its score.
            # 4. Select top_n by score AND above percentile_floor within rejected set.
            # 5. Hydrate via fetch_metadata / enrich_batch if stale.
            # 6. apply_block(candidates, self.screener, FilterContext(source=None)).
            # 7. Passed: stamp source="reinforced", add to ctx.collection, append to ctx.reinforcement_log.
            # 8. Return StepResult(signal=passed, ...).
    ```
    Module docstring labels this as v1; future versions can use betweenness, community-aware, or learned metrics.
  - **Files touched.** New: `src/citeclaw/steps/reinforce_graph.py`. Register in `steps/__init__.py`.
  - **Verify done.** New test `tests/test_reinforce_graph.py` with a hand-built collection + seen set where a high-pagerank rejected paper is restored.

- [ ] **PD-02. `HumanInTheLoop` step v1 (CLI)**
  - **What.** New `src/citeclaw/steps/human_in_the_loop.py`:
    ```python
    class HumanInTheLoop:
        name = "HumanInTheLoop"
        def __init__(self, *, k=10, timeout_sec=120,
                     include_accepted=True, include_rejected=True,
                     balance_by_filter=True): ...

        def run(self, signal, ctx):
            # 1. Build candidate pool: accepted = list(ctx.collection.values()).
            #    rejected = [p for p in known_papers if p.paper_id in ctx.rejection_ledger
            #                and any(cat.startswith("llm_") for cat in ctx.rejection_ledger[p.paper_id])]
            # 2. If balance_by_filter: per LLM filter name, sample roughly equal counts within each half.
            # 3. Shuffle k papers; present each via rich.prompt.Confirm with title/venue/year/abstract snippet.
            # 4. Collect labels; timeout → auto-continue with warning.
            # 5. Compute per-filter agreement (precision/recall vs. user labels).
            # 6. Write report to <data_dir>/hitl_report.json.
            # 7. If any filter's agreement < 0.5, prompt user: continue / stop.
            # 8. Return signal unchanged.
    ```
  - **Files touched.** New: `src/citeclaw/steps/human_in_the_loop.py`. Register in `steps/__init__.py`.
  - **Verify done.** New test mocking `rich.prompt.Confirm` with canned label sequence; asserts report is written and agreement computed.

- [ ] **PD-03. Integration test: HITL + ReinforceGraph in composed pipeline**
  - **What.** Extend `tests/test_expand_family_e2e.py` (or new file) with: `LoadSeeds → ExpandForward → HumanInTheLoop (mocked) → ExpandBySearch → ReinforceGraph → Finalize`.
  - **Verify done.** `pytest tests/test_expand_family_e2e.py -x`. **Phase D DONE.**

---

## Phase E — Web UI (parallel track)

Lives in `web/` subdirectory. Stack: React 18 + Vite + TypeScript + Tailwind v4 + shadcn/ui + sigma.js (ForceAtlas 2) + React Flow + FastAPI + WebSockets. Cron-Claude alternates Phase E with Phase C/D on alternating runs.

- [ ] **PE-01. Tech stack lock-in + monorepo scaffold**
  - **What.** Create `web/` with subdirs `web/backend/` (FastAPI scaffold: `pyproject.toml` or shared, `main.py` with `/health` endpoint, `.env.example`) and `web/frontend/` (Vite + React + TypeScript scaffold via `pnpm create vite`, Tailwind v4 + shadcn/ui installed, one "Hello CiteClaw" page rendering). Add `web/README.md` documenting the stack.
  - **Files touched.** New: `web/**`.
  - **Verify done.** `cd web/backend && uvicorn main:app --port 9999` returns 200 on `/health`; `cd web/frontend && pnpm dev` serves "Hello CiteClaw" at :5173.

- [ ] **PE-02. FastAPI REST endpoints**
  - **What.** Add to `web/backend/`:
    - `GET /api/configs` — list saved YAML configs in project root.
    - `GET /api/configs/{name}` — read YAML, return as JSON.
    - `POST /api/configs/{name}` — write a config from JSON (React Flow → YAML conversion).
    - `GET /api/papers/{paper_id}` — return PaperRecord as JSON (read from `data_bio/cache.db`).
    - `GET /api/runs/{run_id}` — return run state from `data_bio/run_state.json`.
    - `POST /api/runs` — trigger a new run with a config name; return `run_id`.
  - All endpoints reuse Pydantic models from `citeclaw.models` / `citeclaw.config` directly.
  - **Files touched.** `web/backend/api/*.py`, `web/backend/main.py`.
  - **Verify done.** `curl localhost:9999/api/configs` returns JSON.

- [ ] **PE-03. Pipeline event bus + WebSocket stream**
  - **What.** Refactor `src/citeclaw/pipeline.py::run_pipeline` to emit events to an injected `EventSink` abstraction with methods `step_start`, `step_end`, `paper_added`, `paper_rejected`, `shape_table_update`. Default sink is no-op (preserves current CLI behavior). New `src/citeclaw/event_sink.py`. In `web/backend/`, add `ws/run_stream.py` — WebSocket endpoint `ws://localhost:9999/api/runs/{run_id}/stream` that subscribes to the sink and pushes events.
  - **Files touched.** `src/citeclaw/pipeline.py`, new `src/citeclaw/event_sink.py`, new `web/backend/ws/run_stream.py`.
  - **Verify done.** New `tests/test_event_sink.py` with a recording sink asserting the expected event sequence.

- [ ] **PE-04. React scaffold: routing, layout, 3-pane shell**
  - **What.** In `web/frontend/`: React Router v6 with routes `/`, `/run/:runId`, `/configs/:name`. 3-pane layout via shadcn `ResizablePanelGroup` (left=paper detail, center=graph, right=config/run controls). Dark mode toggle. Top bar branded "CiteClaw". Zustand for client state, TanStack Query for server state.
  - **Files touched.** `web/frontend/src/**`.
  - **Verify done.** Visual: `pnpm dev` renders the 3-pane layout with placeholders.

- [ ] **PE-05. Sigma.js graph component with ForceAtlas 2**
  - **What.** New `web/frontend/src/components/Graph.tsx` using `@react-sigma/core` + `graphology` + `graphology-layout-forceatlas2`. Mounts a Sigma canvas, loads initial graph from `GET /api/runs/{run_id}/graph`. ForceAtlas 2 iterative layout running continuously at low intensity. Node click → emits event for `PaperPanel`. Color = cluster or source. Size = log(citation_count). Built-in zoom/pan/select.
  - **Files touched.** `web/frontend/src/components/Graph.tsx`, `web/frontend/src/hooks/useSigmaGraph.ts`.
  - **Verify done.** Visual: loading a cached run renders the citation network interactively.

- [ ] **PE-06. Live graph updates with "bouncing node" animation**
  - **What.** New `web/frontend/src/hooks/usePipelineRun.ts` — WebSocket hook subscribing to `ws://localhost:9999/api/runs/{run_id}/stream`, dispatches to Zustand. Graph reacts: on `paper_added`, add node + edges to graphology, bump ForceAtlas 2 iteration count, briefly pulse the new node (opacity 0→1 over 300ms, scale 0.5→1.2→1.0). On `step_start`, show toast banner. On `step_end`, update shape table.
  - **Files touched.** `web/frontend/src/hooks/usePipelineRun.ts`, `web/frontend/src/components/Graph.tsx`.
  - **Verify done.** Manual: trigger a real run via UI, watch nodes pop in.

- [ ] **PE-07. React Flow pipeline builder**
  - **What.** New `web/frontend/src/components/PipelineBuilder.tsx`. React Flow canvas with draggable nodes for each step type. Left drawer = "block library" with all step types. Drag blocks onto canvas, connect top-to-bottom. Each node has a settings gear opening a right-sidebar form for that step's config. Filter blocks nest inside screener slots. Save button → `POST /api/configs/{name}` (Flow JSON → YAML). Load button → reads YAML, rehydrates Flow.
  - **Files touched.** `web/frontend/src/components/PipelineBuilder.tsx`, `web/frontend/src/lib/pipelineSchema.ts`, `web/frontend/src/lib/yamlBridge.ts`.
  - **Verify done.** Manual: drag blocks, save, reload, verify fidelity.

- [ ] **PE-08. Paper detail sidebar + run controls**
  - **What.** New `web/frontend/src/components/PaperPanel.tsx` (left): title, abstract, venue, year, authors (clickable chips), citation metrics, source tag, rejection history (from `ctx.rejection_ledger`), "Open on S2" link. New `web/frontend/src/components/RunControls.tsx` (right): start/stop/resume buttons, live progress, budget consumed, shape table.
  - **Files touched.** `web/frontend/src/components/PaperPanel.tsx`, `web/frontend/src/components/RunControls.tsx`.
  - **Verify done.** Visual: clicking a graph node shows paper detail; starting a run updates live.

- [ ] **PE-09. HumanInTheLoop web integration**
  - **What.** When `HumanInTheLoop` runs, backend emits `hitl_request` event with the k sampled papers. Frontend shows shadcn `Dialog` modal with paper cards + yes/no buttons + progress bar. User submits → `POST /api/runs/{run_id}/hitl` → backend unblocks the step. Refactor `HumanInTheLoop.run()` to be awaitable on an external signal (asyncio.Event or shared dict).
  - **Files touched.** `src/citeclaw/steps/human_in_the_loop.py`, `web/backend/api/runs.py`, `web/frontend/src/components/HitlModal.tsx`.
  - **Verify done.** Manual e2e: run example YAML with HITL, click through modal, verify report is written.

- [ ] **PE-10. Polish + packaging**
  - **What.** `pnpm build` for production bundle. Wire FastAPI to serve the static files. Package as `python -m citeclaw web` CLI subcommand. Add screenshots / demo GIF to `README.md`.
  - **Files touched.** `src/citeclaw/__main__.py`, `web/README.md`, possibly `pyproject.toml` extras.
  - **Verify done.** `python -m citeclaw web --port 9999` serves the full UI on :9999. **Phase E DONE.**

---

## Phase F — Meta-review agent (HUMAN GATE — DO NOT IMPLEMENT)

**STOP.** Cron-Claude must not execute any Phase F task. Append a feedback entry "reached Phase F human gate, awaiting user approval" and exit immediately.

**Design summary** (for the human review session):
- Two-step pattern: `ReviewCollection` (LLM-driven, **read-only**, writes `meta_review_report.json`) and `ApplyReview` (pure Python, **no LLM**, dispatches bounded actions through existing primitives). User can run just the dry-run, inspect, then optionally apply.
- Action vocabulary (only four): `SuggestSeedSearch`, `SuggestAddPaper`, `SuggestRemovePaper`, `SuggestExpandBackward`. Each routes through an existing primitive (`ExpandBackward`, `ReScreen`, etc.).
- Hard caps enforced by `ApplyReview`: `max_iterations=5`, `max_removals=min(10, 0.1*len(collection))`, `max_additions=30`, `min_remove_confidence=0.85`, `require_rationale_chars=30`.
- Provenance: `PaperRecord.meta_review_notes`, `Context.meta_review_log`.
- Tool use: JSON-schema-enforced single-turn `LLMClient.call` with rolling transcript (no native tool-calling).
- Composition slot: after `Rerank` + `Cluster`, before `Finalize`.

Open questions for the design session:
- Does the agent see cluster labels? How?
- Sampling strategy for "collection preview"?
- Dry-run-by-default flag?
- Interaction with pipeline checkpointing?

---

## Risks and open questions

1. **Agent prompt quality** — unit tests prove plumbing, not reasoning. PB-06's manual script is the only real validation. Reserve a human review session before Phase D.
2. **S2 RPS under multi-expand load** — `s2_rps=0.9` is 1.1s/request. A 5-step expand pipeline can spend 100-250s on S2 calls. Mitigation: raise rps if API key allows; parallelize via existing `Parallel` step; aggressively cache.
3. **Filter `source=None` tolerance** — must be verified in PC-05 before any expand step ships.
4. **Web UI scope creep** — Phase E is ambitious. Ship E-01 to E-05 first as a demo. Don't block Phase C/D on E.
5. **ReinforceGraph v1 is deliberately dumb** — pagerank-rank-in-seen is a heuristic. Expect to iterate on the algorithm in v2.
6. **`ResolveSeeds` sibling cost** — each title-with-preprint-and-published triggers 2-3 extra S2 calls. Acceptable for ≤20 seeds; document.
