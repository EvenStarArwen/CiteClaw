# api-contract.md — the endpoint contract for the new UI

**Status.** Specification only. Nothing was implemented. Companion to
`backend-architecture.md` (storage and process shapes); this file owns wire
shapes.

**Derived from.** Recon report 2 (`v2_1b010c…md`, the demo's mock data
contracts) for what each screen consumes; recon report 3 (`v2_36c79c…md`) and
report 1 (`v2_01796d…md`) for the old endpoints and what can be kept;
`recon/synthesis.json` for the 135 gaps; `scratchpad/decisions-ledger.md` for
every product ruling; `citation-feasibility/REPORT.md` for the citation module;
`web/wiring/import-format-matrix.md` for import; `web/app/src/data/types.ts`
for the front end's own boundary.

**Every judgment call is marked `DECISION-B<n>`.** Index in §17.

---

## 1. Conventions

### 1.1 Base and framing

- Base path `/api/v1`. Streams at `/api/v1/...:stream` over WebSocket.
- JSON in, JSON out. `snake_case` keys. UTF-8.
- Timestamps: RFC 3339 UTC strings (`2026-08-13T09:14:02Z`). **Never
  pre-formatted relative strings.** The demo shipped `date:'yesterday'` and
  `dur:'13:01'` (report 2 §1.1d); the front end does the humanising, because
  the server does not know the reader's clock.
- Durations: seconds, as numbers (`elapsed_s`).

> **DECISION-B1. The wire speaks the backend's vocabulary, not the demo's.**
> `title`, not `ti`; `authors` as `[{author_id, name}]`, not `'A · B · C'`.
> Reason: `web/app/src/data/types.ts` states the rule itself — *"The vocabulary
> is the DEMO's vocabulary … the adapter's job is to translate CiteClaw's
> shapes INTO these."* The translation belongs in the adapter, in one file. It
> also keeps the middot out of the wire, which §1.5 requires anyway.

### 1.2 Errors

```json
{ "error": { "code": "seed_set_empty",
             "message": "Star at least one seed paper before starting a run.",
             "hint": "Papers you star in the search panel become the seed set.",
             "field": "seeds" } }
```

`message` is a complete, actionable sentence intended for direct display —
carried over from the old backend's best convention (report 1 §1.1: *"all error
copy is written as full sentences for the end user"*). `code` is stable and
machine-readable; `hint` and `field` are optional. HTTP status is meaningful
(400 validation, 402-equivalent uses 409 for budget refusal, 404, 409 conflict,
413 too large, 429 upstream rate limit, 503 upstream unreachable).

> **DECISION-B2.** Top-level `error` object rather than FastAPI's default
> `{"detail": …}`, via one exception handler. The old UI read `body.detail ||
> body.message`; the new adapter is new code, so this costs nothing and gives
> the front end a code to switch on for the missing-state work.

### 1.3 Collections and pagination

```json
{ "items": [...], "total": 17440, "offset": 0, "limit": 50,
  "capped": false, "cap_note": null }
```

`capped` + `cap_note` are inherited from the old rejected endpoint and exist
because upstream truncation is real (S2 serves at most 1000 search matches;
rejection detail has a hard ceiling). The note is a full sentence.

> **DECISION-B3. Server-side pagination for rejected papers and for corpus
> paper lists; whole-collection responses for topics, communities and graph
> edges.** The demo generated 17,440 rejected objects in memory at startup and
> paged client-side, and `types.ts` warns against inventing pagination the UI
> does not have. But the UI *does* have real pagers on both paper panels
> (`.rp-pager`, `.sb-pgmore`), so this is not invented UI — it is the same
> pager fed a page at a time. Topics (≈20), communities (≈12) and edges (≈1.2k)
> stay whole because the canvas needs them whole.

### 1.4 Identifiers and display

- Every entity has an opaque `id` (`p_…`, `r_…`, `v_…`, `j_…`).
- Runs additionally carry `number` (integer). The UI renders `Run 37` from
  `number`; the server never sends `"RUN-37"` and never sends a raw id for
  display. This satisfies the design's ban on printing raw ids.
- Papers are addressed by `paper_id` everywhere. **No response uses an array
  index as an identifier another response can refer to** (`backend-architecture.md`
  DECISION-A17). Where a payload carries index-based edges for canvas
  performance, it carries its own ordering in the same response (§11.2).

### 1.5 Display-string rules the server must honour

The design bans monospace, the middot separator `·`, and the em dash in
user-visible text. Any string the server produces that reaches the screen —
filter labels, rejection reasons, step names, error messages, version names,
budget copy — must be written without them. This retires the old backend's
`fmtReason()` output style (`LLM · else · filter 3`).

> **DECISION-B4.** Server-produced display strings are plain sentences or
> comma-separated phrases. Where the old code produced a middot-joined path, the
> new field is structured instead (`{filter_key, filter_label, branch_label}`)
> and the front end composes. Structured beats formatted: it survives a copy
> change without a server release.

### 1.6 Capability discovery

`GET /api/v1/capabilities` returns what is actually wired, so the front end can
render honest disabled states instead of guessing:

```json
{ "agent_panel": false, "library": false, "author_network": false,
  "cost_dashboard": false, "live_log": false, "pause_resume": true,
  "limit_raise": true, "citation_statements": true, "topic_model": true,
  "community_model": true, "import_formats": ["bib","ris","csv","txt","pdf"],
  "engine_fields": { "accepted_at_step": true, "times_hits": true,
                     "relevance": false, "llm_confidence": false } }
```

`agent_panel: false` is the ledger's Q5: the Explore chat panel stays visually
present but dead, and sending anything must produce
`Assistant unreachable — no backend is wired to this panel.` (ledger Q19,
verbatim). `relevance` and `llm_confidence` are `false` because the backend has
no such notion yet and the ledger's Q4 pins them to `1.00` for now — the flag
lets the UI stop pretending later without a redeploy of both sides.

### 1.7 Field maturity markers

Fields marked **`pending-engine`** below depend on engine work listed in
`backend-architecture.md` §5. Until that lands the server returns `null` (never
a fabricated value). The front end must render `null` as absence, not as zero.
This is the direct lesson of synthesis risk 4: the demo's PDF availability was
a title hash, so 62 percent of papers claimed full text at random.

---

## 2. Meta and auth

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/api/v1/health` | `{status, version, data_root_ok, db_ok}` |
| GET | `/api/v1/capabilities` | §1.6 |
| POST | `/api/v1/auth/session` | Decorative. Accepts anything, returns the local session. Not called by the Login screen this phase (ledger Q2, Q16). |
| GET | `/api/v1/auth/me` | `{tenant_id:"local", label:"Local", multi_tenant:false}` |
| POST | `/api/v1/auth/logout` | No-op locally; exists so the tenant layer has a target. |

Health is also what the System Banners layer's connection set drives off. The
banner's countdown and backoff are front-end concerns; the server only has to
answer or not answer.

---

## 3. Settings

### 3.1 Read / write

`GET /api/v1/settings`

```json
{ "screening_model": "gemini-3.1-flash-lite",
  "keys": { "openai":   {"present": true,  "probe": {"state":"valid",  "checked_at":"…"}},
            "gemini":   {"present": true,  "probe": {"state":"valid",  "checked_at":"…"}},
            "anthropic":{"present": false, "probe": {"state":"not_set","checked_at":null}},
            "s2":       {"present": true,  "probe": {"state":"rate_limited",
                                                     "message":"This key is being rate limited. Requests still work, just slower."}} },
  "budgets": { "max_accepted_papers": 200, "max_screened_papers": 5000,
               "max_execution_minutes": 90, "llm_budget_usd": 25 },
  "budget_note": "A run stops gracefully when it reaches any cap." }
```

`PUT /api/v1/settings` takes `screening_model` and `budgets`; **key values are
never returned**, only presence and probe state.

`PUT /api/v1/settings/keys` takes `{openai?, gemini?, anthropic?, s2?}`; empty
or absent fields leave the stored value untouched (matching the old modal's
behaviour), and an explicit `null` clears.

`POST /api/v1/settings/keys/probe` → `{results:[{provider, state, message}]}`,
where `state ∈ not_set | set | valid | invalid | rate_limited | unreachable`,
matching the six chip states the Settings screen already renders. Probing is a
real, cheap upstream call per provider.

> **DECISION-B5. Anthropic gets a fourth key row.** The demo's model menu offers
> `claude-haiku-4` and `claude-sonnet-4` but Settings has only three key fields
> (diff report C15a) — as shipped, picking a Claude model can never work. The
> ledger's Q12 says Anthropic is being wired, so the key row is required. **The
> UI change is not mine to make**: raised as `E-BE-02` for the design batch.

> **DECISION-B6. Model list comes from the server**, `GET /api/v1/models` →
> `[{id, provider, label, group_label, supported, requires_key, reasoning,
> efforts}]`. The new Settings menu shows names only (prices were dropped,
> diff report R27), but `requires_key` lets the UI grey out a model whose
> provider key is missing instead of failing at run start. Prices are still
> returned and simply unused, so restoring the price line later is a UI-only
> change.

### 3.2 The four budgets are enforced, not decorative

Ledger Q12 chose "all four honoured".

| Budget | Enforcement | Behaviour at the limit |
| --- | --- | --- |
| `max_accepted_papers` | engine `max_papers_total` | graceful stop, partial results finalised; if `capabilities.limit_raise`, a `limit_prompt` event is emitted first (§7.3) |
| `max_screened_papers` | new counter over papers seen by any screener | graceful stop |
| `max_execution_minutes` | wall clock from run start | graceful stop, `stop_reason: "time_budget"` |
| `llm_budget_usd` | `budget.py` cost accumulator against `MODEL_PRICING` | graceful stop, `stop_reason: "cost_budget"` |

All four surface live on the `metrics` event as
`budgets:[{key, used, limit, pct, state}]` with `state ∈ ok | warning |
exhausted` (`warning` at 80 percent, which is what drives the monitor's budget
ring turning amber).

> **DECISION-B7. Graceful stop, not hard kill.** Every budget terminates through
> the same `_StopRun` path that the Stop button uses, so partial results are
> finalised and the version-1 corpus still exists. A budget that discards the
> run's work would be worse than no budget.

---

## 4. Projects

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v1/projects` | Home list |
| POST | `/api/v1/projects` | wizard create |
| GET | `/api/v1/projects/{id}` | detail, incl. active job and Explore pointer |
| PATCH | `/api/v1/projects/{id}` | rename, retopic, touch `opened_at` |
| POST | `/api/v1/projects/{id}/duplicate` | Home ⋯ menu |
| DELETE | `/api/v1/projects/{id}` | soft delete |

List row:

```json
{ "id":"p_01J…", "name":"AI agents for scientific discovery",
  "topic":"How autonomous agents are being used to run and design experiments.",
  "paper_count":354, "run_count":8,
  "opened_at":"2026-08-13T07:02:11Z", "created_at":"…",
  "thumbnail": {"kind":"net", "seed":11},
  "active_job_id": null }
```

`thumbnail.kind ∈ net | pipe` and `seed` reproduce the demo's generative
preview: the Home thumbnails are canvas drawings, not screenshots (report 2
§2.2). `kind` is `net` once the project has a completed run, `pipe` before that.
`seed` is stable per project so the picture does not change between visits.

Create (the three-step wizard):

```json
POST /api/v1/projects
{ "topic": "…composer text…",
  "scope":    { "year_from": 2018, "year_to": null, "citation_momentum": 10 },
  "seeds":    { "paper_ids": ["…"], "import_id": null },
  "boundary": { "paper_types": "methodological_only",
                "preprints": "include", "surveys": "exclude" } }
→ 201 { "project": {...}, "pipeline": {...} }
```

> **DECISION-B8. The wizard's three steps are stored as given and translated
> into a starting pipeline server-side**, and the translation is returned so
> Build opens showing exactly what was created. Storing the wizard answers
> separately from the pipeline (schema: `scope_json`, `boundary_json`) means a
> later change to the mapping does not lose what the user actually said.
> Mapping: `scope.year_*` → a `YearFilter`; `citation_momentum` → a
> `CitationFilter` with `beta = momentum`; `boundary.preprints:"exclude"` → a
> venue predicate using the engine's existing `preprint` venue preset;
> `paper_types` and `surveys` → LLM title/abstract criteria. **The last two are
> a genuine judgment call** — they are prose intentions with no mechanical
> equivalent — and the generated criteria text should be reviewed by the product
> owner; raised as `E-BE-03`.

> **DECISION-B9. "Library" gets no endpoints.** Ledger Q9 cut the cross-project
> paper table and Q20 keeps the right-hand column visible but inert. Adding
> endpoints for a dead panel would invite it being wired by accident. The panel
> renders its demo appearance and nothing calls the server.

### 4.1 Pipeline document

| Method | Path | |
| --- | --- | --- |
| GET | `/api/v1/projects/{id}/pipeline` | the Build canvas document |
| PUT | `/api/v1/projects/{id}/pipeline` | save |
| POST | `/api/v1/pipeline/validate` | pre-flight, returns the refusal reasons |
| GET | `/api/v1/pipeline/presets` | Scout / Survey / Dragnet |

Document shape (mirrors the Build canvas, not the engine YAML):

```json
{ "steps": [
    { "uid":"s1", "kind":"source", "code":"SED", "seeds":{"paper_ids":[…]} },
    { "uid":"s2", "kind":"fwd", "params":{…},
      "filters":[ {"uid":"f1","type":"year","config":{"from":2018,"to":null}},
                  {"uid":"f2","type":"llm","config":{"target":"title",
                     "tree":{"op":"and","children":[{"kind":"q","text":"…"}]},
                     "model":null,"reasoning_effort":null}} ] },
    { "kind":"parallel", "branches":[[…],[…]] } ],
  "unsupported": [],
  "raw_preserved": true }
```

> **DECISION-B10. The UI model is a strict subset of the engine's, and the
> untranslatable remainder is preserved, read-only.** Ledger Q10 chose exactly
> this. Concretely: a pipeline that came from hand-written YAML and uses `Route`,
> `Any`, `Not`, nested `Sequential`, linked copies, or an unexposed step kind is
> returned with those parts listed in `unsupported[]` (`{path, kind, label,
> reason}`) and preserved verbatim in `raw_preserved`. `PUT` round-trips them
> untouched. **Saving must never silently drop a block the editor cannot draw** —
> that would be the worst possible failure of this feature.

> **DECISION-B11. The LLM query tree is the wire format; the `formula` string
> is an engine detail.** The new Build editor produces a tree
> (`{op:'and'|'or', children:[{kind:'q'|'group', …}]}`) while the engine takes a
> boolean `formula` plus a named `queries` map (diff report C8). The server
> converts. Round-tripping a hand-written formula back into a tree is
> best-effort; when it fails, the filter appears in `unsupported[]` rather than
> being mangled.

`POST /api/v1/pipeline/validate` returns the Runs screen's refusal banner
content directly:

```json
{ "ok": false,
  "reasons": [ {"code":"llm_criterion_empty","message":"Step 3's abstract screen has no criterion. An empty criterion accepts every paper.","step_uid":"s3"},
               {"code":"missing_key","message":"This pipeline screens with Gemini, and no Gemini key is set. Open Settings to add one.","provider":"gemini"} ] }
```

> **DECISION-B12. The empty-LLM-criterion check is kept.** The old app checked it
> on both sides after a real incident (`e1ede4c`: an empty criterion silently
> accepted every paper). The new UI has no equivalent (diff report R25, listed
> `unclear` in the synthesis). It costs one validation rule and prevents a
> class of silently wrong runs. One to four reasons is exactly what the refusal
> banner renders.

---

## 5. Paper search and abstracts

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/api/v1/papers/search?q&limit&offset&year_from&year_to&min_cites` | reuses `s2_seeds.py` unchanged |
| POST | `/api/v1/papers/abstract` | reuses `abstracts.py`; OpenAlex fallback |
| GET | `/api/v1/papers/{paper_id}` | single record, for the abstract drill |

Search response keeps the old `{total, offset, next, items}` shape plus
`capped`/`cap_note` carrying the honest S2 message (*"Semantic Scholar serves at
most the first 1,000 matches for a query."*). The old UI showed that; the new
one has no place for it yet (diff report R18) — the field exists so the design
lane can place it.

`POST /api/v1/papers/abstract` returns `{abstract, source}` where `source` is
`"semantic_scholar" | "openalex" | null`. The new abstract drill drops the
`via openalex` note (diff report R19); the field is still returned because
provenance is cheap to keep and expensive to re-add.

---

## 6. Import

Formats and phases are fixed by `import-format-matrix.md` §2.9: pilot is
`.bib`, `.ris`, `.csv`, DOI/arXiv `.txt`, and single PDFs; `.zip` and batch PDF
are phase 2; JSON needs `E-IMP-01` sign-off; EndNote and Zotero-native are
`later` with reject-with-guidance copy.

| Method | Path | |
| --- | --- | --- |
| POST | `/api/v1/imports` | multipart upload, returns `{import_id, job_id}` |
| GET | `/api/v1/imports/{id}` | parse + resolve state, per file and per entry |
| WS | `/api/v1/jobs/{job_id}:stream` | determinate progress for large files |
| PATCH | `/api/v1/imports/{id}/entries/{entry_id}` | choose a candidate, tick/untick |
| POST | `/api/v1/imports/{id}/commit` | apply to the target |
| DELETE | `/api/v1/imports/{id}` | abandon; staging files are deleted |

`POST /api/v1/imports` body is multipart: `files[]` plus
`target={"kind":"seed_set","project_id":"p_…"}` or
`{"kind":"corpus","version_id":"v_…"}`. The three demo entry points (Home
wizard step 2, Build sidebar, Runs Add papers) differ only in `target` — the
resolver is one flow, exactly as the demo shares one `KLImport`.

`GET /api/v1/imports/{id}`:

```json
{ "id":"i_…", "status":"review",
  "files":[ {"name":"refs.bib","ext":"bib","bytes":48211,
             "state":"parsed","entry_count":34,"error":null} ],
  "counts":{"extracted":34,"matched":28,"ambiguous":2,"duplicate":1,"no_match":3},
  "entries":[
    { "id":"e_1","state":"ok","source_file":"refs.bib","source_locator":"line 33",
      "parsed":{"title":"…","authors":"…","year":2024,"venue":"…"},
      "resolved":{"paper_id":"…","title":"…","venue":"…","year":2024,"citation_count":41},
      "candidates":null,"checked":true },
    { "id":"e_2","state":"multi","candidates":[
        {"paper_id":"…","venue":"Nature","year":2024,"citation_count":88,"note":"Journal version"},
        {"paper_id":"…","venue":"arXiv","year":2023,"citation_count":35,"note":"Preprint"}],
      "checked":false },
    { "id":"e_3","state":"none","reason":"Title not found on Semantic Scholar.","checked":false },
    { "id":"e_4","state":"dupe","reason":"Already in the corpus.","checked":false } ] }
```

The four states `ok | multi | none | dupe` are a hard UI contract (report 2
§1.8): they drive the four review groups in fixed triage order (Needs a
decision, Couldn't match, Matched, then the duplicate group whose label is
supplied by the caller — "In the seed set" on Build, "Already in the corpus" on
Runs).

> **DECISION-B13. The server asserts `extracted == matched + ambiguous +
> duplicate + no_match` per file and per import and fails loudly if it does
> not.** This is non-negotiable #1 in the format matrix; the promise the review
> screen exists to keep is that nothing is silently dropped.

> **DECISION-B14. Extension decides the parser; content mismatch is a reported
> error, not a rescue.** The demo dispatches purely on extension (`extOf`), so a
> BibTeX file named `.ris` fails. Format matrix non-negotiable #3 says pick a
> rule and pin it. Extension-first is chosen because sniffing turns one clear
> failure into an unpredictable one; the error message names both what the
> extension promised and what the content looked like.

> **DECISION-B15. Resolution is partitioned before it starts.** Id-bearing
> entries go to `POST /paper/batch` at 500 per request (mirror-backed,
> unthrottled); title-only entries queue against `search/match` at roughly 1
> request per second. Never resolve in file order — format matrix §3.1. The job
> event therefore carries two progress tracks, and the UI's per-file spinner is
> a lie for large files (registered as `MS-IMP-02`, not mine to fix).

> **DECISION-B16. Ambiguity comes from relevance search, not from `search_match`.**
> `search_match` returns one best match and cannot produce the `N matches`
> popover. The server issues a relevance search and returns candidates above a
> score threshold, preferring the preprint/published sibling pair that
> `ResolveSeeds(include_siblings=True)` already knows how to find — which is
> exactly the demo's two-candidate shape (Journal version / Preprint).

`POST /api/v1/imports/{id}/commit` body `{entry_ids:[…]}`; response reports what
landed and what did not:

```json
{ "added": 29, "skipped_duplicate": 1, "skipped_unmatched": 3,
  "version_id": "v_…"   // present only when the target was a corpus
}
```

Unmatched entries are **never added** — the demo's own rule, restated in
`import-format-matrix.md`. Staging files are deleted on commit and on abandon;
nothing an import touches becomes a stored library object.

---

## 7. Runs: lifecycle and event stream

### 7.1 Lifecycle

| Method | Path | |
| --- | --- | --- |
| POST | `/api/v1/runs` | `{project_id, pipeline?, seeds?, limits?}`; omitting `pipeline` uses the project's saved one |
| GET | `/api/v1/runs?project_id=&limit=&offset=` | the All-runs popover; grouped client-side by day |
| GET | `/api/v1/runs/{id}` | full snapshot, see §7.2 |
| PATCH | `/api/v1/runs/{id}` | rename |
| DELETE | `/api/v1/runs/{id}` | soft delete; refuses while running |
| POST | `/api/v1/runs/{id}/stop` | |
| POST | `/api/v1/runs/{id}/pause` | |
| POST | `/api/v1/runs/{id}/resume` | |
| POST | `/api/v1/runs/{id}/limit-decision` | `{prompt_id, action:"raise"\|"stop", new_limit?}` |

> **DECISION-B17. Pause, resume and the limit-raise prompt are in the contract.**
> The new demo dropped both (diff report R5, R7) but ledger Q11 rules them back
> in, with the raise prompt redrawn as a centred modal over a dimmed app reusing
> existing modal styling. The backend endpoints and events for both already
> exist in `run_manager` and are simply renamed here.

`POST /api/v1/runs` returns `201 {run, job_id}` or `400` with the same
`reasons[]` array `POST /api/v1/pipeline/validate` produces, so the refusal
banner has one shape to render regardless of which call refused.

> **DECISION-B18. `Run pipeline` gets real behaviour on Build too.** Ledger
> Round 4 records that the dead primary button is a demo bug and must start a
> run when wired. Same endpoint; busy, disabled and failure states are a design
> batch item, not an API one.

### 7.2 Run snapshot

`GET /api/v1/runs/{id}` — everything needed to render the Runs screen cold,
with no stream. This is the fix for debt P1 (refresh used to lose the run).

```json
{ "id":"r_…", "number":37, "label":"AI agents for scientific discovery",
  "project_id":"p_…", "status":"running", "outcome":null,
  "started_at":"…", "ended_at":null, "elapsed_s":781,
  "job_id":"j_…", "last_seq":10432,
  "prisma": { "accepted":354, "rejected":17440, "steps_done":5, "steps_total":7 },
  "steps":[ { "idx":1, "code":"FWD-02", "kind":"fwd", "name":"Forward Searcher",
              "wave":2, "branch":0, "state":"done",
              "papers_in":7, "found":341, "kept":45, "rejected":296,
              "filters":[ {"key":"year","label":"Year","seen":341,"passed":267,"rejected":74,
                           "config":{"from":2018,"to":null}},
                          {"key":"citation","label":"Citation","seen":267,"passed":92,"rejected":175} ],
              "calls":{"graph":2,"reco":0},
              "tokens":{"in_title":4200,"in_abstract":12100,"in_db":0,"out":1675} } ],
  "metrics": { "calls":{"graph":316,"reco":0,"total":316},
               "tokens":{"in_title":36166,"in_abstract":117395,"in_db":0,"out":41360,"total":194921},
               "rejections":[{"key":"citation","label":"Citation","count":10328}],
               "budgets":[{"key":"llm_budget_usd","used":3.90,"limit":25,"pct":15.6,"state":"ok"}] },
  "versions_summary": {"count":3, "latest_version_id":"v_…"},
  "notices":[ {"level":"warning","code":"s2_degraded",
               "message":"Semantic Scholar is unreachable. Fetching is paused; screening continues.",
               "since":"…","retries":6} ] }
```

Field notes, all traceable to the demo's contracts:

- `steps[].papers_in / found / kept / rejected` are the step drill's four hero
  numbers. `found` and the per-filter cascade do not exist in today's
  `shape_summary.json` — **`pending-engine`**.
- `filters[].seen / passed / rejected` are the 16-cell unit bars. The invariant
  the demo asserts holds server-side: the per-filter rejections of a step sum to
  the step's `rejected`, and **duplicates are counted in the filter table but
  not in the run's rejected total** (report 2 §1.1a).
- `wave` groups consecutive steps that share an input. The demo derives waves
  from equal `in` counts; the server knows the pipeline structure and states it
  outright, so the fork copy ("both read the 7 seeds") and the join arithmetic
  ("45 + 8 → 53 papers continue") are computable without inference.
- `tokens` split by scope (title screening, abstract screening, database search,
  output) is what the monitor's LLM group shows. Today's budget accounting has
  categories but not this split — **`pending-engine`**.

### 7.3 Event stream

`WS /api/v1/jobs/{job_id}:stream?since=<seq>`. Every event is
`{seq, type, at, ...payload}`. Types:

| Type | Payload | Notes |
| --- | --- | --- |
| `hello` | `{job, run?, snapshot, last_seq}` | first frame is always a complete snapshot, so a client never renders from a partial replay |
| `phase` | `{phase, detail}` | `starting \| running \| pausing \| paused \| resuming \| stopping \| finalizing \| done`. `detail` carries the staged stop copy ("Finishing the current batch.", "Writing the checkpoint.", "Saving partial results.") as real transitions, not a timer |
| `step` | one `steps[]` element, in full | replaces by `idx`; sent on state change and on counter change (throttled) |
| `paper_accepted` | `{paper, accepted_at_step, times_hits, relevance, llm_confidence, marker}` | **per-step acceptance attribution** — ledger Q4 |
| `paper_rejected` | `{paper_id, step_idx, filter_key, reason_text}` | drives the monitor's rejection bars and the rejected list's `why` chip |
| `hits` | `{updates:[{paper_id, times_hits}]}` | live cumulative, coalesced; frozen at `done` — ledger Q18 |
| `metrics` | as in §7.2 `metrics` | throttled, latest-wins on replay |
| `prisma` | `{accepted, rejected, steps_done, steps_total}` | the hero counters, cheap and frequent |
| `budget` | `{key, used, limit, pct, state}` | emitted on threshold crossings so the ring can animate without polling |
| `graph` | `{paper_ids:[…], edges:[[i,j,w]]}` | canvas; `i`/`j` index `paper_ids` **of this event** (DECISION-A17) |
| `limit_prompt` | `{prompt_id, kind, current, suggested, timeout_s, message}` | the raise modal; ledger Q11 |
| `limit_resolved` | `{prompt_id, action}` | closes it, including on timeout |
| `notice` | `{level, code, message, since, retries}` | the three-tier degradation ladder: silent retries emit nothing, panel-level warnings emit `warning`, only a dead backend is an app-level banner |
| `version_created` | `{version}` | refine jobs; also emitted at run end for version 1 |
| `done` | `{status, summary, version_id, stop_reason}` | `status ∈ completed \| stopped \| failed`; `stop_reason ∈ user \| paper_budget \| screen_budget \| time_budget \| cost_budget \| error` |
| `error` | `{code, message}` | |

> **DECISION-B19. No `log` event.** Ledger Q11 cut the live log panel and Q19
> cut the heartbeat. Emitting a stream nothing consumes is how the old system
> ended up with nine implemented capabilities and no UI. The engine still writes
> `citeclaw.log` to the run directory, and ledger Q17 asks for that file to get
> *richer* (pipeline and filter configuration, PRISMA counts) precisely because
> debugging moved from the screen to the disk.

### 7.4 Guarantees the front end may rely on

1. **Exactly one step is `active` at any time.** Parallel branches report
   `state` and counters but never `active`; the wave is the active unit
   (`backend-architecture.md` DECISION-A10). The design forbids two pulsing
   dots and this is where that is made true.
2. **Counters are monotonic within a run.** No progress ever goes backwards; a
   stop freezes them rather than recomputing.
3. **`seq` is gapless per job.** A client that has seen `seq=n` can resume from
   `n` and miss nothing.
4. **`done` is terminal and is always sent**, including for failures and stops,
   and is present in replay so a client attaching to a finished job sees the
   whole story and then a close.

### 7.5 Rejected papers

`GET /api/v1/runs/{id}/rejected?offset&limit&sort&q&filter_key&step_idx&year_from&year_to&min_cites`

`sort ∈ recent | filter | year | cites | title`. Returns the standard collection
envelope. Rejected rows carry **no abstract** — the engine keeps none for a
reject, and the demo makes that explicit (`abs:''`). The drill must render an
honest empty state, not a spinner.

> **DECISION-B20. `filter_key` and `step_idx` are queryable.** The new filter
> panel has a `Step accepted` control on the accepted side; the rejected side
> needs the mirror ("rejected at which step, by which filter") because the
> monitor's rejection bars are clickable in spirit even if the demo did not wire
> them. Cheap given the telemetry tables already exist.

---

## 8. Corpus versions

### 8.1 Semantics, stated once

- **Append-only.** A version is never mutated after creation. `Restore` does
  not truncate the chain; it appends a new version whose content equals the
  restored one and whose `restored_from_version_id` records the source. Versions
  that a restore jumped over are marked so the UI can fold them into
  `Superseded`.
- **v1 is the run's own result** and is created by the same code path whether
  the run came from the engine or from the legacy importer.
- **Manual verdicts are pins, and pins outrank later re-screens.** A paper the
  user accepted or rejected by hand keeps that state through subsequent
  `rescreen` operations. A re-screen reports how many pins it honoured
  (`pinned_kept`) so the number is visible rather than mysterious. A pin is
  released only by another manual edit on the same paper.
- **Papers added by hand and papers merged in arrive pinned.** They were chosen,
  not screened; a later re-screen must not quietly delete a user's choice.
- **Consecutive edits coalesce.** If the tip version is an `edit` and the next
  operation is also an `edit`, the moves fold into the tip version rather than
  appending a version per click. Any other operation kind closes the edit
  version.

> **DECISION-B21.** The coalescing rule above is a judgment call taken from the
> design's own description of the Edit chain. It has one visible consequence:
> `updated_at` on a version can move even though the chain is append-only. The
> alternative — a version per bulk selection — produces chains of thirty
> versions from one afternoon of triage.

### 8.2 Reading the chain

| Method | Path | |
| --- | --- | --- |
| GET | `/api/v1/runs/{id}/versions` | the whole chain |
| GET | `/api/v1/versions/{vid}` | one version |
| PATCH | `/api/v1/versions/{vid}` | rename |
| GET | `/api/v1/versions/{vid}/papers` | paginated members |
| GET | `/api/v1/versions/{vid}/changes` | the delta versus the parent |
| POST | `/api/v1/versions/{vid}/restore` | appends a new version |
| GET | `/api/v1/versions/{vid}/prisma` | PRISMA flow numbers for the diagram export |

Chain entry:

```json
{ "id":"v_…", "v":3, "kind":"rescreen",
  "name":"Re-screen, accepted, at least 10 cites per year",
  "detail":"176 fetched, 7 accepted, 169 rejected",
  "created_at":"…",
  "summary":{"added":7,"removed":169},
  "accepted_count":354,"rejected_count":17440,
  "parent_version_id":"v_…","restored_from_version_id":null,
  "superseded_by_version_id":null,
  "job_id":"j_…","params":{…} }
```

`kind ∈ run | edit | rescreen | grow | merge | add | seed | restore` — the five
version icons the UI draws plus the two operations that were not in the demo's
inline chain.

`GET /api/v1/versions/{vid}/papers?state=accepted|rejected&sort&q&…` returns
members with their change markers:

```json
{ "items":[ {"paper_id":"…","title":"…","authors":[{"author_id":"…","name":"…"}],
             "venue":"…","year":2024,"citation_count":41,
             "full_text_available":true,"full_text_source":"mirror",
             "accepted_at_step":"FWD-02","times_hits":6,
             "relevance":null,"llm_confidence":null,
             "marker":"rescued","pinned":true,
             "topic_id":2,"community_id":5} ],
  "total":354,"offset":0,"limit":50 }
```

`marker ∈ rescued | added | removed | new | null` — the four change rails, null
when this version did not touch the paper. `relevance` and `llm_confidence` are
`null` here rather than `1.00`: ledger Q4 pins the *displayed* value to 1.00
because the semantics are undecided, and a server that invents 1.00 makes it
impossible to tell later whether a value is real. **The 1.00 is a UI constant,
and this is recorded so it does not become a data lie.**

> **DECISION-B22.** `relevance` and `llm_confidence` are returned as `null` with
> `capabilities.engine_fields.relevance = false`, and the front end renders the
> ledger's 1.00 placeholder. Judgment call; the alternative (server sends 1.00)
> is simpler now and untraceable later.

### 8.3 Draft mode

Re-screen and verdict editing both have a preview state in the UI ("Previewing,
+12 −5 versus v3, Discard / Apply") before a version exists.

> **DECISION-B23. Drafts live on the client; the server offers a stateless
> preview.** `POST /api/v1/versions/{vid}/ops/{op}/preview` returns the full
> move list for a proposed operation without writing anything. Reason: a
> server-side draft is a second mutable state next to an append-only chain, and
> the two disagree the moment a browser tab closes. Metadata-only operations
> (year, citation, keyword, venue) preview instantly and exactly; LLM
> re-screens cannot be previewed exactly and return an *estimate* instead
> (§8.5), which is precisely why the design gates those behind a cost
> confirmation.

### 8.4 The six operations

All six append a version. All six return `202 {job_id, optimistic:{…}}` and
report completion on the job stream with `version_created`. Fast ones
(`verdicts`, `add`) complete before the response returns and include
`version_id` directly.

| Op | Path | Body | Version kind |
| --- | --- | --- | --- |
| Edit verdicts | `POST /api/v1/versions/{vid}/ops/verdicts` | `{moves:[{paper_id,to:"accepted"\|"rejected"}]}` | `edit` |
| Re-screen | `POST …/ops/rescreen` | `{scope:"accepted"\|"rejected"\|"both", filters:[…]}` | `rescreen` |
| Grow | `POST …/ops/grow` | `{directions:["forward","backward","semantic"], from:"accepted"\|"all", apply_criteria:true}` | `grow` |
| Merge a run | `POST …/ops/merge` | `{source_run_id, source_version_id?}` | `merge` |
| Add papers | `POST …/ops/add` | `{paper_ids:[…], import_id?}` | `add` |
| Add seed papers | `POST …/ops/seed` | `{paper_ids:[…], searcher?:{kind:"db"\|"fwd"\|"bwd", params:{…}}}` | `seed` |

Notes per operation:

- **Verdicts** sets pins. Bulk selection and single-row swipe use the same call.
- **Re-screen** reuses the engine's `ReScreen` step, which is why it is first in
  the build order. Its `filters` payload is the same filter list shape as a
  Build step's, so the Runs re-screen editor and the Build config panel share
  one serialisation.
- **Grow** is a delta run over the current corpus, screened with the current
  criteria. `directions` may hold one to three values; the demo defaults to all
  three.
- **Merge** brings another run's accepted set in, deduped, arriving pinned.
  `source_version_id` defaults to that run's latest version.
- **Add papers** is the corpus picker's commit; papers arrive pinned.
- **Add seed papers** widens and re-runs, then merges with dedupe. This is the
  only operation that can produce new steps in the spine.

Dedupe for merge, add and seed uses the engine's existing `MergeDuplicates`
logic (DOI/arXiv, title similarity, SPECTER2 cosine) lifted into the import and
refine paths rather than reimplemented.

### 8.5 Dry-run estimation

`POST /api/v1/versions/{vid}/ops/{op}/estimate` with the same body as the
operation:

```json
{ "papers_affected": 512,
  "would_accept": 7, "would_remove": 169, "pinned_kept": 23,
  "exact": false,
  "cost_usd": {"low": 0.40, "high": 1.10},
  "duration_minutes": {"low": 2, "high": 6},
  "requires_confirmation": true,
  "basis": "Estimated from 512 abstracts at the current screening model's price, plus this project's observed tokens per abstract.",
  "assumptions": {"model":"gemini-3.1-flash-lite","tokens_per_paper_est":1180,"concurrency":8} }
```

> **DECISION-B24. Estimates are ranges with a stated basis, never a single
> number.** The design's copy is "papers, approximate dollars, minutes"; a bare
> `$0.73` reads as a quote. `basis` is displayable so a surprising number can be
> understood rather than distrusted.

> **DECISION-B25. `requires_confirmation` is computed server-side.** It is true
> when the operation calls a model or the network in bulk (LLM, keyword over
> full text, venue lookups, similarity) and false for pure metadata filters,
> which the design already distinguishes by showing an instant delta for the
> latter. Putting the rule on the server keeps the two sides from drifting.

> **DECISION-B26. `exact:false` for anything touching an LLM.** Screening
> outcomes are not deterministic; `would_accept` for an LLM re-screen is a
> projection from the previous run's acceptance rate on comparable papers, and
> the flag tells the UI to phrase it as an estimate. Metadata-only operations
> return `exact:true` with the real numbers, which is what makes the live delta
> in the re-screen panel trustworthy.

---

## 9. Topology suggestions

`GET /api/v1/versions/{vid}/suggestions?measure=cites_corpus|shared_bibliography&limit=5`

```json
{ "measure":"cites_corpus", "corpus_size":354, "candidate_pool":46304,
  "items":[ { "paper_id":"…","title":"…","authors":[…],"venue":"…","year":2025,
              "citation_count":207,
              "evidence":{ "kind":"cites_corpus",
                           "text":"Cites 53 of 354 accepted papers.",
                           "numerator":53,"denominator":354 },
              "linked_paper_ids":["…","…"],
              "best_overlap":{"paper_id":"…","title":"…","shared":24} } ] }
```

> **DECISION-B27. The raw score never crosses the wire as a display field.** The
> design states the score is never shown; only the evidence sentence and its
> ratio. Sending `score: 53` invites it onto the screen. The server sends the
> sentence *and* its two numbers so the UI can draw the proportion bar without
> reconstructing prose.

`linked_paper_ids` are real citation edges into the current corpus, used for the
canvas ghost preview. The demo faked these (`kpMockLinks`, whose own comment
says *"real wiring supplies real links"*); they must be real here or the preview
is decoration.

---

## 10. Citation statements

Ledger Q7 chose sentences only: no section classification, no TLDR. The
feasibility report settles the numbers: of 354 papers, 220 (62 percent) have at
least one statement, 96 have no in-corpus citer at all, and 38 have citers but
no retrievable sentence.

`GET /api/v1/versions/{vid}/papers/{paper_id}/citation-statements?offset&limit&q`

```json
{ "state": "has_statements",
  "cited_paper_id":"…",
  "citer_count": 35, "statement_count": 142,
  "groups":[ { "citing_paper_id":"…", "title":"…", "first_author_surname":"Sur",
               "year":2024, "label":"Sur et al. (2024)",
               "statements":[ {"text":"…the passage…","truncated":true} ] } ],
  "total": 35, "offset": 0, "limit": 8 }
```

`state` is the field that produces the **two distinct empty states** the ledger
requires:

| `state` | Meaning | UI |
| --- | --- | --- |
| `has_statements` | at least one sentence survives filtering | the module renders |
| `no_internal_citers` | no paper in this corpus cites it (96 of 354) | empty state A: nothing in this corpus cites it |
| `no_statements_available` | citers exist, no sentence retrievable (38 of 354) | empty state B: the citation exists, the sentence is not available |
| `not_fetched` | never attempted | loading or a Try again affordance |
| `error` | upstream failure, with `message` | Try again |

> **DECISION-B28. The two empty states are a server-computed field, not a
> client inference.** The client cannot tell them apart without the citation
> edge graph; making it derive the distinction guarantees the wrong copy will
> eventually show. Copy for both is the design lane's (ledger: "two distinct
> empty states, wording by design agent, approved by the user").

Server-side data hygiene, all three approved by the ledger:

1. **Filter junk sentences**: drop `very_short` (fewer than 40 characters) and
   `low_alpha` (letters below 55 percent of characters). Roughly 3 percent of
   sentences. Dropped rows are *kept in the database* with `dropped_reason` set
   so thresholds can be retuned without re-fetching.
2. **Dedupe by statement text** (normalised hash) — the same flattened table
   caption otherwise attaches to several cited papers.
3. **Ellipsis truncation is preserved as-is**; `truncated:true` is a hint, not
   an instruction to repair. About 20 percent of S2 contexts end mid-sentence.

`GET /api/v1/versions/{vid}/citation-statements/availability` →
`{"available_paper_ids":[…], "states":{"<paper_id>":"no_internal_citers", …}}`
so the left-hand list can decide, per row, whether the module is offered at all.
The feasibility report's first launch recommendation is exactly this: do not
show an empty panel for the 96 papers nothing cites.

`POST /api/v1/versions/{vid}/citation-statements/refresh` — the Try again path;
enqueues a fetch job for one paper or for the version.

> **DECISION-B29. Direction comes from `references`, not from the undirected
> graph.** The feasibility report found the run's `graph.json` edges are
> undirected and are also missing about 2 percent of the in-corpus citation
> pairs present in `corpus.json.references`. Building the citation edge table
> from references is both directional and more complete.

> **DECISION-B30. Fetch from the cited side, keyless, at 1 request per second,
> and cache forever.** `/paper/{id}/citations` is used rather than
> `/paper/{id}/references` because publishers elide the `references` field on
> some records (the report reproduced this on a Science paper) while the
> citations view is assembled from other papers' parsed full texts and is not
> affected. Keyless because the report's A/B/A tests showed both of the user's
> S2 keys performing about twice as badly as anonymous access — the key's quota
> bucket is congested. **This is a workaround, not a fix**; the standing action
> item is to check those keys with Semantic Scholar, and the medium-term
> direction is the `citations` bulk dataset (same data, no rate limit,
> ODC-BY redistributable). Raised as `E-BE-04`.

---

## 11. Explore data

The working corpus is one pointer per project: a `(run, version)` pair.

| Method | Path | |
| --- | --- | --- |
| GET | `/api/v1/projects/{id}/explore-pointer` | `{run_id, version_id, version_summary}` |
| PUT | `/api/v1/projects/{id}/explore-pointer` | Show in Explore |
| GET | `/api/v1/versions/{vid}/graph` | canvas |
| GET | `/api/v1/versions/{vid}/topics` | groups, semantic space |
| GET | `/api/v1/versions/{vid}/communities` | groups, citation space |
| POST | `/api/v1/versions/{vid}/topic-model` | on-demand modelling, returns a job |
| POST | `/api/v1/versions/{vid}/community-model` | ditto |

### 11.1 Provenance and staleness

Every grouping response carries `{model_id, built_from_version_id, method,
params, created_at}`. When `built_from_version_id != vid`, the response also
sets `stale:true` and the UI shows its "Built from v2" chip.

> **DECISION-B31. Staleness is reported, never acted on.** No automatic
> recompute, no cache invalidation. The design says staleness is information,
> not failure, and a corpus edit silently triggering a paid modelling job would
> be a bad surprise.

### 11.2 Graph

```json
{ "paper_ids": ["…", "…"],
  "edges": [[0, 7, 1.0], [6, 7, 1.0]],
  "node_count": 354, "edge_count": 1239 }
```

Indices address `paper_ids` **in this response only**. Any other endpoint that
needs to name a paper uses `paper_id`. This is the deliberate containment of
synthesis risk 3.

### 11.3 Groups

```json
{ "model": {"id":"gm_…","space":"community","method":"leiden",
            "params":{"resolution":1.0,"seed":42},
            "modularity":0.61,"nmi_vs_topic":0.44,"ari_vs_topic":0.31,
            "built_from_version_id":"v_…","stale":false},
  "peripheral": {"max_size":1, "id":-2},
  "groups":[ {"id":0,"name":"Autonomous Chemistry Labs",
              "description":"Running synthesis experiments and mining materials literature with robotic platforms.",
              "keywords":["…"],"size":57,"cx":630.28,"cy":549.58,
              "internal_edges":214,"external_edges":88,
              "neighbours":[{"group_id":3,"edges":41}],
              "anchor_paper_id":"…","bridge_paper_id":"…"} ] }
```

> **DECISION-B32. `peripheral.max_size` is a parameter, not a hard-coded rule.**
> The demo's `community-data.js` carries a comment written straight to whoever
> wires the backend: Leiden and friends routinely produce size-one communities,
> and the UI must fold everything at or below `PERIPHERAL_MAX` into one
> synthetic group with id `-2` — *generically, never hard-coding this run's
> count*. Sending the threshold makes that literal.

`modularity`, `nmi_vs_topic` and `ari_vs_topic` feed the "How this was computed"
popover. Leiden itself does not exist in the repo yet (`pending-engine`,
ledger Q6 approves building it).

### 11.4 Topic map coordinates

Per-paper `x`/`y` come back on `/versions/{vid}/papers` (§8.2) rather than as a
separate endpoint, because every consumer that wants coordinates also wants the
paper row. Today the topic model reduces to five dimensions and does not
persist intermediates — `pending-engine`.

### 11.5 What Explore does not get

No agent endpoints, no chat persistence, no skills registry, no literature
review generation or storage (ledger Q5). No author network, no external graph
upload, no arbitrary run browser (ledger Q8). `capabilities` reports all of
these as false and the UI shows the fixed unreachable message.

---

## 12. Exports and downloads

| Method | Path | |
| --- | --- | --- |
| GET | `/api/v1/versions/{vid}/export/papers.csv` | Accepted papers, Corpus papers |
| GET | `/api/v1/versions/{vid}/export/collection.bib` | BibTeX |
| GET | `/api/v1/versions/{vid}/export/graph.graphml` | |
| GET | `/api/v1/versions/{vid}/export/stats.json` | |
| GET | `/api/v1/versions/{vid}/export/prisma.svg` | the version menu's PRISMA diagram |
| GET | `/api/v1/versions/{vid}/export/bundle.zip` | Full bundle |
| GET | `/api/v1/runs/{id}/export/bundle.zip` | the run directory as-is |

Version 1 exports stream the engine's own artifacts; later versions are
regenerated from the database. Every download menu item in the demo was a fake
that flashed "Preparing download" and produced nothing; these are the routes
that make it real. The Literature review Markdown item in Explore's menu has no
backend (agent panel not wired) and must be hidden or disabled rather than
returning an empty file — raised as `E-BE-05`.

---

## 13. Jobs

| Method | Path | |
| --- | --- | --- |
| GET | `/api/v1/jobs/{id}` | snapshot |
| GET | `/api/v1/jobs?project_id=&status=` | what is running |
| POST | `/api/v1/jobs/{id}/cancel` | cooperative |
| WS | `/api/v1/jobs/{id}:stream?since=` | §7.3 |

The top bar's Runs-tab activity dot is `GET /api/v1/jobs?status=running` scoped
to the project, or the `active_job_id` already present on the project record —
no separate endpoint.

---

## 14. Legacy run importer

Ledger Q17: existing runs become version-1 corpora.

| Method | Path | |
| --- | --- | --- |
| GET | `/api/v1/legacy-runs` | scan and report candidates |
| POST | `/api/v1/legacy-runs/import` | `{path, project_id?, project_name?}` |

```json
{ "candidates":[
   { "path":"runs/data", "detected":"citeclaw_run",
     "artifacts":{"collection":true,"bib":true,"citation_graphml":true,
                  "rejections":true,"shape_summary":true,"run_state":true,
                  "pipeline_config":false},
     "papers":21, "generations":["", "exp2", "exp3"],
     "modified_at":"…", "importable":true, "note":null },
   { "path":"runs/webui/7985659b126f", "detected":"citeclaw_run",
     "artifacts":{"collection":false},
     "importable":false,
     "note":"This run stopped before writing its results. Only its cache remains." } ] }
```

Import semantics:

- Creates `run(origin='legacy_import')` plus `corpus_version(v=1, kind='run')`
  through the same writer as a fresh run (DECISION-A18).
- Fields the old artifacts do not contain are `null`, never invented:
  `accepted_at_step`, `times_hits`, per-step `found/kept`, per-filter cascade.
  `rejections.json` gives category counts, so run-level rejection totals survive
  while per-step attribution does not.
- Continuation generations (`literature_collection.exp2.json` …) map to
  **additional versions** of the same run, in order, with
  `kind='grow'` and a `detail` naming the generation. This is the one place
  where a legacy artifact genuinely represents a chain, and flattening it would
  throw away real history.
- `runs/web/<ts>_<hex>` directories (the dead first-generation backend) are
  listed with `importable:false` and a note, rather than hidden — a directory
  that silently does not appear reads as a bug.

> **DECISION-B33.** Import is a copy-by-reference: `run.data_dir` points at the
> existing directory and nothing is moved or rewritten. A legacy import is
> therefore reversible by deleting rows. If the orchestrator prefers copying
> into a managed layout, that is a one-line change but it makes the operation
> destructive-adjacent, so the safe default is chosen.

---

## 15. Mapping table: demo mock to endpoint

For the adapter author. Left column is what the screens consume today.

| Demo source | Replaced by |
| --- | --- |
| `KLRunMock.PIPELINE` / `RUN37.steps` | `GET /runs/{id}` `.steps` |
| `KLRunMock.RUN37.filters` | `.steps[].filters`, plus `metrics.rejections` for the monitor |
| `KLRunMock.RUN_LIBRARY`, `NEXT_RUN_NO` | `GET /runs?project_id`; `number` is assigned server-side |
| `makeReplay(...).frameAt(ms)` | the job event stream (§7.3). There is no frame function; the UI accumulates events |
| `KLRunMock.RECO` | `GET /versions/{vid}/suggestions` |
| `topic-data.js` `PAPERS` | `GET /versions/{vid}/papers` |
| `topic-data.js` `TOPICS` + `topic-desc.js` | `GET /versions/{vid}/topics` (description is a field, not a second module) |
| `graph-data.js` `EDGES` | `GET /versions/{vid}/graph` |
| `community-data.js` `LEIDEN` + colours + `PERIPHERAL_*` | `GET /versions/{vid}/communities`; colours stay in the front end |
| `citation-context.js` `SUBJECT`/`GROUPS` | `GET /versions/{vid}/papers/{pid}/citation-statements` |
| `citation-context.js` `SUMMARY` | **nothing** — the synthesised summary is agent work, not wired (ledger Q5, Q7) |
| `review-draft.js` `REVIEW_MD` | **nothing** — not wired |
| `import-resolver.js` `parse/groups` | §6 |
| Runs inline `mkRej()` (17,440 synthetic rejects) | `GET /runs/{id}/rejected` |
| Runs inline `rnChainFor()` | `GET /runs/{id}/versions` |
| Home `_defProjects()` | `GET /projects` |
| Home `_defCollections()` | **nothing** — library cut (ledger Q9, Q20) |
| Settings hard-coded keys and budgets | §3 |
| `paper-row.js` `prPdf()` title hash | `full_text_available` on the paper record. Never hash-derived |
| Explore's random `step`/`rel`/`hits`/`llm` | `accepted_at_step` and `times_hits` are real; `relevance` and `llm_confidence` are `null` plus a UI constant (DECISION-B22) |

---

## 16. Error codes

| Code | Status | Where |
| --- | --- | --- |
| `seed_set_empty` | 400 | run start |
| `llm_criterion_empty` | 400 | run start, validate |
| `missing_key` | 400 | run start, validate |
| `model_unsupported` | 400 | settings, run start |
| `pipeline_untranslatable` | 400 | pipeline save |
| `run_already_active` | 409 | run start |
| `run_not_running` | 409 | stop, pause, resume |
| `version_not_tip` | 409 | any op on a non-latest version |
| `budget_exhausted` | 409 | run start when a budget is already spent |
| `import_too_large` | 413 | import, per the ceilings in the format matrix |
| `import_format_unsupported` | 415 | import |
| `upstream_rate_limited` | 429 | S2 |
| `upstream_unreachable` | 503 | S2, OpenAlex, model provider |
| `not_wired` | 501 | any agent-panel surface, so a mis-wired client fails loudly |

---

## 17. DECISION index

| Id | Decision |
| --- | --- |
| B1 | Wire speaks backend vocabulary; the front-end adapter translates |
| B2 | Top-level `error` envelope with a stable `code` |
| B3 | Server-side paging for papers and rejects; whole collections for topics, groups, edges |
| B4 | Structured fields instead of pre-formatted middot strings |
| B5 | Anthropic key row is required (UI change escalated) |
| B6 | Model catalogue served, including unused price fields |
| B7 | Budgets stop gracefully through the existing stop path |
| B8 | Wizard answers stored separately from the generated pipeline |
| B9 | Library gets no endpoints |
| B10 | Untranslatable pipeline blocks preserved read-only, never dropped |
| B11 | LLM query tree is the wire format |
| B12 | Empty-LLM-criterion validation is kept |
| B13 | Import conservation invariant asserted server-side |
| B14 | Extension decides the parser; mismatch is a reported error |
| B15 | Import resolution partitioned by identifier availability |
| B16 | Ambiguous candidates from relevance search plus sibling logic |
| B17 | Pause, resume and limit-raise are in the contract |
| B18 | Build's Run pipeline button starts a real run |
| B19 | No `log` event; richer on-disk logs instead |
| B20 | Rejected list is queryable by step and filter |
| B21 | Consecutive edits coalesce into the tip version |
| B22 | `relevance` and `llm_confidence` are `null`; 1.00 is a UI constant |
| B23 | Drafts are client-side; the server offers a stateless preview |
| B24 | Estimates are ranges with a stated basis |
| B25 | `requires_confirmation` computed server-side |
| B26 | `exact:false` for anything involving a model |
| B27 | Raw suggestion scores never cross the wire as display fields |
| B28 | The two citation empty states are a server-computed field |
| B29 | Citation direction from `references`, not the undirected graph |
| B30 | Keyless S2 citation fetch as a documented workaround |
| B31 | Staleness is reported, never acted on |
| B32 | Peripheral threshold is a parameter |
| B33 | Legacy import references the existing directory, does not move it |

Escalations from this lane are in `escalations.md` under `E-BE-*`.
