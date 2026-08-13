# backend-architecture.md — the server that will feed the new UI

**Status.** Specification only. No code was written, no code was changed. This
document and `api-contract.md` are the deliverables; everything here is a
proposal for the orchestrator and the product owner to accept or reject.

**Sources this is derived from** (read in full, nothing inferred from memory):

| Source | Used for |
| --- | --- |
| `scratchpad/decisions-ledger.md` (Rounds 1–4) | binding product decisions; every "the ledger says" below is quoting it |
| recon report 1 — `v2_01796d…md` | old `web/live` architecture, endpoint list, WS protocol, debt list P1–P15 |
| recon report 2 — `v2_1b010c…md` | the new UI's mock data contracts (what the screens actually consume) |
| recon report 3 — `v2_36c79c…md` | engine capability inventory, run artifact schemas, old endpoints |
| recon report 4 — `v2_225802…md` | screen-by-screen feature inventory |
| recon report 5 — `v2_b3a013…md` | new-vs-old feature diff, the 29 removed capabilities |
| `recon/synthesis.json` | 135 gaps (48 `backend_missing`, 50 `needs_adaptation`), 20 risks |
| `scratchpad/citation-feasibility/REPORT.md` | citation-statement coverage, quality, filter thresholds |
| `web/wiring/import-format-matrix.md` | import formats, phases, non-negotiables |
| `web/live/backend/*.py`, `web/public/backend/*.py`, `web/app/src/data/types.ts` | read directly |

Cross-reference: `api-contract.md` is the endpoint-level companion. Where the
two disagree, `api-contract.md` wins on wire shapes and this file wins on
storage and process shapes.

---

## 0. The one-paragraph answer

Build a **new server package** that **imports the existing `web/live/backend`
engine-facing modules unchanged, as a library**, and owns everything else
itself: HTTP surface, persistence, versioning, jobs, tenancy seam. Do not
extend `web/live/backend/server.py` in place, and do not start from zero.
Persistence is **one SQLite file** (WAL, single writer) that indexes and
versions what the run directories already contain on disk; run directories stay
the artifact store. The invite-code tenant layer plugs in later through five
named seams that are present from day one and cost nothing while there is only
one local user.

---

## 1. Extend `web/live/backend`, or a new server package?

### 1.1 What is actually in `web/live/backend` (measured)

| Module | Lines | Verdict for the new UI | Why |
| --- | --- | --- | --- |
| `run_manager.py` | 1146 | **Reuse, extend additively** | The only implementation of stop / pause / resume / cap that exists. Stop is not task cancellation: a flag is checked in the dashboard hook and `_StopRun` is raised so it unwinds through the pipeline into `finalize_partial` (recon 1 §2.2). Rewriting this is re-earning a year of bug fixes (`43c3981` "truthful stop flow" alone). |
| `config_translate.py` | 465 | **Reuse as the base of a new translator** | Encodes real, painful knowledge: empty-LLM-criterion rejection (`e1ede4c`), alias resolution, per-`LLMFilter` key validation. But the new Build model is a flat filter list with a query **tree**, not a `formula` string (diff report C8), so the front half must be rewritten. Keep the file, add a new entry point. |
| `s2_seeds.py` | 112 | **Reuse verbatim** | Search + the 1000-result reach cap + three user-facing 429/401 messages. Nothing about it is old-UI-specific. |
| `abstracts.py` | 115 | **Reuse verbatim** | OpenAlex abstract fallback. The new abstract drill still needs it. |
| `models_catalog.py` | 174 | **Reuse, extend** | Hard-won per-model API quirks (2.5-gen rejects `thinking_level`, some ids 404). Needs Anthropic entries (ledger Q12). |
| `keys_store.py` | 91 | **Reuse behind an interface** | Fine locally. Writes `.env.local` and force-overwrites `os.environ`, which the multi-tenant layer must not do — hence the `KeyProvider` seam in §4. |
| `snapshots.py` | 242 | **Reuse the metrics half, replace the graph half** | `build_metrics` maps well. `build_graph` emits index-based edges `{a,b,w}` while `explore_runs` emits id-based `{source,target}` and the demo uses `[i,j,w]` — three incompatible shapes, one of the top synthesis risks. New server emits exactly one. |
| `explore_runs.py` | 645 | **Do not reuse** | It is a directory scanner standing in for a database, and it serves the two features the ledger cut (author network Q8, external graph upload Q8). Its job is taken over by the SQLite index (§3) plus the legacy-run importer (`api-contract.md` §14). |
| `server.py` | 519 | **Do not reuse** | Its whole top half assembles the old single-page HTML (`ASSEMBLY_ORDER`, CDN `<head>`, `window.__TWEAKS`). Its API half is shaped by old-UI vocabulary (`GET /api/explore/run?path=`) and holds process-global mutable defaults (`_defaults`, `server.py:165`). |

### 1.2 The three options

**Option A — extend `web/live/backend/server.py` in place.**
Cheapest on day one, and wrong for four reasons.

1. **`web/public` imports these modules.** `web/public/backend/server.py`
   imports `web.live.backend.*` and layers auth/tenancy/limits on top; that
   deployment is live at `cola-lab--citeclaw-web-serve.modal.run`. Editing
   `server.py`'s routes edits the deployed public app by construction.
2. `server.py` serves the old UI from `/`. The new UI is a built Vite SPA
   (`web/app`, React 18 + react-router, `dist/`). One file cannot be both
   an assembler of in-browser-Babel JSX and an SPA host without becoming a
   fork with an `if`.
3. The old route vocabulary is wrong at the root. There is no project, no
   version, no job; `runs/` is addressed by filesystem path
   (`/api/explore/run?path=`). Every new route would be a second vocabulary
   living next to the first.
4. Process-global mutable state (`_defaults`) and CWD-relative `runs/`
   (debt P2, P8) get inherited rather than fixed.

**Option B — a clean-start server with no reuse.**
Rejected. It means reimplementing `run_manager` (stop/pause/cap semantics
coupled to the engine's `dashboard` and `event_sink` Protocols — synthesis
risk 10), the model-quirk catalog, and the S2 error copy. That is the highest
regression-risk work in the repo and none of it is what the new UI needs
changed.

**Option C — new package, old modules imported as a library. RECOMMENDED.**

```
web/server/                      NEW — owns HTTP, storage, versioning, jobs
  app.py                         FastAPI app factory, SPA host, error envelope
  context.py                     RequestContext: tenant_id, data_root, keys, limits
  db/
    schema.sql                   §3 schema
    migrate.py                   schema_migrations, forward-only
    store.py                     single writer connection, WAL, busy_timeout
  domain/
    projects.py  runs.py  versions.py  corpus.py  pins.py
    citations.py imports.py     settings.py legacy_import.py
  jobs/
    registry.py                  one job abstraction for run + refine + model + import
    events.py                    the event catalogue in api-contract.md §7
  engine/
    adapter.py                   the ONLY module that imports web.live.backend.*
    translate.py                 new Build model  ->  CiteClaw config
    telemetry.py                 per-step / per-filter counters -> DB rows
  routes/                        thin; one file per api-contract.md section
```

`web/live/backend` stays where it is, keeps running on port 8787, and keeps
serving `web/public`. `web/server/engine/adapter.py` imports it by package
path — `from web.live.backend import run_manager, s2_seeds, abstracts,
models_catalog` — which is the import style `web/public` already uses and which
works because `web/live/backend/__init__.py` exists while `web/` and
`web/live/` are PEP 420 namespace packages (verified: no `__init__.py` at those
levels, yet `web/public/backend/server.py` imports this way today).

> **DECISION-A1.** New package under `web/server/`. Name chosen to sit beside
> `web/app` (the frontend) and to avoid `web/backend` (dead first generation)
> and `web/api` (reads like a route folder). If the orchestrator prefers
> `web/omniknowledge` or similar, only this path token changes.

> **DECISION-A2.** `web/live/backend` is treated as a **vendored library, not a
> shared codebase**. Rules: (a) the new server may import from it; (b) it may
> not import from the new server; (c) changes to it must be **purely additive**
> and must keep `tests/test_web_live_translate.py`, `test_web_live_models.py`,
> `test_web_live_stop_flow.py`, `test_web_rejected_page.py`,
> `test_web_public_auth.py`, `test_web_public_server.py` green. Rationale for
> allowing any change at all: the ledger's Q4 approves changing the engine
> event protocol, and `run_manager` is where those events are emitted.

> **DECISION-A3.** Additive-only changes to `run_manager` are safe for the old
> UI because `live-store.jsx::_handleEvent` is a switch on `type` that ignores
> unknown types (recon 1 §1.3). New event types and new fields on existing
> events therefore cannot break `web/live` or `web/public`. Removing or
> renaming anything is forbidden.

> **DECISION-A4.** Port **8788** for the new server (8787 is `web/live`;
> 5273/5274 are the Vite dev/preview ports pinned by `web/app/vite.config.ts`
> with `strictPort`). Env override `OK_SERVER_PORT` / `OK_SERVER_HOST`.

> **DECISION-A5.** Serving model: production serves `web/app/dist` as static
> files with an SPA fallback to `index.html`, single origin, no CORS — the one
> genuinely good property of the old server (recon 1 §5.1). Development runs
> Vite on 5273 with a proxy for `/api` and `/ws` to 8788. **The proxy entry
> must be added to `web/app/vite.config.ts`, which this lane must not touch**
> — raised as `E-BE-01`.

### 1.3 What the new server must own that nothing today does

From `synthesis.json` (48 `backend_missing`) and diff report §4, the new-build
surface is: projects, corpus versions and the six refinement operations, dry-run
estimation, per-step × per-filter telemetry, citation statements, bulk import
resolution, topology suggestions, topic 2D coordinates + Leiden communities,
budgets and their enforcement, export/download, and the legacy-run importer.
None of these have a home in `web/live/backend`; all of them are §3 tables plus
`web/server/domain/*`.

---

## 2. Runtime shape

### 2.1 One job abstraction, not five

The UI renders progress for six different long operations: a run, a re-screen,
a grow, a merge, an add-seeds re-run, an import resolution, and (Explore) topic
/ community modeling. The demo fakes all of them with `setTimeout`. If each
gets its own ad-hoc progress mechanism, the front end grows five stream
readers.

> **DECISION-A6.** One `job` record and one event channel shape for all of
> them. A run is a job of `kind='run'`; a re-screen is `kind='rescreen'`. The
> run-specific event types (`step`, `paper_accepted`, …) are a superset used
> only by run-shaped jobs; every job emits `hello` / `phase` / `progress` /
> `notice` / `done` / `error`. See `api-contract.md` §7.

### 2.2 Transport: WebSocket, with a REST fallback and a resume cursor

Keep the existing WebSocket rather than moving to SSE: `run_manager.subscribe()`
already implements backlog replay with high-frequency-stream compaction
(latest-only for `metrics`/`graph`/`activity`, last-100 for `log`), and the
public deployment's Modal container already proxies WS.

The old system's worst bug is P1: `runId` lived only in React memory, so a
refresh lost the run while it kept running on the server, and the two endpoints
written for that case (`/api/run/{id}/status`, `/graph`) were never called.

> **DECISION-A7.** Every job event carries a monotonic `seq`. The stream
> accepts `?since=<seq>` and replays from there. `GET /api/v1/jobs/{id}` returns
> a complete snapshot so a cold client never needs the stream to render. The
> active job id is discoverable from `GET /api/v1/projects/{id}` so a refresh
> reattaches without any client-side storage.

> **DECISION-A8.** `rs.events` grows without bound today (one entry per accepted
> paper, debt P5). The new server persists events to the `job_event` table with
> the same `seq` and keeps only a bounded in-memory tail; replay past the tail
> reads the table. This also makes "reopen a finished run and watch its stream"
> work, which the UI's version-chain PRISMA views want.

### 2.3 Run execution stays a thread

`RunManager` starts a daemon thread per run and stop/pause are cooperative
flag checks inside the engine's dashboard hook. That coupling is deep
(synthesis risk 10) and out of scope to change.

> **DECISION-A9.** Keep the thread model. The job registry wraps it rather than
> replacing it. Concurrency limit: one *engine* job at a time per project by
> default (a second POST returns `409` with the running job id); modeling and
> import jobs are cheap and may run alongside.

### 2.4 The strict-serial progress invariant

`design-system.md` requires that exactly one step is `active` at any moment;
parallel branches are expressed in words and arithmetic, never two pulsing
dots. The engine's `Parallel` step is genuinely concurrent.

> **DECISION-A10.** The server, not the client, enforces the invariant. The
> telemetry layer maps a concurrent wave onto one `active` step: the wave is the
> active unit, its branches report `state` and counts but never `active`. This
> is stated as an API guarantee in `api-contract.md` §7.4 so the front end can
> rely on it instead of normalising defensively.

### 2.5 Paths and working directory

Debt P2: `runs/` is resolved relative to the process CWD, so starting the
server anywhere but the repo root silently reads and writes the wrong place.

> **DECISION-A11.** A single `data_root` resolved once at startup, in this
> order: `OK_DATA_ROOT` env → `<repo>/runs` if it exists → platform user-data
> dir. Everything (SQLite file, run dirs, import temp dirs, export staging)
> hangs off it. No module resolves a relative path at call time. This is
> deliberately shaped like `web/public/backend/paths.py`, which is what makes
> §4 cheap.

---

## 3. Persistence

### 3.1 The choice

> **DECISION-A12. SQLite**, one file at `<data_root>/omniknowledge.db`, WAL
> mode, `busy_timeout=5000`, foreign keys on.

Considered and rejected:

- **JSON files per project** (what `web/public` does with `session.json`).
  Fine for a session blob; wrong here. The version chain needs
  "which papers are in version 4 of run 12, filtered by year, sorted by
  citations, page 3" — that is a query, and the rejected side is ~17k rows
  per run.
- **Postgres.** The deployment target this phase is a single local user
  (ledger Q16). Requiring a server process to open the app is a product
  regression. The schema below is deliberately portable if it ever moves.
- **Reusing the engine's `cache.db`.** That file is a read-through cache of
  external data with its own 9-table schema and is symlinked/shared across
  runs (`cache_sync.py`). Mixing application state into it would make cache
  eviction destructive. Separate file.

Precedent: the engine already depends on SQLite and the public deployment
already ships a SQLite file, so this adds no dependency.

**The one operational caveat, and it matters for §4:** `web/public/backend/paths.py`
records that *SQLite must not live on the Modal volume — FUSE file locking is
unreliable*, which is why the cache lives on container-local disk and is
snapshotted by `cache_sync.py`. The same rule applies to
`omniknowledge.db`. This is why `data_root` is a function and not a constant.

### 3.2 What the database is, and what it is not

> **DECISION-A13. The database is an index and a version ledger. Run
> directories remain the artifact store.** Blobs the engine already writes —
> `literature_collection.json`, `.bib`, `citation_network.graphml`,
> `rejections.json`, `shape_summary.json`, `pipeline_config.json`,
> `citeclaw.log` — are not copied into the DB. The DB stores the columns the UI
> queries, filters, sorts and paginates on, plus everything the engine does not
> persist at all (versions, pins, projects, citation statements, telemetry).
> Rule of thumb: if only "Download full bundle" needs it, it stays a file.

Consequence for exports: version 1 downloads stream the run directory as-is;
downloads of later versions are regenerated from the DB at request time.

### 3.3 Schema sketch

Column types are SQLite affinities. `id` columns are opaque text
(`p_`/`r_`/`v_`/`j_` prefix + ULID); UI-facing numbering is a separate integer
column so the front end can render `Run 37` without parsing an id
(`api-contract.md` §1.4).

```sql
-- ── meta ────────────────────────────────────────────────────────────────
CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT);

-- ── tenancy seam (§4) ───────────────────────────────────────────────────
CREATE TABLE tenant (
  id TEXT PRIMARY KEY,              -- 'local' in this phase
  label TEXT, created_at TEXT
);

-- ── projects ────────────────────────────────────────────────────────────
CREATE TABLE project (
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL REFERENCES tenant(id),
  name TEXT NOT NULL,               -- derived from topic, user-renamable
  topic TEXT NOT NULL,              -- the composer text
  scope_json TEXT,                  -- wizard step 1: years, citation momentum
  boundary_json TEXT,               -- wizard step 3: paper types, preprints, surveys
  pipeline_json TEXT,               -- the Build canvas document (see §3.4)
  pipeline_raw_json TEXT,           -- untranslatable config preserved verbatim (ledger Q10)
  thumbnail_kind TEXT,              -- 'net' | 'pipe'   (Home preview painter)
  thumbnail_seed INTEGER,
  explore_version_id TEXT,          -- the working-corpus pointer (one per project)
  created_at TEXT, updated_at TEXT, opened_at TEXT,
  deleted_at TEXT                   -- soft delete; Home's Delete is confirmable
);
CREATE INDEX ix_project_tenant ON project(tenant_id, deleted_at, opened_at DESC);

-- ── runs ────────────────────────────────────────────────────────────────
CREATE TABLE run (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES project(id),
  number INTEGER NOT NULL,          -- per tenant, monotonic; UI renders "Run 37"
  label TEXT,                       -- defaults to the project topic
  status TEXT NOT NULL,             -- queued|running|paused|stopping|completed|stopped|failed
  outcome TEXT,                     -- Completed|Stopped|Failed  (UI vocabulary)
  data_dir TEXT NOT NULL,           -- artifact directory under data_root
  origin TEXT NOT NULL,             -- 'engine' | 'legacy_import'
  pipeline_snapshot_json TEXT NOT NULL,  -- what was actually run, frozen
  started_at TEXT, ended_at TEXT, elapsed_s REAL,
  error TEXT,
  deleted_at TEXT
);
CREATE UNIQUE INDEX ux_run_number ON run(project_id, number);

-- per-step telemetry: the step drill's four hero numbers, per step
CREATE TABLE run_step (
  run_id TEXT NOT NULL REFERENCES run(id),
  idx INTEGER NOT NULL,             -- position in the spine
  code TEXT NOT NULL,               -- 'FWD-02'  (prefix from _STEP_META + ordinal)
  kind TEXT NOT NULL,               -- seed|fwd|bwd|db|sem|rrk|rsc|sink
  name TEXT NOT NULL,
  wave INTEGER NOT NULL,            -- parallel grouping; equal wave = one fork block
  branch INTEGER,                   -- null on the trunk
  state TEXT NOT NULL,              -- queued|active|done|skipped
  papers_in INTEGER, found INTEGER, kept INTEGER, rejected INTEGER,
  calls_graph INTEGER, calls_reco INTEGER,
  tokens_in_title INTEGER, tokens_in_abs INTEGER, tokens_in_db INTEGER, tokens_out INTEGER,
  started_at TEXT, ended_at TEXT,
  PRIMARY KEY (run_id, idx)
);

-- per-filter stage cascade inside a step (the 16-cell unit bars)
CREATE TABLE run_step_filter (
  run_id TEXT NOT NULL, step_idx INTEGER NOT NULL,
  ord INTEGER NOT NULL,
  key TEXT NOT NULL,                -- year|citation|keyword|venue|llm_title|llm_abstract|similarity|duplicate
  label TEXT NOT NULL,              -- display string; no middot, no em dash, no mono
  config_json TEXT,                 -- mirrors the Build editor fields for the popover
  seen INTEGER, passed INTEGER, rejected INTEGER,
  PRIMARY KEY (run_id, step_idx, ord)
);

-- ── papers (metadata cache, deduped across everything) ──────────────────
CREATE TABLE paper (
  paper_id TEXT PRIMARY KEY,        -- S2 id, or 'local:<hash>' for unresolved imports
  title TEXT, authors_json TEXT,    -- [{author_id,name}] — keep the structure, do not flatten
  venue TEXT, year INTEGER, citation_count INTEGER,
  abstract TEXT, doi TEXT, arxiv_id TEXT, url TEXT,
  full_text_available INTEGER,      -- 0/1/NULL=unknown. NEVER hash-derived (synthesis risk 4)
  full_text_source TEXT,            -- 'mirror' | 'user'
  refreshed_at TEXT
);

-- ── corpus versions: the append-only chain ──────────────────────────────
CREATE TABLE corpus_version (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES run(id),
  v INTEGER NOT NULL,               -- 1-based; v1 = the run's own result
  kind TEXT NOT NULL,               -- run|edit|rescreen|grow|merge|add|seed|restore
  name TEXT NOT NULL,               -- 'Re-screen, accepted, at least 10 cites per year'
  detail TEXT,                      -- '176 fetched, 7 accepted, 169 rejected'
  params_json TEXT,                 -- exactly what the op was given (replayable)
  job_id TEXT,                      -- the job that produced it, if any
  parent_version_id TEXT REFERENCES corpus_version(id),
  restored_from_version_id TEXT REFERENCES corpus_version(id),
  added_count INTEGER, removed_count INTEGER,   -- the "+a -r" summary
  accepted_count INTEGER, rejected_count INTEGER,
  created_at TEXT,
  UNIQUE (run_id, v)
);

-- the delta: what this version changed. Append-only, never updated.
CREATE TABLE corpus_move (
  version_id TEXT NOT NULL REFERENCES corpus_version(id),
  paper_id TEXT NOT NULL REFERENCES paper(paper_id),
  to_state TEXT NOT NULL,           -- accepted | rejected
  marker TEXT NOT NULL,             -- rescued | added | removed | new
  reason_key TEXT,                  -- manual | rescreen | grow | merge | <filter key>
  reason_text TEXT,
  PRIMARY KEY (version_id, paper_id)
);

-- materialised membership, rebuilt when a version is created. Read path.
CREATE TABLE corpus_member (
  version_id TEXT NOT NULL REFERENCES corpus_version(id),
  paper_id TEXT NOT NULL REFERENCES paper(paper_id),
  state TEXT NOT NULL,              -- accepted | rejected
  marker TEXT,                      -- null unless changed by THIS version
  pinned INTEGER NOT NULL DEFAULT 0,
  accepted_at_step TEXT,            -- 'FWD-02' — new engine field (ledger Q4)
  times_hits INTEGER,               -- live cumulative during a run, frozen at end (Q18)
  relevance REAL,                   -- 1.00 placeholder this phase (Q4)
  llm_confidence REAL,              -- 1.00 placeholder this phase (Q4)
  reject_filter_key TEXT,           -- rejected side only
  topic_id INTEGER, community_id INTEGER, x REAL, y REAL,
  PRIMARY KEY (version_id, paper_id)
);
CREATE INDEX ix_member_list ON corpus_member(version_id, state, citation_sort_hint);
-- (citation_sort_hint denormalises paper.citation_count for the default sort;
--  see DECISION-A16 for why the denormalisation is deliberate)

-- manual verdicts survive later re-screens
CREATE TABLE verdict_pin (
  run_id TEXT NOT NULL REFERENCES run(id),
  paper_id TEXT NOT NULL REFERENCES paper(paper_id),
  state TEXT NOT NULL,              -- accepted | rejected
  origin TEXT NOT NULL,             -- manual | added | merged
  set_in_version_id TEXT NOT NULL REFERENCES corpus_version(id),
  released_in_version_id TEXT,      -- non-null once a later manual edit overrides it
  PRIMARY KEY (run_id, paper_id)
);

-- ── citation statements ─────────────────────────────────────────────────
CREATE TABLE citation_edge (             -- direction matters; built from references
  citing_id TEXT NOT NULL, cited_id TEXT NOT NULL,
  PRIMARY KEY (citing_id, cited_id)
);
CREATE TABLE citation_statement (
  cited_id TEXT NOT NULL, citing_id TEXT NOT NULL,
  text_hash TEXT NOT NULL,          -- sha1 of normalised text; dedupe key
  text TEXT NOT NULL,
  source TEXT NOT NULL,             -- 's2_graph_api' | 's2_bulk'
  dropped_reason TEXT,              -- 'very_short' | 'low_alpha' | null
  fetched_at TEXT,
  PRIMARY KEY (cited_id, citing_id, text_hash)
);
CREATE TABLE citation_fetch_state (
  cited_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,             -- ok | partial | error | never
  citer_count INTEGER, statement_count INTEGER,
  attempted_at TEXT, error TEXT
);

-- ── grouping models, versioned by corpus version (provenance / staleness) ─
CREATE TABLE grouping_model (
  id TEXT PRIMARY KEY,
  version_id TEXT NOT NULL REFERENCES corpus_version(id),
  space TEXT NOT NULL,              -- 'topic' | 'community'
  method TEXT NOT NULL,             -- 'umap_hdbscan' | 'leiden'
  params_json TEXT,
  modularity REAL, nmi_vs_topic REAL, ari_vs_topic REAL,   -- the "How this was computed" popover
  group_count INTEGER, noise_count INTEGER,
  created_at TEXT
);
CREATE TABLE grouping_group (
  model_id TEXT NOT NULL REFERENCES grouping_model(id),
  group_id INTEGER NOT NULL,        -- -1 = noise, -2 = synthetic Peripheral
  name TEXT, description TEXT, keywords_json TEXT,
  size INTEGER, cx REAL, cy REAL,
  PRIMARY KEY (model_id, group_id)
);

-- ── imports (transient files, durable audit) ────────────────────────────
CREATE TABLE import_session (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL,
  target_kind TEXT, target_id TEXT,     -- seed_set:<project> | corpus:<version>
  status TEXT,                          -- parsing|review|committing|done|abandoned
  staging_dir TEXT,                     -- deleted on done AND on abandon
  extracted INTEGER, matched INTEGER, ambiguous INTEGER,
  duplicate INTEGER, no_match INTEGER,  -- the invariant: extracted == sum(rest)
  created_at TEXT, expires_at TEXT
);
CREATE TABLE import_entry (
  id TEXT PRIMARY KEY, import_id TEXT NOT NULL REFERENCES import_session(id),
  source_file TEXT, source_locator TEXT,  -- 'refs.bib line 33'
  state TEXT NOT NULL,                    -- ok|multi|none|dupe
  reason TEXT,
  parsed_json TEXT, candidates_json TEXT, resolved_paper_id TEXT,
  chosen INTEGER
);

-- ── settings, jobs ──────────────────────────────────────────────────────
CREATE TABLE settings (
  tenant_id TEXT PRIMARY KEY REFERENCES tenant(id),
  screening_model TEXT,
  budget_max_accepted INTEGER, budget_max_screened INTEGER,
  budget_max_minutes INTEGER, budget_llm_usd REAL,
  updated_at TEXT
  -- NO api key columns. See DECISION-A15.
);
CREATE TABLE key_probe (            -- last verification result per provider
  tenant_id TEXT NOT NULL, provider TEXT NOT NULL,
  state TEXT NOT NULL,              -- not_set|set|valid|invalid|rate_limited|unreachable
  checked_at TEXT, message TEXT,
  PRIMARY KEY (tenant_id, provider)
);
CREATE TABLE job (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL,
  kind TEXT NOT NULL,               -- run|rescreen|grow|merge|add|seed|import|topic_model|community_model|legacy_import
  project_id TEXT, run_id TEXT, version_id TEXT,
  status TEXT NOT NULL,             -- queued|running|paused|stopping|done|stopped|failed
  phase TEXT, progress_json TEXT,
  params_json TEXT, result_json TEXT, error TEXT,
  created_at TEXT, started_at TEXT, ended_at TEXT
);
CREATE TABLE job_event (
  job_id TEXT NOT NULL REFERENCES job(id),
  seq INTEGER NOT NULL,
  type TEXT NOT NULL, payload_json TEXT NOT NULL, at TEXT NOT NULL,
  PRIMARY KEY (job_id, seq)
);
```

### 3.4 Notes on the modelling choices

> **DECISION-A14. Store both the delta and the materialised membership.**
> `corpus_move` is the truth and is append-only; `corpus_member` is a
> derived snapshot written once when a version is created. Reason: the UI needs
> both shapes at once — "Changes · N" and the four change markers read the
> delta, while the paginated, filtered, sorted paper list reads the snapshot.
> Reconstructing membership by folding the chain on every list request is
> O(chain) per page. Cost is roughly 354 accepted + up to ~20k rejected rows per
> version, which for a realistic chain of five versions is a few hundred
> thousand narrow rows — nothing for SQLite. If a chain ever grows past ~30
> versions, snapshot only every Nth version and fold forward; the schema does
> not change.

> **DECISION-A15. API keys never go in the database.** Local phase keeps
> `web/live/backend/keys_store.py` (`.env.local`, chmod 600, git-ignored). The
> DB stores only the probe *state* per provider. This is what makes the tenant
> layer a drop-in: `web/public/backend/tenants.py` already AES-GCM-encrypts
> per-session keys and deliberately never exports them to `os.environ`, and both
> implementations satisfy the same `KeyProvider` interface (§4).

> **DECISION-A16. `paper` is global, membership is per version.** Paper
> metadata is deduped once across all projects and runs; the version tables hold
> only membership and per-run judgements. The `citation_sort_hint` denormalised
> column on `corpus_member` exists so the default "Most cited" sort does not
> join 20k rows; it is written at snapshot time and is allowed to go stale
> relative to `paper.citation_count`, which is honest — a version is a
> photograph of a moment.

> **DECISION-A17. Array indices never cross the wire as identifiers.** The demo
> uses "index into `PAPERS`" as the cross-module primary key (edges, citation
> groups, `RECO.cites`) — synthesis risk 3, the single most likely source of
> "showed the wrong paper" bugs. The API always speaks `paper_id`. Where a
> payload genuinely needs compact index-based edges for canvas performance, the
> indices are **local to that one response** and the response carries its own
> `paper_ids[]` ordering (`api-contract.md` §11.2).

> **DECISION-A18. Version 1 of an engine run is written by the same code path
> as the legacy importer.** Both produce `corpus_version(v=1, kind='run')` from
> a run directory. That guarantees imported historical runs and fresh runs are
> indistinguishable downstream, which is exactly what ledger Q17 asks for.

### 3.5 Concurrency

One process, one writer connection guarded by a lock; readers open their own
connections. Run threads never touch SQLite directly — they push events onto
the job queue and a single consumer writes. WAL keeps readers unblocked during
the write bursts of a run. `PRAGMA synchronous=NORMAL` (WAL makes this safe
enough for an app database, and a run writes thousands of small rows).

---

## 4. How the invite-code tenant layer plugs in later, without rework

The ledger (Q2, Q16) says: invite-code direction is chosen, the login page is
**not** wired this phase, any input signs you in, deployment is local
single-user. The requirement is that choosing that later is a **layer**, not a
refactor. `web/public` already proves the pattern for the old backend; the new
server should be built so the same trick works.

Five seams, all present from day one, all no-ops locally:

**Seam 1 — `tenant_id` on every root row.** Present in the §3 schema already,
always `'local'`. Adding real tenants later inserts rows; it does not migrate
tables. Every query in `domain/*` filters on it from the start, so there is no
"find the queries that forgot" pass later.

**Seam 2 — `RequestContext`.** One FastAPI dependency builds
`RequestContext(tenant_id, data_root, keys: KeyProvider, limits: Limits)` and
**no route handler or domain function reads `os.environ`, a module-level
default, or a global path.** Locally the dependency returns the fixed local
context. Publicly it becomes `require_session` — the exact function
`web/public/backend/auth.py` already implements (HMAC-signed HttpOnly cookie,
90-day TTL, sha256-hashed invite codes in `invites.json`).

**Seam 3 — `KeyProvider` protocol.**

```python
class KeyProvider(Protocol):
    def presence(self) -> dict[str, bool]: ...
    def get(self, field: str) -> str: ...
    def set(self, values: dict[str, str]) -> None: ...
    def as_settings_overrides(self) -> dict: ...   # never touches os.environ
```

Local implementation wraps `keys_store`. Public implementation wraps
`tenants.py`. The critical rule, learned from debt P9 and encoded in
`tenants.py`'s own docstring: keys reach the engine as `Settings` overrides,
never through the process environment. Building the local implementation to
*also* return overrides (rather than relying on `load_into_environ()`) means the
public swap changes one class and nothing else.

**Seam 4 — `data_root` is a function of the context.** `paths.py`-shaped
resolution (§2.5). The public layout is already designed:
`DATA_ROOT/sessions/<sid>/runs/<rid>/`. Locally the tenant segment collapses to
nothing. **Plus the FUSE rule**: `omniknowledge.db` resolves through a separate
`db_path(context)` that on a volume-backed deployment points at container-local
disk with `cache_sync`-style snapshot/restore — because `web/public/backend/paths.py`
records that SQLite on the Modal volume is unreliable. Getting this wrong later
is a data-loss bug; getting it right now costs one function.

**Seam 5 — `Limits` checked before every job start.** Locally the limits object
returns "no limit" for everything. Publicly it becomes
`web/public/backend/limits.py` (max papers ceiling, per-session and global
concurrency, runs per day, retained runs, TTL, body size). The check point is
one function call in `jobs/registry.py::start`.

> **DECISION-A19. The decorative login is a real endpoint returning a fixed
> local identity**, not a front-end-only fiction: `POST /api/v1/auth/session`
> accepts anything and returns the local session; `GET /api/v1/auth/me` returns
> it. The Login screen still does not call them this phase (ledger Q2/Q16 — any
> input signs you in, front-end only). Having the routes exist and be exercised
> by the tenant layer's tests means turning auth on later is a middleware
> change, not a new API surface. **This is a judgment call**: the alternative
> is no auth routes at all, which is simpler now and a bigger step later.

> **DECISION-A20. Nothing user-visible is keyed on the account.** No "owner"
> label, no per-user copy, no avatar sourced from the server this phase. The
> account menu's `Ada Lovelace / ada@lovelace.edu` stays demo-static; if the
> server fed it, it would be a lie in a single-user local deployment. Recorded
> on the "not wired" list.

---

## 5. Engine-side dependencies (other lanes, listed so nothing is assumed)

The server cannot deliver the contract alone. These are engine changes
(`src/citeclaw`) already approved by the ledger, tracked here as blockers:

| Need | Ledger | Today | Consumed by |
| --- | --- | --- | --- |
| `accepted_at_step` per paper | Q4 (A: change the event protocol) | absent; only `source`/`depth` | step chip, Runs and Explore filters |
| per-step × per-filter counters | Q4 | `shape_summary.json` has `in/out/delta` only; no per-step tokens or calls | step drill cascade, monitor by-step popovers |
| `times_hits` live cumulative, frozen at end | Q18 | absent | paper row |
| Leiden + modularity + NMI/ARI | Q6 (A: implement Leiden) | zero hits repo-wide | Explore communities, "How this was computed" |
| 2D coordinates exported per paper | — | UMAP runs at `n_components=5`, intermediate not persisted | topic map |
| dry-run estimation | Q12 (cost estimation) | none | every refine confirmation |
| budget enforcement for 4 budgets | Q12 (A: all four honoured) | only `max_papers_total` and `max_llm_tokens` exist | Settings, budget ring, graceful stop |
| Anthropic provider | Q12 | catalog is Gemini + OpenAI only | Settings model menu |
| richer on-disk run log (pipeline + filter config, PRISMA counts) | Q17 | partial | debugging "why did this run miss that paper" |

`api-contract.md` marks every field that depends on one of these as
`pending-engine`, so the front end can be wired against a server that returns
`null` for them without a second integration pass.

---

## 6. Rollout

1. **Skeleton + read paths.** App factory, `RequestContext`, DB + migrations,
   projects CRUD, settings, `/capabilities`. Front end can leave the design-data
   provider mounted and switch page by page.
2. **Legacy importer.** Turn the existing `runs/` directories into projects +
   runs + version 1. This gives every later screen real data without waiting
   for a live run, and it is ledger Q17's explicit ask.
3. **Run lifecycle.** Job registry over `RunManager`, event stream with `seq`,
   telemetry tables. Old `web/live` must still pass its tests after the additive
   `run_manager` changes.
4. **Versions + the six operations**, verdicts and re-screen first (they reuse
   `ReScreen`), then grow/merge/add/seed.
5. **Citation statements**, then grouping models, then import, then export.

Regression net: the six existing `tests/test_web_*` files stay green throughout
(they cover the modules being reused). New coverage lives in
`tests/test_ok_server_*`; the version-chain semantics in §3 deserve a
property-style test (fold the delta chain, compare against the materialised
snapshot — they must never disagree).

---

## 7. DECISION index

| Id | Decision |
| --- | --- |
| A1 | New package at `web/server/` |
| A2 | `web/live/backend` treated as a vendored library; additive changes only |
| A3 | Additive events are safe because the old store ignores unknown types |
| A4 | Port 8788 |
| A5 | Serve `web/app/dist` single-origin; Vite proxy in dev (needs `E-BE-01`) |
| A6 | One job abstraction for all long operations |
| A7 | Monotonic `seq` + `?since=` resume + full snapshot endpoint |
| A8 | Events persisted to `job_event`, bounded memory tail |
| A9 | Keep the thread run model; one engine job per project |
| A10 | Server enforces the strict-serial progress invariant |
| A11 | Single `data_root` resolved once at startup |
| A12 | SQLite, one file, WAL |
| A13 | DB indexes; run directories remain the artifact store |
| A14 | Store both delta and materialised membership |
| A15 | API keys never in the DB |
| A16 | Global `paper` table; per-version membership; deliberate sort denormalisation |
| A17 | Array indices never used as cross-response identifiers |
| A18 | Engine v1 and legacy import share one code path |
| A19 | Decorative auth routes exist and return a fixed local identity |
| A20 | Nothing user-visible is keyed on the account this phase |

Escalations raised by this lane are in `escalations.md` under `E-BE-*`.
