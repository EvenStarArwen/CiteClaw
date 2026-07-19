# CiteClaw UI — redesign contract

**The one hard rule: the redesigned frontend must keep speaking to the
current backend exactly as today.** Everything visual is negotiable —
markup, layout, component structure, placement, styling, motion. What is
frozen is the API traffic below and the existence (in some form) of the
affordances that produce it. Engineering re-wires approved designs back
into the React components and is the compatibility gate; when a proposal
can't be wired without changing the API, it goes back for revision rather
than being adopted.

## Fixed API surface (do not require changes to any of this)

REST (all under the same origin):

- `GET/POST /api/settings` — fields: `gemini_api_key`, `openai_api_key`,
  `s2_api_key` (write-only; UI shows presence booleans), `model`,
  `reasoning_effort` (`minimal|low|medium|high`), `max_papers`.
- `GET /api/models` — model catalog rows `{id, provider, label, input,
  output, reasoning, note, supported, requires_key}` for the picker.
- `GET /api/seeds/search?q&limit&year&min_cites&offset` — paged
  `{total, offset, next, items[]}`; items feed the seed list.
- `POST /api/seeds/abstract` — abstract fallback for a selected seed.
- `POST /api/run` — body `{pipeline, seeds, limits, topic, model,
  reasoning_effort}` produced by the builder's data model. The pipeline
  editor may look ANY way, but must still produce this structure.
- `POST /api/run/{id}/stop` · `POST /api/run/{id}/cap` (`{action:
  "stop"|"raise", max}`) · `GET /api/run/{id}/status` · `GET
  /api/run/{id}/graph`.
- `GET /api/explore/runs` · `GET /api/explore/run?path=` · `GET
  /api/explore/collab?path=` · `POST /api/explore/upload`.
- Public build only: `POST /api/auth/join {code}` · `GET /api/auth/me` ·
  `GET /api/session/runs` · `GET /api/download/{run}/{collection|bib|
  citation|collab|zip}`.

WebSocket `GET /api/run/{id}/stream` — event types the UI renders:
`hello`, `progress` (step list), `activity` (outer/inner bars, lanes,
retry banner), `metrics`, `graph` (nodes/edges), `paper_added`, `log`,
`cap_reached`/`cap_resolved`, `error`, `done`.

## Affordances that must exist in some form

Search seeds and star/unstar them · edit pipeline steps and their filter
chains · run / stop · answer the cap dialog (stop, or raise with a number,
under a 30s timeout) · watch progress (steps, current activity, live log)
· see accepted papers streaming and open one · the citation graph with
zoom/fit, labels toggle, and graph settings · Explore finished runs with
list filters and a paper detail view · settings (keys, model, effort, max
papers) · theme toggle · public build: invite entry, results/downloads,
"keep tab open" warning while a run is active.

## Practical guidance

- `assets/app.css` is the real production stylesheet — the preferred edit
  target. Design-token changes (`--cc-*`) restyle the whole app at once.
- The app ships the `data-palette="mono"` variant (every card's root
  attributes reproduce it). Prefer editing tokens over hardcoding colors
  so the light/dark theme toggle keeps working.
- `src/jsx/` holds the actual component sources; propose markup changes
  against these or as annotated HTML — either is adoptable.
- The graph itself is canvas-rendered (sigma.js) — its *chrome* (toolbar,
  legend, popover, stats) is HTML and fully redesignable; node/edge
  visuals are parameterized (sizes, palette ramps) and can be re-themed
  via the settings popover's defaults rather than CSS.
