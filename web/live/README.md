# CiteClaw — Live Web UI

The CiteClaw_WebUI (v3) design, made **live**: configure a search pipeline,
launch a real run against Semantic Scholar + an LLM, and watch it stream in
— live dashboard, accepted-paper feed, and a growing citation graph. Runs
entirely on your laptop; nothing is exposed to the internet.

This is the *design you already had* connected to the *real pipeline*. The
look and the visual builder are unchanged — only the data behind them is now
real.

---

## Run it

```bash
python web/live/run.py
```

That's it. It starts a local server and opens your browser at
`http://127.0.0.1:8787`. Stop it with `Ctrl-C`.

There is **no build step** — the page runs directly in the browser, so you
never touch `npm`/`node`. (You do need internet, because the search itself
calls Semantic Scholar and the LLM.)

### First time: add your API keys

Click the **gear** (top-right) → paste your keys → **Save**:

- **Gemini API key** — required to screen papers. Get one at
  [aistudio.google.com](https://aistudio.google.com/apikey).
- **Semantic Scholar key** — optional but recommended (searches are much
  faster and won't rate-limit). Free from
  [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api).
- **OpenAI key** — only needed if OpenAI models become selectable later.

Keys are saved to `CiteClaw/.env.local` on your computer (git-ignored, never
uploaded).

---

## Use it

1. **Build tab**
   - **Seeds** (left): type a query, press **Enter** to search Semantic
     Scholar, click papers to ⭐ them. Starred papers seed the search.
   - **Pipeline** (center-top): the chain of steps (Seed set → Forward /
     Backward screeners → Rerank). Click a block to configure it below.
   - **Config** (center-bottom): the selected block's filter tree
     (year / citation / keyword / similarity / LLM filters).
2. Press **Run pipeline** (top-right).
3. **Run tab** shows it live: pipeline progress + log (left), the citation
   graph growing (center), metrics + rejections + cost (dashboard), and the
   accepted-paper stream (right). Click a node or a paper to inspect it.
4. **Explore tab** — the citation network as a full page, for digging into a
   collection after (or during) a run:
   - **Papers** (left): pick the data source — the current session or any
     finished run found under `runs/` — then sort, and narrow with the
     Filters bar (year window / min citations / seeds only). Filters apply
     to the list *and* the graph. Gephi exports work too: drop a
     `citation_network.gexf` into `runs/<name>/` and it appears as an
     explorable dataset (node attrs `paper_id`/`title`/`year`/`venue`/
     `abstract`/`citation_count` are picked up when present).
   - **Graph** (center): the same engine as the Run view, plus labels, a
     growth replay (⟲ replays how the collection was accepted), a force-
     layout options popover (spacing / gravity / LinLog / overlap), zoom
     controls, and hover tooltips.
   - **Details** (right): the selected paper — abstract (with an OpenAlex
     fallback when Semantic Scholar has none), metadata, a Semantic Scholar
     link, and **Explore subtree**, which trims the graph to the paper's
     2-hop citation neighbourhood.

Each run's outputs (`literature_collection.json`, `.bib`,
`citation_network.graphml`, …) are written to `runs/webui/<run-id>/`.

Both graph views share one renderer: a **graphology** graph laid out by
**ForceAtlas2** (the Gephi algorithm) in a web worker and drawn by
**sigma.js** on WebGL — so layout never blocks the UI, and new papers are
inserted incrementally during a live run instead of re-laying-out from
scratch. The libraries load from esm.sh at page load; the Explore/Run graph
therefore needs internet access (as do the searches themselves).

---

## Model support (this version)

This first release only actually runs **`gemini-3.1-flash-lite`** with
`minimal` reasoning effort. The Settings picker lists every current OpenAI +
Gemini model with prices, but selecting any other one will report
"not supported yet" when you press Run.

> Note: the id originally requested — `gemini-3.1-flash-lite-preview` — no
> longer exists (the model reached general availability), so it's accepted as
> an **alias** and runs against the GA id `gemini-3.1-flash-lite`
> ($0.25 in / $1.50 out per 1M tokens).

---

## Troubleshooting

- **Port already in use** → `CITECLAW_WEBUI_PORT=9000 python web/live/run.py`.
- **Searches are slow / "rate limited"** → add a Semantic Scholar key in
  Settings (keyless access is heavily throttled).
- **"Gemini API key not set"** → add it in Settings (gear icon).
- **Nothing happens on Run** → open the browser console; also check the
  terminal where you launched the server for errors.

---

## How it fits together (for the curious)

- **Back end** (`web/live/backend/`, Python/FastAPI): serves the page, runs
  the real `citeclaw` pipeline in a background thread, and streams live
  events to the browser over a WebSocket. It translates the visual pipeline
  into a real CiteClaw config, searches Semantic Scholar, and enforces the
  model rules.
- **Front end** (`web/live/static/`): the original v3 design's components,
  unchanged except that their data now comes from a small live store
  (`live-store.jsx`) fed by the back end, instead of the old demo data.

Everything speaks over `http://127.0.0.1:8787` — one local address, one
command.
