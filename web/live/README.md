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
     Results come 100 at a time — the header shows how many of the total
     matches are loaded, and **Load more** pages deeper (S2 serves the first
     1,000 matches per query; past that, refine the query).
   - **Pipeline** (center-top): the chain of steps (Seed set → Forward /
     Backward screeners → Rerank). Click a block to configure it below.
     Every "add step" menu also offers **Reuse an existing step**: pick one
     and a **linked copy** is inserted there — it always runs with the
     original's parameters and filters (edit the original once, every copy
     follows), wears a **⇄ ORIGINAL-ID** badge in the pipeline, and its own
     config panel is locked with an explanatory note. **Unsync** in that
     note makes the copy independent (it keeps a snapshot and becomes
     editable; **Re-sync** goes back to mirroring). Deleting the original
     automatically turns its copies independent — nothing loses its
     screener. The **⧉** button in a step's header is a shortcut for
     "linked copy right after this step".
   - **Config** (center-bottom): the selected block's filter tree
     (year / citation / keyword / similarity / LLM filters). Hovering a
     filter reveals **↑ / ↓** (reorder — add a filter at the end, walk it
     up to where it belongs, e.g. a cheap citation cap *before* an
     expensive LLM screen), **⧉ copy** and **×**; every "Add" menu then
     offers **Paste** of the copied filter (whole groups too), in this step
     or any other. Rows show as much of a filter's summary as fits and hide
     it entirely when only a meaningless fragment would fit (the tooltip
     always has the full text). LLM query prompts are multi-line boxes —
     drag the lower-right corner to enlarge them.
2. Press **Run pipeline** (top-right).
3. **Run tab** shows it live: pipeline progress + log (left), the citation
   graph growing (center), metrics + rejections + cost (dashboard), and the
   accepted-paper stream (right). Click a node or a paper to inspect it.
4. **Explore tab** — the citation network as a full page, for digging into a
   collection after (or during) a run:
   - **Papers** (left): pick the data source — the current session, any
     finished run found under `runs/`, or a **local graph file** (the ⭱
     button opens your system file picker; GraphML/GEXF citation *or*
     collaboration networks are auto-detected, anything else is rejected
     with an explanation of why). Then sort, and narrow with the Filters
     bar: year window, min/max citations, seeds only, and a **keyword
     formula** over title + abstract using the search pipeline's DSL
     (`(graph | network) & !survey`, quotes for phrases). Filters apply to
     the list *and* the graph. Papers cut from the network view by the
     *graph-side* filters (min degree / edge weight / largest component)
     stay listed but render dimmed with an unlink mark — they're still in
     the collection, just not in the current network. Gephi exports work
     too: drop a `citation_network.gexf` into `runs/<name>/`.
   - **Graph** (center): the same engine as the Run view, plus a growth
     replay (⟲), a label toggle (off by default), hover tooltips, a
     network-stats card (nodes / edges / avg degree / density / components /
     diameter — exact BFS on small graphs, farthest-sweep pseudo-diameter on
     big ones so filters never stall the UI), a **Citation ↔ Authors**
     switch (the co-authorship network, from Finalize's
     `collaboration_network.graphml` or derived from the collection JSON —
     **sized by the author's citations inside the collection, coloured by
     the year their first paper entered it, edge width = collaboration
     strength**), and a **graph-settings panel** (sliders icon):
       - *Force layout*: scaling, gravity, strong gravity, dissuade hubs,
         LinLog clustering, prevent node overlap, edge-weight influence.
       - *Appearance*: **exact min/max node size** (Gephi-style — 1 → 100
         is a real 100× difference) with a **size-scale transformation**
         (linear / √ / ∛ / log / ² / ³), **edge width min/max** mapped from
         edge weight with its own transformation, colour palette (Ember
         default + Moss / Slate / Dusk / Ash), labels.
       - *Graph filters*: min degree, min edge weight, largest component
         only. These are Gephi-style — filtered nodes/edges are **removed
         from the simulation**, so the layout re-flows live (debounced, and
         positions restore when you loosen a filter). Filters are
         dataset-scoped: switching source clears them.
   - **Details** (right): the selected paper — abstract (with an OpenAlex
     fallback when Semantic Scholar has none), metadata, a Semantic Scholar
     link, and **Explore subtree** (papers) / **ego network** (authors),
     which trims the graph to the 2-hop neighbourhood.

Each run's outputs (`literature_collection.json`, `.bib`,
`citation_network.graphml`, …) are written to `runs/webui/<run-id>/`.

Both graph views share one renderer: a **graphology** graph laid out by
**ForceAtlas2** (the Gephi algorithm) and drawn by **sigma.js** on WebGL.
The same algorithm runs in two modes: opening a static dataset does a quick
synchronous pre-warm and then a bounded **fast-settle** in a web worker
(double step size for a few seconds, then the gentle configured dynamics) —
so a <1k-node graph reaches its layout in seconds without ever blocking the
UI — while a live run keeps the worker warm and inserts new papers
incrementally instead of re-laying-out from scratch. The libraries load
from esm.sh at page load; the Explore/Run graph therefore needs internet
access (as do the searches themselves).

---

## Model support (this version)

This first release only actually runs **`gemini-3.1-flash-lite`** with
`minimal` reasoning effort. The Settings picker lists every current OpenAI +
Gemini model with prices, but selecting any other one will report
"not supported yet" when you press Run.

The model in Settings is the **default** — every LLM filter can override the
model and reasoning effort in its own config (Build tab → click the filter),
so different filters can screen with different models once more are
supported. The run guard checks support + the matching API key for every
override, not just the default.

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
