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

Each run's outputs (`literature_collection.json`, `.bib`,
`citation_network.graphml`, …) are written to `runs/webui/<run-id>/`.

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
