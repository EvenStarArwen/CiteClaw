# CiteClaw local Web UI

This is the first live, laptop-only version of the CiteClaw interface. It uses
the visual language of the original static `CiteClaw_WebUI` demo, but its
buttons now call the real CiteClaw pipeline.

In plain language, the **frontend** is the page you see in the browser. The
**backend** is the local Python process that validates configs, holds API keys
in memory, runs CiteClaw, and sends live updates to the page. Both run on your
own computer; no CiteClaw server is deployed on the internet.

## One-time setup

From the CiteClaw repository root:

```bash
pip install -e '.[web]'

cd web/frontend
pnpm install
pnpm build
cd ../..
```

The first command installs the Python web dependencies. The next two install
and build the browser application. Re-run `pnpm build` after changing frontend
code.

## Start the app

```bash
python -m citeclaw web
```

Leave that terminal open and visit <http://127.0.0.1:9999>. Stop the server with
<kbd>Ctrl</kbd>+<kbd>C</kbd>. The default address is loopback-only, so another
computer cannot connect to it.

## Run a search

1. Choose an existing config in the left column or edit the raw YAML.
2. Use the guided fields for common settings. The YAML editor remains the
   authoritative place where every `config.yaml` option can be changed.
3. Enter a Semantic Scholar key and a Gemini key in the credentials panel.
   An OpenAI field is also available for future model support.
4. Select `gemini-3.1-flash-lite-preview` and `minimal` reasoning.
5. Select **Validate** to catch configuration errors without starting a run.
6. Select **Run pipeline**. The page switches to live step progress, metrics,
   logs, accepted papers, and the citation network.
7. Click a graph node to inspect the paper. Drag to pan, scroll to zoom, or use
   graph search to find a paper by title.

Run files are written to `runs/web/<run-id>/`. Closing the browser does not stop
an active run as long as the Python server remains open; reconnecting to the run
replays its events.

## API key handling

- Keys entered in the page are sent only to the local Python process and kept in
  memory for the run.
- They are not added to saved YAML, the run's config snapshot, or log messages.
- You can instead set `SEMANTIC_SCHOLAR_API_KEY` (or `S2_API_KEY`),
  `GEMINI_API_KEY`, and `OPENAI_API_KEY` in the server environment. The page
  reports whether each environment key is available without exposing its value.
- Keep using the default `127.0.0.1` host unless you intentionally want to make
  the app reachable on your network.

## First-release limits

- The price browser lists current CiteClaw-suitable text models from the official
  OpenAI and Gemini pricing pages, with those source links and a verification
  date. Only `gemini-3.1-flash-lite-preview` with `minimal` reasoning is runnable
  in this version. Selecting another entry produces a deliberate unsupported
  configuration error before any API call.
- One pipeline can run at a time. Pause and cancellation are not implemented.
- Live run history is held in memory and is cleared when the Python server
  restarts. Completed run artifacts remain under `runs/web/`.
- The graph is refreshed after each completed pipeline step rather than after
  every individual paper.

## Development

For frontend work, use two terminals. Start the Python backend in the first:

```bash
python -m citeclaw web
```

Start Vite's development server in the second:

```bash
cd web/frontend
pnpm dev
```

Open <http://127.0.0.1:5173>. Vite proxies API and WebSocket traffic to the
Python server on port 9999.

Useful verification commands:

```bash
pytest -q tests/test_web_backend.py
ruff check web/backend tests/test_web_backend.py src/citeclaw/web_server.py
cd web/frontend && pnpm build
```

The main implementation is split between:

- `web/backend/runtime.py` — run lifecycle and real pipeline bridge
- `web/backend/api/` — config, model catalog, run, and WebSocket endpoints
- `web/frontend/src/components/BuildView.tsx` — configuration workspace
- `web/frontend/src/components/RunView.tsx` — live monitoring workspace
- `web/frontend/src/components/GraphView.tsx` — sigma.js citation graph
