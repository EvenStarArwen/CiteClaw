"""s2mirror — a self-hosted Semantic Scholar *graph API* mirror.

Builds read-only SQLite shards from the official S2AG dataset dumps
(papers / abstracts / citations / paper-ids / authors) and serves the
subset of the graph API that CiteClaw's expansion hot path uses:

* ``GET  /graph/v1/paper/{id}``
* ``POST /graph/v1/paper/batch``
* ``GET  /graph/v1/paper/{id}/references`` / ``.../citations``
* ``POST /graph/v1/author/batch``
* ``GET  /graph/v1/author/{id}/papers``

Everything else (search, recommendations, embedding fields, papers
newer than the loaded release) is proxied to the real API with a
server-side key at a polite 1 rps.

The package is deliberately dependency-light (stdlib + numpy + orjson +
fastapi/httpx for serving) so the same code runs in pytest on a laptop
and inside the Modal ingest/serve containers (see ``modal_s2_mirror.py``
at the repo root).
"""
