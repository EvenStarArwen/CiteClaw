"""Self-hosted S2ORC full-text mirror for CiteClaw.

Serves parsed open-access full text (body + GROBID annotation spans +
per-paper open-access license/status) from the Semantic Scholar ``s2orc``
bulk dataset, keyed by integer ``corpusid``, over a small HTTP surface::

    GET  /s2orc/v1/paper/{id}       id = CorpusId:.. | DOI:.. | ARXIV:.. | PMID:..
    POST /s2orc/v1/paper/batch      {"ids": [...]}   (aligned list, null = miss)

A miss (paper not in S2ORC) is a plain 404 / null — S2ORC has no
per-paper upstream to fall back to, so unlike the graph mirror there is
no proxy. Ingest and serving share one Modal app + the ``citeclaw-s2orc``
volume; see ``modal_s2orc_mirror.py``.

The library is import-clean on a laptop (stdlib ``json`` fallback in
``jsonio``) so the map/reduce/store logic is unit-testable without Modal.
"""
