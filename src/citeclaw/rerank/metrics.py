"""``compute_metric`` — score each paper in a signal by a named ranking metric.

Two metrics are registered:

* ``"citation"`` — raw citation count. ``None`` becomes ``0.0`` so
  papers without S2-side citation data still get a numeric score.
* ``"pagerank"`` — the PageRank score over the cumulative citation
  graph (``ctx.collection``), restricted to the signal-paper subset
  on the way out. Papers absent from the graph (no incident edges)
  default to ``0.0``.

The returned ``{paper_id: score}`` map is consumed by
:class:`citeclaw.steps.rerank.Rerank` for top-K selection. Unknown
metric names raise :class:`ValueError` so a typo in YAML fails fast
rather than silently scoring everything as zero.
"""

from __future__ import annotations

from typing import Literal

from citeclaw.models import PaperRecord
from citeclaw.network import build_citation_graph, compute_pagerank


def compute_metric(
    name: Literal["citation", "pagerank"] | str,
    signal: list[PaperRecord],
    ctx,
) -> dict[str, float]:
    """Return ``{paper_id: score}`` for every paper in ``signal``.

    See module docstring for the supported metric names. The signature
    accepts a bare ``str`` (not just the ``Literal``) so callers
    forwarding a YAML-derived name don't need to narrow the type
    themselves; the unknown-name guard still raises ``ValueError``.
    """
    if name == "citation":
        return {p.paper_id: float(p.citation_count or 0) for p in signal}
    if name == "pagerank":
        sig_ids = {p.paper_id for p in signal}
        g = build_citation_graph(ctx.collection)
        scores = dict(compute_pagerank(g))
        return {pid: scores.get(pid, 0.0) for pid in sig_ids}
    raise ValueError(f"Unknown rerank metric {name!r}")
