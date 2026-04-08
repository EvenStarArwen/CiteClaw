"""compute_metric — score a signal of papers by name (citation, pagerank, ...)."""

from __future__ import annotations

from citeclaw.models import PaperRecord
from citeclaw.network import build_citation_graph, compute_pagerank


def compute_metric(name: str, signal: list[PaperRecord], ctx) -> dict[str, float]:
    if name == "citation":
        return {p.paper_id: float(p.citation_count or 0) for p in signal}
    if name == "pagerank":
        sig_ids = {p.paper_id for p in signal}
        g = build_citation_graph(ctx.collection)
        ranked = compute_pagerank(g)
        scores = dict(ranked)
        return {pid: scores.get(pid, 0.0) for pid in sig_ids}
    raise ValueError(f"Unknown rerank metric {name!r}")
