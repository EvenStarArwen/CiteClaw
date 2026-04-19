"""Auto-analysis primitives for V3 workers.

Each worker iteration runs a query, then this module turns the raw
paper list into the data view an agent sees:

- ``decompose_query``: tokenise top-level Lucene clauses
- ``build_query_tree``: count-only S2 queries per clause → counts
- ``topic_model``: TF-IDF + MiniBatchKMeans on title+abstract
- ``diff_vs_prev``: new / seen counts relative to previous iterations
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from citeclaw.agents.v3.state import QueryTreeNode, TopicCluster
from citeclaw.cluster.representation import (
    extract_keywords_ctfidf,
    select_representative_papers,
)
from citeclaw.cluster.topic_model import cluster_embeddings

log = logging.getLogger("citeclaw.agents.v3.analysis")

# Inside ExpandBySearch all papers share the topic's anchor vocabulary,
# so the pipeline-wide adaptive min_cluster_size collapses everything
# into one huge cluster. Halve the threshold so sub-areas surface.
_EXPAND_SIZE_FACTOR = 0.5


# ---------------------------------------------------------------------------
# Query tokeniser
# ---------------------------------------------------------------------------


def decompose_query(query: str) -> list[str]:
    """Split a Lucene query into top-level clauses.

    A clause is a quoted string or a parenthesised group, optionally
    prefixed with ``+`` or ``-``. For ``("A" | "B") +"C" -"D"`` returns
    ``['("A" | "B")', '+"C"', '-"D"']``. Best-effort: bare tokens are
    kept as-is.
    """
    clauses: list[str] = []
    i = 0
    n = len(query)
    while i < n:
        while i < n and query[i].isspace():
            i += 1
        if i >= n:
            break
        start = i
        if query[i] in "+-":
            i += 1
        while i < n and query[i].isspace():
            i += 1
        if i >= n:
            break
        ch = query[i]
        if ch == '"':
            i += 1
            while i < n and query[i] != '"':
                i += 1
            if i < n:
                i += 1
        elif ch == "(":
            depth = 1
            i += 1
            while i < n and depth > 0:
                if query[i] == "(":
                    depth += 1
                elif query[i] == ")":
                    depth -= 1
                i += 1
        else:
            while i < n and not query[i].isspace():
                i += 1
        clauses.append(query[start:i].strip())
    return [c for c in clauses if c]


# ---------------------------------------------------------------------------
# Query tree (per-clause counts)
# ---------------------------------------------------------------------------


def build_query_tree(
    query: str,
    full_total: int,
    *,
    s2_client: Any,
) -> list[QueryTreeNode]:
    """For each top-level clause, issue a count-only S2 search and
    return ``QueryTreeNode(clause, count)``. The final row is the full
    query's own total, so the agent can see how the clauses combine.

    Uses S2 ``search_bulk`` with ``limit=1`` — only the ``total`` field
    matters. S2 caches by query hash so repeated calls are cheap.
    """
    nodes: list[QueryTreeNode] = []
    clauses = decompose_query(query)
    for cl in clauses:
        stripped = cl.lstrip("+-").strip()
        if not stripped:
            continue
        try:
            resp = s2_client.search_bulk(query=stripped, limit=1)
            count = int(resp.get("total") or 0)
        except Exception as exc:  # noqa: BLE001
            log.debug("query_tree probe failed for %r: %s", stripped, exc)
            count = -1
        nodes.append(QueryTreeNode(clause=cl, count=count))
    nodes.append(QueryTreeNode(clause=f"<full query>", count=full_total))
    return nodes


def render_query_tree(nodes: list[QueryTreeNode]) -> str:
    """Pretty-print a query tree for the prompt."""
    if not nodes:
        return "(unable to decompose query)"
    # Find max clause width for alignment
    width = max(len(n.clause) for n in nodes) + 2
    lines = []
    for n in nodes:
        count_str = str(n.count) if n.count >= 0 else "(probe failed)"
        lines.append(f"  {n.clause.ljust(width)} {count_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Topic modelling
# ---------------------------------------------------------------------------


def topic_model(
    papers: list[dict],
    *,
    s2_client: Any = None,
    size_factor: float = _EXPAND_SIZE_FACTOR,
    k: int | None = None,  # kept for back-compat; unused (HDBSCAN is density-based)
) -> list[TopicCluster]:
    """Cluster ``papers`` via UMAP + HDBSCAN on cached SPECTER2 embeddings,
    then fill in c-TF-IDF keywords + centroid-closest representative
    titles per cluster.

    ``papers`` is a list of ``{"paperId", "title", "abstract"}`` dicts.
    ``s2_client`` is required — it supplies the SPECTER2 embedding
    (:meth:`SemanticScholarClient.fetch_embeddings_batch`). Papers with
    no embedding end up in the noise (-1) bucket and are dropped from
    the returned list. ``size_factor`` defaults to 0.5 which halves
    the pipeline-wide adaptive ``min_cluster_size`` / ``min_samples``
    so shared-anchor corpora still surface sub-clusters.
    """
    if not papers or s2_client is None:
        return []
    ids = [p["paperId"] for p in papers if p.get("paperId")]
    if not ids:
        return []
    try:
        embeddings = s2_client.fetch_embeddings_batch(ids)
    except Exception as exc:  # noqa: BLE001
        log.warning("topic_model: fetch_embeddings_batch failed: %s", exc)
        return []

    membership, eff = cluster_embeddings(
        ids=ids,
        embeddings_by_id=embeddings,
        size_factor=size_factor,
    )
    # Drop noise papers; we only surface real clusters.
    clustered = {pid: cid for pid, cid in membership.items() if cid >= 0}
    if not clustered:
        return []

    # c-TF-IDF keywords per cluster (shared helper)
    paper_proxies = {
        p["paperId"]: SimpleNamespace(
            paper_id=p["paperId"],
            title=p.get("title") or "",
            abstract=p.get("abstract") or "",
        )
        for p in papers if p.get("paperId")
    }
    keywords_by_cid = extract_keywords_ctfidf(
        clustered, paper_proxies, n_keywords=8, ngram_range=(1, 3),
    )

    # Representative papers per cluster (centroid-closest in embedding space)
    rep_ids_by_cid = select_representative_papers(clustered, embeddings, n=3)
    titles_by_pid = {p["paperId"]: (p.get("title") or "") for p in papers if p.get("paperId")}

    sizes: dict[int, int] = {}
    for cid in clustered.values():
        sizes[cid] = sizes.get(cid, 0) + 1

    out: list[TopicCluster] = []
    for cid, size in sizes.items():
        rep_titles = [titles_by_pid.get(pid, "") for pid in rep_ids_by_cid.get(cid, [])]
        rep_titles = [t for t in rep_titles if t][:3]
        out.append(
            TopicCluster(
                cluster_id=int(cid),
                count=size,
                keywords=list(keywords_by_cid.get(cid, [])[:8]),
                representative_titles=rep_titles,
            )
        )
    # Sort by size desc, re-id contiguously so the display is stable.
    out.sort(key=lambda c: -c.count)
    for new_id, c in enumerate(out):
        c.cluster_id = new_id
    if eff:
        log.debug(
            "topic_model: %d clusters (mcs=%d, min_samples=%d, n_neighbors=%d)",
            len(out), eff.get("min_cluster_size", 0),
            eff.get("min_samples", 0), eff.get("n_neighbors", 0),
        )
    return out


def render_clusters(clusters: list[TopicCluster]) -> str:
    if not clusters:
        return "(no clusters computed)"
    lines = []
    for c in clusters:
        kw = ", ".join(c.keywords[:6])
        lines.append(f"  cluster {c.cluster_id} ({c.count} papers) — keywords: {kw}")
        for t in c.representative_titles:
            short = t if len(t) <= 120 else t[:117] + "..."
            lines.append(f"      · {short}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Diff vs previous iterations (by paperId; doi-equiv via paperId works
# because S2 dedupes by paperId internally)
# ---------------------------------------------------------------------------


def diff_vs_prev(
    current_ids: list[str],
    prior_ids: set[str],
) -> tuple[int, int]:
    """Return (new_count, seen_count) relative to the prior cumulative set.

    ``new`` = papers in current not in any prior iter.
    ``seen`` = papers in current that were also in a prior iter.
    """
    new = sum(1 for pid in current_ids if pid not in prior_ids)
    seen = len(current_ids) - new
    return new, seen
