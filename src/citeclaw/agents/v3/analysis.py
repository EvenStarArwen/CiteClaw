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
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from citeclaw.agents.v3.state import QueryTreeNode, TopicCluster

log = logging.getLogger("citeclaw.agents.v3.analysis")


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
    k: int | None = None,
) -> list[TopicCluster]:
    """TF-IDF + MiniBatchKMeans over title+abstract.

    Returns clusters sorted by size descending. Each cluster has its
    top-8 TF-IDF terms and top-3 centroid-closest titles.
    """
    if not papers:
        return []
    texts: list[str] = []
    for p in papers:
        title = (p.get("title") or "").strip()
        abstract = (p.get("abstract") or "").strip()
        texts.append(f"{title} {abstract}".strip() or "untitled")
    n = len(texts)
    if k is None:
        k = max(4, min(10, int(np.sqrt(n) / 2) or 4))
    k = max(2, min(k, n))
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=2000,
        min_df=2 if n > 30 else 1,
        ngram_range=(1, 2),
    )
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        return []
    try:
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            batch_size=min(256, max(32, n // 4)),
            n_init=3,
        )
        labels = km.fit_predict(X)
    except Exception as exc:  # noqa: BLE001
        log.debug("kmeans failed: %s", exc)
        return []
    terms = vectorizer.get_feature_names_out()
    out: list[TopicCluster] = []
    for c_id in range(k):
        mask = labels == c_id
        count = int(mask.sum())
        if count == 0:
            continue
        centroid = km.cluster_centers_[c_id]
        top_t = np.argsort(centroid)[::-1][:8]
        keywords = [str(terms[i]) for i in top_t if centroid[i] > 0]
        member_idx = np.where(mask)[0]
        member_vecs = X[member_idx]
        sims = np.asarray(member_vecs.dot(centroid)).ravel()
        local_order = np.argsort(-sims)[:3]
        rep_titles = [(papers[member_idx[i]].get("title") or "").strip() for i in local_order]
        rep_titles = [t for t in rep_titles if t][:3]
        out.append(
            TopicCluster(
                cluster_id=int(c_id),
                count=count,
                keywords=keywords[:8],
                representative_titles=rep_titles,
            )
        )
    out.sort(key=lambda c: -c.count)
    # Re-id clusters to be contiguous in display order (0..N-1 by size)
    for new_id, c in enumerate(out):
        c.cluster_id = new_id
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
