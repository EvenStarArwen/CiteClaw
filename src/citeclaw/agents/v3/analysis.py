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

from citeclaw.agents.v3.query_translate import (
    decompose_query,
    parse_or_alternatives,
    term_matches,
    to_natural,
)
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
# Query tree (per-clause + per-OR-alternative counts)
# ---------------------------------------------------------------------------


def build_query_tree(
    query: str,
    full_total: int,
    *,
    s2_client: Any,
    enriched_papers: list[dict] | None = None,
) -> list[QueryTreeNode]:
    """Decompose a Lucene query into its top-level facets and count each.

    Top-level clause counts (how many papers each facet matches alone)
    come from count-only S2 probes. Inside each OR-group clause, we
    break it into its alternatives and count — IN MEMORY, against the
    already-fetched ``enriched_papers`` — how many papers match each
    individual alternative. That in-memory pass needs no S2 round-trip
    and lets the agent see, for ``+("ESM" | "ProtBERT" | "ProtT5")``,
    whether one of the three names is carrying all the recall.

    A trailing ``<full query>`` row shows the overall intersection total.
    """
    nodes: list[QueryTreeNode] = []
    clauses = decompose_query(query)

    # Build a lower-cased corpus index once — reused for every
    # in-memory OR-alternative probe.
    corpus: list[str] = []
    if enriched_papers:
        for p in enriched_papers:
            title = (p.get("title") or "")
            abstract = (p.get("abstract") or "")
            corpus.append((title + " " + abstract).lower())

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
        # Per-OR-alternative breakdown (in-memory regex on fetched papers).
        children: list[QueryTreeNode] = []
        alts = parse_or_alternatives(cl)
        if alts and corpus:
            for alt in alts:
                n_hits = sum(1 for text in corpus if term_matches(alt, text))
                children.append(QueryTreeNode(clause=alt, count=n_hits))
        nodes.append(QueryTreeNode(clause=cl, count=count, children=children))
    nodes.append(QueryTreeNode(clause="<full query>", count=full_total))
    return nodes


def build_in_cluster_tree(
    cluster_papers: list[dict],
    query_lucene: str,
) -> dict[str, Any]:
    """Break down how the query's OR-alternative COMBINATIONS contribute
    to a single cluster.

    All papers in a cluster already satisfy every AND facet (that's how
    they passed the query in the first place), so we only enumerate
    OR-group alternatives and count the Cartesian product of
    (alt1 of facet1, alt2 of facet2, …). For ``(A | B) AND (C | D)``
    the cells are ``A+C, A+D, B+C, B+D``. For ``(A | B) AND C``
    (single OR facet) the cells are ``A, B`` (``C`` is implicit).

    Returns::

        {
          "total": N,
          "or_facets": [{"clause": str_natural, "alternatives": [str, ...]}, ...],
          "marginals": [                  # per-facet single-term breakdown
              {"facet": natural, "per_alt": {alt: hits_in_cluster}},
              ...
          ],
          "combinations": [                # Cartesian product counts
              {"alts": [alt_facet1, alt_facet2, ...], "count": int},
              ...
          ],
        }

    Matching is in-memory regex over title+abstract — no S2 calls.
    """
    from itertools import product

    total = len(cluster_papers)
    out: dict[str, Any] = {
        "total": total,
        "or_facets": [],
        "marginals": [],
        "combinations": [],
    }
    if total == 0:
        return out

    clauses = decompose_query(query_lucene)
    or_facets: list[tuple[str, list[str]]] = []
    for cl in clauses:
        alts = parse_or_alternatives(cl)
        if alts:
            or_facets.append((cl, alts))

    corpus: list[str] = [
        ((p.get("title") or "") + " " + (p.get("abstract") or "")).lower()
        for p in cluster_papers
    ]

    if not or_facets:
        return out  # nothing OR-shaped to break down

    for clause, alts in or_facets:
        try:
            label = to_natural(clause) or clause
        except Exception:  # noqa: BLE001
            label = clause
        per_alt = {alt: sum(1 for t in corpus if term_matches(alt, t)) for alt in alts}
        out["or_facets"].append({"clause": label, "alternatives": list(alts)})
        out["marginals"].append({"facet": label, "per_alt": per_alt})

    combo_alts = [alts for _, alts in or_facets]
    for combo in product(*combo_alts):
        count = sum(1 for t in corpus if all(term_matches(a, t) for a in combo))
        out["combinations"].append({"alts": list(combo), "count": count})

    # Sort combinations by count desc for easier reading.
    out["combinations"].sort(key=lambda r: -r["count"])
    return out


_IN_CLUSTER_COMBO_CAP = 40


def render_in_cluster_tree(tree: dict[str, Any]) -> str:
    """Pretty-print an in-cluster query tree for the worker prompt.

    Per-facet marginals are always fully shown — they're bounded by
    the number of OR alternatives the worker wrote. The Cartesian-
    product combinations section is capped at the top
    ``_IN_CLUSTER_COMBO_CAP`` rows by count (plus an explicit
    "X more combinations" footer), because with 3 facets × 15 alts
    the full grid would be ~3.4K rows — enough to push the prompt
    into rate-limit territory on OpenAI.
    """
    total = tree.get("total", 0)
    or_facets = tree.get("or_facets") or []
    marginals = tree.get("marginals") or []
    combos = tree.get("combinations") or []
    if total == 0:
        return "  (cluster is empty)"
    lines: list[str] = [f"  cluster_total = {total}"]
    if not or_facets:
        lines.append("  (query has no OR facets — no breakdown available)")
        return "\n".join(lines)
    lines.append("")
    lines.append("  Per-facet marginals (papers matching each alternative alone):")
    for m in marginals:
        lines.append(f"    facet: {m['facet']}")
        items = sorted(m["per_alt"].items(), key=lambda kv: -kv[1])
        for alt, n in items:
            pct = 100 * n / total
            lines.append(f"      · {alt:<40s} {n:5d}  ({pct:5.1f}%)")
    lines.append("")
    shown = combos[:_IN_CLUSTER_COMBO_CAP]
    truncated = len(combos) - len(shown)
    lines.append(
        "  Combinations (how many papers match every listed alternative; "
        f"top {len(shown)} by count):"
    )
    for r in shown:
        n = r["count"]
        pct = 100 * n / total
        combo_str = "  +  ".join(r["alts"])
        lines.append(f"      · {combo_str}  →  {n}  ({pct:5.1f}%)")
    if truncated > 0:
        lines.append(f"      · ... {truncated} more combinations (all smaller)")
    return "\n".join(lines)


def render_query_tree(nodes: list[QueryTreeNode], *, as_natural: bool = True) -> str:
    """Pretty-print a query tree for the worker prompt.

    ``as_natural`` (default) shows every clause in AND/OR/NOT form so it
    matches the natural-language query the worker wrote. Turning it off
    surfaces the raw Lucene form — useful for debug / tests.

    Top-level clauses print at 2-space indent; OR-alternative children
    (if any) print at 6-space indent with a ``·`` marker. Children note
    the hit count from in-memory matching on the enriched sample.
    """
    if not nodes:
        return "(unable to decompose query)"

    def _label(clause: str) -> str:
        if not as_natural:
            return clause
        if clause.startswith("<") and clause.endswith(">"):
            return clause  # the sentinel rows like <full query>
        try:
            return to_natural(clause) or clause
        except Exception:  # noqa: BLE001
            return clause

    all_labels: list[str] = []
    for n in nodes:
        all_labels.append(_label(n.clause))
        for ch in n.children:
            all_labels.append(_label(ch.clause))
    width = max((len(l) for l in all_labels), default=10) + 4

    lines: list[str] = []
    for n in nodes:
        label = _label(n.clause)
        count_str = str(n.count) if n.count >= 0 else "(probe failed)"
        lines.append(f"  {label.ljust(width)} {count_str}")
        for ch in n.children:
            ch_label = _label(ch.clause)
            ch_count = str(ch.count) if ch.count >= 0 else "(probe failed)"
            note = " (in-memory on fetched sample)" if ch == n.children[0] else ""
            lines.append(f"      · {ch_label.ljust(width - 4)} {ch_count}{note}")
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
    pids_by_cid: dict[int, list[str]] = {}
    for pid, cid in clustered.items():
        sizes[cid] = sizes.get(cid, 0) + 1
        pids_by_cid.setdefault(cid, []).append(pid)

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
                paper_ids=list(pids_by_cid.get(cid, [])),
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
