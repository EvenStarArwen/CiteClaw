"""GraphML export via igraph."""

from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import Any

from citeclaw.filters.measures.semantic_sim import _cosine
from citeclaw.models import PaperRecord

log = logging.getLogger("citeclaw.output.graphml")

_MIN_SIM = 1e-5  # avoid exact zero (Gephi drops weight=0 edges)


def _compute_edge_weights(
    raw_ref: list[float],
    raw_cit: list[float],
) -> list[float]:
    """Compute the per-edge ``weight`` attribute used by Gephi rendering.

    Weight is computed from REF / CIT similarity ONLY — semantic
    similarity is intentionally excluded so the visual layout reflects
    citation-graph structure rather than abstract embedding similarity.
    Semantic similarity is still stored on each edge as its own
    attribute; it just doesn't drive the layout.

    Algorithm:
        1. norm_ref[i] = raw_ref[i] / global_max(raw_ref)
        2. norm_cit[i] = raw_cit[i] / global_max(raw_cit)
        3. weight[i]   = max(norm_ref[i], norm_cit[i])
        4. For edges where BOTH raw_ref[i] AND raw_cit[i] are zero (no
           citation-graph signal at all), replace the weight with the
           lower 25% quantile (Q1) of the non-zero edge weights. This
           keeps "no signal" edges visible at a meaningful baseline
           rather than at the previous 1e-5 floor (which Gephi
           rendered as essentially invisible) or at the median (which
           overstated their importance).

    Edge cases:
        - Empty input → empty output
        - All edges zero (no real signal) → fall back to ``_MIN_SIM``
          for every edge so Gephi still keeps them
        - Only one valid weight → Q1 = that single value
    """
    if not raw_ref:
        return []
    n = len(raw_ref)
    assert len(raw_cit) == n, "raw_ref and raw_cit must be the same length"

    max_ref = max(raw_ref) if raw_ref else 0.0
    max_cit = max(raw_cit) if raw_cit else 0.0

    weights: list[float] = []
    is_zero: list[bool] = []
    for rs, cs in zip(raw_ref, raw_cit):
        nr = (rs / max_ref) if max_ref > 0 else 0.0
        nc = (cs / max_cit) if max_cit > 0 else 0.0
        weights.append(max(nr, nc))
        is_zero.append(rs <= 0 and cs <= 0)

    valid_weights = [w for w, z in zip(weights, is_zero) if not z]
    if not valid_weights:
        # Whole graph has no ref/cit overlap — keep edges visible at floor.
        return [_MIN_SIM] * n

    # Q1 (lower quartile) of the non-zero edge weights. ``inclusive``
    # matches numpy's default and handles small samples gracefully:
    # for a single valid weight, Q1 == that weight.
    if len(valid_weights) >= 2:
        q1 = statistics.quantiles(valid_weights, n=4, method="inclusive")[0]
    else:
        q1 = valid_weights[0]

    return [q1 if z else w for w, z in zip(weights, is_zero)]


def _semantic_cosine(
    src_pid: str,
    tgt_pid: str,
    embeddings: dict[str, list[float] | None],
) -> float:
    """Cosine similarity between two papers' embeddings, clamped to [0, 1].

    Returns 0.0 when either embedding is missing. We deliberately don't
    collapse "no data" to ``_MIN_SIM``: missing embeddings should not
    masquerade as "slightly similar". The edge still gets a floor via the
    combined ``weight`` computation downstream.
    """
    src = embeddings.get(src_pid)
    tgt = embeddings.get(tgt_pid)
    if not src or not tgt:
        return 0.0
    cos = _cosine(src, tgt)
    if cos is None:
        return 0.0
    return max(0.0, cos)


def _join_authors_names(authors: list[dict] | None) -> str:
    """Render authors as a ``"; "``-joined string of names (Gephi-friendly)."""
    if not authors:
        return ""
    return "; ".join((a.get("name") or "") for a in authors if isinstance(a, dict))


def _join_author_ids(authors: list[dict] | None) -> str:
    """Render authors as a ``","``-joined string of S2 author IDs.

    Drops authors without an ``authorId`` (S2 occasionally returns
    name-only entries for unresolved authors).
    """
    if not authors:
        return ""
    return ",".join(
        (a.get("authorId") or "") for a in authors if isinstance(a, dict) and a.get("authorId")
    )


def _set_vertex_attributes(g, papers: list[PaperRecord]) -> None:
    """Set the per-paper vertex attributes on ``g`` from ``papers``.

    13 attributes total — paper_id / label / title / year / venue /
    abstract / citation_count / influential_citation_count / depth /
    source / pdf_url / authors (joined names) / author_ids (joined
    S2 ids). Coerces None to safe defaults so igraph's GraphML writer
    doesn't fail on missing fields.
    """
    g.vs["paper_id"] = [p.paper_id for p in papers]
    g.vs["label"] = [p.title or "" for p in papers]
    g.vs["title"] = [p.title or "" for p in papers]
    g.vs["year"] = [p.year or 0 for p in papers]
    g.vs["venue"] = [p.venue or "" for p in papers]
    g.vs["abstract"] = [p.abstract or "" for p in papers]
    g.vs["citation_count"] = [p.citation_count or 0 for p in papers]
    g.vs["influential_citation_count"] = [
        int(p.influential_citation_count or 0) for p in papers
    ]
    g.vs["depth"] = [p.depth for p in papers]
    g.vs["source"] = [p.source or "" for p in papers]
    g.vs["pdf_url"] = [p.pdf_url or "" for p in papers]
    g.vs["authors"] = [_join_authors_names(p.authors) for p in papers]
    g.vs["author_ids"] = [_join_author_ids(p.authors) for p in papers]


def _set_cluster_attributes(
    g, clusters: dict[str, Any], node_ids: list[str],
) -> None:
    """Add ``cluster_<name>`` (+ optional ``cluster_<name>_label``) per cluster.

    For every entry in ``clusters`` (a
    :class:`citeclaw.cluster.base.ClusterResult`):
      - ``cluster_<name>`` carries the integer cluster id (``-1`` for
        noise / unassigned papers).
      - ``cluster_<name>_label`` carries the human-readable label
        string when at least one cluster in the result was labelled;
        the column is omitted when no labels were generated so empty
        attributes don't pollute the GraphML.
    """
    for cluster_name, result in clusters.items():
        membership = getattr(result, "membership", None) or {}
        metadata_map = getattr(result, "metadata", None) or {}
        cluster_ids: list[int] = []
        cluster_labels: list[str] = []
        any_label = False
        for pid in node_ids:
            cid = int(membership.get(pid, -1))
            cluster_ids.append(cid)
            md = metadata_map.get(cid)
            label = (getattr(md, "label", "") or "") if md is not None else ""
            if label:
                any_label = True
            cluster_labels.append(label)
        attr_key = f"cluster_{cluster_name}"
        g.vs[attr_key] = cluster_ids
        if any_label:
            g.vs[f"{attr_key}_label"] = cluster_labels


def export_graphml(
    collection: dict[str, PaperRecord],
    graphml_path: Path,
    *,
    metadata: dict[str, Any] | None = None,
    edge_meta: dict[tuple[str, str], dict] | None = None,
    s2: Any = None,
    clusters: dict[str, Any] | None = None,
) -> None:
    """Build a citation graph from ``collection`` and export as GraphML.

    ``edge_meta`` maps ``(src_paper_id, dst_paper_id)`` to
    ``{contexts, intents, is_influential}``. Missing entries fall back to
    empty / "false".

    When ``s2`` is provided, SPECTER2 embeddings are prefetched for every
    paper in the collection (via ``s2.fetch_embeddings_batch``) and the
    resulting cosine similarity is stored as a per-edge ``semantic_similarity``
    attribute.

    The edge-level ``weight`` attribute is computed by
    :func:`_compute_edge_weights` from REF and CIT similarity only —
    semantic similarity is excluded from the weight (still stored as
    its own ``semantic_similarity`` attribute). Each measure is
    normalised by its global max for a comparable [0, 1] scale, then
    ``weight = max(norm_ref, norm_cit)``. Edges where both
    ref_similarity and cit_similarity are zero get the Q1 of the
    non-zero edge weights so "no signal" edges stay visible in Gephi.

    When ``clusters`` is provided, each entry (a
    :class:`~citeclaw.cluster.base.ClusterResult`) becomes two node attributes
    on every paper: ``cluster_<name>`` (the integer cluster id, with ``-1``
    for noise / unassigned) and, when the cluster has a label, a parallel
    ``cluster_<name>_label`` carrying the human-readable label string.
    Papers not present in the cluster's membership map default to ``-1`` /
    empty string.
    """
    import igraph as ig

    edge_meta = edge_meta or {}
    clusters = clusters or {}

    # ------------------------------------------------------------------
    # Prefetch embeddings for semantic similarity, if an S2 client was
    # provided. Everything already in the cache serves from disk; confirmed
    # misses (``[]`` sentinels) are persisted, so subsequent calls are free.
    # ------------------------------------------------------------------
    embeddings: dict[str, list[float] | None] = {}
    if s2 is not None:
        try:
            embeddings = s2.fetch_embeddings_batch(list(collection.keys())) or {}
        except Exception as exc:
            log.warning("embedding prefetch failed: %s — semantic_similarity will be 0", exc)
            embeddings = {}

    log.info("Building citation graph for GraphML export ...")

    papers = list(collection.values())
    node_ids = [p.paper_id for p in papers]
    id_to_idx = {pid: i for i, pid in enumerate(node_ids)}
    node_set = set(node_ids)

    g = ig.Graph(n=len(papers), directed=True)
    if metadata:
        for k, v in metadata.items():
            g[k] = v
    _set_vertex_attributes(g, papers)
    _set_cluster_attributes(g, clusters, node_ids)

    edge_list: list[tuple[int, int]] = []
    edge_pair_keys: list[tuple[str, str]] = []
    edge_ref_sim: list[float] = []
    edge_cit_sim: list[float] = []
    edge_sem_sim: list[float] = []
    seen_edges: set[tuple[int, int]] = set()

    def _add_edge(src_pid: str, tgt_pid: str) -> None:
        key = (id_to_idx[src_pid], id_to_idx[tgt_pid])
        if key in seen_edges:
            return
        seen_edges.add(key)
        ref_a = set(collection[src_pid].references) & node_set
        ref_b = set(collection[tgt_pid].references) & node_set
        ref_union = ref_a | ref_b
        rs = (len(ref_a & ref_b) / len(ref_union)) if ref_union else 0.0
        citers_a = {p2.paper_id for p2 in collection.values() if src_pid in p2.references}
        citers_b = {p2.paper_id for p2 in collection.values() if tgt_pid in p2.references}
        cit_union = citers_a | citers_b
        cs = (len(citers_a & citers_b) / len(cit_union)) if cit_union else 0.0
        ss = _semantic_cosine(src_pid, tgt_pid, embeddings)
        edge_list.append(key)
        edge_pair_keys.append((src_pid, tgt_pid))
        # Store raw ref / cit similarity (no floor) so the attribute
        # reflects the true Jaccard score. ``_compute_edge_weights``
        # applies the Q1-of-non-zero floor to ``weight`` after all
        # edges are built.
        edge_ref_sim.append(rs)
        edge_cit_sim.append(cs)
        edge_sem_sim.append(ss)

    for p in papers:
        for ref_id in p.references:
            if ref_id in node_set and ref_id != p.paper_id:
                _add_edge(ref_id, p.paper_id)
        for sp_id in p.supporting_papers:
            if sp_id in node_set and sp_id != p.paper_id:
                if p.source == "forward":
                    _add_edge(sp_id, p.paper_id)
                else:
                    _add_edge(p.paper_id, sp_id)

    if edge_list:
        g.add_edges(edge_list)
        g.es["ref_similarity"] = edge_ref_sim
        g.es["cit_similarity"] = edge_cit_sim
        g.es["semantic_similarity"] = edge_sem_sim
        # weight = max of normalised ref / cit similarity, with
        # zero-signal edges promoted to Q1 of the non-zero edge weights.
        g.es["weight"] = _compute_edge_weights(edge_ref_sim, edge_cit_sim)

        contexts_attr: list[str] = []
        intents_attr: list[str] = []
        is_inf_attr: list[str] = []
        for src_pid, tgt_pid in edge_pair_keys:
            meta = edge_meta.get((src_pid, tgt_pid)) or {}
            ctxs = meta.get("contexts") or []
            ints = meta.get("intents") or []
            inf = bool(meta.get("is_influential", False))
            contexts_attr.append(" | ".join(str(c) for c in ctxs) if ctxs else "")
            intents_attr.append(",".join(str(i) for i in ints) if ints else "")
            is_inf_attr.append("true" if inf else "false")
        g.es["contexts"] = contexts_attr
        g.es["intents"] = intents_attr
        g.es["is_influential"] = is_inf_attr

        # PDF-extraction provenance (ExpandByPDF).
        pdf_rel_attr: list[str] = []
        pdf_marker_attr: list[str] = []
        for src_pid, tgt_pid in edge_pair_keys:
            meta = edge_meta.get((src_pid, tgt_pid)) or {}
            pdf_rel_attr.append(str(meta.get("pdf_relevance", "")))
            pdf_marker_attr.append(str(meta.get("pdf_citation_marker", "")))
        g.es["pdf_relevance"] = pdf_rel_attr
        g.es["pdf_citation_marker"] = pdf_marker_attr

    log.info("  Graph: %d nodes, %d edges", g.vcount(), g.ecount())
    graphml_path.parent.mkdir(parents=True, exist_ok=True)
    g.write_graphml(str(graphml_path))
    log.info("Wrote GraphML: %s (%d nodes, %d edges)", graphml_path, g.vcount(), g.ecount())
