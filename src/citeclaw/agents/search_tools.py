"""Tool implementations for the v2 ExpandBySearch worker.

Each tool is a ``(handler, pre_hook, post_hook)`` triple registered
with a :class:`~citeclaw.agents.tool_dispatch.WorkerDispatcher`.
Handlers return plain Python dicts; errors are raised as
:class:`DispatcherError` and translated to ``{error, hint}``
envelopes by the dispatcher.

Token discipline (see the v2 design doc, *Token budget discipline*
section): **abstracts appear in exactly one tool's output** —
:func:`get_paper`. Every listing / sampling / clustering / search
tool returns titles + year + venue + citation count only. The agent
can always call ``get_paper(paper_id)`` when it specifically wants
an abstract.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from citeclaw.agents.state import AngleState, query_fingerprint
from citeclaw.agents.tool_dispatch import (
    DispatcherError,
    ToolSpec,
    WorkerDispatcher,
    error_envelope,
    find_angle_by_df_id,
    require_df_id,
)

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger("citeclaw.agents.search_tools")

_FETCH_TOTAL_CAP = 50_000  # fetch_results refuses above this
_FETCH_TOTAL_WARN = 5_000


# ===========================================================================
# 1. check_query_size
# ===========================================================================


def _handle_check_query_size(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any]:
    query = args.get("query")
    filters = args.get("filters")
    if not isinstance(query, str) or not query.strip():
        raise DispatcherError(
            "missing or empty 'query' string",
            "query is a Lucene-style expression; quote multi-word phrases",
        )
    if filters is not None and not isinstance(filters, dict):
        raise DispatcherError(
            "'filters' must be a JSON object or null",
            "see the fieldsOfStudy / year / venue list in the system prompt",
        )

    # Pre-flight syntax lint — reject obvious broken queries before S2
    # sees them. A reject here is a teaching signal; the agent fixes
    # and re-calls in one turn. Unlike a true S2 call it costs no
    # network budget, so we can afford strict-ish rules.
    from citeclaw.agents.s2_query_lint import lint_s2_query
    lint = lint_s2_query(query)
    if not lint.ok:
        raise DispatcherError(
            f"S2 query lint: {lint.message}",
            lint.hint,
        )

    merged_filters = _merge_priors_into_filters(
        filters if isinstance(filters, dict) else None, d
    )
    payload = d.ctx.s2.search_bulk(
        query=query,
        filters=merged_filters or None,
        limit=3,
    )
    total = 0
    first_titles: list[str] = []
    if isinstance(payload, dict):
        total_raw = payload.get("total")
        if isinstance(total_raw, int) and total_raw >= 0:
            total = total_raw
        data = payload.get("data")
        if isinstance(data, list):
            for row in data[:3]:
                if isinstance(row, dict):
                    t = row.get("title")
                    if isinstance(t, str):
                        first_titles.append(t)

    fp = query_fingerprint(query, filters)
    return {
        "total": total,
        "first_3_titles": first_titles,
        "query_fingerprint": fp,
    }


def _pre_check_query_size(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any] | None:
    """Pre-hook: angle transition enforcement.

    If the fingerprint of this call differs from the active angle's
    fingerprint, the active angle's checklist must be complete.
    """
    query = args.get("query")
    filters = args.get("filters")
    if not isinstance(query, str) or not query.strip():
        return None  # let handler raise the proper error
    fp = query_fingerprint(query, filters if isinstance(filters, dict) else None)
    active = d.state.active_angle
    if active is not None and active.fingerprint != fp and not active.is_checklist_complete():
        missing = []
        if active.df_id is not None:
            if not active.checked_top_cited:
                missing.append("sample_titles(strategy='top_cited')")
            if not active.checked_random:
                missing.append("sample_titles(strategy='random')")
            if not active.checked_years:
                missing.append("year_distribution")
            if active.requires_topic_model and not active.checked_topic_model:
                missing.append("topic_model")
        if missing:
            return error_envelope(
                "current angle incomplete — cannot open a new angle yet",
                (
                    f"outstanding checks for angle {active.fingerprint}: "
                    f"{', '.join(missing)}. Run these on df_id={active.df_id!r} "
                    f"before size-checking a new (query, filters) tuple."
                ),
            )
    # Angle-cap pre-check: would this open a 5th angle?
    if fp not in d.state.angles and len(d.state.angles) >= d.config.max_angles_per_worker:
        return error_envelope(
            f"angle cap reached ({d.config.max_angles_per_worker})",
            (
                "you've opened the maximum number of distinct query angles "
                "for this sub-topic; finish inspection on existing angles "
                "and call done()"
            ),
        )
    return None


def _post_check_query_size(args: dict[str, Any], result: dict[str, Any], d: WorkerDispatcher) -> None:
    query = args.get("query")
    filters = args.get("filters")
    if not isinstance(query, str):
        return
    try:
        angle = d.get_or_create_angle(query, filters if isinstance(filters, dict) else None)
    except DispatcherError:
        return  # cap hit — the handler still returned a size, but don't register
    angle.total_in_corpus = result.get("total", angle.total_in_corpus)
    d.set_active(angle.fingerprint)


# ===========================================================================
# 2. fetch_results
# ===========================================================================


def _handle_fetch_results(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise DispatcherError(
            "pandas is not installed",
            "install via the topic_model extras: pip install 'citeclaw[topic_model]'",
        ) from exc

    query = args.get("query")
    filters = args.get("filters")
    if not isinstance(query, str) or not query.strip():
        raise DispatcherError("missing 'query' string", "see check_query_size")

    fp = query_fingerprint(query, filters if isinstance(filters, dict) else None)
    angle = d.state.angles.get(fp)
    if angle is None:
        raise DispatcherError(
            "fetch_results called for a query that was not size-checked",
            "call check_query_size with this exact (query, filters) pair first",
        )

    total = angle.total_in_corpus or 0
    if total > _FETCH_TOTAL_CAP:
        raise DispatcherError(
            f"query matches {total:,} papers (cap {_FETCH_TOTAL_CAP:,}) — refine first",
            (
                "add a structural prior (year, fieldsOfStudy, venue) or "
                "tighten the query with more specific terms"
            ),
        )

    limit = d.config.fetch_results_limit_per_strategy
    merged_filters = _merge_priors_into_filters(
        filters if isinstance(filters, dict) else None, d
    )
    # Two S2 calls: top-cited and paperId-order (~random).
    top_cited_payload = d.ctx.s2.search_bulk(
        query=query,
        filters=merged_filters or None,
        sort="citationCount:desc",
        limit=limit,
    )
    random_payload = d.ctx.s2.search_bulk(
        query=query,
        filters=merged_filters or None,
        sort="paperId",
        limit=limit,
    )

    paper_ids: list[str] = []
    seen: set[str] = set()
    for payload in (top_cited_payload, random_payload):
        data = (payload or {}).get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            continue
        for row in data:
            if not isinstance(row, dict):
                continue
            pid = row.get("paperId")
            if isinstance(pid, str) and pid and pid not in seen:
                seen.add(pid)
                paper_ids.append(pid)

    # Enrich via batch metadata.
    hydrated = d.ctx.s2.enrich_batch([{"paper_id": pid} for pid in paper_ids])

    # Build a DataFrame. Always construct with explicit columns so an
    # empty fetch still produces a well-shaped (0-row) DataFrame —
    # downstream tools that iterate df["paper_id"] then gracefully
    # return empty results instead of raising KeyError.
    columns = ["paper_id", "title", "abstract", "year", "venue", "citations"]
    rows = []
    for rec in hydrated:
        rows.append({
            "paper_id": rec.paper_id,
            "title": rec.title or "",
            "abstract": rec.abstract or "",
            "year": rec.year,
            "venue": rec.venue or "",
            "citations": rec.citation_count if rec.citation_count is not None else 0,
        })
    df = pd.DataFrame(rows, columns=columns)

    df_id = f"df_{d.worker_id}_{angle.fingerprint[7:15]}_t{d.state.turn_index}"
    d.store.put(df_id, df, worker_id=d.worker_id, metadata={
        "query": query,
        "filters": filters,
        "fingerprint": angle.fingerprint,
    })

    # Update cumulative set.
    for pid in df["paper_id"]:
        if isinstance(pid, str) and pid:
            d.state.cumulative_paper_ids.add(pid)

    n_fetched = len(df)
    strategy_label = f"top_cited:{limit} + paperId_order:{limit} (merged → {n_fetched})"
    return {
        "df_id": df_id,
        "n_fetched": n_fetched,
        "total_in_corpus": total,
        "fetch_strategy": strategy_label,
    }


def _pre_fetch_results(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any] | None:
    """Pre-hook: fingerprint must match a prior check_query_size."""
    query = args.get("query")
    filters = args.get("filters")
    if not isinstance(query, str) or not query.strip():
        return error_envelope(
            "missing or empty 'query' string",
            "fetch_results needs the same (query, filters) you size-checked",
        )
    if filters is not None and not isinstance(filters, dict):
        return error_envelope(
            "'filters' must be a JSON object or null",
            "reuse exactly the filters you passed to check_query_size",
        )
    fp = query_fingerprint(query, filters if isinstance(filters, dict) else None)
    if fp not in d.state.angles:
        return error_envelope(
            "this (query, filters) tuple was not size-checked",
            (
                "call check_query_size on this exact query first; "
                "the dispatcher verifies object identity, not temporal proximity"
            ),
        )
    return None


def _post_fetch_results(args: dict[str, Any], result: dict[str, Any], d: WorkerDispatcher) -> None:
    query = args.get("query")
    filters = args.get("filters")
    fp = query_fingerprint(
        query if isinstance(query, str) else "",
        filters if isinstance(filters, dict) else None,
    )
    angle = d.state.angles.get(fp)
    if angle is None:
        return
    angle.df_id = result.get("df_id")
    angle.n_fetched = result.get("n_fetched")
    angle.total_in_corpus = result.get("total_in_corpus", angle.total_in_corpus)
    d.set_active(angle.fingerprint)


# ===========================================================================
# 3. sample_titles
# ===========================================================================


def _handle_sample_titles(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any]:
    df_id = args["df_id"]
    strategy = args.get("strategy", "top_cited")
    n = int(args.get("n") or 20)
    if strategy not in ("top_cited", "random"):
        raise DispatcherError(
            f"unknown strategy {strategy!r}",
            "strategy must be 'top_cited' or 'random'",
        )
    df = d.store.get(df_id)
    n = max(1, min(n, len(df)))
    if strategy == "top_cited":
        sampled = df.sort_values("citations", ascending=False).head(n)
    else:
        # paperId-order was already applied at fetch time. For "random"
        # we sort by paper_id (stable, data-agnostic) and take head(n) —
        # this is reproducible and "random enough" for multi-angle
        # inspection purposes.
        sampled = df.sort_values("paper_id").head(n)
    rows = []
    for _, row in sampled.iterrows():
        rows.append({
            "paper_id": str(row.get("paper_id", "")),
            "title": str(row.get("title", "")),
            "year": int(row["year"]) if row.get("year") is not None and not _pd_is_nan(row.get("year")) else None,
            "venue": str(row.get("venue", "")),
            "citations": int(row.get("citations", 0) or 0),
        })
    return {"samples": rows, "strategy": strategy, "n_returned": len(rows)}


def _post_sample_titles(args: dict[str, Any], result: dict[str, Any], d: WorkerDispatcher) -> None:
    df_id = args.get("df_id")
    strategy = args.get("strategy", "top_cited")
    angle = find_angle_by_df_id(d, df_id) if isinstance(df_id, str) else None
    if angle is None:
        return
    if strategy == "top_cited":
        angle.checked_top_cited = True
    elif strategy == "random":
        angle.checked_random = True


# ===========================================================================
# 4. year_distribution
# ===========================================================================


def _handle_year_distribution(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any]:
    df_id = args["df_id"]
    df = d.store.get(df_id)
    counts: dict[str, int] = {}
    for y in df["year"]:
        if y is None or _pd_is_nan(y):
            continue
        key = str(int(y))
        counts[key] = counts.get(key, 0) + 1
    return {"year_counts": dict(sorted(counts.items()))}


def _post_year_distribution(args: dict[str, Any], result: dict[str, Any], d: WorkerDispatcher) -> None:
    df_id = args.get("df_id")
    angle = find_angle_by_df_id(d, df_id) if isinstance(df_id, str) else None
    if angle is not None:
        angle.checked_years = True


# ===========================================================================
# 5. venue_distribution
# ===========================================================================


def _handle_venue_distribution(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any]:
    df_id = args["df_id"]
    top_k = int(args.get("top_k") or 20)
    df = d.store.get(df_id)
    counts: dict[str, int] = {}
    for v in df["venue"]:
        if not v:
            continue
        counts[v] = counts.get(v, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return {"venue_counts": [{"venue": k, "count": v} for k, v in ranked]}


# ===========================================================================
# 6. topic_model
# ===========================================================================


def _handle_topic_model(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any]:
    df_id = args["df_id"]
    df = d.store.get(df_id)
    paper_ids = [str(p) for p in df["paper_id"].tolist() if isinstance(p, str) and p]

    embeddings_map = d.ctx.s2.fetch_embeddings_batch(paper_ids)
    # Build aligned arrays: paper_ids with non-None embeddings.
    kept_ids: list[str] = []
    vectors: list[list[float]] = []
    for pid in paper_ids:
        v = embeddings_map.get(pid)
        if isinstance(v, list) and v:
            kept_ids.append(pid)
            vectors.append(v)

    if len(kept_ids) < 10:
        return {
            "clusters": [],
            "reason": (
                f"only {len(kept_ids)} embeddings available (need >=10 for "
                f"topic model); skip topic_model on this angle"
            ),
        }

    # Lazy-import clustering deps so the tool module itself stays importable
    # without umap/hdbscan (e.g. tests that only exercise other tools).
    try:
        import hdbscan  # type: ignore[import-untyped]
        import numpy as np
        import umap  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DispatcherError(
            "topic_model requires umap-learn + hdbscan + numpy",
            "install: pip install 'citeclaw[topic_model]'",
        ) from exc

    X = np.asarray(vectors, dtype=float)
    n = X.shape[0]
    n_neighbors = max(5, min(15, n - 1))
    min_cluster_size = max(5, n // 25)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(2, min_cluster_size // 5),
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)

    # Build membership map, then reuse extract_keywords_ctfidf.
    from citeclaw.cluster.representation import extract_keywords_ctfidf
    from citeclaw.models import PaperRecord

    paper_rows = df.set_index("paper_id").to_dict(orient="index")
    papers_objs: dict[str, PaperRecord] = {}
    membership: dict[str, int] = {}
    for pid, lbl in zip(kept_ids, labels):
        row = paper_rows.get(pid) or {}
        papers_objs[pid] = PaperRecord(
            paper_id=pid,
            title=str(row.get("title", "")),
            abstract=str(row.get("abstract", "")),
            year=row.get("year") if not _pd_is_nan(row.get("year")) else None,
            venue=str(row.get("venue", "")),
        )
        membership[pid] = int(lbl)

    keywords = extract_keywords_ctfidf(membership, papers_objs, n_keywords=6)

    # Per-cluster sample titles (titles-only, no abstracts).
    cluster_to_ids: dict[int, list[str]] = {}
    for pid, lbl in membership.items():
        cluster_to_ids.setdefault(lbl, []).append(pid)
    clusters_out = []
    for cid, pids in sorted(cluster_to_ids.items()):
        if cid == -1:
            continue
        kw = keywords.get(cid, [])
        samples = []
        for spid in pids[:5]:
            row = paper_rows.get(spid) or {}
            samples.append({
                "paper_id": spid,
                "title": str(row.get("title", "")),
                "year": row.get("year") if not _pd_is_nan(row.get("year")) else None,
                "venue": str(row.get("venue", "")),
            })
        clusters_out.append({
            "cluster_id": cid,
            "size": len(pids),
            "ctfidf_label": ", ".join(kw[:5]) if kw else "(no label)",
            "keywords": kw,
            "sample_titles": samples,
        })
    noise_size = len(cluster_to_ids.get(-1, []))
    return {
        "clusters": clusters_out,
        "noise_size": noise_size,
        "n_embedded": len(kept_ids),
        "n_total": len(paper_ids),
    }


def _post_topic_model(args: dict[str, Any], result: dict[str, Any], d: WorkerDispatcher) -> None:
    df_id = args.get("df_id")
    angle = find_angle_by_df_id(d, df_id) if isinstance(df_id, str) else None
    if angle is not None:
        angle.checked_topic_model = True


# ===========================================================================
# 7. search_match
# ===========================================================================


def _handle_search_match(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any]:
    title = args.get("title")
    if not isinstance(title, str) or not title.strip():
        raise DispatcherError("missing 'title' string", "pass the exact paper title you want to locate")
    try:
        match = d.ctx.s2.search_match(title)
    except Exception as exc:  # noqa: BLE001
        raise DispatcherError(
            "S2 search_match failed",
            str(exc)[:200] or "(no message)",
        ) from exc
    if not match:
        return {"match": None}
    return {
        "match": {
            "paper_id": match.get("paperId"),
            "title": match.get("title"),
            "year": match.get("year"),
            "venue": match.get("venue"),
        },
    }


# ===========================================================================
# 8. contains
# ===========================================================================


def _handle_contains(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any]:
    pid = args.get("paper_id")
    if not isinstance(pid, str) or not pid.strip():
        raise DispatcherError(
            "missing 'paper_id' string",
            "contains(paper_id) checks if this paper is in the worker's cumulative fetch set",
        )
    present = pid in d.state.cumulative_paper_ids
    return {"contains": present, "cumulative_size": len(d.state.cumulative_paper_ids)}


def _pre_contains(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any] | None:
    if not d.state.cumulative_paper_ids:
        return error_envelope(
            "worker cumulative set is empty",
            "call fetch_results at least once before checking containment",
        )
    return None


def _post_contains(args: dict[str, Any], result: dict[str, Any], d: WorkerDispatcher) -> None:
    if result.get("contains") is False:
        pid = args.get("paper_id")
        if isinstance(pid, str):
            d.state.verification_misses.append(pid)


# ===========================================================================
# 9. search_within_df
# ===========================================================================


def _handle_search_within_df(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any]:
    df_id = args["df_id"]
    pattern = args.get("pattern")
    fields = args.get("fields") or ["title"]
    if not isinstance(pattern, str) or not pattern:
        raise DispatcherError("missing 'pattern'", "pass a regex or literal substring")
    if not isinstance(fields, list):
        raise DispatcherError("'fields' must be a list", "fields=['title'] or ['title','abstract']")
    valid_fields = {"title", "abstract", "venue"}
    bad = [f for f in fields if f not in valid_fields]
    if bad:
        raise DispatcherError(
            f"unknown fields: {bad}",
            f"valid fields: {sorted(valid_fields)}",
        )
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise DispatcherError(f"invalid regex {pattern!r}", str(exc)[:100]) from exc

    df = d.store.get(df_id)
    matches: list[dict[str, Any]] = []
    max_out = 50
    for _, row in df.iterrows():
        hit_field = None
        for f in fields:
            text = str(row.get(f, "") or "")
            if text and rx.search(text):
                hit_field = f
                break
        if hit_field is None:
            continue
        matches.append({
            "paper_id": str(row.get("paper_id", "")),
            "title": str(row.get("title", "")),
            "matching_field": hit_field,
            "year": int(row["year"]) if row.get("year") is not None and not _pd_is_nan(row.get("year")) else None,
            "venue": str(row.get("venue", "")),
        })
        if len(matches) >= max_out:
            break
    return {"matches": matches, "n_matches": len(matches)}


# ===========================================================================
# 10. get_paper (THE ONLY TOOL THAT RETURNS AN ABSTRACT)
# ===========================================================================


def _handle_get_paper(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any]:
    pid = args.get("paper_id")
    if not isinstance(pid, str) or not pid.strip():
        raise DispatcherError("missing 'paper_id'", "pass a Semantic Scholar paper_id")
    try:
        rec = d.ctx.s2.fetch_metadata(pid)
    except Exception as exc:  # noqa: BLE001
        raise DispatcherError(
            "paper not in S2",
            "search_match may have mis-resolved; try an alternate title",
        ) from exc
    # Fill abstract from OpenAlex fallback if S2 doesn't have it.
    try:
        d.ctx.s2.enrich_with_abstracts([rec])
    except Exception:  # noqa: BLE001 — best-effort
        pass
    authors = []
    for a in (rec.authors or []):
        authors.append({
            "authorId": getattr(a, "author_id", None),
            "name": getattr(a, "name", ""),
        })
    return {
        "paper_id": rec.paper_id,
        "title": rec.title or "",
        "abstract": rec.abstract or "",
        "year": rec.year,
        "venue": rec.venue or "",
        "citations": rec.citation_count if rec.citation_count is not None else 0,
        "authors": authors,
    }


# ===========================================================================
# 11. diagnose_miss
# ===========================================================================


_VALID_ACTIONS = {
    "accept_gap",
    "add_angle",
    "refine_current_angle",
    "relax_prior",
    "no_action",
}


def _handle_diagnose_miss(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any]:
    target = args.get("target_title")
    hypotheses = args.get("hypotheses")
    action = args.get("action_taken")
    angles_used = args.get("query_angles_used")
    if not isinstance(target, str) or not target.strip():
        raise DispatcherError("missing 'target_title'", "the paper title you were verifying")
    if not isinstance(hypotheses, list) or not hypotheses:
        raise DispatcherError(
            "'hypotheses' must be a non-empty list of strings",
            "give >=1 hypothesis for why the paper was missed",
        )
    if not isinstance(action, str) or action not in _VALID_ACTIONS:
        raise DispatcherError(
            f"invalid 'action_taken' — got {action!r}",
            f"valid actions: {sorted(_VALID_ACTIONS)}",
        )
    entry = {
        "target_title": target,
        "hypotheses": [str(h) for h in hypotheses if h],
        "action_taken": action,
        "query_angles_used": [str(q) for q in (angles_used or []) if q],
    }
    d.state.miss_diagnoses.append(entry)
    return {"acknowledged": True, "diagnoses_recorded": len(d.state.miss_diagnoses)}


def _pre_diagnose_miss(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any] | None:
    if not d.recent_contains_miss():
        return error_envelope(
            "no recent miss to diagnose",
            (
                "diagnose_miss is only valid immediately after a contains(...) "
                "that returned False; run verification first"
            ),
        )
    return None


def _post_diagnose_miss(args: dict[str, Any], result: dict[str, Any], d: WorkerDispatcher) -> None:
    action = args.get("action_taken")
    if action == "refine_current_angle" and d.state.active_angle is not None:
        angle = d.state.active_angle
        angle.refinement_count += 1
        if angle.refinement_count > d.config.max_refinement_per_angle:
            # Soft signal — the worker's own prompt should push it to a new
            # angle. We don't reject the diagnose itself (it's still
            # information), but the next refine attempt on this angle will
            # be rejected by _pre_check_query_size via the angle_incomplete
            # path if the worker tries to go around.
            log.info(
                "angle %s exceeded refinement cap (%d)",
                angle.fingerprint, d.config.max_refinement_per_angle,
            )


# ===========================================================================
# 12. done (worker)
# ===========================================================================


def _handle_done(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any]:
    paper_ids = args.get("paper_ids")
    assessment = args.get("coverage_assessment")
    summary = args.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise DispatcherError("missing 'summary'", "one-paragraph self-report on what was done")
    if assessment not in ("comprehensive", "acceptable", "limited"):
        raise DispatcherError(
            f"invalid 'coverage_assessment' — got {assessment!r}",
            "must be 'comprehensive', 'acceptable', or 'limited'",
        )
    # paper_ids in args is advisory — the authoritative set is always
    # the worker's cumulative_paper_ids. We take the union with any
    # supplied ids so the worker can't accidentally drop its own work.
    final_ids = set(d.state.cumulative_paper_ids)
    if isinstance(paper_ids, list):
        for p in paper_ids:
            if isinstance(p, str) and p:
                final_ids.add(p)
    return {
        "paper_ids": sorted(final_ids),
        "coverage_assessment": assessment,
        "summary": summary,
        "n_paper_ids": len(final_ids),
    }


def _pre_done(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any] | None:
    # At least one fetch_results must have run.
    fetched_angles = [a for a in d.state.angles.values() if a.df_id is not None]
    if not fetched_angles:
        return error_envelope(
            "no fetch_results completed yet",
            "run at least one full angle (check_query_size → fetch_results → inspect → verify) before done()",
        )
    # Every fetched angle must have its per-angle checklist complete.
    incomplete = [a for a in fetched_angles if not a.is_checklist_complete()]
    if incomplete:
        parts = []
        for a in incomplete:
            missing = []
            if not a.checked_top_cited:
                missing.append("sample_titles(top_cited)")
            if not a.checked_random:
                missing.append("sample_titles(random)")
            if not a.checked_years:
                missing.append("year_distribution")
            if a.requires_topic_model and not a.checked_topic_model:
                missing.append("topic_model (required: n_fetched >= 500)")
            parts.append(f"df_id={a.df_id} missing {', '.join(missing) or '(unknown)'}")
        return error_envelope(
            "per-angle inspection checklist incomplete",
            "; ".join(parts),
        )
    # At least one verification cycle.
    contains_calls = [e for e in d.state.call_log if e.get("tool") == "contains"]
    if not contains_calls:
        return error_envelope(
            "no verification cycle performed",
            (
                "before done() you must run at least one search_match -> "
                "contains cycle on a reference paper (from the sub_topic spec "
                "OR your own domain knowledge)"
            ),
        )
    # Every miss must have a corresponding diagnose_miss.
    misses = sum(1 for e in contains_calls if (e.get("result") or {}).get("contains") is False)
    if misses > len(d.state.miss_diagnoses):
        return error_envelope(
            f"{misses} verification miss(es) without corresponding diagnose_miss",
            "run diagnose_miss(...) for each miss before done()",
        )
    return None


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def _handle_abandon_angle(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any]:
    """Drop the current active angle without completing its checklist.

    Removes the AngleState from the registry, drops its DataFrame from
    the store, and removes its fetched paper_ids from the worker's
    cumulative set (so verification doesn't count them). The angle
    slot is freed — len(angles) decreases, so this does NOT count
    against ``max_angles_per_worker``.

    After abandon, the worker has no active angle; next
    check_query_size opens a fresh one.
    """
    active = d.state.active_angle
    if active is None:
        raise DispatcherError(
            "no active angle to abandon",
            "abandon_angle is only valid when an angle is active (after check_query_size)",
        )
    fp = active.fingerprint
    abandoned_df = active.df_id
    abandoned_n = active.n_fetched or 0
    # Remove fetched papers from the cumulative set. We need the
    # paper_ids that were added by THIS angle. The simplest correct
    # approach: re-derive from the DataFrame (if we still have it),
    # then subtract.
    removed_ids = 0
    if abandoned_df and abandoned_df in d.store:
        try:
            df = d.store.get(abandoned_df)
            for pid in df["paper_id"]:
                if isinstance(pid, str) and pid in d.state.cumulative_paper_ids:
                    d.state.cumulative_paper_ids.remove(pid)
                    removed_ids += 1
        except Exception:  # noqa: BLE001
            pass
        d.store.drop(abandoned_df)
    # Remove the angle entirely and clear the active pointer.
    d.state.angles.pop(fp, None)
    d.state.active_fingerprint = None
    return {
        "abandoned_fingerprint": fp,
        "n_papers_removed_from_cumulative": removed_ids,
        "n_fetched_in_abandoned_df": abandoned_n,
        "angles_remaining": len(d.state.angles),
    }


def register_worker_tools(dispatcher: WorkerDispatcher) -> None:
    """Register all 13 worker tools on ``dispatcher``."""
    specs = [
        ToolSpec("check_query_size", _handle_check_query_size,
                 pre_hook=_pre_check_query_size, post_hook=_post_check_query_size),
        ToolSpec("fetch_results", _handle_fetch_results,
                 pre_hook=_pre_fetch_results, post_hook=_post_fetch_results),
        ToolSpec("sample_titles", _handle_sample_titles,
                 pre_hook=require_df_id, post_hook=_post_sample_titles),
        ToolSpec("year_distribution", _handle_year_distribution,
                 pre_hook=require_df_id, post_hook=_post_year_distribution),
        ToolSpec("venue_distribution", _handle_venue_distribution,
                 pre_hook=require_df_id),
        ToolSpec("topic_model", _handle_topic_model,
                 pre_hook=require_df_id, post_hook=_post_topic_model),
        ToolSpec("search_match", _handle_search_match),
        ToolSpec("contains", _handle_contains,
                 pre_hook=_pre_contains, post_hook=_post_contains),
        ToolSpec("search_within_df", _handle_search_within_df,
                 pre_hook=require_df_id),
        ToolSpec("get_paper", _handle_get_paper),
        ToolSpec("diagnose_miss", _handle_diagnose_miss,
                 pre_hook=_pre_diagnose_miss, post_hook=_post_diagnose_miss),
        ToolSpec("abandon_angle", _handle_abandon_angle),
        ToolSpec("done", _handle_done, pre_hook=_pre_done),
    ]
    dispatcher.register_many(specs)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _pd_is_nan(v: Any) -> bool:
    """True iff ``v`` is pandas/numpy NaN (not just any falsy value)."""
    try:
        import math
        return isinstance(v, float) and math.isnan(v)
    except Exception:  # noqa: BLE001
        return False


def _merge_priors_into_filters(
    per_call_filters: dict[str, Any] | None,
    d: WorkerDispatcher,
) -> dict[str, Any]:
    """Merge the worker's structural priors into the per-call filter dict.

    Per-call filters take precedence when they overlap a prior — the
    worker can locally narrow (say, year) without changing the priors.
    Priors are read from ``d.state.structural_priors`` which the
    worker loop populates at construction.
    """
    priors = getattr(d.state, "structural_priors", None)
    merged: dict[str, Any] = {}
    if priors is not None:
        merged.update(priors.to_s2_filters())
    if per_call_filters:
        merged.update(per_call_filters)
    return merged


__all__ = [
    "register_worker_tools",
]
