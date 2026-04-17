"""Tool implementations for the ExpandBySearch worker (post-refactor).

Shape change from v2 → v2.1 (this file): the LLM only speaks out
**decisions** — which query to try, which miss to diagnose, which
paper to deep-dive, when to close. Deterministic post-fetch steps
(sampling, distributions, topic model, reference-paper verification)
are baked INTO ``fetch_results``, which returns a single inspection
digest. The dispatcher no longer orchestrates a per-angle checklist;
there are no ``checked_*`` flags, no ``inspect_angle``, no
``sample_titles`` / ``year_distribution`` / ``venue_distribution`` /
``topic_model`` / ``search_match`` / ``contains`` / ``abandon_angle``
tools. The worker's tool surface is:

    check_query_size, fetch_results, query_diagnostics,
    search_within_df, get_paper, diagnose_miss, done

Seven tools. See docstrings below for per-tool semantics. Token
discipline is unchanged: abstracts appear only in seed papers and
via :func:`_handle_get_paper`.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

from citeclaw.agents.state import query_fingerprint
from citeclaw.agents.tool_dispatch import (
    DispatcherError,
    ToolSpec,
    WorkerDispatcher,
    error_envelope,
    find_query_by_df_id,
    require_df_id,
)

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger("citeclaw.agents.search_tools")

_FETCH_TOTAL_CAP_DEFAULT = 50_000


# ===========================================================================
# 1. check_query_size — size-check before committing to a fetch
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
    # and re-calls in one turn.
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
    """Reject at-cap opens of NEW queries. Same-query re-checks are free."""
    query = args.get("query")
    filters = args.get("filters")
    if not isinstance(query, str) or not query.strip():
        return None  # handler will raise the proper error
    fp = query_fingerprint(query, filters if isinstance(filters, dict) else None)
    if fp not in d.state.queries and len(d.state.queries) >= d.config.max_queries_per_worker:
        return error_envelope(
            f"query cap reached ({d.config.max_queries_per_worker})",
            (
                f"you've opened {len(d.state.queries)} of "
                f"{d.config.max_queries_per_worker} distinct queries. "
                "Diagnose outstanding misses and call done() — more "
                "refinement past the cap isn't productive."
            ),
        )
    return None


def _post_check_query_size(args: dict[str, Any], result: dict[str, Any], d: WorkerDispatcher) -> None:
    query = args.get("query")
    filters = args.get("filters")
    if not isinstance(query, str):
        return
    try:
        q = d.get_or_create_query(query, filters if isinstance(filters, dict) else None)
    except DispatcherError:
        return
    q.total_in_corpus = result.get("total", q.total_in_corpus)
    d.set_active(q.fingerprint)


# ===========================================================================
# 2. fetch_results — fetch + inspect + verify, all in one dispatch
# ===========================================================================
#
# This is the refactor's core: one call does everything deterministic.
# Return shape:
#     {
#       "df_id", "n_fetched", "total_in_corpus", "fetch_strategy",
#       "top_cited_titles": [...],        # ~100 items
#       "random_titles": [...],           # ~100 items
#       "year_distribution": {...},
#       "venue_distribution": [...],      # top 20
#       "topic_model": {clusters, ...} | {"skipped": ..., "reason": ...},
#       "frequent_ngrams": [{ngram, frequency}, ...],
#       "reference_coverage": {...} | None,
#     }


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
    q = d.state.queries.get(fp)
    if q is None:
        raise DispatcherError(
            "fetch_results called for a query that was not size-checked",
            "call check_query_size with this exact (query, filters) pair first",
        )

    total = q.total_in_corpus or 0
    cap = getattr(d.config, "fetch_total_cap", _FETCH_TOTAL_CAP_DEFAULT)
    if total > cap:
        raise DispatcherError(
            f"query matches {total:,} papers (cap {cap:,}) — refine first",
            (
                "add a structural filter (year / fieldsOfStudy / venue) — "
                "the sampled fetch would be unreliable at this size. "
                "Prefer filters over extra '+' clauses."
            ),
        )

    # --- Fetch (2 S2 calls: top_cited + paperId-order) --------------
    limit = d.config.fetch_results_limit_per_strategy
    merged_filters = _merge_priors_into_filters(
        filters if isinstance(filters, dict) else None, d
    )
    top_cited_payload = d.ctx.s2.search_bulk(
        query=query, filters=merged_filters or None,
        sort="citationCount:desc", limit=limit,
    )
    random_payload = d.ctx.s2.search_bulk(
        query=query, filters=merged_filters or None,
        sort="paperId", limit=limit,
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
    hydrated = d.ctx.s2.enrich_batch([{"paper_id": pid} for pid in paper_ids])

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

    df_id = f"df_{d.worker_id}_{q.fingerprint[7:15]}_t{d.state.turn_index}"
    d.store.put(df_id, df, worker_id=d.worker_id, metadata={
        "query": query, "filters": filters, "fingerprint": q.fingerprint,
    })
    # Update worker state.
    for pid in df["paper_id"]:
        if isinstance(pid, str) and pid:
            d.state.cumulative_paper_ids.add(pid)
    q.df_id = df_id
    q.n_fetched = len(df)
    d.set_active(q.fingerprint)

    n_fetched = len(df)

    # --- Auto-inspection --------------------------------------------
    sample_n = getattr(d.config, "inspection_sample_size", 100)
    top_titles = _internal_sample(df, strategy="top_cited", n=sample_n)
    random_titles = _internal_sample(df, strategy="random", n=sample_n)
    year_counts = _internal_year_distribution(df)
    venue_counts = _internal_venue_distribution(df, top_k=20)

    # Extract frequent n-grams from titles — cheap co-sampling that
    # often surfaces synonyms before topic_model's n≥500 threshold.
    title_texts = [r["title"] for r in top_titles] + [r["title"] for r in random_titles]
    ngrams = _internal_extract_ngrams(title_texts, n_min=2, n_max=3, top_k=20)

    # Topic model when the corpus is big enough. Silent skip below
    # the threshold.
    if n_fetched >= 500:
        try:
            tm = _internal_topic_model(df, d)
        except DispatcherError as exc:
            tm = {"skipped": True, "reason": str(exc.error)}
        except Exception as exc:  # noqa: BLE001
            tm = {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}
    else:
        tm = {
            "skipped": True,
            "reason": f"n_fetched={n_fetched} < 500 threshold",
        }

    # --- Auto-verification against reference papers -----------------
    ref_coverage = _internal_auto_verify_references(d)

    fetch_strategy = f"top_cited:{limit} + paperId_order:{limit} (merged → {n_fetched})"
    return {
        "df_id": df_id,
        "n_fetched": n_fetched,
        "total_in_corpus": total,
        "fetch_strategy": fetch_strategy,
        "top_cited_titles": top_titles,
        "random_titles": random_titles,
        "year_distribution": year_counts,
        "venue_distribution": venue_counts,
        "topic_model": tm,
        "frequent_ngrams": ngrams,
        "reference_coverage": ref_coverage,
    }


def _pre_fetch_results(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any] | None:
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
    if fp not in d.state.queries:
        return error_envelope(
            "this (query, filters) tuple was not size-checked",
            "call check_query_size on this exact query first",
        )
    return None


# ===========================================================================
# 3. diagnose_miss — explain a reference paper that fell outside coverage
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
    queries_used = args.get("query_angles_used") or args.get("queries_used")
    if not isinstance(target, str) or not target.strip():
        raise DispatcherError(
            "missing 'target_title'",
            "the title of the reference paper that was flagged as a miss",
        )
    if isinstance(hypotheses, str) and hypotheses.strip():
        hyp_list = [hypotheses.strip()]
    elif isinstance(hypotheses, list):
        hyp_list = [str(h) for h in hypotheses if h]
    else:
        hyp_list = []
    if not hyp_list:
        hyp_list = ["(no hypothesis provided)"]
    if not isinstance(action, str) or action not in _VALID_ACTIONS:
        raise DispatcherError(
            f"invalid 'action_taken' — got {action!r}",
            f"valid actions: {sorted(_VALID_ACTIONS)}",
        )
    if isinstance(queries_used, str):
        q_list = [queries_used]
    elif isinstance(queries_used, list):
        q_list = [str(q) for q in queries_used if q]
    else:
        q_list = []
    entry = {
        "target_title": target,
        "hypotheses": hyp_list,
        "action_taken": action,
        "queries_used": q_list,
    }
    d.state.miss_diagnoses.append(entry)
    remaining = max(0, len(d.state.pending_miss_titles) - len(d.state.miss_diagnoses))
    return {
        "acknowledged": True,
        "diagnoses_recorded": len(d.state.miss_diagnoses),
        "pending_misses_remaining": remaining,
    }


def _pre_diagnose_miss(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any] | None:
    if d.pending_miss_count() <= 0:
        return error_envelope(
            "no pending verification misses to diagnose",
            (
                "diagnose_miss consumes an entry from pending_miss_titles. "
                "Those are flagged automatically by fetch_results' "
                "reference_coverage section. Either run fetch_results (so "
                "the auto-verifier surfaces misses) or skip this call."
            ),
        )
    return None


# ===========================================================================
# 4. search_within_df — regex over a fetched DataFrame
# ===========================================================================


def _handle_search_within_df(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any]:
    df_id = args["df_id"]
    pattern = args.get("pattern")
    fields = args.get("fields") or ["title"]
    if not isinstance(pattern, str) or not pattern:
        raise DispatcherError("missing 'pattern'", "pass a regex or literal substring")
    if not isinstance(fields, list):
        raise DispatcherError("'fields' must be a list", "fields=['title'] or ['title','abstract']")
    valid = {"title", "abstract", "venue"}
    bad = [f for f in fields if f not in valid]
    if bad:
        raise DispatcherError(
            f"unknown fields: {bad}", f"valid fields: {sorted(valid)}",
        )
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise DispatcherError(f"invalid regex {pattern!r}", str(exc)[:100]) from exc
    df = d.store.get(df_id)
    matches: list[dict[str, Any]] = []
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
        if len(matches) >= 50:
            break
    return {"matches": matches, "n_matches": len(matches)}


# ===========================================================================
# 5. get_paper — full metadata + abstract
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
            "the paper_id may be wrong; try resolving via reference_coverage",
        ) from exc
    try:
        d.ctx.s2.enrich_with_abstracts([rec])
    except Exception:  # noqa: BLE001
        pass
    authors = [{"authorId": getattr(a, "author_id", None), "name": getattr(a, "name", "")}
               for a in (rec.authors or [])]
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
# 6. query_diagnostics — per-OR-leaf hit counts (in-context + raw)
# ===========================================================================


_DIAGNOSTICS_MAX_LEAVES = 12


def _handle_query_diagnostics(args: dict[str, Any], d: WorkerDispatcher) -> dict[str, Any]:
    query = args.get("query")
    filters = args.get("filters")
    if not isinstance(query, str) or not query.strip():
        raise DispatcherError(
            "missing or empty 'query' string",
            "pass the full (possibly-complex) query you want to break down",
        )
    if filters is not None and not isinstance(filters, dict):
        raise DispatcherError(
            "'filters' must be a JSON object or null",
            "pass the same filter shape you'd use in check_query_size",
        )

    from citeclaw.agents.s2_query_lint import lint_s2_query
    lint = lint_s2_query(query)
    if not lint.ok:
        raise DispatcherError(f"S2 query lint: {lint.message}", lint.hint)

    from citeclaw.agents.query_parser import enumerate_or_leaves, tree_signature
    try:
        tree, substitutions = enumerate_or_leaves(query)
    except Exception as exc:  # noqa: BLE001
        raise DispatcherError(
            f"could not parse query: {exc}",
            "use Lucene syntax with balanced parens and quoted phrases",
        ) from exc

    sig = tree_signature(tree)
    merged_filters = _merge_priors_into_filters(
        filters if isinstance(filters, dict) else None, d
    )
    # Full-query total.
    full_resp = d.ctx.s2.search_bulk(query=query, filters=merged_filters or None, limit=1)
    full_total = 0
    if isinstance(full_resp, dict):
        tot = full_resp.get("total")
        if isinstance(tot, int) and tot >= 0:
            full_total = tot

    if not substitutions:
        return {
            "total_full_query": full_total,
            "tree_shape": sig,
            "or_groups": [],
            "note": (
                "query contains no OR (|) operators — per-branch breakdown "
                "is only meaningful when at least one | is present"
            ),
        }
    if len(substitutions) > _DIAGNOSTICS_MAX_LEAVES:
        return {
            "total_full_query": full_total,
            "tree_shape": sig,
            "or_groups": [],
            "note": (
                f"query has {len(substitutions)} OR leaves across "
                f"{sig['or_groups']} groups — exceeds diagnostics cap of "
                f"{_DIAGNOSTICS_MAX_LEAVES}. Simplify the query."
            ),
        }

    groups: dict[int, list[dict[str, Any]]] = {}
    raw_breakdown: dict[str, int | None] = {}
    for sub in substitutions:
        # In-context count.
        total: int | None = None
        try:
            resp = d.ctx.s2.search_bulk(
                query=sub.substituted_query, filters=merged_filters or None, limit=1,
            )
            if isinstance(resp, dict):
                r = resp.get("total")
                if isinstance(r, int) and r >= 0:
                    total = r
        except Exception:  # noqa: BLE001
            total = None
        # Raw count (leaf alone).
        total_raw: int | None = None
        try:
            raw_resp = d.ctx.s2.search_bulk(
                query=sub.leaf_text, filters=merged_filters or None, limit=1,
            )
            if isinstance(raw_resp, dict):
                r = raw_resp.get("total")
                if isinstance(r, int) and r >= 0:
                    total_raw = r
        except Exception:  # noqa: BLE001
            total_raw = None
        if sub.leaf_text not in raw_breakdown:
            raw_breakdown[sub.leaf_text] = total_raw
        groups.setdefault(sub.group_id, []).append({
            "leaf": sub.leaf_text,
            "substituted_query": sub.substituted_query,
            "total_in_context": total,
            "total_raw": total_raw,
        })

    or_groups_out = []
    for gid in sorted(groups.keys()):
        leaves = sorted(groups[gid], key=lambda e: -(e.get("total_in_context") or -1))
        or_groups_out.append({
            "group_id": gid,
            "n_leaves": len(leaves),
            "leaves": leaves,
        })
    return {
        "total_full_query": full_total,
        "tree_shape": sig,
        "raw_breakdown": raw_breakdown,
        "or_groups": or_groups_out,
        "n_leaves_probed": len(substitutions),
    }


# ===========================================================================
# 7. done — closure
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
    # Must have at least one successful fetch.
    fetched = [q for q in d.state.queries.values() if q.df_id is not None]
    if not fetched:
        return error_envelope(
            "no fetch_results completed yet",
            "run at least one query (check_query_size → fetch_results) before done()",
        )
    # Every pending miss must have a matching diagnose_miss.
    pending = d.pending_miss_count()
    if pending > 0:
        return error_envelope(
            f"{pending} auto-detected reference miss(es) not diagnosed",
            (
                "fetch_results' reference_coverage surfaced papers "
                "resolved by S2 but not in your cumulative set. Each must "
                "be explained via diagnose_miss before done() is accepted."
            ),
        )
    return None


# ===========================================================================
# Internal helpers — DETERMINISTIC post-fetch operations (not LLM tools)
# ===========================================================================


def _internal_sample(df: "pd.DataFrame", *, strategy: str, n: int) -> list[dict[str, Any]]:
    """Return up to ``n`` rows per strategy as lightweight dicts.

    ``top_cited`` sorts by citations desc; ``random`` sorts by
    paper_id (which the fetch already ordered via ``sort=paperId``,
    so this is deterministic and representative).
    """
    n = max(1, min(n, len(df)))
    if strategy == "top_cited":
        sampled = df.sort_values("citations", ascending=False).head(n)
    else:
        sampled = df.sort_values("paper_id").head(n)
    out: list[dict[str, Any]] = []
    for _, row in sampled.iterrows():
        out.append({
            "paper_id": str(row.get("paper_id", "")),
            "title": str(row.get("title", "")),
            "year": int(row["year"]) if row.get("year") is not None and not _pd_is_nan(row.get("year")) else None,
            "venue": str(row.get("venue", "")),
            "citations": int(row.get("citations", 0) or 0),
        })
    return out


def _internal_year_distribution(df: "pd.DataFrame") -> dict[str, int]:
    counts: dict[str, int] = {}
    for y in df["year"]:
        if y is None or _pd_is_nan(y):
            continue
        key = str(int(y))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _internal_venue_distribution(df: "pd.DataFrame", *, top_k: int) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for v in df["venue"]:
        if not v:
            continue
        counts[v] = counts.get(v, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return [{"venue": k, "count": v} for k, v in ranked]


# Small English stoplist for n-gram extraction — short enough to eyeball,
# big enough to remove the most dominant grammatical noise. We intentionally
# KEEP tokens like "learning" / "neural" / "transformer" because they're
# topic-signalling in CS/ML titles.
_NGRAM_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to",
    "via", "with", "we", "our", "using", "use", "based", "toward", "towards",
    "study", "analysis", "approach", "method", "paper", "new",
})

_NGRAM_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")


def _internal_extract_ngrams(
    titles: list[str], *, n_min: int, n_max: int, top_k: int,
) -> list[dict[str, Any]]:
    """Top-K bi/trigrams across titles. Cheap lexical inspection.

    Serves the same purpose as topic_model at smaller scales where
    embeddings-based clustering would be skipped (n<500). Filters an
    English stoplist and tokens < 2 chars.
    """
    counter: Counter[str] = Counter()
    for title in titles:
        if not isinstance(title, str):
            continue
        toks = [t.lower() for t in _NGRAM_TOKEN_RE.findall(title)]
        toks = [t for t in toks if t not in _NGRAM_STOPWORDS and len(t) > 1]
        for n in range(n_min, n_max + 1):
            for i in range(len(toks) - n + 1):
                window = toks[i:i + n]
                # Skip windows dominated by stopwords already filtered
                # above (guard against residual single-char tokens).
                if all(len(t) <= 2 for t in window):
                    continue
                ng = " ".join(window)
                counter[ng] += 1
    return [
        {"ngram": ng, "frequency": freq}
        for ng, freq in counter.most_common(top_k)
    ]


def _internal_topic_model(df: "pd.DataFrame", d: WorkerDispatcher) -> dict[str, Any]:
    """UMAP + HDBSCAN over SPECTER2 embeddings for the fetched df.

    Hyperparameter notes (post-refactor):

    - ``n_neighbors = max(5, min(15, n - 1))`` — same as before.
    - ``min_cluster_size = max(5, n // 50)`` — HALVED from the prior
      ``n // 25`` so clusters are more granular. User feedback: the
      coarser setting was collapsing related-but-distinct slices into
      single clusters, hiding the outliers that drive "add a new
      angle" decisions.
    - ``n_samples_per_cluster = 5`` unchanged. Each cluster now has
      ~5-15 members (instead of 10-30), so 5 samples is still a
      representative slice.
    """
    paper_ids = [str(p) for p in df["paper_id"].tolist() if isinstance(p, str) and p]
    embeddings_map = d.ctx.s2.fetch_embeddings_batch(paper_ids)
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
                f"topic model); skipped"
            ),
            "skipped": True,
        }
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
    # Halved from n//25 (user note): more granular clusters expose
    # outlier slices that inform new-angle decisions. Floor of 5 keeps
    # clusters interpretable.
    min_cluster_size = max(5, n // 50)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, n_components=5, min_dist=0.0,
        metric="cosine", random_state=42,
    )
    reduced = reducer.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(2, min_cluster_size // 5),
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)

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


def _internal_auto_verify_references(d: WorkerDispatcher) -> dict[str, Any] | None:
    """Resolve every ``state.reference_papers`` title against S2, check
    against ``state.cumulative_paper_ids``, and return a report.

    Also APPENDS matched-but-not-in-cumulative titles to
    ``state.pending_miss_titles`` so ``_pre_done`` can require
    diagnose_miss before closure. The append is *de-duplicated* so
    re-running fetch_results on a new query doesn't multiply the
    pending list — only NEW misses count.

    Returns ``None`` when the sub_topic has no reference_papers
    (nothing to verify against). Returns a structured report
    otherwise.
    """
    refs = tuple(d.state.reference_papers or ())
    if not refs:
        return None
    matched_in: list[dict[str, Any]] = []
    matched_not_in: list[dict[str, Any]] = []
    not_in_s2: list[str] = []
    for title in refs:
        try:
            match = d.ctx.s2.search_match(title)
        except Exception:  # noqa: BLE001
            match = None
        if not match:
            not_in_s2.append(title)
            continue
        pid = match.get("paperId") if isinstance(match, dict) else None
        if not isinstance(pid, str):
            not_in_s2.append(title)
            continue
        record = {
            "title": title,
            "paper_id": pid,
            "matched_title": match.get("title"),
        }
        if pid in d.state.cumulative_paper_ids:
            matched_in.append(record)
        else:
            matched_not_in.append(record)
            # Dedup against pending list so repeated fetches don't
            # inflate the diagnose-miss backlog.
            if title not in d.state.pending_miss_titles:
                d.state.pending_miss_titles.append(title)
    return {
        "matched_in_cumulative": matched_in,
        "matched_not_in_cumulative": matched_not_in,
        "not_in_s2": not_in_s2,
        "summary": {
            "matched_in_cumulative": len(matched_in),
            "matched_not_in_cumulative": len(matched_not_in),
            "not_in_s2": len(not_in_s2),
        },
    }


# ===========================================================================
# Registration
# ===========================================================================


def register_worker_tools(dispatcher: WorkerDispatcher) -> None:
    """Register all 7 worker tools on ``dispatcher``."""
    specs = [
        ToolSpec("check_query_size", _handle_check_query_size,
                 pre_hook=_pre_check_query_size, post_hook=_post_check_query_size),
        ToolSpec("fetch_results", _handle_fetch_results,
                 pre_hook=_pre_fetch_results),
        ToolSpec("diagnose_miss", _handle_diagnose_miss,
                 pre_hook=_pre_diagnose_miss),
        ToolSpec("search_within_df", _handle_search_within_df,
                 pre_hook=require_df_id),
        ToolSpec("get_paper", _handle_get_paper),
        ToolSpec("query_diagnostics", _handle_query_diagnostics),
        ToolSpec("done", _handle_done, pre_hook=_pre_done),
    ]
    dispatcher.register_many(specs)


# ===========================================================================
# Small helpers
# ===========================================================================


def _pd_is_nan(v: Any) -> bool:
    try:
        return isinstance(v, float) and math.isnan(v)
    except Exception:  # noqa: BLE001
        return False


def _merge_priors_into_filters(
    per_call_filters: dict[str, Any] | None, d: WorkerDispatcher,
) -> dict[str, Any]:
    priors = getattr(d.state, "structural_priors", None)
    merged: dict[str, Any] = {}
    if priors is not None:
        merged.update(priors.to_s2_filters())
    if per_call_filters:
        merged.update(per_call_filters)
    return merged


__all__ = ["register_worker_tools"]
