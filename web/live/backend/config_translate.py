"""Translate the v3 design's pipeline model into a real CiteClaw config.

The design UI (``web/live/static/jsx/data.jsx`` → ``INITIAL_PIPELINE``)
already speaks CiteClaw's vocabulary almost verbatim: pipeline nodes with
a ``kind`` of ``seed / fwd / bwd / rerank / rsc / sink`` and a screener
"filter tree" whose node ``kind``s are ``Sequential / Any / Not / Route``
plus leaf atoms (``YearFilter``, ``CitationFilter``, ``LLMFilter``,
``SimilarityFilter``, ``*KeywordFilter``) with the same param names.

This module walks that client-side model and emits the dict that
``citeclaw.config.load_settings(None, overrides=...)`` accepts. It never
imports front-end code — the browser POSTs the plain JSON model.
"""

from __future__ import annotations

from typing import Any

# Design leaf-kind -> CiteClaw block ``type``. Names already match 1:1, but
# we keep an explicit table so an unknown kind fails loudly rather than
# silently producing an invalid config.
_ATOM_TYPES = {
    "YearFilter",
    "CitationFilter",
    "LLMFilter",
    "SimilarityFilter",
    "TitleKeywordFilter",
    "AbstractKeywordFilter",
    "VenueKeywordFilter",
}
_MEASURE_TYPES = {"RefSim", "CitSim", "SemanticSim"}
_COMPOSITE = {"Sequential", "Any", "Not", "Route", "Parallel"}


class TranslationError(ValueError):
    """Raised when the design model can't be mapped to a valid config."""


def _clean(d: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is ``None`` / ``""`` so defaults apply."""
    return {k: v for k, v in d.items() if v not in (None, "")}


def _translate_measure(m: dict[str, Any]) -> dict[str, Any]:
    kind = m.get("kind") or m.get("type")
    if kind not in _MEASURE_TYPES:
        raise TranslationError(f"Unknown similarity measure: {kind!r}")
    out: dict[str, Any] = {"type": kind}
    if kind == "CitSim" and m.get("pass_if_cited_at_least") is not None:
        out["pass_if_cited_at_least"] = int(m["pass_if_cited_at_least"])
    if kind == "SemanticSim":
        out["embedder"] = m.get("embedder", "s2")
    return out


def _translate_filter(node: dict[str, Any]) -> dict[str, Any]:
    """Recursively translate one filter-tree node into a CiteClaw block."""
    if not isinstance(node, dict):
        raise TranslationError(f"Filter node must be an object, got {type(node)}")
    kind = node.get("kind") or node.get("type")
    params = node.get("params") or {}

    if kind in _COMPOSITE:
        children = node.get("children") or []
        translated = [_translate_filter(c) for c in children]
        if kind == "Not":
            if not translated:
                raise TranslationError("Not filter needs exactly one child")
            return {"type": "Not", "layer": translated[0]}
        if kind == "Route":
            # Design Route children carry {if, pass_to} / {default}; pass them
            # through unchanged (predicate keys already match builder.py).
            return {"type": "Route", "routes": node.get("routes", [])}
        # Sequential / Any / Parallel all use ``layers``
        return {"type": kind, "layers": translated}

    if kind not in _ATOM_TYPES:
        raise TranslationError(f"Unknown filter kind: {kind!r}")

    out: dict[str, Any] = {"type": kind}

    if kind == "YearFilter":
        out.update(_clean({"min": params.get("min"), "max": params.get("max")}))
    elif kind == "CitationFilter":
        out.update(
            _clean(
                {
                    "beta": params.get("beta"),
                    "exemption_years": params.get("exemption_years"),
                    "reference_year": params.get("reference_year"),
                }
            )
        )
    elif kind in ("TitleKeywordFilter", "AbstractKeywordFilter", "VenueKeywordFilter"):
        out.update(_clean({"match": params.get("match")}))
        if params.get("formula"):
            out["formula"] = params["formula"]
            out["keywords"] = params.get("keywords") or {}
        elif params.get("keyword"):
            out["keyword"] = params["keyword"]
        if params.get("case_sensitive") is not None:
            out["case_sensitive"] = bool(params["case_sensitive"])
    elif kind == "SimilarityFilter":
        out.update(_clean({"threshold": params.get("threshold"), "on_no_data": params.get("on_no_data")}))
        measures = params.get("measures") or []
        out["measures"] = [_translate_measure(m) for m in measures]
    elif kind == "LLMFilter":
        out.update(_clean({"scope": params.get("scope")}))
        if params.get("formula"):
            out["formula"] = params["formula"]
            out["queries"] = params.get("queries") or {}
        elif params.get("prompt"):
            out["prompt"] = params["prompt"]
        else:
            raise TranslationError("LLMFilter needs a prompt or a formula+queries")
        # per-filter model/effort overrides are intentionally NOT honored here:
        # the whole run is pinned to the single supported model (see run guard).
    return out


def _translate_screener(screener: Any) -> Any:
    if not screener:
        return None
    return _translate_filter(screener)


def _translate_step(node: dict[str, Any]) -> dict[str, Any]:
    """Translate one regular pipeline node into a single CiteClaw step dict."""
    kind = node.get("kind")
    cfg = node.get("config") or {}
    screener = _translate_screener(node.get("screener"))
    if kind == "seed":
        return {"step": "LoadSeeds"}
    if kind == "fwd":
        step: dict[str, Any] = {"step": "ExpandForward"}
        if cfg.get("maxChildren") is not None:
            step["max_citations"] = int(cfg["maxChildren"])
        if screener:
            step["screener"] = screener
        return step
    if kind == "bwd":
        step = {"step": "ExpandBackward"}
        if screener:
            step["screener"] = screener
        return step
    if kind == "rerank":
        step = {"step": "Rerank", "metric": cfg.get("metric", "citation")}
        if cfg.get("targetN") is not None:
            step["k"] = int(cfg["targetN"])
        # MMR lambda in the design ~ "want diversity"; map any positive
        # lambda to cluster-diverse reranking via walktrap.
        lam = cfg.get("lambda")
        if lam is not None and float(lam) > 0:
            step["diversity"] = {"type": "walktrap", "n_communities": 3}
        return step
    if kind == "rsc":
        step = {"step": "ReScreen"}
        if screener:
            step["screener"] = screener
        return step
    if kind == "sink":
        return {"step": "Finalize"}
    raise TranslationError(f"Unknown pipeline node kind: {kind!r}")


def translate_pipeline(model: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn the design's ``pipeline`` node list into CiteClaw ``pipeline:``.

    A node with ``kind == "parallel"`` carries ``branches`` (each branch is a
    single regular node in this version) and maps to CiteClaw's ``Parallel``
    step — the incoming signal is broadcast to every branch and the outputs
    are unioned. A degenerate single-branch parallel collapses to a plain
    serial step.
    """
    nodes = model.get("pipeline") or []
    steps: list[dict[str, Any]] = []
    has_finalize = False
    for node in nodes:
        if node.get("kind") == "parallel":
            branches: list[list[dict[str, Any]]] = []
            for b in node.get("branches") or []:
                branches.append([_translate_step(b)])
            if not branches:
                continue
            if len(branches) == 1:
                steps.append(branches[0][0])
            else:
                steps.append({"step": "Parallel", "branches": branches})
            continue
        step = _translate_step(node)
        steps.append(step)
        if step.get("step") == "Finalize":
            has_finalize = True
    if not has_finalize:
        steps.append({"step": "Finalize"})
    return steps


def _seed_entries(seeds: list[dict[str, Any]], max_seeds: int | None) -> tuple[list[dict[str, Any]], bool]:
    """Build ``seed_papers`` entries; return (entries, needs_resolve).

    Prefer a real Semantic Scholar / DOI id when present; otherwise fall
    back to title (resolved by the ``ResolveSeeds`` step).
    """
    entries: list[dict[str, Any]] = []
    needs_resolve = False
    for s in seeds:
        pid = (s.get("paper_id") or s.get("s2_id") or "").strip()
        title = (s.get("title") or "").strip()
        # demo ids like "s1".."s8" are not real — treat them as title-only
        is_real_id = bool(pid) and not (len(pid) <= 3 and pid[:1] == "s" and pid[1:].isdigit())
        if is_real_id:
            entries.append({"paper_id": pid})
        elif title:
            entries.append({"title": title})
            needs_resolve = True
        if max_seeds and len(entries) >= max_seeds:
            break
    return entries, needs_resolve


def build_config(model: dict[str, Any], *, data_dir: str, screening_model: str,
                 reasoning_effort: str) -> dict[str, Any]:
    """Assemble the full CiteClaw config dict from the design payload.

    ``model`` is the JSON the browser POSTs:
    ``{pipeline: [...], seeds: [...], limits: {...}, topic: "..."}``.
    """
    pipeline = translate_pipeline(model)

    # find the seed node to read its max-seeds cap + query→topic default
    seed_query = ""
    max_seeds = None
    for node in model.get("pipeline") or []:
        if node.get("kind") == "seed":
            cfg = node.get("config") or {}
            seed_query = (cfg.get("query") or "").strip()
            if cfg.get("maxSeeds") is not None:
                max_seeds = int(cfg["maxSeeds"])
            break

    seeds = model.get("seeds") or []
    seed_papers, needs_resolve = _seed_entries(seeds, max_seeds)
    if not seed_papers:
        raise TranslationError(
            "No seed papers selected. Star at least one paper in the Seeds panel."
        )

    if needs_resolve:
        # ResolveSeeds must run before LoadSeeds
        pipeline = [{"step": "ResolveSeeds"}] + pipeline

    limits = model.get("limits") or {}
    topic = (model.get("topic") or seed_query or "").strip()

    cfg: dict[str, Any] = {
        "screening_model": screening_model,
        "reasoning_effort": reasoning_effort,
        "data_dir": data_dir,
        "topic_description": topic,
        "seed_papers": seed_papers,
        "max_papers_total": int(limits.get("max_papers", 200)),
        "max_llm_tokens": int(limits.get("max_llm_tokens", 5_000_000)),
        "s2_requests_per_second": float(limits.get("s2_rps", 0.9)),
        "llm_batch_size": int(limits.get("llm_batch_size", 10)),
        "llm_concurrency": int(limits.get("llm_concurrency", 8)),
        "blocks": {},
        "pipeline": pipeline,
    }
    return cfg
