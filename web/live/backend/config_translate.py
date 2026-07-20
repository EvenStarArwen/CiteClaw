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

import re
from typing import Any

from .models_catalog import resolve_model

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


_EXPR_TOKEN = re.compile(
    r'"([^"]*)"'                      # quoted phrase
    r"|([A-Za-z0-9_À-￿][A-Za-z0-9_À-￿-]*)"  # bare word
    r"|([&|!()])"                     # operator
    r"|(\s+)"                         # whitespace
    r"|(.)"                           # anything else = error
)


def _compile_keyword_expression(expr: str) -> tuple[str, dict[str, str]]:
    """Compile a direct keyword expression into the CLI's formula shape.

    ``("large language model" | LLM) & "scientific discovery"`` becomes
    ``("k1 | k2) & k3"`` plus ``{"k1": "large language model", ...}`` —
    the named-formula form ``TitleKeywordFilter`` & friends execute. The
    UI never shows the generated names. Wildcards are rejected loudly:
    the keyword filters match plain substrings / whole words / prefixes.
    """
    parts: list[str] = []
    names: dict[str, str] = {}   # lowercased term -> generated name
    keywords: dict[str, str] = {}
    for m in _EXPR_TOKEN.finditer(expr):
        quoted, bare, op, ws, bad = m.groups()
        if ws is not None:
            continue
        if bad is not None:
            if bad == "*":
                raise TranslationError(
                    "Keyword expressions don't support * wildcards — terms match "
                    "as substrings (or whole words / prefixes via the match mode)."
                )
            raise TranslationError(
                f"Unexpected character {bad!r} in keyword expression — allowed: "
                'words, "quoted phrases", & | ! and parentheses.'
            )
        if op is not None:
            parts.append(op)
            continue
        term = quoted if quoted is not None else bare
        if not term.strip():
            raise TranslationError("Empty quotes in keyword expression — put a phrase between them.")
        name = names.get(term.lower())
        if name is None:
            name = f"k{len(names) + 1}"
            names[term.lower()] = name
            keywords[name] = term
        parts.append(name)
    formula = " ".join(parts)
    if not keywords:
        raise TranslationError("Keyword expression has no terms — enter at least one keyword.")
    try:
        from citeclaw.screening.formula import BooleanFormula
        BooleanFormula(formula)
    except TranslationError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface the parser's message
        raise TranslationError(f"Keyword expression is not valid: {exc}") from exc
    return formula, keywords


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
        emb = m.get("embedder", "s2")
        # Voyage embeddings aren't enabled in the WebUI yet — always run on s2.
        out["embedder"] = "s2" if emb == "voyage" else emb
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
            # UI shape: routes[{id, if: {kind, values|n}, pass_to: node}] +
            # else. CLI shape: routes[{if: {VenueIn: [...]}, pass_to: block},
            # ..., {default: block}].
            out_routes: list[dict[str, Any]] = []
            for r in node.get("routes") or []:
                pred = (r or {}).get("if") or {}
                pk = pred.get("kind")
                if pk == "VenueIn":
                    values = [str(v).strip() for v in (pred.get("values") or []) if str(v).strip()]
                    if not values:
                        raise TranslationError(
                            "A Route venue condition has no venues — add at least "
                            "one (click the Route to edit its conditions)."
                        )
                    cond: dict[str, Any] = {"VenueIn": values}
                elif pk in ("CitAtLeast", "YearAtLeast"):
                    cond = {pk: int(pred.get("n") or 0)}
                else:
                    raise TranslationError(f"Route condition has an unknown predicate: {pk!r}")
                if not r.get("pass_to"):
                    raise TranslationError(
                        "A Route condition has no branch filters — add a filter "
                        "under it in the filter tree."
                    )
                out_routes.append({"if": cond, "pass_to": _translate_filter(r["pass_to"])})
            if node.get("else"):
                out_routes.append({"default": _translate_filter(node["else"])})
            if not out_routes:
                raise TranslationError("Route has no conditions and no else branch.")
            return {"type": "Route", "routes": out_routes}
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
                    "curve": params.get("curve"),
                    "exp_base": params.get("exp_base"),
                }
            )
        )
    elif kind in ("TitleKeywordFilter", "AbstractKeywordFilter", "VenueKeywordFilter"):
        out.update(_clean({"match": params.get("match")}))
        expr = str(params.get("expression") or "").strip()
        if expr:
            # Direct expression, e.g. ("large language model" | LLM) & agent
            out["formula"], out["keywords"] = _compile_keyword_expression(expr)
        elif params.get("formula"):
            # Legacy named form (older saved pipelines)
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
        # Per-filter model/effort overrides — the CLI's per-block ``model:`` /
        # ``reasoning_effort:``, so different filters can screen with different
        # models. Aliases resolve to the GA id here; the run endpoint validates
        # support + key presence for every override before starting.
        m = str(params.get("model") or "").strip()
        if m:
            out["model"] = resolve_model(m)
        eff = str(params.get("effort") or params.get("reasoning_effort") or "").strip()
        if eff:
            out["reasoning_effort"] = eff
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
        # ``maxChildren`` is the pre-rename key; accept it for older payloads.
        cap = cfg.get("maxCitations", cfg.get("maxChildren"))
        if cap is not None:
            step["max_citations"] = int(cap)
        if screener:
            step["screener"] = screener
        return step
    if kind == "bwd":
        step = {"step": "ExpandBackward"}
        if screener:
            step["screener"] = screener
        return step
    if kind == "search":
        # ExpandBySearch — the CLI agent is a placeholder today; wired through
        # faithfully so it starts working the moment the CLI side does.
        step = {"step": "ExpandBySearch"}
        agent: dict[str, Any] = {}
        if cfg.get("maxIterations") is not None:
            agent["max_iterations"] = int(cfg["maxIterations"])
        if cfg.get("maxPerIteration") is not None:
            agent["max_papers_per_iteration"] = int(cfg["maxPerIteration"])
        if agent:
            step["agent"] = agent
        if screener:
            step["screener"] = screener
        return step
    if kind == "rerank":
        step = {"step": "Rerank", "metric": cfg.get("metric") or "citation"}
        if cfg.get("targetN") is not None:
            step["k"] = int(cfg["targetN"])
        div = cfg.get("diversity")
        # Legacy: a positive lambda used to mean "turn diversity on" (walktrap).
        if div is None and cfg.get("lambda") is not None and float(cfg["lambda"]) > 0:
            div = "walktrap"
        if div and div not in ("off", "none"):
            # Inline clusterer spec — the CLI runs the same registry the
            # standalone Cluster step uses (louvain / walktrap / topic_model).
            if div == "walktrap":
                step["diversity"] = {"type": "walktrap",
                                     "n_communities": int(cfg.get("divCommunities") or 3)}
            elif div == "topic_model":
                d: dict[str, Any] = {"type": "topic_model",
                                     "min_cluster_size": int(cfg.get("divMinCluster") or 5)}
                if int(cfg.get("divNeighbors") or 0) > 0:
                    d["n_neighbors"] = int(cfg["divNeighbors"])
                step["diversity"] = d
            else:
                step["diversity"] = {"type": div}
        return step
    if kind == "rsc":
        step = {"step": "ReScreen"}
        if screener:
            step["screener"] = screener
        return step
    if kind == "sink":
        return {"step": "Finalize"}
    raise TranslationError(f"Unknown pipeline node kind: {kind!r}")


def _translate_seq(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recursively translate one design sequence into a list of CiteClaw steps.

    A ``parallel`` element becomes CiteClaw's ``Parallel`` step whose branches
    are themselves translated sequences — so nested parallels and any steps that
    follow a merge all round-trip. Empty branches are dropped; a degenerate
    single-branch parallel collapses back to a plain serial run.
    """
    steps: list[dict[str, Any]] = []
    for node in seq:
        if node.get("kind") == "parallel":
            branches = [_translate_seq(b) for b in (node.get("branches") or [])]
            branches = [b for b in branches if b]
            if not branches:
                continue
            if len(branches) == 1:
                steps.extend(branches[0])
            else:
                steps.append({"step": "Parallel", "branches": branches})
            continue
        steps.append(_translate_step(node))
    return steps


def translate_pipeline(model: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn the design's ``pipeline`` sequence into CiteClaw ``pipeline:``.

    The design model is a series-parallel *sequence*: an ordered list whose
    elements are either regular step nodes or ``parallel`` nodes (each carrying
    its own ``branches`` sub-sequences). Branches are broadcast the incoming
    signal and their outputs unioned; whatever element follows the parallel in
    the parent sequence therefore operates on the merged set (the "merge" step).
    """
    steps = _translate_seq(model.get("pipeline") or [])
    if not any(s.get("step") == "Finalize" for s in steps):
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
