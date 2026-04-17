"""Render a built screener block as a human-readable calibration context.

Supervisor + workers see this as static context at the top of their
first user message. Knowing what the downstream screener will accept
or reject lets agents make smarter trade-offs: if the downstream
filter already rejects surveys, the worker doesn't need a ``-survey``
exclusion in its query; if the screener demands a ``YearFilter >=
2018``, there's no point broadening the search with earlier papers.

The walker traverses a *built* Filter tree (composed of Sequential /
Any_ / Not_ / Route / SimilarityFilter / LLMFilter / YearFilter /
CitationFilter / *KeywordFilter objects produced by
``filters.builder``), NOT the YAML dict. This is what
``ctx.config.blocks_built`` returns, and what ``ExpandBySearch``
already receives via its ``screener`` parameter.

The output format is one bulleted line per terminal filter, with
the composition operator clearly marked (`AND:` for Sequential,
`OR:` for Any_, `NOT:` for Not_, `IF/ELSE:` for Route). Example::

    Downstream filters configured in this run:
    AND:
      - YearFilter: year in [2018, 2026]
      - CitationFilter: beta=30, exemption_years=1
      - AbstractKeywordFilter required: "evolutionary" | "genetic algorithm" | "fitness"
      - LLMFilter (title+abstract): "the paper proposes a novel method"

    These are fixed and appropriate — do not try to second-guess them.
"""

from __future__ import annotations

from typing import Any

_CALIBRATION_HEADER = "Downstream filters configured in this run:"
_CALIBRATION_FOOTER = (
    "These filters run AFTER this step. Use them as calibration for how "
    "strict your query needs to be: if a downstream filter will already "
    "remove off-topic / low-citation / wrong-year papers, your queries "
    "can stay broad on those dimensions. Do NOT try to second-guess or "
    "duplicate the filter logic in your queries."
)


def render_filter_summary(screener: Any | None) -> str:
    """Return a human-readable summary of the built screener block.

    Pass the same ``screener`` object the ExpandBySearch step receives
    (after ``config.blocks_built`` expansion). ``None`` returns a
    one-line "no downstream screener configured" note so callers can
    always print something.
    """
    if screener is None:
        return f"{_CALIBRATION_HEADER}\n  (no downstream screener configured)\n"
    body = _render_block(screener, indent=0)
    return f"{_CALIBRATION_HEADER}\n{body}\n\n{_CALIBRATION_FOOTER}"


# ---------------------------------------------------------------------------
# Dispatch by type — isinstance checks on the concrete block/atom classes
# ---------------------------------------------------------------------------


def _render_block(block: Any, *, indent: int) -> str:
    """Render one block or atom as indented text."""
    from citeclaw.filters.atoms.citation import CitationFilter
    from citeclaw.filters.atoms.keyword import (
        AbstractKeywordFilter,
        TitleKeywordFilter,
        VenueKeywordFilter,
    )
    from citeclaw.filters.atoms.llm_query import LLMFilter
    from citeclaw.filters.atoms.year import YearFilter
    from citeclaw.filters.blocks.any_block import Any_
    from citeclaw.filters.blocks.not_block import Not_
    from citeclaw.filters.blocks.route import Route
    from citeclaw.filters.blocks.sequential import Sequential
    from citeclaw.filters.blocks.similarity import SimilarityFilter

    pad = "  " * indent
    bullet = f"{pad}- "

    if isinstance(block, Sequential):
        layers = getattr(block, "layers", []) or []
        if not layers:
            return f"{pad}AND: (empty)"
        parts = [f"{pad}AND:"]
        for layer in layers:
            parts.append(_render_block(layer, indent=indent + 1))
        return "\n".join(parts)

    if isinstance(block, Any_):
        layers = getattr(block, "layers", []) or []
        if not layers:
            return f"{pad}OR: (empty)"
        parts = [f"{pad}OR:"]
        for layer in layers:
            parts.append(_render_block(layer, indent=indent + 1))
        return "\n".join(parts)

    if isinstance(block, Not_):
        inner = getattr(block, "layer", None)
        parts = [f"{pad}NOT:"]
        if inner is not None:
            parts.append(_render_block(inner, indent=indent + 1))
        else:
            parts.append(f"{pad}  (empty)")
        return "\n".join(parts)

    if isinstance(block, Route):
        return _render_route(block, indent=indent)

    if isinstance(block, YearFilter):
        lo = getattr(block, "_min", None)
        hi = getattr(block, "_max", None)
        if lo is None and hi is None:
            return f"{bullet}YearFilter: (no bounds)"
        if lo is not None and hi is not None:
            return f"{bullet}YearFilter: year in [{lo}, {hi}]"
        if lo is not None:
            return f"{bullet}YearFilter: year >= {lo}"
        return f"{bullet}YearFilter: year <= {hi}"

    if isinstance(block, CitationFilter):
        beta = getattr(block, "_beta", None)
        exemption = getattr(block, "_exemption_years", None)
        ref_year = getattr(block, "_reference_year", None)
        bits = []
        if beta is not None:
            bits.append(f"beta={beta}")
        if exemption is not None:
            bits.append(f"exemption_years={exemption}")
        if ref_year is not None:
            bits.append(f"reference_year={ref_year}")
        return f"{bullet}CitationFilter: " + (", ".join(bits) or "(defaults)")

    if isinstance(block, LLMFilter):
        scope = getattr(block, "scope", "title")
        prompt = getattr(block, "prompt", "")
        formula = getattr(block, "formula", None)
        queries = getattr(block, "queries", None) or {}
        if formula:
            qs = "; ".join(f"{k}: {v!r}" for k, v in queries.items())
            return (
                f"{bullet}LLMFilter ({scope}) formula {formula!r} over: {qs}"
            )
        return f"{bullet}LLMFilter ({scope}): {prompt!r}"

    if isinstance(block, SimilarityFilter):
        thresh = getattr(block, "threshold", None)
        measures = getattr(block, "measures", []) or []
        mtypes = [type(m).__name__ for m in measures]
        on_nd = getattr(block, "on_no_data", None)
        return (
            f"{bullet}SimilarityFilter: threshold={thresh}, "
            f"measures=[{', '.join(mtypes)}], on_no_data={on_nd!r}"
        )

    if isinstance(block, (TitleKeywordFilter, AbstractKeywordFilter, VenueKeywordFilter)):
        return _render_keyword_filter(block, bullet=bullet)

    # Unknown atom — fall back to type name + repr for operator visibility.
    tn = type(block).__name__
    name = getattr(block, "name", "")
    return f"{bullet}{tn}{f' ({name!r})' if name else ''}"


def _render_route(block: Any, *, indent: int) -> str:
    """Render a Route's if/elif/else chain."""
    pad = "  " * indent
    routes = getattr(block, "routes", []) or []
    parts = [f"{pad}IF/ELIF/ELSE:"]
    for i, r in enumerate(routes):
        # Each Route.routes entry is a dict with either ``if`` / ``pass_to``
        # keys or a lone ``default`` key. The builder normalises to keyword
        # names, so we read those directly.
        default_target = r.get("default") if isinstance(r, dict) else None
        pred = r.get("if") if isinstance(r, dict) else None
        target = r.get("pass_to") if isinstance(r, dict) else None
        if default_target is not None:
            parts.append(f"{pad}  ELSE ->")
            parts.append(_render_block(default_target, indent=indent + 2))
            continue
        label = "IF" if i == 0 else "ELIF"
        pred_str = _render_predicate(pred)
        parts.append(f"{pad}  {label} {pred_str} ->")
        if target is not None:
            parts.append(_render_block(target, indent=indent + 2))
    return "\n".join(parts)


def _render_predicate(pred: Any) -> str:
    """Render a single Route.if predicate dict as compact text."""
    if not isinstance(pred, dict):
        return "(empty predicate)"
    items = []
    for key, value in pred.items():
        if isinstance(value, (list, tuple)):
            items.append(f"{key}={list(value)!r}")
        else:
            items.append(f"{key}={value!r}")
    return ", ".join(items)


def _render_keyword_filter(block: Any, *, bullet: str) -> str:
    """Render TitleKeywordFilter / AbstractKeywordFilter / VenueKeywordFilter."""
    tn = type(block).__name__
    match_mode = getattr(block, "match", getattr(block, "_match", "substring"))
    case_sensitive = getattr(block, "case_sensitive", getattr(block, "_case_sensitive", False))
    simple_kw = getattr(block, "keyword", getattr(block, "_keyword", ""))
    formula = getattr(block, "formula", getattr(block, "_formula", None))
    keywords = getattr(block, "keywords", getattr(block, "_keywords", None))
    suffix = f" (match={match_mode}, case_sensitive={case_sensitive})"
    if formula and keywords:
        kw_strs = "; ".join(f"{k}={v!r}" for k, v in keywords.items())
        return f"{bullet}{tn} formula {formula!r} over: {kw_strs}{suffix}"
    if simple_kw:
        return f"{bullet}{tn}: {simple_kw!r}{suffix}"
    return f"{bullet}{tn}: (no keyword configured){suffix}"


__all__ = ["render_filter_summary"]
