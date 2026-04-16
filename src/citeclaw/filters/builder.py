"""Block builder: turn raw block dicts into Filter instances.

Topological build: each name resolves on demand, recursively. Cycle protection
via an "in-progress" set raises a clear error.
"""

from __future__ import annotations

from typing import Any

from citeclaw.filters.atoms.citation import CitationFilter
from citeclaw.filters.atoms.keyword import AbstractKeywordFilter, TitleKeywordFilter
from citeclaw.filters.atoms.llm_query import LLMFilter
from citeclaw.filters.atoms.predicates import CitAtLeast, VenueIn, YearAtLeast
from citeclaw.filters.atoms.year import YearFilter
from citeclaw.filters.blocks.any_block import Any_
from citeclaw.filters.blocks.not_block import Not_
from citeclaw.filters.blocks.route import Route, RouteCase
from citeclaw.filters.blocks.sequential import Sequential
from citeclaw.filters.blocks.similarity import SimilarityFilter
from citeclaw.filters.measures import MEASURE_TYPES

ATOM_TYPES = {
    "YearFilter": YearFilter,
    "CitationFilter": CitationFilter,
    "LLMFilter": LLMFilter,
    "TitleKeywordFilter": TitleKeywordFilter,
    "AbstractKeywordFilter": AbstractKeywordFilter,
}

PREDICATE_KEYS = {
    "venue_in": VenueIn,
    "cit_at_least": CitAtLeast,
    "year_at_least": YearAtLeast,
}


def _build_predicate(d: dict) -> Any:
    if not isinstance(d, dict) or len(d) != 1:
        raise ValueError(f"Predicate must be one-key dict, got {d!r}")
    key, val = next(iter(d.items()))
    cls = PREDICATE_KEYS.get(key)
    if cls is None:
        raise ValueError(f"Unknown predicate {key!r}")
    if key == "venue_in":
        return cls(name=key, values=list(val))
    return cls(name=key, n=int(val))


def _build_measure(d: dict) -> Any:
    if not isinstance(d, dict) or "type" not in d:
        raise ValueError(f"Measure must be a dict with 'type', got {d!r}")
    cls = MEASURE_TYPES.get(d["type"])
    if cls is None:
        raise ValueError(f"Unknown measure type {d['type']!r}")
    kwargs = {k: v for k, v in d.items() if k != "type"}
    return cls(**kwargs)


def build_blocks(raw: dict[str, dict]) -> dict[str, Any]:
    built: dict[str, Any] = {}
    in_progress: set[str] = set()

    def resolve(ref: Any, name_hint: str = "anon") -> Any:
        if isinstance(ref, str):
            if ref in built:
                return built[ref]
            if ref not in raw:
                raise KeyError(f"Block reference {ref!r} not defined")
            if ref in in_progress:
                raise ValueError(f"Cyclic block reference: {ref}")
            in_progress.add(ref)
            built[ref] = _build_one(raw[ref], ref)
            in_progress.discard(ref)
            return built[ref]
        if isinstance(ref, dict):
            return _build_one(ref, name_hint)
        raise ValueError(f"Bad block ref: {ref!r}")

    def _build_one(d: dict, name: str) -> Any:
        t = d.get("type")
        if t is None:
            raise ValueError(f"Block {name!r} missing 'type'")
        if t == "Sequential":
            layers = [resolve(x, f"{name}.layer{i}") for i, x in enumerate(d.get("layers", []))]
            return Sequential(name=name, layers=layers)
        if t == "Any":
            layers = [resolve(x, f"{name}.layer{i}") for i, x in enumerate(d.get("layers", []))]
            return Any_(name=name, layers=layers)
        if t == "Not":
            if "layer" not in d:
                raise ValueError(f"Not block {name!r} requires 'layer:' (singular)")
            layer = resolve(d["layer"], f"{name}.inner")
            return Not_(name=name, layer=layer)
        if t == "Route":
            cases: list[RouteCase] = []
            for i, c in enumerate(d.get("routes", [])):
                if "default" in c:
                    target = resolve(c["default"], f"{name}.default")
                    cases.append(RouteCase(predicate=None, target=target, is_default=True))
                else:
                    pred = _build_predicate(c["if"])
                    target = resolve(c["pass_to"], f"{name}.case{i}")
                    cases.append(RouteCase(predicate=pred, target=target))
            return Route(name=name, cases=cases)
        if t == "SimilarityFilter":
            measures = [_build_measure(m) for m in d.get("measures", [])]
            kwargs = {k: v for k, v in d.items() if k not in ("type", "measures")}
            return SimilarityFilter(name=name, measures=measures, **kwargs)
        cls = ATOM_TYPES.get(t)
        if cls is None:
            raise ValueError(f"Unknown block type {t!r} in {name!r}")
        kwargs = {k: v for k, v in d.items() if k != "type"}
        return cls(name=name, **kwargs)

    for name in raw:
        if name not in built:
            built[name] = _build_one(raw[name], name)
    return built
