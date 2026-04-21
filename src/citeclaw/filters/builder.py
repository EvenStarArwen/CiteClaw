"""Block builder: turn raw block dicts into Filter instances.

The user describes filters in YAML as a flat ``{name: {type: ..., ...}}``
mapping; :func:`build_blocks` walks that mapping and produces a parallel
``{name: Filter}`` dict that the pipeline (and other blocks) can
reference by name.

Resolution is *lazy + topological*. A compositor block (Sequential, Any,
Not, Route) may name another block by string in its ``layers:`` /
``layer:`` / ``pass_to:`` field; the builder resolves each reference on
first use, caches the result, and detects cycles via an in-progress
set. Inline anonymous blocks (a dict where a name was expected) are
also legal — they get a synthesised name like ``"parent.layer0"`` for
debugging.

Two registries plug new filter types into the builder:

* :data:`ATOM_TYPES` — leaf filter classes (`YearFilter`, `CitationFilter`,
  `LLMFilter`, the keyword filters).
* :data:`PREDICATE_KEYS` — Route-predicate classes (`VenueIn`,
  `VenuePreset`, `CitAtLeast`, `YearAtLeast`).

Compositor blocks (Sequential / Any / Not / Route / SimilarityFilter)
are dispatched directly in :func:`_build_one` because each has a
distinct schema.
"""

from __future__ import annotations

from typing import Any

from citeclaw.filters.atoms.citation import CitationFilter
from citeclaw.filters.atoms.keyword import (
    AbstractKeywordFilter,
    TitleKeywordFilter,
    VenueKeywordFilter,
)
from citeclaw.filters.atoms.llm_query import LLMFilter
from citeclaw.filters.atoms.predicates import CitAtLeast, VenueIn, VenuePreset, YearAtLeast
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
    "VenueKeywordFilter": VenueKeywordFilter,
}

PREDICATE_KEYS = {
    "VenueIn": VenueIn,
    "VenuePreset": VenuePreset,
    "CitAtLeast": CitAtLeast,
    "YearAtLeast": YearAtLeast,
}

# Compositor blocks that take an identical ``layers: [...]`` schema.
# Keyed by YAML ``type:`` discriminator.
_LAYERED_BLOCKS = {
    "Sequential": Sequential,
    "Any": Any_,
}


def _build_predicate(d: dict) -> Any:
    """Build one Route predicate from a single-key YAML dict.

    The predicate dict must contain exactly one entry whose key is in
    :data:`PREDICATE_KEYS`. Value shape varies per predicate:
    ``VenueIn`` and ``VenuePreset`` take a list; ``CitAtLeast`` /
    ``YearAtLeast`` take an int.
    """
    if not isinstance(d, dict) or len(d) != 1:
        raise ValueError(f"Predicate must be one-key dict, got {d!r}")
    key, val = next(iter(d.items()))
    cls = PREDICATE_KEYS.get(key)
    if cls is None:
        raise ValueError(f"Unknown predicate {key!r}")
    if key == "VenueIn":
        return cls(name=key, values=list(val))
    if key == "VenuePreset":
        return cls(name=key, presets=list(val))
    return cls(name=key, n=int(val))


def _build_measure(d: dict) -> Any:
    """Build one SimilarityMeasure from a ``{type: ..., ...}`` dict."""
    if not isinstance(d, dict) or "type" not in d:
        raise ValueError(f"Measure must be a dict with 'type', got {d!r}")
    cls = MEASURE_TYPES.get(d["type"])
    if cls is None:
        raise ValueError(f"Unknown measure type {d['type']!r}")
    kwargs = {k: v for k, v in d.items() if k != "type"}
    return cls(**kwargs)


def build_blocks(raw: dict[str, dict]) -> dict[str, Any]:
    """Build the ``{name: Filter}`` dict from raw YAML block definitions.

    See module docstring for the lazy / topological resolution strategy.
    Cycles raise :class:`ValueError`; references to undefined block
    names raise :class:`KeyError`; unknown block types raise
    :class:`ValueError`.
    """
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
        if t in _LAYERED_BLOCKS:
            cls = _LAYERED_BLOCKS[t]
            layers = [
                resolve(x, f"{name}.layer{i}")
                for i, x in enumerate(d.get("layers", []) or [])
            ]
            return cls(name=name, layers=layers)
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
