"""Closed operator set the worker picks from on every refinement iteration.

Seven transformation types. The worker's output schema names the op +
facet + term; :func:`apply_transformation` mutates the QueryPlan. This
replaces the old free-form ``WORKER_WRITE_NEXT`` — the worker no
longer retypes the whole query, so structure survives across iters.

The ops split into two families:

- **Cast-widening** — ``add_or_alternative``, ``loosen_term``,
  ``swap_operator`` (to OR), ``add_facet``.
- **Cast-narrowing** — ``remove_or_alternative``, ``tighten_term``,
  ``swap_operator`` (to AND), ``add_exclusion`` (last resort).

Typical iter applies up to ``_MAX_TRANSFORMATIONS_PER_ITER`` ops — one
to fill coverage gaps, one to kill noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from citeclaw.agents.v3.state import (
    MutableFacet,
    QueryPlan,
    TermSpec,
)


log = logging.getLogger("citeclaw.agents.v3.transformations")

_MAX_TRANSFORMATIONS_PER_ITER = 2
_VALID_STRICTNESS_TIGHT = {"proximity", "phrase"}
_VALID_STRICTNESS_LOOSE = {"and_words", "proximity"}

SUPPORTED_OPS = {
    "add_or_alternative",
    "remove_or_alternative",
    "tighten_term",
    "loosen_term",
    "swap_operator",
    "add_facet",
    "add_exclusion",
}


@dataclass
class TransformResult:
    applied: list[dict[str, Any]]
    rejected: list[dict[str, Any]]  # [{op, reason}]


# ---------------------------------------------------------------------------
# Individual op handlers
# ---------------------------------------------------------------------------


def _find_facet(plan: QueryPlan, facet_id: str) -> MutableFacet | None:
    for f in plan.facets:
        if f.id == facet_id:
            return f
    return None


def _find_term(facet: MutableFacet, raw: str) -> TermSpec | None:
    raw_l = raw.strip().lower()
    for t in facet.terms:
        if t.raw.strip().lower() == raw_l:
            return t
    return None


def _op_add_or_alternative(plan: QueryPlan, args: dict) -> str | None:
    fid = str(args.get("facet_id") or "").strip()
    terms = args.get("terms") or []
    facet = _find_facet(plan, fid)
    if facet is None:
        return f"facet_id {fid!r} not in plan"
    if not isinstance(terms, list) or not terms:
        return "'terms' must be a non-empty list"
    for entry in terms:
        if isinstance(entry, str):
            raw, strictness, slop = entry.strip(), "phrase", 0
        elif isinstance(entry, dict):
            raw = str(entry.get("raw") or entry.get("term") or "").strip()
            strictness = str(entry.get("strictness") or "phrase").strip().lower()
            if strictness == "exact_phrase":
                strictness = "phrase"
            if strictness == "and":
                strictness = "and_words"
            if strictness not in ("and_words", "proximity", "phrase"):
                strictness = "phrase"
            slop = int(entry.get("slop") or 0)
            if strictness != "proximity":
                slop = 0
            else:
                slop = max(slop, 1)
        else:
            continue
        if not raw:
            continue
        if _find_term(facet, raw) is not None:
            continue
        facet.terms.append(TermSpec(raw=raw, strictness=strictness, slop=slop))
    return None


def _op_remove_or_alternative(plan: QueryPlan, args: dict) -> str | None:
    fid = str(args.get("facet_id") or "").strip()
    raw = str(args.get("term") or "").strip()
    facet = _find_facet(plan, fid)
    if facet is None:
        return f"facet_id {fid!r} not in plan"
    if not raw:
        return "'term' is empty"
    target = _find_term(facet, raw)
    if target is None:
        return f"term {raw!r} not in facet {fid!r}"
    if len(facet.terms) <= 1:
        return f"facet {fid!r} has only one term — removing it would empty the facet"
    facet.terms = [t for t in facet.terms if t is not target]
    return None


def _op_tighten_term(plan: QueryPlan, args: dict) -> str | None:
    raw = str(args.get("term") or "").strip()
    to = str(args.get("to") or "").strip().lower()
    if to.startswith("proximity_"):
        try:
            slop = int(to.split("_", 1)[1])
        except (IndexError, ValueError):
            return f"malformed 'to' {to!r}"
        new_strictness, new_slop = "proximity", max(slop, 1)
    elif to in ("proximity", "phrase", "exact_phrase"):
        new_strictness = "phrase" if to != "proximity" else "proximity"
        new_slop = int(args.get("slop") or 3) if new_strictness == "proximity" else 0
    else:
        return f"'to' must be proximity_N or exact_phrase (got {to!r})"
    if new_strictness not in _VALID_STRICTNESS_TIGHT:
        return f"{new_strictness!r} is not a tightening target"
    hit = _mutate_term_everywhere(plan, raw, new_strictness, new_slop)
    if not hit:
        return f"term {raw!r} not found in any facet or exclusion"
    return None


def _op_loosen_term(plan: QueryPlan, args: dict) -> str | None:
    raw = str(args.get("term") or "").strip()
    to = str(args.get("to") or "").strip().lower()
    if to.startswith("proximity_"):
        try:
            slop = int(to.split("_", 1)[1])
        except (IndexError, ValueError):
            return f"malformed 'to' {to!r}"
        new_strictness, new_slop = "proximity", max(slop, 1)
    elif to in ("and_words", "and", "proximity"):
        if to == "and":
            to = "and_words"
        new_strictness = to
        new_slop = int(args.get("slop") or 3) if new_strictness == "proximity" else 0
    else:
        return f"'to' must be and_words or proximity_N (got {to!r})"
    if new_strictness not in _VALID_STRICTNESS_LOOSE:
        return f"{new_strictness!r} is not a loosening target"
    hit = _mutate_term_everywhere(plan, raw, new_strictness, new_slop)
    if not hit:
        return f"term {raw!r} not found in any facet or exclusion"
    return None


def _mutate_term_everywhere(
    plan: QueryPlan,
    raw: str,
    new_strictness: str,
    new_slop: int,
) -> bool:
    changed = False
    for f in plan.facets:
        t = _find_term(f, raw)
        if t is not None:
            t.strictness = new_strictness  # type: ignore[assignment]
            t.slop = new_slop
            changed = True
    return changed


def _op_swap_operator(plan: QueryPlan, args: dict) -> str | None:
    """Merge two facets (to OR) or split an OR group (to AND).

    Rare — when worker picks this the skeleton was almost certainly
    wrong. We merge two facets by id into a single facet that
    OR-unions their terms. Splitting is the inverse: a facet_id plus
    a ``split_on`` list moves those terms into a new facet. Swapping
    is logged so supervisor can see the worker flagged the skeleton.
    """
    to = str(args.get("to") or "").strip().upper()
    if to not in ("AND", "OR"):
        return f"'to' must be AND or OR (got {to!r})"
    if to == "OR":
        a_id = str(args.get("facet_id") or args.get("a") or "").strip()
        b_id = str(args.get("b") or args.get("merge_with") or "").strip()
        if not a_id or not b_id or a_id == b_id:
            return "merging to OR requires distinct facet_id + b"
        a = _find_facet(plan, a_id)
        b = _find_facet(plan, b_id)
        if a is None or b is None:
            return f"facet ids {a_id!r} / {b_id!r} not both in plan"
        a.concept = f"{a.concept} | {b.concept}"
        seen = {t.raw.strip().lower() for t in a.terms}
        for t in b.terms:
            if t.raw.strip().lower() not in seen:
                a.terms.append(t)
                seen.add(t.raw.strip().lower())
        plan.facets = [f for f in plan.facets if f.id != b_id]
        return None
    # to == "AND": split a facet in two
    fid = str(args.get("facet_id") or "").strip()
    split_on = args.get("split_on") or []
    facet = _find_facet(plan, fid)
    if facet is None:
        return f"facet_id {fid!r} not in plan"
    if not isinstance(split_on, list) or not split_on:
        return "splitting to AND requires 'split_on' list of term raws"
    split_keys = {str(s).strip().lower() for s in split_on if str(s).strip()}
    carved = [t for t in facet.terms if t.raw.strip().lower() in split_keys]
    kept = [t for t in facet.terms if t.raw.strip().lower() not in split_keys]
    if not carved or not kept:
        return "split must leave both facets non-empty"
    facet.terms = kept
    new_id = str(args.get("new_id") or (fid + "_b"))
    new_concept = str(args.get("new_concept") or facet.concept)
    plan.facets.append(MutableFacet(id=new_id, concept=new_concept, terms=carved))
    return None


def _op_add_facet(plan: QueryPlan, args: dict) -> str | None:
    fid = str(args.get("facet_id") or args.get("id") or "").strip()
    concept = str(args.get("concept") or fid).strip()
    terms_raw = args.get("terms") or []
    if not fid:
        return "'facet_id' is required"
    if any(f.id == fid for f in plan.facets):
        return f"facet_id {fid!r} already in plan"
    terms: list[TermSpec] = []
    for entry in terms_raw:
        if isinstance(entry, str):
            raw = entry.strip()
            if raw:
                terms.append(TermSpec(raw=raw, strictness="phrase"))
        elif isinstance(entry, dict):
            raw = str(entry.get("raw") or entry.get("term") or "").strip()
            if not raw:
                continue
            strictness = str(entry.get("strictness") or "phrase").strip().lower()
            if strictness == "exact_phrase":
                strictness = "phrase"
            if strictness not in ("and_words", "proximity", "phrase"):
                strictness = "phrase"
            slop = int(entry.get("slop") or 0)
            if strictness != "proximity":
                slop = 0
            terms.append(TermSpec(raw=raw, strictness=strictness, slop=slop))
    if not terms:
        return "new facet needs at least one term"
    plan.facets.append(MutableFacet(id=fid, concept=concept, terms=terms))
    return None


def _op_add_exclusion(plan: QueryPlan, args: dict) -> str | None:
    raw = str(args.get("term") or "").strip()
    if not raw:
        return "'term' is empty"
    for t in plan.exclusions:
        if t.raw.strip().lower() == raw.lower():
            return None  # idempotent
    strictness = str(args.get("strictness") or "phrase").strip().lower()
    if strictness == "exact_phrase":
        strictness = "phrase"
    if strictness not in ("and_words", "proximity", "phrase"):
        strictness = "phrase"
    slop = int(args.get("slop") or 0) if strictness == "proximity" else 0
    plan.exclusions.append(TermSpec(raw=raw, strictness=strictness, slop=slop))
    return None


_OP_DISPATCH = {
    "add_or_alternative": _op_add_or_alternative,
    "remove_or_alternative": _op_remove_or_alternative,
    "tighten_term": _op_tighten_term,
    "loosen_term": _op_loosen_term,
    "swap_operator": _op_swap_operator,
    "add_facet": _op_add_facet,
    "add_exclusion": _op_add_exclusion,
}


# ---------------------------------------------------------------------------
# Apply a list of transformations
# ---------------------------------------------------------------------------


def apply_transformations(
    plan: QueryPlan,
    transformations: list[dict[str, Any]],
    *,
    cap: int = _MAX_TRANSFORMATIONS_PER_ITER,
) -> TransformResult:
    """Mutate ``plan`` in place by applying up to ``cap`` transformations.

    Returns a :class:`TransformResult` listing which ops applied and
    which were rejected (with a reason). Rejected ops do NOT stop the
    loop — we apply as many as we can and let the caller surface the
    error block to the worker for the next iter.
    """
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for raw in transformations[:cap]:
        if not isinstance(raw, dict):
            rejected.append({"op": str(raw), "reason": "not an object"})
            continue
        op = str(raw.get("type") or raw.get("op") or "").strip()
        if op not in SUPPORTED_OPS:
            rejected.append({"op": op, "reason": f"unknown op {op!r}"})
            continue
        handler = _OP_DISPATCH[op]
        err = handler(plan, raw)
        if err is None:
            applied.append(raw)
        else:
            rejected.append({"op": op, "reason": err})
    if len(transformations) > cap:
        rejected.append({
            "op": "(extra)",
            "reason": f"received {len(transformations)} ops but cap is {cap}",
        })
    return TransformResult(applied=applied, rejected=rejected)


def render_transformations(transformations: list[dict[str, Any]]) -> str:
    if not transformations:
        return "  (no transformations applied)"
    lines: list[str] = []
    for t in transformations:
        op = t.get("type") or t.get("op") or "?"
        rest = {k: v for k, v in t.items() if k not in ("type", "op")}
        lines.append(f"  - {op}: {rest}")
    return "\n".join(lines)
