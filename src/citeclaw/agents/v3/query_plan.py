"""Structured query plan for V3 worker.

The flat query string the worker used to write by hand now lives as a
:class:`~citeclaw.agents.v3.state.QueryPlan` — a mutable tree of
facets, each with an OR group of :class:`TermSpec`. Transformations
(`add_or_alternative`, `tighten_term`, `swap_operator`, …) mutate
this tree, and we render it to natural-language / Lucene at the edge.

Why a structured plan: free-form rewrites across iterations threw
away state from prior rounds (gpt-5.4-nano's 49-paper degenerate
trajectory re-wrote the query every iter and kept shuffling the
same subset). A closed operator set keeps every iter's edit visible
and composable.
"""

from __future__ import annotations

import re

from citeclaw.agents.v3.state import (
    FacetSkeleton,
    MutableFacet,
    QueryPlan,
    TermSpec,
)


# ---------------------------------------------------------------------------
# Term rendering
# ---------------------------------------------------------------------------


def _looks_multiword(raw: str) -> bool:
    return bool(re.search(r"\s", raw.strip()))


def render_term_natural(t: TermSpec) -> str:
    """One OR-alternative in AND/OR/NOT notation."""
    raw = t.raw.strip()
    if not raw:
        return ""
    if not _looks_multiword(raw):
        return raw
    if t.strictness == "and_words":
        parts = raw.split()
        return "(" + " AND ".join(parts) + ")"
    if t.strictness == "proximity":
        slop = max(t.slop, 1)
        return f'"{raw}"~{slop}'
    return f'"{raw}"'


def render_term_lucene(t: TermSpec) -> str:
    """One OR-alternative in Lucene notation (the `+` is attached by
    the facet renderer, not here)."""
    raw = t.raw.strip()
    if not raw:
        return ""
    if not _looks_multiword(raw):
        return raw
    if t.strictness == "and_words":
        # Inside an OR group, AND'd words need to be grouped as a
        # nested mandatory sub-query.
        parts = raw.split()
        return "(" + " ".join(f"+{p}" for p in parts) + ")"
    if t.strictness == "proximity":
        slop = max(t.slop, 1)
        return f'"{raw}"~{slop}'
    return f'"{raw}"'


# ---------------------------------------------------------------------------
# Plan rendering
# ---------------------------------------------------------------------------


def _dedupe_preserve_order(terms: list[TermSpec]) -> list[TermSpec]:
    seen: set[str] = set()
    out: list[TermSpec] = []
    for t in terms:
        key = t.raw.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def render_plan_natural(plan: QueryPlan) -> str:
    groups: list[str] = []
    for facet in plan.facets:
        terms = _dedupe_preserve_order(facet.terms)
        if not terms:
            continue
        rendered = [render_term_natural(t) for t in terms]
        rendered = [r for r in rendered if r]
        if not rendered:
            continue
        if len(rendered) == 1:
            groups.append(rendered[0])
        else:
            groups.append("(" + " OR ".join(rendered) + ")")
    if not groups:
        return ""
    body = " AND ".join(groups)
    exclusions = _dedupe_preserve_order(plan.exclusions)
    for t in exclusions:
        rendered = render_term_natural(t)
        if rendered:
            body += f" AND NOT {rendered}"
    return body


def render_plan_lucene(plan: QueryPlan) -> str:
    groups: list[str] = []
    for facet in plan.facets:
        terms = _dedupe_preserve_order(facet.terms)
        if not terms:
            continue
        rendered = [render_term_lucene(t) for t in terms]
        rendered = [r for r in rendered if r]
        if not rendered:
            continue
        if len(rendered) == 1:
            groups.append(f"+{rendered[0]}")
        else:
            groups.append("+(" + " | ".join(rendered) + ")")
    body = " ".join(groups)
    for t in _dedupe_preserve_order(plan.exclusions):
        rendered = render_term_lucene(t)
        if rendered:
            body += f" -{rendered}"
    return body.strip()


# ---------------------------------------------------------------------------
# Build plan from propose_first JSON + skeleton seeding
# ---------------------------------------------------------------------------


_VALID_STRICTNESS = {"and_words", "proximity", "phrase"}


def _normalise_strictness(raw: str | None, slop: int) -> tuple[str, int]:
    s = (raw or "").strip().lower()
    if s == "exact_phrase":
        s = "phrase"
    if s == "and":
        s = "and_words"
    if s not in _VALID_STRICTNESS:
        s = "phrase"
    if s != "proximity":
        slop = 0
    else:
        slop = max(int(slop or 0), 1)
    return s, slop


def _parse_term_entry(entry: object) -> TermSpec | None:
    if isinstance(entry, str):
        raw = entry.strip()
        if not raw:
            return None
        return TermSpec(raw=raw, strictness="phrase")
    if not isinstance(entry, dict):
        return None
    raw = str(entry.get("raw") or entry.get("term") or "").strip()
    if not raw:
        return None
    strictness, slop = _normalise_strictness(
        entry.get("strictness") or entry.get("mode"),
        int(entry.get("slop") or entry.get("proximity") or 0),
    )
    return TermSpec(raw=raw, strictness=strictness, slop=slop)


def plan_from_propose_first(
    response: dict,
    *,
    skeleton: FacetSkeleton | None,
) -> QueryPlan:
    """Parse a ``WORKER_PROPOSE_FIRST`` JSON reply into a QueryPlan.

    Expected shape::

        {
          "facets": [
            {"id": "technology",
             "terms": ["prime editing", {"raw": "PE", "strictness": "phrase"}, ...]},
            ...
          ],
          "exclusions": ["review"]           # optional, rare
        }

    Missing ``id`` entries are matched positionally against the skeleton
    when one is available; missing ``concept`` falls back to the
    skeleton's concept. Facets not present in the response but in the
    skeleton are kept empty so downstream transformations can see the
    missing dimension.
    """
    plan = QueryPlan()
    raw_facets = response.get("facets") if isinstance(response, dict) else None
    raw_facets = raw_facets if isinstance(raw_facets, list) else []
    skeleton_facets = list(skeleton.facets) if skeleton else []
    by_id: dict[str, MutableFacet] = {}

    for i, item in enumerate(raw_facets):
        if not isinstance(item, dict):
            continue
        fid = str(item.get("id") or "").strip()
        concept = str(item.get("concept") or "").strip()
        if not fid and i < len(skeleton_facets):
            fid = skeleton_facets[i].id
            concept = concept or skeleton_facets[i].concept
        if not fid:
            fid = f"facet_{i+1}"
        concept = concept or fid
        terms_raw = item.get("terms") or []
        terms_raw = terms_raw if isinstance(terms_raw, list) else []
        terms: list[TermSpec] = []
        for entry in terms_raw:
            spec = _parse_term_entry(entry)
            if spec:
                terms.append(spec)
        facet = MutableFacet(id=fid, concept=concept, terms=terms)
        by_id[fid] = facet
        plan.facets.append(facet)

    # Re-align to skeleton order: any skeleton facet missing from the
    # response gets an empty placeholder at the end so the worker can
    # add_or_alternative to it later.
    for f in skeleton_facets:
        if f.id not in by_id:
            plan.facets.append(MutableFacet(id=f.id, concept=f.concept, terms=[]))

    raw_excl = response.get("exclusions") if isinstance(response, dict) else None
    raw_excl = raw_excl if isinstance(raw_excl, list) else []
    for entry in raw_excl:
        spec = _parse_term_entry(entry)
        if spec:
            plan.exclusions.append(spec)

    return plan


def render_plan_tree(plan: QueryPlan) -> str:
    """Diagnostic-block rendering that surfaces every facet + term
    with its strictness tag. Used in worker prompts when we want the
    model to see the parse-tree view, not just the flat query."""
    if not plan.facets and not plan.exclusions:
        return "  (empty plan)"
    lines: list[str] = []
    for f in plan.facets:
        terms = _dedupe_preserve_order(f.terms)
        if not terms:
            lines.append(f"  facet {f.id} ({f.concept}): (empty)")
            continue
        lines.append(f"  facet {f.id} ({f.concept}):")
        for t in terms:
            tag = t.strictness
            if tag == "proximity":
                tag = f"proximity~{t.slop}"
            lines.append(f"      · {t.raw}  [{tag}]")
    if plan.exclusions:
        lines.append("  exclusions:")
        for t in _dedupe_preserve_order(plan.exclusions):
            lines.append(f"      · {t.raw}  [{t.strictness}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Skeleton rendering (supervisor → worker) + amendment application
# ---------------------------------------------------------------------------


def render_skeleton(skeleton: FacetSkeleton | None) -> str:
    if skeleton is None or not skeleton.facets:
        return "  (no skeleton — worker designs facets from the description alone)"
    lines: list[str] = []
    for f in skeleton.facets:
        seeds = ", ".join(f.seed_terms) if f.seed_terms else "(no seeds)"
        lines.append(f"  facet {f.id} — {f.concept}")
        lines.append(f"      seed terms: {seeds}")
    return "\n".join(lines)


def apply_skeleton_amendments(
    skeleton: FacetSkeleton | None,
    amendments: list[dict],
) -> FacetSkeleton | None:
    """Apply a worker's amendment list to the supervisor's skeleton.

    Supported ops: ``add_facet``, ``remove_facet``, ``reshape_terms``.
    Malformed entries are skipped silently — the caller decides
    whether to log them."""
    if not skeleton:
        return skeleton
    facets = list(skeleton.facets)
    from citeclaw.agents.v3.state import Facet

    for amend in amendments or []:
        if not isinstance(amend, dict):
            continue
        op = str(amend.get("op") or "").strip()
        fid = str(amend.get("facet_id") or amend.get("id") or "").strip()
        if not fid:
            continue
        if op == "remove_facet":
            facets = [f for f in facets if f.id != fid]
        elif op == "add_facet":
            if any(f.id == fid for f in facets):
                continue
            concept = str(amend.get("concept") or fid).strip()
            seeds = tuple(
                str(s).strip() for s in (amend.get("seed_terms") or []) if str(s).strip()
            )
            facets.append(Facet(id=fid, concept=concept, seed_terms=seeds))
        elif op == "reshape_terms":
            new_seeds = tuple(
                str(s).strip() for s in (amend.get("seed_terms") or []) if str(s).strip()
            )
            new_concept = str(amend.get("concept") or "").strip()
            facets = [
                Facet(
                    id=f.id,
                    concept=new_concept or f.concept,
                    seed_terms=new_seeds if new_seeds else f.seed_terms,
                ) if f.id == fid else f
                for f in facets
            ]
    return FacetSkeleton(facets=tuple(facets))
