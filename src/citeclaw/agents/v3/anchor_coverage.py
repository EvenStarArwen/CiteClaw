"""Anchor-coverage check for V3 worker iterations.

Given a set of titles the worker *should* have retrieved (either the
auto-injected anchors from :mod:`anchor_discovery`, or titles the
worker names during diagnosis), fuzzy-match each against the current
iter's fetched paper titles and report present / absent / ambiguous.

Anchor coverage replaces total-count-based effectiveness guessing as
the refinement signal. Literature on query performance prediction
(Scells 2018) shows internal query statistics are poor predictors on
recall tasks; actual expected-paper coverage is the direct test.

Matching is deliberately lenient: Unicode-strip, lowercase, drop
punctuation, squeeze whitespace, then check whether every word of the
reference title appears (as a whole word) in the fetched title. This
tolerates trailing subtitles and minor rewording without being
blindsided by punctuation.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from citeclaw.agents.v3.state import AnchorPaper


CoverageStatus = Literal["present", "absent", "ambiguous"]


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _normalise(text: str) -> str:
    s = unicodedata.normalize("NFKD", text or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(_normalise(text)) if w]


def _significant_tokens(text: str) -> list[str]:
    """Words long enough to contribute matching signal."""
    return [w for w in _tokens(text) if len(w) >= 3]


def _match_score(ref_tokens: list[str], cand_tokens: set[str]) -> float:
    if not ref_tokens:
        return 0.0
    hits = sum(1 for w in ref_tokens if w in cand_tokens)
    return hits / len(ref_tokens)


def check_anchor_coverage(
    anchor_titles: list[str],
    fetched_titles: list[str],
    *,
    present_threshold: float = 0.85,
    absent_threshold: float = 0.5,
) -> dict[str, CoverageStatus]:
    """For each anchor, return ``present`` / ``absent`` / ``ambiguous``.

    - ``present``: at least one fetched title shares ``>=`` ``present_threshold``
      of the anchor's significant-word tokens.
    - ``absent``: the best fetched overlap is below ``absent_threshold``.
    - ``ambiguous``: the best overlap is in between — the anchor
      might be there, but no single fetched title matched cleanly.
    """
    out: dict[str, CoverageStatus] = {}
    cand_token_sets = [set(_tokens(t)) for t in fetched_titles if t]
    for title in anchor_titles:
        if not title.strip():
            continue
        ref = _significant_tokens(title)
        if not ref:
            out[title] = "ambiguous"
            continue
        best = 0.0
        for cand in cand_token_sets:
            score = _match_score(ref, cand)
            if score > best:
                best = score
                if best >= 1.0:
                    break
        if best >= present_threshold:
            out[title] = "present"
        elif best <= absent_threshold:
            out[title] = "absent"
        else:
            out[title] = "ambiguous"
    return out


def render_anchor_coverage(coverage: dict[str, CoverageStatus]) -> str:
    if not coverage:
        return "  (no anchors to check)"
    n_total = len(coverage)
    n_present = sum(1 for v in coverage.values() if v == "present")
    n_absent = sum(1 for v in coverage.values() if v == "absent")
    n_ambig = n_total - n_present - n_absent
    lines = [
        f"  present: {n_present}/{n_total}   absent: {n_absent}   ambiguous: {n_ambig}",
    ]
    for title, status in coverage.items():
        short = title if len(title) <= 100 else title[:97] + "..."
        lines.append(f"  [{status:9s}] {short}")
    return "\n".join(lines)


def coverage_ratio(coverage: dict[str, CoverageStatus]) -> float:
    if not coverage:
        return 0.0
    present = sum(1 for v in coverage.values() if v == "present")
    return present / len(coverage)


def titles_for_coverage(
    anchors: list[AnchorPaper],
    extra_titles: list[str] | None = None,
) -> list[str]:
    """Merge auto-injected anchor titles with worker-proposed extras."""
    out = [a.title for a in anchors if a.title.strip()]
    if extra_titles:
        out.extend(t for t in extra_titles if t and t.strip())
    # Dedupe while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for t in out:
        k = _normalise(t)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(t)
    return deduped
