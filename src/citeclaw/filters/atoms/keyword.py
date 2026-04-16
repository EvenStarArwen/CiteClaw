"""Keyword-based filters over paper title / abstract / venue.

Three atoms share the same DSL: :class:`TitleKeywordFilter` checks the
paper's title, :class:`AbstractKeywordFilter` checks the abstract,
:class:`VenueKeywordFilter` checks the venue. All three run a plain
substring search per keyword and combine the per-keyword booleans with
the same Boolean formula DSL used by :class:`LLMFilter`
(:mod:`citeclaw.screening.formula`).

**Simple mode** — single keyword or phrase::

    title_has_ml:
      type: TitleKeywordFilter
      keyword: "deep learning"          # case-insensitive substring

**Formula mode** — Boolean expression over named keywords (operators ``&``
AND, ``|`` OR, ``!`` NOT, parenthesised)::

    abstract_topic:
      type: AbstractKeywordFilter
      formula: "(deep_learning | transformer) & !survey"
      keywords:
        deep_learning: "deep learning"
        transformer:   "transformer"
        survey:        "survey"

Each keyword is an independent substring search; the per-keyword booleans
are then fed into the formula. Names in ``keywords`` are arbitrary
identifiers used by the formula; the *value* is the actual string searched
for in the field.

Knobs:
    ``case_sensitive`` — default ``False``. Match exact case when ``True``.
    ``match``          — default ``"substring"``. Match strategy:
                         ``"substring"``    plain ``kw in text``;
                         ``"whole_word"``   ``\\b<kw>\\b`` regex, so
                                            ``"learn"`` won't match
                                            ``"learning"``;
                         ``"starts_with"``  ``^<kw>\\b`` regex (anchored
                                            at the start of the field
                                            with a word boundary), so
                                            ``"Cell"`` matches
                                            ``"Cell Reports"`` but not
                                            ``"Stem Cell Reports"`` or
                                            ``"Cellulose"``. Use this
                                            for journal allow-lists like
                                            "venue starts with Nature /
                                            Science / Cell".

Missing / empty fields are treated as the empty string. A required
keyword (e.g. ``ml``) won't match against an empty field, but a negation
(e.g. ``!survey``) will pass — same semantics as evaluating the formula
on an all-False keyword vector.
"""

from __future__ import annotations

import re
from typing import Any

from citeclaw.filters.base import PASS, FilterContext, FilterOutcome
from citeclaw.models import PaperRecord


class _KeywordFilterBase:
    """Shared logic for title / abstract keyword filters.

    Subclasses set :attr:`_scope_label` (used in rejection messages) and
    :attr:`_category` (the rejection bucket) and implement :meth:`_content`
    to extract the field they screen.
    """

    _scope_label: str = "field"
    _category: str = "keyword"

    _MATCH_MODES = ("substring", "whole_word", "starts_with")

    def __init__(
        self,
        name: str,
        *,
        keyword: str | None = None,
        formula: str | None = None,
        keywords: dict[str, str] | None = None,
        case_sensitive: bool = False,
        match: str = "substring",
    ) -> None:
        self.name = name
        self.case_sensitive = bool(case_sensitive)
        if match not in self._MATCH_MODES:
            raise ValueError(
                f"{type(self).__name__}: 'match' must be one of "
                f"{self._MATCH_MODES}, got {match!r}"
            )
        self.match = match

        cls_name = type(self).__name__
        if keyword is not None and formula is not None:
            raise ValueError(
                f"{cls_name}: 'keyword' and 'formula' are mutually exclusive"
            )
        if keyword is None and formula is None:
            raise ValueError(
                f"{cls_name}: provide either 'keyword' or 'formula' + 'keywords'"
            )

        self.keyword: str | None = None
        self.formula_expr: str | None = None
        self.keywords: dict[str, str] = {}
        self._formula: Any = None

        if keyword is not None:
            if not isinstance(keyword, str) or not keyword.strip():
                raise ValueError(f"{cls_name}: 'keyword' must be a non-empty string")
            if keywords:
                raise ValueError(
                    f"{cls_name}: 'keywords' is only valid together with 'formula'"
                )
            self.keyword = keyword
            return

        # Formula mode — parse eagerly so config errors surface at build time.
        if not keywords:
            raise ValueError(
                f"{cls_name}: 'formula' requires a non-empty 'keywords' mapping"
            )
        from citeclaw.screening.formula import BooleanFormula, FormulaError

        try:
            self._formula = BooleanFormula(formula)
        except FormulaError as exc:
            raise ValueError(f"{cls_name}: bad formula {formula!r}: {exc}") from exc
        referenced = self._formula.query_names()
        missing = sorted(referenced - set(keywords.keys()))
        if missing:
            raise ValueError(
                f"{cls_name}: formula references undefined keywords: {missing}"
            )
        for kname, kw in keywords.items():
            if not isinstance(kw, str) or not kw.strip():
                raise ValueError(
                    f"{cls_name}: keyword {kname!r} must be a non-empty string"
                )
        extras = sorted(set(keywords.keys()) - referenced)
        if extras:
            import logging

            logging.getLogger("citeclaw.filters.atoms.keyword").warning(
                "%s %r defines unused keywords: %s", cls_name, name, extras,
            )
        self.formula_expr = formula
        self.keywords = dict(keywords)

    def _content(self, paper: PaperRecord) -> str:
        raise NotImplementedError

    def _match_one(self, kw: str, text: str) -> bool:
        if not text:
            return False
        if self.match == "starts_with":
            flags = 0 if self.case_sensitive else re.IGNORECASE
            # ``re.match`` is anchored at the start of the string.
            # ``\b`` after the keyword forbids matches like Cell→Cellulose.
            # ``lstrip`` defends against stray leading whitespace in venue.
            return re.match(rf"{re.escape(kw)}\b", text.lstrip(), flags) is not None
        if self.match == "whole_word":
            flags = 0 if self.case_sensitive else re.IGNORECASE
            return re.search(rf"\b{re.escape(kw)}\b", text, flags) is not None
        if self.case_sensitive:
            return kw in text
        return kw.lower() in text.lower()

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        text = self._content(paper) or ""
        if self.keyword is not None:
            if self._match_one(self.keyword, text):
                return PASS
            return FilterOutcome(
                False,
                f"{self._scope_label} missing keyword {self.keyword!r}",
                self._category,
            )
        values = {name: self._match_one(kw, text) for name, kw in self.keywords.items()}
        if self._formula.evaluate(values):
            return PASS
        seen = ", ".join(f"{n}={'1' if v else '0'}" for n, v in sorted(values.items()))
        return FilterOutcome(
            False,
            f"{self._scope_label} formula {self.formula_expr!r} false ({seen})",
            self._category,
        )


class TitleKeywordFilter(_KeywordFilterBase):
    """Pass papers whose title satisfies a keyword (or Boolean formula)."""

    _scope_label = "title"
    _category = "title_keyword"

    def __init__(self, name: str = "title_keyword", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)

    def _content(self, paper: PaperRecord) -> str:
        return paper.title or ""


class AbstractKeywordFilter(_KeywordFilterBase):
    """Pass papers whose abstract satisfies a keyword (or Boolean formula)."""

    _scope_label = "abstract"
    _category = "abstract_keyword"

    def __init__(self, name: str = "abstract_keyword", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)

    def _content(self, paper: PaperRecord) -> str:
        return paper.abstract or ""


class VenueKeywordFilter(_KeywordFilterBase):
    """Pass papers whose venue satisfies a keyword (or Boolean formula).

    Pair with ``match: starts_with`` for hard journal allow-lists like
    "venue starts with Nature / Science / Cell" — that accepts
    ``Nature Methods``, ``Science Advances``, and ``Cell Reports`` but
    rejects ``Royal Society Open Science``, ``Chemical Science``,
    ``Energy & Environmental Science``, ``Stem Cell Reports``, and
    ``Cellulose`` (none of which begin with the keyword).
    """

    _scope_label = "venue"
    _category = "venue_keyword"

    def __init__(self, name: str = "venue_keyword", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)

    def _content(self, paper: PaperRecord) -> str:
        return paper.venue or ""
