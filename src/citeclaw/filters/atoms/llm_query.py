"""LLMFilter atom — yes/no LLM query, scoped to title / abstract / venue.

Note: ``check()`` is implemented but the runner short-circuits and dispatches
batches via ``screening.llm_runner.dispatch_batch`` for efficiency. ``check()``
exists so the Filter Protocol is satisfied for unit / Route fall-through use.

A filter runs in one of two modes:

**Single-prompt mode** (default) — the ``prompt`` field is a single
criterion and the filter accepts the paper iff the LLM says "match".

**Formula mode** — the ``formula`` field is a Boolean expression over
named sub-queries (``queries``). Each sub-query runs as an independent
batched LLM call; the filter's final verdict is the Boolean evaluation
of ``formula`` substituting each sub-query's result. Operators: ``&``
AND, ``|`` OR, ``!`` NOT, parenthesised. Example::

    abstract_llm:
      type: LLMFilter
      scope: title_abstract
      formula: "(q_ml | q_stats) & !q_survey"
      queries:
        q_ml: "the paper proposes a new ML/DL method"
        q_stats: "the paper proposes a new statistical method"
        q_survey: "the paper is a pure survey or review"

Formula mode inherits every other per-filter setting (``model``,
``reasoning_effort``, ``votes``, ``min_accepts``) — sub-queries use the
same model, the same voting threshold, etc.

Per-filter overrides (all optional, all default to ``None``/``1`` so
zero-change YAML configs keep working):
    ``model``            — override the global ``screening_model`` for this
                           filter only. Cross-provider is allowed: e.g. the
                           base model can be ``gpt-4o`` and one filter can use
                           ``gemini-2.5-flash``.
    ``reasoning_effort`` — override the global reasoning effort.
    ``votes``            — number of independent LLM calls per paper
                           (default 1 = today's behavior).
    ``min_accepts``      — minimum "accept" votes to let a paper pass.
                           Must satisfy ``1 <= min_accepts <= votes``.
    ``formula``          — Boolean expression over named sub-queries.
                           Mutually exclusive with ``prompt``.
    ``queries``          — mapping ``{name: sub-prompt}``. Required when
                           ``formula`` is set; must cover every name
                           referenced by the formula.
"""

from __future__ import annotations

from typing import Any

from citeclaw.filters.base import PASS, FilterContext, FilterOutcome
from citeclaw.models import PaperRecord


class LLMFilter:
    def __init__(
        self,
        name: str = "llm",
        *,
        scope: str = "title",
        prompt: str = "",
        model: str | None = None,
        reasoning_effort: str | None = None,
        votes: int = 1,
        min_accepts: int = 1,
        formula: str | None = None,
        queries: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        if scope not in ("title", "title_abstract", "venue"):
            raise ValueError(f"LLMFilter.scope must be title|title_abstract|venue, got {scope!r}")
        self.scope = scope
        self.model = model
        self.reasoning_effort = reasoning_effort
        if votes < 1:
            raise ValueError(f"LLMFilter.votes must be >= 1, got {votes}")
        if min_accepts < 1:
            raise ValueError(f"LLMFilter.min_accepts must be >= 1, got {min_accepts}")
        if min_accepts > votes:
            raise ValueError(
                f"LLMFilter.min_accepts ({min_accepts}) cannot exceed votes ({votes})"
            )
        self.votes = votes
        self.min_accepts = min_accepts

        # Formula mode ↔ single-prompt mode are mutually exclusive.
        self.prompt = prompt
        self.formula_expr: str | None = formula
        self.queries: dict[str, str] = dict(queries or {})
        self._formula: Any = None
        if formula is not None:
            if prompt:
                raise ValueError(
                    "LLMFilter: 'formula' is mutually exclusive with 'prompt'; "
                    "use one or the other"
                )
            if not self.queries:
                raise ValueError(
                    "LLMFilter: 'formula' requires a non-empty 'queries' mapping"
                )
            # Parse eagerly so config errors surface at build time, not at
            # first-batch time when the pipeline is already mid-run.
            from citeclaw.screening.formula import BooleanFormula, FormulaError

            try:
                self._formula = BooleanFormula(formula)
            except FormulaError as exc:
                raise ValueError(f"LLMFilter: bad formula {formula!r}: {exc}") from exc
            referenced = self._formula.query_names()
            missing = sorted(referenced - set(self.queries.keys()))
            if missing:
                raise ValueError(
                    f"LLMFilter: formula references undefined queries: {missing}"
                )
            extras = sorted(set(self.queries.keys()) - referenced)
            if extras:
                # Extra queries are tolerated but worth flagging.
                import logging
                logging.getLogger("citeclaw.filters.atoms.llm_query").warning(
                    "LLMFilter %r defines unused sub-queries: %s", name, extras,
                )
        elif self.queries:
            raise ValueError(
                "LLMFilter: 'queries' is only valid together with 'formula'"
            )

    def content_for(self, paper: PaperRecord) -> str:
        if self.scope == "title":
            return paper.title or ""
        if self.scope == "venue":
            return paper.venue or ""
        # title_abstract
        return f"Title: {paper.title}\nAbstract: {paper.abstract or '(no abstract)'}"

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        # Synchronous fallback path: dispatch a batch of 1.
        from citeclaw.screening.llm_runner import dispatch_batch

        verdicts = dispatch_batch([paper], self, fctx.ctx)
        if verdicts.get(paper.paper_id, False):
            return PASS
        return FilterOutcome(False, f"llm:{self.name}", f"llm_{self.name}")
