"""Deterministic offline stub LLM client.

Used in tests / CI / `screening_model: stub` so the full pipeline runs
without spending real LLM tokens. The stub inspects each prompt's expected
output schema and returns matching JSON with deterministic verdicts:

  - Query screening (``{"index": N, "match": true}``)   → match=true
    Emitted as ``{"results": [...]}`` to match the structured-output shape
    used by OpenAI ``response_format``/Gemini ``response_schema``. The
    parser in :mod:`citeclaw.screening.llm_runner` also accepts the legacy
    flat-array shape so older callers keep working.
  - Re-screen scope (``{"index": N, "reject": false}``) → reject=false
  - LLM reranker   (``{"index": N, "score": <int>}``)   → score=3
  - Topic naming   (``{"topic_label": ..., "summary": ...}``)
                                                        → label="stub-topic", summary="stub summary"
  - Annotate label (free-text "Label:" prompt)          → "stub-label"
"""

from __future__ import annotations

import json
import re
from typing import Any

from citeclaw.clients.llm.base import LLMResponse
from citeclaw.config import BudgetTracker, Settings

_RE_EXACTLY_N = re.compile(r"exactly\s+(\d+)\s+objects", re.IGNORECASE)


def _extract_n(user: str) -> int:
    m = _RE_EXACTLY_N.search(user)
    if m:
        return int(m.group(1))
    n = len(re.findall(r"^\s*\d+\.\s", user, re.MULTILINE))
    return max(n, 1)


def stub_respond(system: str, user: str) -> str:
    """Return a deterministic JSON / text response for the given prompt."""
    if '"topic_label"' in user:
        return json.dumps({"topic_label": "stub-topic", "summary": "stub summary"})
    if '"score"' in user:
        n = _extract_n(user)
        return json.dumps([{"index": i, "score": 3} for i in range(1, n + 1)])
    if '"reject"' in user:
        n = _extract_n(user)
        return json.dumps([{"index": i, "reject": False} for i in range(1, n + 1)])
    if '"match"' in user:
        n = _extract_n(user)
        return json.dumps(
            {"results": [{"index": i, "match": True} for i in range(1, n + 1)]}
        )
    if "Label:" in user or system.startswith("You are labelling"):
        m = re.search(r"Title:\s*(.+)", user)
        title = (m.group(1).strip() if m else "stub").split()
        return " ".join(title[:2])[:30] or "stub"
    return "[]"


class StubClient:
    """Concrete LLMClient using the deterministic stub responder."""

    supports_logprobs = False

    def __init__(
        self,
        config: Settings,
        budget: BudgetTracker,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self._config = config
        self._budget = budget
        # Stub ignores model / reasoning_effort; kwargs accepted so the
        # factory can call every client uniformly.
        self._model = model or config.screening_model
        self._reasoning_effort = (
            reasoning_effort if reasoning_effort is not None else config.reasoning_effort
        )

    def call(
        self,
        system: str,
        user: str,
        *,
        with_logprobs: bool = False,
        category: str = "other",
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        from citeclaw.models import BudgetExhaustedError

        if self._budget.is_exhausted(self._config):
            raise BudgetExhaustedError(f"Budget exhausted: {self._budget.summary()}")
        # The stub ignores ``response_schema`` — its deterministic output is
        # already well-formed. ``stub_respond`` returns the wrapped shape
        # for match-queries so it parses identically to a real structured
        # response.
        text = stub_respond(system, user)
        # Deterministic fake token bookkeeping
        self._budget.record_llm(len(user) // 4, len(text) // 4, category)
        return LLMResponse(text=text, logprob_tokens=[])
