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
  - Topic naming — two shapes:
      * Single-cluster legacy prompt ``{"topic_label": ..., "summary": ...}``
        → label="stub-topic", summary="stub summary"
      * Batched prompt (detected by ``cluster_id=`` markers in the user
        text) → ``{"results": [{"cluster_id": N, "topic_label": ..., "summary": ...}, ...]}``
        with one entry per cluster found in the prompt.
  - Iterative search agent (``{"thinking": ..., "query": ..., "agent_decision": ..., "reasoning": ...}``)
                                                        → three-state lifecycle initial → refine → satisfied,
    advanced by counting prior ``"query":`` JSON keys in the transcript.
  - Annotate label (free-text "Label:" prompt)          → "stub-label"
"""

from __future__ import annotations

import json
import re
from typing import Any

from citeclaw.clients.llm.base import LLMResponse
from citeclaw.config import BudgetTracker, Settings

_RE_EXACTLY_N = re.compile(r"exactly\s+(\d+)\s+objects", re.IGNORECASE)
_RE_CLUSTER_ID = re.compile(r"cluster_id=(-?\d+)")


def _extract_n(user: str) -> int:
    m = _RE_EXACTLY_N.search(user)
    if m:
        return int(m.group(1))
    n = len(re.findall(r"^\s*\d+\.\s", user, re.MULTILINE))
    return max(n, 1)


def stub_respond(system: str, user: str) -> str:
    """Return a deterministic JSON / text response for the given prompt."""
    if '"relevant_references"' in user:
        # PDF reference extraction (citeclaw.agents.pdf_reference_extractor).
        # Return one deterministic reference so the pipeline has something
        # to resolve via search_match.
        return json.dumps({
            "relevant_references": [
                {
                    "citation_marker": "[1]",
                    "reference_text": "Stub Author. Stub Reference Title. Journal 2023.",
                    "title": "Stub Reference Title",
                    "mentions": [
                        {"quote": "We build on [1] which established the baseline.", "relevance": "baseline"}
                    ],
                    "relevance_explanation": "Foundational baseline work.",
                }
            ]
        })
    if '"topic_label"' in user:
        # Batched form (see citeclaw.prompts.topic_naming.BATCH_USER_TEMPLATE)
        # embeds ``cluster_id=<int>`` markers — one per cluster. When we
        # see them, return the structured-output shape with one entry per
        # cluster found in the prompt.
        cluster_ids = _RE_CLUSTER_ID.findall(user)
        if cluster_ids:
            return json.dumps({
                "results": [
                    {
                        "cluster_id": int(cid),
                        "topic_label": "stub-topic",
                        "summary": "stub summary",
                    }
                    for cid in cluster_ids
                ]
            })
        return json.dumps({"topic_label": "stub-topic", "summary": "stub summary"})
    if '"agent_decision"' in user or '"should_stop"' in user:
        # Iterative search agent (citeclaw.agents.iterative_search). Each
        # iteration's user prompt embeds the prior turns' raw JSON in the
        # transcript section, so the count of ``"query":`` JSON keys grows
        # by exactly 1 per completed iteration. Map that count to a
        # deterministic three-state lifecycle:
        #
        #   0 prior queries → "initial"  (first turn)
        #   1 prior query   → "refine"   (narrow after seeing first results)
        #  ≥2 prior queries → "satisfied" (terminate the loop)
        #
        # PH-08: the trigger now matches BOTH the v1 schema (which had
        # ``agent_decision`` as a top-level field) and the v2 schema
        # (which has ``should_stop`` as a top-level boolean). The
        # response carries the v1 fields (thinking + agent_decision) so
        # the existing test suite — which asserts on those exact field
        # names — keeps passing. The agent code's backward-compat
        # fallback in iterative_search.py reads agent_decision when
        # should_stop is missing.
        n_prior_queries = user.count('"query":')
        if n_prior_queries == 0:
            return json.dumps({
                "thinking": "stub: initial exploration",
                "query": {"text": "test topic"},
                "agent_decision": "initial",
                "reasoning": "stub initial",
            })
        if n_prior_queries == 1:
            return json.dumps({
                "thinking": "stub: prior was too broad, narrowing",
                "query": {"text": "test topic narrowed"},
                "agent_decision": "refine",
                "reasoning": "stub refine",
            })
        return json.dumps({
            "thinking": "stub: results saturated",
            "query": {"text": "test topic narrowed"},
            "agent_decision": "satisfied",
            "reasoning": "stub satisfied",
        })
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
