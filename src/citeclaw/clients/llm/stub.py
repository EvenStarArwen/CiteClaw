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
  - Annotate label (free-text "Label:" prompt)          → "stub-label"
"""

from __future__ import annotations

import json
import re
from typing import Any

from citeclaw.clients.llm._token_extract import estimate_stub_usage
from citeclaw.clients.llm.base import LLMResponse
from citeclaw.budget import BudgetTracker
from citeclaw.config import Settings

_RE_EXACTLY_N = re.compile(r"exactly\s+(\d+)\s+objects", re.IGNORECASE)
_RE_CLUSTER_ID = re.compile(r"cluster_id=(-?\d+)")
# Matches the per-paper marker in citeclaw.prompts.annotation.PAPER_BLOCK_TEMPLATE.
_RE_PAPER_INDEX = re.compile(r"Paper index=(-?\d+)")
_RE_TITLE_LINE = re.compile(r"Title:\s*(.+)")


def _extract_n(user: str) -> int:
    """Infer the number of items the prompt expects in the response.

    First looks for an explicit ``"exactly N objects"`` declaration in
    the prompt template; falls back to counting numbered list items
    (``^N. ``). Defaults to 1 so a malformed prompt still yields a
    valid (single-item) stub response.
    """
    m = _RE_EXACTLY_N.search(user)
    if m:
        return int(m.group(1))
    n = len(re.findall(r"^\s*\d+\.\s", user, re.MULTILINE))
    return max(n, 1)


def stub_respond(system: str, user: str) -> str:
    """Return a deterministic JSON / text response for the given prompt."""
    if '"relevant_references"' in user:
        # PDF reference extraction (citeclaw.steps._pdf_reference_extractor).
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
    if system.startswith("You are labelling") or "Label:" in user:
        # Batched annotation (see citeclaw.prompts.annotation.BATCH_USER_TEMPLATE)
        # interleaves ``Paper index=<int>`` and ``Title: <str>`` blocks —
        # one per paper. Walk the user text and match them in order so each
        # index gets the title from *its* block rather than the first one.
        indices = [int(x) for x in _RE_PAPER_INDEX.findall(user)]
        titles = [m.group(1).strip() for m in _RE_TITLE_LINE.finditer(user)]
        if indices and len(titles) >= len(indices):
            results = []
            for idx, title in zip(indices, titles):
                words = (title or "stub").split()
                label = (" ".join(words[:2])[:30]) or "stub"
                results.append({"index": idx, "label": label})
            return json.dumps({"results": results})
        # Legacy single-paper path (USER_TEMPLATE ends with "Label:").
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
        """Run the deterministic responder, billing tokens via the budget.

        Honours :class:`Settings`-driven budget exhaustion so the stub
        path mirrors a real client's failure mode (the pipeline must
        still observe budget caps even on offline runs).
        ``with_logprobs`` and ``response_schema`` are accepted for
        signature compatibility but ignored — the stub's output is
        already well-formed JSON / text.
        """
        from citeclaw.models import BudgetExhaustedError

        if self._budget.is_exhausted(self._config):
            raise BudgetExhaustedError(f"Budget exhausted: {self._budget.summary()}")
        text = stub_respond(system, user)
        usage = estimate_stub_usage(user, text)
        self._budget.record_llm(
            usage.prompt, usage.completion, category, model=self._model,
        )
        return LLMResponse(text=text, logprob_tokens=[])
