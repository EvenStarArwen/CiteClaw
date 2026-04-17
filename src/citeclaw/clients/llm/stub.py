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
# Matches the per-paper marker in citeclaw.prompts.annotation.PAPER_BLOCK_TEMPLATE.
_RE_PAPER_INDEX = re.compile(r"Paper index=(-?\d+)")
_RE_TITLE_LINE = re.compile(r"Title:\s*(.+)")


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
    if "SUPERVISOR" in system and "set_strategy" in system:
        # v2 ExpandBySearch supervisor. Three-state lifecycle driven by
        # what the most recent user message says:
        #   1. No prior strategy set   → set_strategy (1 sub-topic)
        #   2. Strategy set, nothing dispatched → dispatch worker
        #   3. Worker dispatched → done()
        if '"acknowledged": true' in user and "n_sub_topics" in user and "sub_topic_ids" in user:
            # set_strategy result is in the user message; dispatch next.
            m = re.search(r'"sub_topic_ids":\s*\[\s*"([^"]+)"', user)
            spec_id = m.group(1) if m else "stub_sub_topic"
            return json.dumps({
                "reasoning": "stub: dispatching the one sub-topic",
                "tool_name": "dispatch_sub_topic_worker",
                "tool_args": {"spec_id": spec_id},
            })
        if '"status":' in user and ('"success"' in user or '"failed"' in user or '"budget_exhausted"' in user):
            # Worker result seen — close the run.
            return json.dumps({
                "reasoning": "stub: wrapping up",
                "tool_name": "done",
                "tool_args": {"summary": "stub supervisor: 1 worker dispatched"},
            })
        # Fresh run — lock in a minimal strategy.
        return json.dumps({
            "reasoning": "stub: one-shot strategy for e2e test",
            "tool_name": "set_strategy",
            "tool_args": {
                "structural_priors": {},
                "sub_topics": [
                    {
                        "id": "stub_sub_topic",
                        "description": "stub sub-topic for e2e test",
                        "initial_query_sketch": "test topic",
                        "reference_papers": [],
                    }
                ],
            },
        })
    if "WORKER" in system and "check_query_size" in system and "fetch_results" in system:
        # v2 ExpandBySearch sub-topic worker. Drive the checklist to
        # completion by inspecting the previous tool's result in the user
        # message and picking the next step.
        #
        # The continuation user message renders the previous call as
        # ``**Previous call**: `<tool_name>` `` (backticks), then the
        # result dict inline. We match on the backtick-wrapped form so
        # the ladder actually advances.
        last_df_match = re.search(r'"df_id":\s*"([^"]+)"', user)
        last_df = last_df_match.group(1) if last_df_match else None
        has_prev = "Previous call" in user
        # Most recent tool name, derived from the marker line.
        m_prev = re.search(r"Previous call\*?\*?:\s*`([^`]+)`", user)
        last_tool = m_prev.group(1) if m_prev else ""
        last_strategy_match = re.search(r'"strategy":\s*"([^"]+)"', user)
        # Decision ladder in order of checklist.
        if not has_prev:
            return json.dumps({
                "reasoning": "stub: initial size check",
                "tool_name": "check_query_size",
                "tool_args": {"query": "test topic"},
            })
        if last_tool == "check_query_size":
            return json.dumps({
                "reasoning": "stub: fetch what we just sized",
                "tool_name": "fetch_results",
                "tool_args": {"query": "test topic"},
            })
        if last_tool == "fetch_results":
            return json.dumps({
                "reasoning": "stub: sample top cited",
                "tool_name": "sample_titles",
                "tool_args": {"df_id": last_df, "strategy": "top_cited", "n": 10},
            })
        if last_tool == "sample_titles" and last_strategy_match and last_strategy_match.group(1) == "top_cited":
            return json.dumps({
                "reasoning": "stub: sample random",
                "tool_name": "sample_titles",
                "tool_args": {"df_id": last_df, "strategy": "random", "n": 10},
            })
        if last_tool == "sample_titles" and last_strategy_match and last_strategy_match.group(1) == "random":
            return json.dumps({
                "reasoning": "stub: year dist",
                "tool_name": "year_distribution",
                "tool_args": {"df_id": last_df},
            })
        if last_tool == "year_distribution":
            return json.dumps({
                "reasoning": "stub: resolve reference title",
                "tool_name": "search_match",
                "tool_args": {"title": "Stub Reference Paper"},
            })
        if last_tool == "search_match":
            return json.dumps({
                "reasoning": "stub: check containment",
                "tool_name": "contains",
                "tool_args": {"paper_id": "stub_ref_paper"},
            })
        if last_tool == "contains" and '"contains": false' in user:
            return json.dumps({
                "reasoning": "stub: accept gap",
                "tool_name": "diagnose_miss",
                "tool_args": {
                    "target_title": "Stub Reference Paper",
                    "hypotheses": ["stub S2 coverage gap"],
                    "action_taken": "accept_gap",
                    "query_angles_used": ["test topic"],
                },
            })
        # Default: close.
        return json.dumps({
            "reasoning": "stub: done",
            "tool_name": "done",
            "tool_args": {
                "paper_ids": [],
                "coverage_assessment": "acceptable",
                "summary": "stub worker: one angle covered",
            },
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
