"""End-to-end regression tests for ExpandBySearch worker + supervisor.

Post-refactor tool surface: 7 tools (check_query_size, fetch_results,
query_diagnostics, search_within_df, get_paper, diagnose_miss, done).
Deterministic post-fetch work (sampling, distributions, topic_model,
reference-paper verification) lives inside ``fetch_results`` and is
returned as a single inspection digest — the worker no longer
orchestrates a per-angle checklist, so the scripts here are much
shorter than in v2.

Uses a scripted stub LLM that replays a canned tool-call sequence
offline — no S2 traffic, no LLM traffic, deterministic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from citeclaw.agents.dataframe_store import DataFrameStore
from citeclaw.agents.search_logging import NullSearchLogger
from citeclaw.agents.state import (
    AgentConfig,
    StructuralPriors,
    SubTopicSpec,
    WorkerState,
)
from citeclaw.agents.supervisor import run_supervisor
from citeclaw.agents.worker import run_sub_topic_worker
from citeclaw.clients.llm.base import LLMResponse
from citeclaw.models import PaperRecord


# ---------------------------------------------------------------------------
# Scripted stub LLM
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """Replays a list of tool-call dicts."""

    supports_logprobs = False

    def __init__(self, script: list[dict[str, Any]]):
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def call(
        self, system: str, user: str, *,
        with_logprobs: bool = False, category: str = "other",
        response_schema: dict | None = None,
    ) -> LLMResponse:
        self.calls.append({"user_len": len(user), "system_len": len(system)})
        if not self._script:
            raise AssertionError(
                f"script exhausted after {len(self.calls)} calls"
            )
        item = dict(self._script.pop(0))
        return LLMResponse(text=json.dumps(item), reasoning_content="")


# ---------------------------------------------------------------------------
# Minimal ctx for worker/supervisor tests
# ---------------------------------------------------------------------------


class _FakeBudget:
    llm_total_tokens = 0
    s2_requests = 0

    def record_llm(self, *_a, **_k): self.llm_total_tokens += 10
    def record_s2(self, *_a, **_k): self.s2_requests += 1
    def is_exhausted(self, _cfg): return False


class _FakeCfg:
    max_llm_tokens = 1_000_000
    max_s2_requests = 1_000

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir


class _Ctx:
    def __init__(self, s2, data_dir: Path):
        self.s2 = s2
        self.budget = _FakeBudget()
        self.config = _FakeCfg(data_dir)


class _FakeS2:
    """Tiny fake that returns canned papers for any query.

    ``search_match`` returns a paper_id derived from the title so
    reference-coverage can distinguish "in cumulative" (matches a
    fetched paper) vs "not in cumulative" (different paper_id).
    """

    def __init__(
        self, *, total: int = 120, n_papers: int = 120,
        ref_pid: str = "ref_001",
    ):
        self._total = total
        self._n = n_papers
        self._ref_pid = ref_pid

    def search_bulk(self, query: str, *, filters=None, sort=None, token=None, limit=1000):
        n = min(limit, self._n)
        prefix = f"paper_{abs(hash(query)) % 10_000:04d}"
        data = [{"paperId": f"{prefix}_{i:03d}", "title": f"Title {i}"} for i in range(n)]
        return {"total": self._total, "data": data, "token": None}

    def search_match(self, title: str):
        return {"paperId": self._ref_pid, "title": title,
                "year": 2023, "venue": "NeurIPS"}

    def enrich_batch(self, candidates):
        return [
            PaperRecord(
                paper_id=c.get("paper_id", ""), title=f"T_{c.get('paper_id', '')}",
                year=2022, venue="NeurIPS", citation_count=50, abstract="",
            )
            for c in candidates
        ]

    def fetch_metadata(self, pid):
        return PaperRecord(
            paper_id=pid, title=f"T_{pid}", year=2023,
            venue="X", citation_count=1, abstract="abs",
        )

    def enrich_with_abstracts(self, recs):
        return recs

    def fetch_embeddings_batch(self, ids):
        return {pid: None for pid in ids}


@pytest.fixture
def tmp_ctx(tmp_path):
    return _Ctx(s2=_FakeS2(), data_dir=tmp_path)


def _spec(reference_papers=("DARTS: Differentiable Architecture Search",)):
    return SubTopicSpec(
        id="darts_core",
        description="DARTS core papers",
        initial_query_sketch='"differentiable architecture search"',
        reference_papers=reference_papers,
    )


def _run_worker(*, script, ctx, priors=None, cfg=None, spec=None):
    return run_sub_topic_worker(
        worker_id="w1",
        spec=spec or _spec(),
        priors=priors or StructuralPriors(),
        topic_description="Neural architecture search methods.",
        filter_summary="(no filters)",
        seed_papers=[],
        llm_client=ScriptedLLM(script),
        ctx=ctx,
        dataframe_store=DataFrameStore(),
        agent_config=cfg or AgentConfig(worker_max_turns=15),
        logger=NullSearchLogger(),
    )


# ---------------------------------------------------------------------------
# Happy path (no reference papers → no verification gate)
# ---------------------------------------------------------------------------


def test_worker_happy_path_no_refs(tmp_ctx):
    """With zero reference_papers the auto-verifier is a no-op, so the
    minimal successful trajectory is just size → fetch → done."""
    script = [
        {"reasoning": "size", "tool_name": "check_query_size",
         "tool_args": {"query": '"differentiable architecture search"'}},
        {"reasoning": "fetch", "tool_name": "fetch_results",
         "tool_args": {"query": '"differentiable architecture search"'}},
        {"reasoning": "wrap", "tool_name": "done",
         "tool_args": {"paper_ids": [], "coverage_assessment": "acceptable",
                       "summary": "covered via one query"}},
    ]
    result = _run_worker(
        script=script, ctx=tmp_ctx,
        spec=SubTopicSpec(
            id="darts_core", description="no refs",
            initial_query_sketch='"differentiable architecture search"',
            reference_papers=(),
        ),
    )
    assert result.status == "success"
    assert result.coverage_assessment == "acceptable"
    assert result.turns_used == 3
    assert len(result.query_results) == 1
    assert result.query_results[0].n_fetched == 120
    assert not result.auto_closed


# ---------------------------------------------------------------------------
# Reference-paper verification gate: done rejects when misses are
# pending; diagnose_miss consumes them.
# ---------------------------------------------------------------------------


def test_worker_done_gated_on_pending_misses(tmp_ctx):
    """spec has one reference paper. The fake S2 resolves it to
    ``ref_001``, which is NOT in the fetched set (fetch uses
    ``paper_XXXX_NNN`` ids) — so fetch_results surfaces exactly one
    miss, and done() rejects until diagnose_miss consumes it."""
    script = [
        {"reasoning": "size", "tool_name": "check_query_size",
         "tool_args": {"query": '"differentiable architecture search"'}},
        {"reasoning": "fetch", "tool_name": "fetch_results",
         "tool_args": {"query": '"differentiable architecture search"'}},
        # First done attempt: should fail — 1 pending miss.
        {"reasoning": "early done", "tool_name": "done",
         "tool_args": {"paper_ids": [], "coverage_assessment": "acceptable",
                       "summary": "x"}},
        # Diagnose the miss.
        {"reasoning": "diagnose", "tool_name": "diagnose_miss",
         "tool_args": {
             "target_title": "DARTS: Differentiable Architecture Search",
             "hypotheses": ["S2 coverage gap on that title"],
             "action_taken": "accept_gap",
             "queries_used": ['"differentiable architecture search"'],
         }},
        # Second done attempt: now succeeds.
        {"reasoning": "wrap", "tool_name": "done",
         "tool_args": {"paper_ids": [], "coverage_assessment": "acceptable",
                       "summary": "verified"}},
    ]
    result = _run_worker(script=script, ctx=tmp_ctx)
    assert result.status == "success"
    assert result.turns_used == 5
    # The rejected done attempt is visible in the event log.
    done_events = [
        e for e in tmp_ctx.s2.search_bulk and []  # placeholder
    ]
    # Not asserting call_log shape here — see test_done_rejected_without_fetch.


# ---------------------------------------------------------------------------
# fetch_results requires a prior check_query_size on the same fingerprint
# ---------------------------------------------------------------------------


def test_fetch_without_size_check_rejected(tmp_ctx):
    from citeclaw.agents.search_tools import register_worker_tools
    from citeclaw.agents.tool_dispatch import WorkerDispatcher

    ws = WorkerState(sub_topic_id="t", structural_priors=StructuralPriors())
    d = WorkerDispatcher(
        worker_state=ws, dataframe_store=DataFrameStore(),
        agent_config=AgentConfig(), ctx=tmp_ctx, worker_id="w1",
    )
    register_worker_tools(d)
    r = d.dispatch("fetch_results", {"query": '"foo"'})
    assert "error" in r
    assert "not size-checked" in r["error"]


# ---------------------------------------------------------------------------
# Query-cap: opening the Nth+1 query when cap is N is rejected
# ---------------------------------------------------------------------------


def test_query_cap_rejects_next_distinct_query(tmp_ctx):
    from citeclaw.agents.search_tools import register_worker_tools
    from citeclaw.agents.tool_dispatch import WorkerDispatcher

    ws = WorkerState(sub_topic_id="t", structural_priors=StructuralPriors())
    d = WorkerDispatcher(
        worker_state=ws, dataframe_store=DataFrameStore(),
        agent_config=AgentConfig(max_queries_per_worker=2),
        ctx=tmp_ctx, worker_id="w1",
    )
    register_worker_tools(d)
    # Open 2 distinct queries (hits the cap).
    d.dispatch("check_query_size", {"query": '"A"'})
    d.dispatch("check_query_size", {"query": '"B"'})
    assert len(ws.queries) == 2
    # 3rd distinct query — rejected.
    r3 = d.dispatch("check_query_size", {"query": '"C"'})
    assert "error" in r3
    assert "cap reached" in r3["error"]
    # Re-checking an existing query is still allowed (no new slot).
    r_dup = d.dispatch("check_query_size", {"query": '"A"'})
    assert "error" not in r_dup


# ---------------------------------------------------------------------------
# done before any fetch_results → rejected
# ---------------------------------------------------------------------------


def test_done_rejected_without_fetch(tmp_ctx):
    from citeclaw.agents.search_tools import register_worker_tools
    from citeclaw.agents.tool_dispatch import WorkerDispatcher

    ws = WorkerState(sub_topic_id="t", structural_priors=StructuralPriors())
    d = WorkerDispatcher(
        worker_state=ws, dataframe_store=DataFrameStore(),
        agent_config=AgentConfig(), ctx=tmp_ctx, worker_id="w1",
    )
    register_worker_tools(d)
    r = d.dispatch("done", {
        "paper_ids": [], "coverage_assessment": "acceptable", "summary": "x",
    })
    assert "error" in r
    assert "no fetch_results" in r["error"]


# ---------------------------------------------------------------------------
# Off-topic drift is not silently accepted — the call_log captures
# the query verbatim for postmortem.
# ---------------------------------------------------------------------------


def test_drift_is_logged(tmp_ctx):
    from citeclaw.agents.search_tools import register_worker_tools
    from citeclaw.agents.tool_dispatch import WorkerDispatcher

    ws = WorkerState(sub_topic_id="darts_core", structural_priors=StructuralPriors())
    d = WorkerDispatcher(
        worker_state=ws, dataframe_store=DataFrameStore(),
        agent_config=AgentConfig(), ctx=tmp_ctx, worker_id="w1",
    )
    register_worker_tools(d)
    d.dispatch("check_query_size", {"query": "protein structure prediction"})
    matches = [e for e in ws.call_log if e["tool"] == "check_query_size"]
    assert len(matches) == 1
    assert matches[0]["args"]["query"] == "protein structure prediction"


# ---------------------------------------------------------------------------
# Supervisor happy-path with enriched dispatch payload (queries list)
# ---------------------------------------------------------------------------


SUPERVISOR_SCRIPT = [
    # Supervisor turn 1: set_strategy.
    {"reasoning": "plan", "tool_name": "set_strategy",
     "tool_args": {
         "structural_priors": {"year_min": 2018, "fields_of_study": ["Computer Science"]},
         "sub_topics": [
             {
                 "id": "darts_core",
                 "description": "DARTS core papers",
                 "initial_query_sketch": '"differentiable architecture search"',
                 "reference_papers": [],
             },
         ],
     }},
    # Supervisor turn 2: dispatch.
    {"reasoning": "go", "tool_name": "dispatch_sub_topic_worker",
     "tool_args": {"spec_id": "darts_core"}},
    # ---- Worker script (3 turns: size, fetch, done — no refs so no gate) ----
    {"reasoning": "size", "tool_name": "check_query_size",
     "tool_args": {"query": '"differentiable architecture search"'}},
    {"reasoning": "fetch", "tool_name": "fetch_results",
     "tool_args": {"query": '"differentiable architecture search"'}},
    {"reasoning": "done", "tool_name": "done",
     "tool_args": {"paper_ids": [], "coverage_assessment": "acceptable",
                   "summary": "OK"}},
    # Supervisor turn 3: close.
    {"reasoning": "aggregate", "tool_name": "done",
     "tool_args": {"summary": "1 worker dispatched"}},
]


def test_supervisor_happy_path_returns_enriched_payload(tmp_ctx):
    llm = ScriptedLLM(SUPERVISOR_SCRIPT)
    sup_state, aggregate = run_supervisor(
        topic_description="Neural arch search.",
        filter_summary="(no filters)",
        seed_papers=[],
        llm_client=llm,
        ctx=tmp_ctx,
        agent_config=AgentConfig(worker_max_turns=15, supervisor_max_turns=10),
        logger=NullSearchLogger(),
    )
    assert len(sup_state.sub_topic_results) == 1
    r = sup_state.sub_topic_results[0]
    assert r.status == "success"
    # Dispatch payload carries per-query detail.
    dispatch_entries = [
        e for e in sup_state.call_log
        if e.get("tool") == "dispatch_sub_topic_worker" and not (
            isinstance(e.get("result"), dict) and "error" in e["result"]
        )
    ]
    assert len(dispatch_entries) == 1
    result = dispatch_entries[0]["result"]
    assert result["n_queries"] == 1
    assert len(result["queries"]) == 1
    assert result["queries"][0]["n_fetched"] == 120
    assert result["stop_reason"] == "called_done"
    assert result["auto_closed"] is False
    assert len(aggregate) == 120
