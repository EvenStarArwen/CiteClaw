"""End-to-end regression tests for v2 ExpandBySearch worker + supervisor.

Uses a scripted stub LLM that replays a canned tool-call sequence so
every hook / dispatcher path can be exercised offline — no S2 traffic,
no LLM traffic, deterministic. Covers:

- Happy path: full per-angle checklist + verify → done accepted.
- abandon_angle: worker drops an angle mid-inspection, opens a fresh
  one, completes it, done accepted.
- Angle transition with incomplete outgoing angle → rejection.
- fetch_results fingerprint mismatch → rejection.
- Angle-cap → rejection when opening the 5th angle.
- done before any fetch → rejection.
- Supervisor set_strategy → dispatch → done happy path with enriched
  payload (angles list, stop_reason).
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
)
from citeclaw.agents.supervisor import run_supervisor
from citeclaw.agents.worker import run_sub_topic_worker
from citeclaw.clients.llm.base import LLMResponse
from citeclaw.models import PaperRecord


# ---------------------------------------------------------------------------
# Scripted stub LLM
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """Replays a list of tool-call dicts. Substitutes ``__LAST_DF__`` in
    ``tool_args`` with the most recent df_id observed in the prior user
    message, so scripts don't need to hard-code auto-generated ids.
    """

    supports_logprobs = False

    def __init__(self, script: list[dict[str, Any]]):
        self._script = list(script)
        self._last_df_id: str | None = None
        self.calls: list[dict[str, Any]] = []

    def call(
        self, system: str, user: str, *,
        with_logprobs: bool = False, category: str = "other",
        response_schema: dict | None = None,
    ) -> LLMResponse:
        # Scrape df_id from the user message BEFORE substituting.
        m = re.search(r'"df_id":\s*"([^"]+)"', user)
        if m:
            self._last_df_id = m.group(1)
        self.calls.append({"user_len": len(user), "system_len": len(system)})
        if not self._script:
            raise AssertionError(
                f"script exhausted after {len(self.calls)} calls "
                f"(last system={system[:40]!r})"
            )
        item = dict(self._script.pop(0))
        if isinstance(item.get("tool_args"), dict):
            substituted = {}
            for k, v in item["tool_args"].items():
                if v == "__LAST_DF__":
                    substituted[k] = self._last_df_id or "missing"
                else:
                    substituted[k] = v
            item["tool_args"] = substituted
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
    """Tiny fake that returns canned papers for any query."""

    def __init__(self, *, total: int = 120, n_papers: int = 120):
        self._total = total
        self._n = n_papers

    def search_bulk(self, query: str, *, filters=None, sort=None, token=None, limit=1000):
        n = min(limit, self._n)
        # Deterministic paper_ids derived from query so different queries
        # produce different sets.
        prefix = f"paper_{abs(hash(query)) % 10_000:04d}"
        data = [{"paperId": f"{prefix}_{i:03d}", "title": f"Title {i}"} for i in range(n)]
        return {"total": self._total, "data": data, "token": None}

    def search_match(self, title: str):
        return {"paperId": "ref_001", "title": title, "year": 2023, "venue": "NeurIPS"}

    def enrich_batch(self, candidates):
        out = []
        for c in candidates:
            pid = c.get("paper_id", "")
            out.append(PaperRecord(
                paper_id=pid, title=f"T_{pid}", year=2022,
                venue="NeurIPS", citation_count=50, abstract="",
            ))
        return out

    def fetch_metadata(self, pid):
        return PaperRecord(paper_id=pid, title=f"T_{pid}", year=2023,
                           venue="X", citation_count=1, abstract="abs")

    def enrich_with_abstracts(self, recs):
        return recs

    def fetch_embeddings_batch(self, ids):
        return {pid: None for pid in ids}


@pytest.fixture
def tmp_ctx(tmp_path):
    return _Ctx(s2=_FakeS2(), data_dir=tmp_path)


def _spec():
    return SubTopicSpec(
        id="darts_core",
        description="DARTS core papers",
        initial_query_sketch='"differentiable architecture search"',
        reference_papers=("DARTS: Differentiable Architecture Search",),
    )


def _run_worker(*, script, ctx, priors=None, cfg=None):
    return run_sub_topic_worker(
        worker_id="w1",
        spec=_spec(),
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
# Happy path
# ---------------------------------------------------------------------------


HAPPY_SCRIPT = [
    {"reasoning": "size", "tool_name": "check_query_size",
     "tool_args": {"query": '"differentiable architecture search"'}},
    {"reasoning": "fetch", "tool_name": "fetch_results",
     "tool_args": {"query": '"differentiable architecture search"'}},
    {"reasoning": "top", "tool_name": "sample_titles",
     "tool_args": {"df_id": "__LAST_DF__", "strategy": "top_cited", "n": 10}},
    {"reasoning": "rand", "tool_name": "sample_titles",
     "tool_args": {"df_id": "__LAST_DF__", "strategy": "random", "n": 10}},
    {"reasoning": "yrs", "tool_name": "year_distribution",
     "tool_args": {"df_id": "__LAST_DF__"}},
    {"reasoning": "resolve ref", "tool_name": "search_match",
     "tool_args": {"title": "DARTS: Differentiable Architecture Search"}},
    {"reasoning": "check cum", "tool_name": "contains",
     "tool_args": {"paper_id": "ref_001"}},
    {"reasoning": "diagnose miss", "tool_name": "diagnose_miss",
     "tool_args": {"target_title": "DARTS", "hypotheses": ["S2 coverage"],
                   "action_taken": "accept_gap",
                   "query_angles_used": ['"differentiable architecture search"']}},
    {"reasoning": "wrap", "tool_name": "done",
     "tool_args": {"paper_ids": [], "coverage_assessment": "acceptable",
                   "summary": "covered via one angle"}},
]


def test_worker_happy_path(tmp_ctx):
    result = _run_worker(script=HAPPY_SCRIPT, ctx=tmp_ctx)
    assert result.status == "success"
    assert result.coverage_assessment == "acceptable"
    assert result.turns_used == 9
    assert len(result.query_angles) == 1
    assert result.query_angles[0].n_fetched == 120


# ---------------------------------------------------------------------------
# abandon_angle path
# ---------------------------------------------------------------------------


ABANDON_SCRIPT = [
    # Angle A: size + fetch + look at top-cited, see it's off-topic, abandon.
    {"reasoning": "size A", "tool_name": "check_query_size",
     "tool_args": {"query": '"architecture search"'}},
    {"reasoning": "fetch A", "tool_name": "fetch_results",
     "tool_args": {"query": '"architecture search"'}},
    {"reasoning": "sample A", "tool_name": "sample_titles",
     "tool_args": {"df_id": "__LAST_DF__", "strategy": "top_cited", "n": 5}},
    {"reasoning": "noisy, drop it", "tool_name": "abandon_angle",
     "tool_args": {}},
    # Angle B: full checklist.
    {"reasoning": "size B", "tool_name": "check_query_size",
     "tool_args": {"query": '"differentiable architecture search"'}},
    {"reasoning": "fetch B", "tool_name": "fetch_results",
     "tool_args": {"query": '"differentiable architecture search"'}},
    {"reasoning": "top B", "tool_name": "sample_titles",
     "tool_args": {"df_id": "__LAST_DF__", "strategy": "top_cited", "n": 10}},
    {"reasoning": "rand B", "tool_name": "sample_titles",
     "tool_args": {"df_id": "__LAST_DF__", "strategy": "random", "n": 10}},
    {"reasoning": "yrs B", "tool_name": "year_distribution",
     "tool_args": {"df_id": "__LAST_DF__"}},
    {"reasoning": "verify", "tool_name": "search_match",
     "tool_args": {"title": "DARTS"}},
    {"reasoning": "cum", "tool_name": "contains",
     "tool_args": {"paper_id": "ref_001"}},
    {"reasoning": "diag", "tool_name": "diagnose_miss",
     "tool_args": {"target_title": "DARTS", "hypotheses": ["x"],
                   "action_taken": "accept_gap", "query_angles_used": ["A", "B"]}},
    {"reasoning": "done", "tool_name": "done",
     "tool_args": {"paper_ids": [], "coverage_assessment": "comprehensive",
                   "summary": "abandoned A, B covered"}},
]


def test_worker_abandon_then_new_angle(tmp_ctx):
    result = _run_worker(script=ABANDON_SCRIPT, ctx=tmp_ctx)
    assert result.status == "success"
    assert result.coverage_assessment == "comprehensive"
    # Only the surviving angle remains in angles.
    assert len(result.query_angles) == 1
    assert result.query_angles[0].query == '"differentiable architecture search"'


# ---------------------------------------------------------------------------
# Angle transition blocked when outgoing checklist incomplete
# ---------------------------------------------------------------------------


TRANSITION_BLOCKED_SCRIPT = [
    {"reasoning": "size A", "tool_name": "check_query_size",
     "tool_args": {"query": "q_a"}},
    {"reasoning": "fetch A", "tool_name": "fetch_results",
     "tool_args": {"query": "q_a"}},
    # Try to skip inspection and open a new angle — dispatcher rejects.
    {"reasoning": "skip to B", "tool_name": "check_query_size",
     "tool_args": {"query": "q_b"}},
    # Model gets the error, recovers: sample A first.
    {"reasoning": "top A", "tool_name": "sample_titles",
     "tool_args": {"df_id": "__LAST_DF__", "strategy": "top_cited", "n": 5}},
    {"reasoning": "rand A", "tool_name": "sample_titles",
     "tool_args": {"df_id": "__LAST_DF__", "strategy": "random", "n": 5}},
    {"reasoning": "yrs A", "tool_name": "year_distribution",
     "tool_args": {"df_id": "__LAST_DF__"}},
    # NOW the transition is allowed.
    {"reasoning": "size B", "tool_name": "check_query_size",
     "tool_args": {"query": "q_b"}},
    {"reasoning": "bail out", "tool_name": "abandon_angle", "tool_args": {}},
    {"reasoning": "verify", "tool_name": "search_match",
     "tool_args": {"title": "DARTS"}},
    {"reasoning": "cum", "tool_name": "contains",
     "tool_args": {"paper_id": "ref_001"}},
    {"reasoning": "diag", "tool_name": "diagnose_miss",
     "tool_args": {"target_title": "DARTS", "hypotheses": ["x"],
                   "action_taken": "accept_gap", "query_angles_used": ["q_a", "q_b"]}},
    {"reasoning": "done", "tool_name": "done",
     "tool_args": {"paper_ids": [], "coverage_assessment": "limited",
                   "summary": "B abandoned; A covered"}},
]


def test_angle_transition_blocked_until_checklist(tmp_ctx):
    result = _run_worker(script=TRANSITION_BLOCKED_SCRIPT, ctx=tmp_ctx)
    assert result.status == "success"
    # Angle B was abandoned so only A survives in the result.
    assert len(result.query_angles) == 1
    assert result.query_angles[0].query == "q_a"


# ---------------------------------------------------------------------------
# fetch_results fingerprint mismatch
# ---------------------------------------------------------------------------


MISMATCH_SCRIPT = [
    {"reasoning": "size q1", "tool_name": "check_query_size",
     "tool_args": {"query": "q_one"}},
    # Try to fetch a totally different query — rejected.
    {"reasoning": "wrong fetch", "tool_name": "fetch_results",
     "tool_args": {"query": "q_totally_different"}},
    {"reasoning": "recover", "tool_name": "fetch_results",
     "tool_args": {"query": "q_one"}},
    {"reasoning": "top", "tool_name": "sample_titles",
     "tool_args": {"df_id": "__LAST_DF__", "strategy": "top_cited", "n": 5}},
    {"reasoning": "rand", "tool_name": "sample_titles",
     "tool_args": {"df_id": "__LAST_DF__", "strategy": "random", "n": 5}},
    {"reasoning": "yrs", "tool_name": "year_distribution",
     "tool_args": {"df_id": "__LAST_DF__"}},
    {"reasoning": "verify", "tool_name": "search_match",
     "tool_args": {"title": "Target"}},
    {"reasoning": "cum", "tool_name": "contains",
     "tool_args": {"paper_id": "ref_001"}},
    {"reasoning": "diag", "tool_name": "diagnose_miss",
     "tool_args": {"target_title": "Target", "hypotheses": ["x"],
                   "action_taken": "accept_gap", "query_angles_used": ["q_one"]}},
    {"reasoning": "done", "tool_name": "done",
     "tool_args": {"paper_ids": [], "coverage_assessment": "acceptable",
                   "summary": "done"}},
]


def test_fingerprint_mismatch_rejected_but_recoverable(tmp_ctx):
    # Use a ScriptedLLM directly so we can inspect call_log for the error.
    from citeclaw.agents.search_tools import register_worker_tools
    from citeclaw.agents.tool_dispatch import WorkerDispatcher
    from citeclaw.agents.state import WorkerState

    ws = WorkerState(sub_topic_id="t", structural_priors=StructuralPriors())
    d = WorkerDispatcher(
        worker_state=ws, dataframe_store=DataFrameStore(),
        agent_config=AgentConfig(), ctx=tmp_ctx, worker_id="w1",
    )
    register_worker_tools(d)
    # Simulate just the size + bad fetch sequence via dispatch directly.
    d.dispatch("check_query_size", {"query": "q_one"})
    bad = d.dispatch("fetch_results", {"query": "q_totally_different"})
    assert "error" in bad
    assert "not size-checked" in bad["error"]
    good = d.dispatch("fetch_results", {"query": "q_one"})
    assert "error" not in good
    assert "df_id" in good


# ---------------------------------------------------------------------------
# Angle-cap enforcement
# ---------------------------------------------------------------------------


def test_angle_cap_rejects_fifth_angle(tmp_ctx):
    from citeclaw.agents.search_tools import register_worker_tools
    from citeclaw.agents.tool_dispatch import WorkerDispatcher
    from citeclaw.agents.state import WorkerState

    ws = WorkerState(sub_topic_id="t", structural_priors=StructuralPriors())
    d = WorkerDispatcher(
        worker_state=ws, dataframe_store=DataFrameStore(),
        agent_config=AgentConfig(max_angles_per_worker=4),
        ctx=tmp_ctx, worker_id="w1",
    )
    register_worker_tools(d)
    for i in range(4):
        r = d.dispatch("check_query_size", {"query": f"q_{i}"})
        # Since checklist on each is incomplete when we switch, the
        # second+ transitions are rejected. Abandon to keep going.
        if "error" in r:
            d.dispatch("abandon_angle", {})
            r = d.dispatch("check_query_size", {"query": f"q_{i}"})
        assert "error" not in r
        d.dispatch("abandon_angle", {})
    # After abandoning four angles, angles dict is empty -> cap not hit.
    assert len(ws.angles) == 0
    # But if we keep them around:
    d.dispatch("check_query_size", {"query": "q_a"})
    d.dispatch("fetch_results", {"query": "q_a"})
    d.dispatch("sample_titles", {"df_id": ws.active_angle.df_id, "strategy": "top_cited", "n": 5})
    d.dispatch("sample_titles", {"df_id": ws.active_angle.df_id, "strategy": "random", "n": 5})
    d.dispatch("year_distribution", {"df_id": ws.active_angle.df_id})
    d.dispatch("check_query_size", {"query": "q_b"})
    d.dispatch("fetch_results", {"query": "q_b"})
    d.dispatch("sample_titles", {"df_id": ws.active_angle.df_id, "strategy": "top_cited", "n": 5})
    d.dispatch("sample_titles", {"df_id": ws.active_angle.df_id, "strategy": "random", "n": 5})
    d.dispatch("year_distribution", {"df_id": ws.active_angle.df_id})
    d.dispatch("check_query_size", {"query": "q_c"})
    d.dispatch("fetch_results", {"query": "q_c"})
    d.dispatch("sample_titles", {"df_id": ws.active_angle.df_id, "strategy": "top_cited", "n": 5})
    d.dispatch("sample_titles", {"df_id": ws.active_angle.df_id, "strategy": "random", "n": 5})
    d.dispatch("year_distribution", {"df_id": ws.active_angle.df_id})
    d.dispatch("check_query_size", {"query": "q_d"})
    # Now 4 angles exist (a, b, c, d); opening a fifth should reject.
    cap_err = d.dispatch("check_query_size", {"query": "q_e"})
    assert "error" in cap_err
    assert "angle cap" in cap_err["error"].lower()


# ---------------------------------------------------------------------------
# done rejection (no fetch_results)
# ---------------------------------------------------------------------------


def test_done_rejected_without_fetch(tmp_ctx):
    from citeclaw.agents.search_tools import register_worker_tools
    from citeclaw.agents.tool_dispatch import WorkerDispatcher
    from citeclaw.agents.state import WorkerState

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
# Topic drift regression — worker trying to query off-topic does NOT
# silently drift. The dispatcher accepts any query text the model sends
# (it's the prompt's job to keep the model on-topic), but off-topic
# queries remain traceable in the event log. This test documents the
# contract: the agent CAN call check_query_size with any query; the
# logging captures it for postmortem even if the model misbehaves.
# ---------------------------------------------------------------------------


def test_drift_is_logged_not_silently_accepted(tmp_ctx):
    from citeclaw.agents.search_tools import register_worker_tools
    from citeclaw.agents.tool_dispatch import WorkerDispatcher
    from citeclaw.agents.state import WorkerState

    ws = WorkerState(sub_topic_id="darts_core", structural_priors=StructuralPriors())
    d = WorkerDispatcher(
        worker_state=ws, dataframe_store=DataFrameStore(),
        agent_config=AgentConfig(), ctx=tmp_ctx, worker_id="w1",
    )
    register_worker_tools(d)
    # Model drifts — calls check_query_size with "protein structure prediction".
    d.dispatch("check_query_size", {"query": "protein structure prediction"})
    # Recorded in call_log with full query text visible.
    matches = [e for e in ws.call_log if e["tool"] == "check_query_size"]
    assert len(matches) == 1
    assert matches[0]["args"]["query"] == "protein structure prediction"


# ---------------------------------------------------------------------------
# Supervisor happy-path with enriched dispatch payload
# ---------------------------------------------------------------------------


SUPERVISOR_SCRIPT_AND_WORKER = [
    # Supervisor turn 1: set_strategy.
    {"reasoning": "plan", "tool_name": "set_strategy",
     "tool_args": {
         "structural_priors": {"year_min": 2018, "fields_of_study": ["Computer Science"]},
         "sub_topics": [
             {
                 "id": "darts_core",
                 "description": "DARTS core papers",
                 "initial_query_sketch": '"differentiable architecture search"',
                 "reference_papers": ["DARTS"],
             },
         ],
     }},
    # Supervisor turn 2: dispatch the worker.
    {"reasoning": "go", "tool_name": "dispatch_sub_topic_worker",
     "tool_args": {"spec_id": "darts_core"}},
    # ---- Worker script for spec darts_core ----
    {"reasoning": "size", "tool_name": "check_query_size",
     "tool_args": {"query": '"differentiable architecture search"'}},
    {"reasoning": "fetch", "tool_name": "fetch_results",
     "tool_args": {"query": '"differentiable architecture search"'}},
    {"reasoning": "top", "tool_name": "sample_titles",
     "tool_args": {"df_id": "__LAST_DF__", "strategy": "top_cited", "n": 5}},
    {"reasoning": "rand", "tool_name": "sample_titles",
     "tool_args": {"df_id": "__LAST_DF__", "strategy": "random", "n": 5}},
    {"reasoning": "yrs", "tool_name": "year_distribution",
     "tool_args": {"df_id": "__LAST_DF__"}},
    {"reasoning": "sm", "tool_name": "search_match",
     "tool_args": {"title": "DARTS"}},
    {"reasoning": "cum", "tool_name": "contains",
     "tool_args": {"paper_id": "ref_001"}},
    {"reasoning": "diag", "tool_name": "diagnose_miss",
     "tool_args": {"target_title": "DARTS", "hypotheses": ["x"],
                   "action_taken": "accept_gap", "query_angles_used": ["q"]}},
    {"reasoning": "worker done", "tool_name": "done",
     "tool_args": {"paper_ids": [], "coverage_assessment": "acceptable",
                   "summary": "OK"}},
    # Supervisor turn 3: close.
    {"reasoning": "aggregate", "tool_name": "done",
     "tool_args": {"summary": "1 worker dispatched, 120 papers accumulated"}},
]


def test_supervisor_happy_path_returns_enriched_payload(tmp_ctx):
    llm = ScriptedLLM(SUPERVISOR_SCRIPT_AND_WORKER)
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
    assert r.coverage_assessment == "acceptable"
    # Enriched payload: the supervisor's dispatch tool result should
    # carry angles + stop_reason. Check via the supervisor's call_log.
    dispatch_entries = [
        e for e in sup_state.call_log
        if e.get("tool") == "dispatch_sub_topic_worker" and not (
            isinstance(e.get("result"), dict) and "error" in e["result"]
        )
    ]
    assert len(dispatch_entries) == 1
    result = dispatch_entries[0]["result"]
    assert "angles" in result
    assert len(result["angles"]) == 1
    assert result["angles"][0]["n_fetched"] == 120
    assert "stop_reason" in result
    assert result["stop_reason"] == "called_done"
    assert len(aggregate) == 120
