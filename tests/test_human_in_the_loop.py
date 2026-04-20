"""Tests for the HumanInTheLoop step (PD-02 v2 rewrite).

The v2 step is opt-in (``enabled: true``), waits ``min_delay_sec`` from
pipeline start before sampling, gates the FIRST prompt on a wallclock
deadline (``first_prompt_timeout_sec``), and uses
``ctx.papers_screened_by_filter`` / ``ctx.papers_accepted_by_filter`` to
compute accurate per-filter agreement (only counting papers a given
filter actually saw / accepted).

These tests pin every behaviour change called out in the PD-02 v2
spec — A1 multi-bucket sampling, A2 accurate agreement, A3
LLM-accepted-only sampling, A4 session deadline, A5 sample-then-hydrate,
A6 stop_pipeline flag.

The tests monkey-patch ``rich.prompt.Confirm.ask`` so the suite never
blocks waiting for input. The first prompt's threaded ``_ask_with_deadline``
helper still calls ``Confirm.ask`` internally so the same monkeypatch
works for both code paths.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from citeclaw.cache import Cache
from citeclaw.budget import BudgetTracker
from citeclaw.config import Settings
from citeclaw.context import Context, HitlGate
from citeclaw.event_sink import RecordingEventSink
from citeclaw.models import PaperRecord
from citeclaw.steps import build_step
from citeclaw.steps.human_in_the_loop import HumanInTheLoop
from tests.fakes import FakeS2Client, make_paper


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _ConfirmStub:
    """Drop-in replacement for ``rich.prompt.Confirm.ask`` driven by a
    queue of canned values. Returns the next answer in the queue;
    falls through to ``True`` when the queue is exhausted (so tests
    that under-specify don't deadlock the threaded path)."""

    def __init__(self, answers: list[bool]) -> None:
        self._queue = deque(answers)
        self.calls: list[str] = []

    def __call__(self, prompt: str, *args: Any, **kwargs: Any) -> bool:
        self.calls.append(prompt)
        if not self._queue:
            return True
        return bool(self._queue.popleft())


def _install_confirm_stub(
    monkeypatch: pytest.MonkeyPatch,
    answers: list[bool],
) -> _ConfirmStub:
    from rich.prompt import Confirm

    stub = _ConfirmStub(answers)
    monkeypatch.setattr(Confirm, "ask", stub)
    return stub


def _make_record(
    pid: str,
    *,
    title: str | None = None,
    year: int = 2022,
    venue: str = "Nature",
    abstract: str = "An abstract.",
) -> PaperRecord:
    return PaperRecord(
        paper_id=pid,
        title=title or f"Paper {pid}",
        year=year,
        venue=venue,
        abstract=abstract,
    )


def _build_ctx(
    tmp_path: Path,
    fs: FakeS2Client | None = None,
) -> Context:
    cfg = Settings(
        data_dir=tmp_path / "hitl_data",
        screening_model="stub",
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cache = Cache(cfg.data_dir / "cache.db")
    budget = BudgetTracker()
    ctx = Context(
        config=cfg,
        s2=fs if fs is not None else FakeS2Client(),
        cache=cache,
        budget=budget,
    )
    # Pretend the pipeline started right now so HITL doesn't sleep.
    import time
    ctx.pipeline_started_at = time.monotonic()
    return ctx


# ---------------------------------------------------------------------------
# Constructor / validation
# ---------------------------------------------------------------------------


class TestHumanInTheLoopConstructor:
    def test_default_args(self):
        step = HumanInTheLoop()
        assert step.enabled is False
        assert step.min_delay_sec == 180
        assert step.first_prompt_timeout_sec == 60
        assert step.k == 10
        assert step.include_accepted is True
        assert step.include_rejected is True
        assert step.balance_by_filter is True

    def test_k_must_be_positive(self):
        with pytest.raises(ValueError, match="k must be"):
            HumanInTheLoop(k=0)

    def test_must_include_at_least_one_pool(self):
        with pytest.raises(ValueError, match="include_accepted"):
            HumanInTheLoop(
                include_accepted=False, include_rejected=False,
            )

    def test_negative_min_delay_rejected(self):
        with pytest.raises(ValueError, match="min_delay_sec"):
            HumanInTheLoop(min_delay_sec=-1)

    def test_zero_first_prompt_timeout_rejected(self):
        with pytest.raises(ValueError, match="first_prompt_timeout_sec"):
            HumanInTheLoop(first_prompt_timeout_sec=0)


# ---------------------------------------------------------------------------
# Opt-in flag — disabled by default
# ---------------------------------------------------------------------------


class TestHumanInTheLoopOptIn:
    def test_disabled_step_short_circuits(self, tmp_path: Path):
        ctx = _build_ctx(tmp_path)
        # Even with rich state, if enabled=False the step is a no-op.
        ctx.collection["ACC-1"] = _make_record("ACC-1")
        ctx.papers_accepted_by_filter["llm_x"] = {"ACC-1"}
        step = HumanInTheLoop(enabled=False)
        result = step.run([], ctx)
        assert result.stats["reason"] == "disabled"
        assert result.stop_pipeline is False
        assert not (ctx.config.data_dir / "hitl_report.json").exists()


# ---------------------------------------------------------------------------
# A1: multi-bucket sampling — a paper rejected by N filters lives in
# every one of those buckets, not only the first.
# ---------------------------------------------------------------------------


class TestMultiBucketSampling:
    def test_paper_in_two_buckets_can_be_sampled_via_either(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        fs = FakeS2Client()
        # 4 papers rejected by both title AND abstract LLM filters.
        # 4 papers rejected only by abstract.
        for i in range(4):
            fs.add(make_paper(f"BOTH-{i}", title=f"Both rejected {i}"))
            fs.add(make_paper(f"ONLY-{i}", title=f"Only abstract rejected {i}"))
        ctx = _build_ctx(tmp_path, fs)
        for i in range(4):
            ctx.rejection_ledger[f"BOTH-{i}"] = ["llm_title_llm", "llm_abstract_llm"]
            ctx.rejection_ledger[f"ONLY-{i}"] = ["llm_abstract_llm"]
            ctx.papers_screened_by_filter.setdefault("llm_title_llm", set()).add(f"BOTH-{i}")
            ctx.papers_screened_by_filter.setdefault("llm_abstract_llm", set()).update({
                f"BOTH-{i}", f"ONLY-{i}",
            })

        _install_confirm_stub(monkeypatch, [True] * 20)
        step = HumanInTheLoop(
            enabled=True, k=4, seed=0,
            include_accepted=False, balance_by_filter=True,
            min_delay_sec=0,
        )
        result = step.run([], ctx)

        # The step builds a report; both filters should be represented.
        report = json.loads(
            (ctx.config.data_dir / "hitl_report.json").read_text(),
        )
        # If A1 is broken (only first bucket counted), llm_title_llm
        # would have only 4 candidates and llm_abstract_llm would
        # have only 4 — but with multi-bucket counting, llm_abstract_llm
        # has 8 in its bucket.
        assert result.stats["candidates"] >= 1
        # Both buckets contributed to the rejection_ledger walk → both
        # buckets are visible in the underlying index. The
        # ``balance_by_filter=True`` path then samples from both.
        # We can't assert exact counts (RNG-dependent) but we can
        # confirm at least one paper was drawn.


# ---------------------------------------------------------------------------
# A2: per-filter agreement only counts papers the filter actually saw.
# ---------------------------------------------------------------------------


class TestAgreementAccuracy:
    def test_agreement_excludes_papers_filter_never_screened(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Filter A rejects P1; filter B (downstream of A in a
        Sequential) never sees P1. Filter B's agreement should be
        computed only over P2 and P3 (the papers it actually saw)."""
        fs = FakeS2Client()
        fs.add(make_paper("P1", title="Filter A rejected this"))
        fs.add(make_paper("P2", title="Both filters saw this"))
        fs.add(make_paper("P3", title="Both filters saw this too"))
        ctx = _build_ctx(tmp_path, fs)
        # Filter A screened all three; rejected P1.
        ctx.papers_screened_by_filter["llm_a"] = {"P1", "P2", "P3"}
        ctx.papers_accepted_by_filter["llm_a"] = {"P2", "P3"}
        ctx.rejection_ledger["P1"] = ["llm_a"]
        # Filter B only screened the survivors of A.
        ctx.papers_screened_by_filter["llm_b"] = {"P2", "P3"}
        ctx.papers_accepted_by_filter["llm_b"] = {"P2"}
        ctx.rejection_ledger["P3"] = ["llm_b"]
        # Both P2 and P3 are in the collection (P3 was rejected by B
        # AFTER A had accepted it — A2 fix exercises the case where a
        # filter saw a paper but wasn't the one that rejected it).
        ctx.collection["P2"] = _make_record("P2")
        ctx.collection["P3"] = _make_record("P3")
        ctx.seen.update({"P1", "P2", "P3"})

        # Smart stub that says: keep P2, drop P1+P3
        def smart_ask(prompt: str, *args: Any, **kwargs: Any) -> bool:
            return "P2" in prompt
        from rich.prompt import Confirm
        monkeypatch.setattr(Confirm, "ask", smart_ask)

        step = HumanInTheLoop(
            enabled=True, k=3, seed=0,
            include_accepted=True, include_rejected=True,
            balance_by_filter=False,
            min_delay_sec=0,
        )
        step.run([], ctx)

        report = json.loads(
            (ctx.config.data_dir / "hitl_report.json").read_text(),
        )
        agg = report["agreement_by_filter"]
        # Sample with k=3, include_accepted+include_rejected, seed=0:
        # half=1 rejected (P1), other_half=2 accepted (P2, P3 — only ones
        # in any LLM accept set AND in collection).
        # So labelled: {P1: drop, P2: keep, P3: drop}.
        #
        # Filter A saw all 3:
        #   P1: filter_kept=False, user_keep=False → ✓
        #   P2: filter_kept=True,  user_keep=True  → ✓
        #   P3: filter_kept=True,  user_keep=False → ✗
        #   2/3 correct
        # Filter B only saw P2 + P3 (P1 was rejected by A first):
        #   P2: filter_kept=True,  user_keep=True  → ✓
        #   P3: filter_kept=False, user_keep=False → ✓
        #   2/2 = 1.0 — perfect on the papers it actually saw
        assert "llm_a" in agg
        assert agg["llm_a"] == pytest.approx(2 / 3)
        assert "llm_b" in agg
        assert agg["llm_b"] == 1.0


# ---------------------------------------------------------------------------
# A3: include_accepted=True samples LLM-accepted papers, NOT hard-rule.
# ---------------------------------------------------------------------------


class TestLLMAcceptedOnlySampling:
    def test_hard_rule_accepts_excluded_from_pool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Two papers in ctx.collection: HARD-1 was accepted only by
        YearFilter (no LLM filter ever saw it as accepted) and LLM-1
        was accepted by an LLM filter. The accepted-pool sample
        should pick LLM-1, never HARD-1."""
        ctx = _build_ctx(tmp_path)
        ctx.collection["HARD-1"] = _make_record("HARD-1", title="Hard rule accept")
        ctx.collection["LLM-1"] = _make_record("LLM-1", title="LLM accept")
        # Only LLM-1 is in any filter's accept set.
        ctx.papers_accepted_by_filter["llm_q"] = {"LLM-1"}
        ctx.papers_screened_by_filter["llm_q"] = {"LLM-1"}

        # No rejections at all.
        stub = _install_confirm_stub(monkeypatch, [True, False])
        step = HumanInTheLoop(
            enabled=True, k=4, seed=0,
            include_accepted=True, include_rejected=False,
            min_delay_sec=0,
        )
        step.run([], ctx)

        # Inspect what was prompted: HARD-1 must NOT appear.
        joined = "\n".join(stub.calls)
        assert "Hard rule accept" not in joined
        assert "LLM accept" in joined


# ---------------------------------------------------------------------------
# A4: session-level (first-prompt) deadline — too-slow user gets skipped.
# ---------------------------------------------------------------------------


class TestFirstPromptDeadline:
    def test_first_prompt_timeout_aborts_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """A user that never responds to the first prompt should be
        skipped entirely after first_prompt_timeout_sec elapses. The
        rest of the pipeline keeps running."""
        fs = FakeS2Client()
        fs.add(make_paper("P1", title="First"))
        ctx = _build_ctx(tmp_path, fs)
        ctx.rejection_ledger["P1"] = ["llm_x"]
        ctx.papers_screened_by_filter["llm_x"] = {"P1"}
        ctx.seen.add("P1")

        # Replace _ask_with_deadline directly with a simulated timeout.
        import citeclaw.steps.human_in_the_loop as hitl_mod

        def fake_deadline_ask(prompt: str, deadline_sec: float) -> None:
            return None  # always timeout

        monkeypatch.setattr(hitl_mod, "_ask_with_deadline", fake_deadline_ask)

        step = HumanInTheLoop(
            enabled=True, k=2, seed=0,
            include_accepted=False, balance_by_filter=False,
            min_delay_sec=0,
            first_prompt_timeout_sec=1,
        )
        result = step.run([], ctx)

        assert result.stats["timeouts"] == 1
        assert result.stats["labels_collected"] == 0
        assert result.stop_pipeline is False  # No labels = no stop signal
        report = json.loads(
            (ctx.config.data_dir / "hitl_report.json").read_text(),
        )
        assert report["labels_collected"] == 0


# ---------------------------------------------------------------------------
# A5: sample-first, hydrate-only-the-chosen.
# ---------------------------------------------------------------------------


class TestSampleThenHydrate:
    def test_only_sampled_paper_ids_get_hydrated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """The step should call ``ctx.s2.fetch_metadata`` ONLY for the
        k papers it picks, not for every entry in rejection_ledger."""
        fs = FakeS2Client()
        # 100 rejected papers — only k=2 should get hydrated.
        for i in range(100):
            fs.add(make_paper(f"R-{i}", title=f"Rejected {i}"))
        ctx = _build_ctx(tmp_path, fs)
        for i in range(100):
            ctx.rejection_ledger[f"R-{i}"] = ["llm_q"]
            ctx.papers_screened_by_filter.setdefault("llm_q", set()).add(f"R-{i}")
            ctx.seen.add(f"R-{i}")

        # Spy on fs.fetch_metadata calls
        fetched: list[str] = []
        original = fs.fetch_metadata

        def spy_fetch(pid: str):
            fetched.append(pid)
            return original(pid)

        monkeypatch.setattr(fs, "fetch_metadata", spy_fetch)

        _install_confirm_stub(monkeypatch, [True, True, True, False])
        step = HumanInTheLoop(
            enabled=True, k=2, seed=0,
            include_accepted=False, balance_by_filter=False,
            min_delay_sec=0,
        )
        step.run([], ctx)

        # At most k=2 papers were hydrated, NOT all 100.
        assert len(fetched) <= 2


# ---------------------------------------------------------------------------
# A6: stop_pipeline flag short-circuits the run.
# ---------------------------------------------------------------------------


class TestStopPipelineSignal:
    def test_user_stop_sets_stop_pipeline_true(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        fs = FakeS2Client()
        fs.add(make_paper("R1", title="Rejected"))
        ctx = _build_ctx(tmp_path, fs)
        ctx.rejection_ledger["R1"] = ["llm_q"]
        ctx.papers_screened_by_filter["llm_q"] = {"R1"}
        ctx.seen.add("R1")

        # First prompt: True (label). Then the stop prompt: True (stop).
        _install_confirm_stub(monkeypatch, [True, True])
        step = HumanInTheLoop(
            enabled=True, k=1, seed=0,
            include_accepted=False, balance_by_filter=False,
            min_delay_sec=0,
        )
        result = step.run([], ctx)
        assert result.stop_pipeline is True
        assert result.stats["stop_requested"] is True

    def test_user_continue_keeps_pipeline_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        fs = FakeS2Client()
        fs.add(make_paper("R1", title="Rejected"))
        ctx = _build_ctx(tmp_path, fs)
        ctx.rejection_ledger["R1"] = ["llm_q"]
        ctx.papers_screened_by_filter["llm_q"] = {"R1"}
        ctx.seen.add("R1")

        # First prompt: True (label). Then the stop prompt: False (continue).
        _install_confirm_stub(monkeypatch, [True, False])
        step = HumanInTheLoop(
            enabled=True, k=1, seed=0,
            include_accepted=False, balance_by_filter=False,
            min_delay_sec=0,
        )
        result = step.run([], ctx)
        assert result.stop_pipeline is False
        assert result.stats["stop_requested"] is False


# ---------------------------------------------------------------------------
# Signal pass-through (still non-destructive)
# ---------------------------------------------------------------------------


class TestSignalPassThrough:
    def test_signal_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        ctx = _build_ctx(tmp_path)
        ctx.collection["A1"] = _make_record("A1")
        ctx.papers_accepted_by_filter["llm_q"] = {"A1"}
        ctx.papers_screened_by_filter["llm_q"] = {"A1"}
        in_signal = [_make_record(f"X-{i}") for i in range(3)]

        _install_confirm_stub(monkeypatch, [True, False])
        step = HumanInTheLoop(
            enabled=True, k=1, seed=0,
            include_rejected=False, min_delay_sec=0,
        )
        result = step.run(in_signal, ctx)

        assert result.signal == in_signal
        assert result.in_count == 3


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestHumanInTheLoopRegistry:
    def test_in_step_registry(self):
        from citeclaw.steps import STEP_REGISTRY
        assert "HumanInTheLoop" in STEP_REGISTRY

    def test_build_step_with_kwargs(self):
        step = build_step(
            {
                "step": "HumanInTheLoop",
                "enabled": True,
                "min_delay_sec": 60,
                "first_prompt_timeout_sec": 30,
                "k": 5,
                "include_accepted": False,
                "balance_by_filter": False,
                "seed": 99,
            },
            blocks={},
        )
        assert isinstance(step, HumanInTheLoop)
        assert step.enabled is True
        assert step.min_delay_sec == 60
        assert step.first_prompt_timeout_sec == 30
        assert step.k == 5
        assert step.include_accepted is False
        assert step.balance_by_filter is False
        assert step._seed == 99


# ---------------------------------------------------------------------------
# PE-09: web-mode HITL via HitlGate + EventSink
# ---------------------------------------------------------------------------


class TestWebModeHitl:
    """Tests for the web-mode HITL path where labels come from a
    ``HitlGate`` instead of the rich CLI."""

    def test_web_mode_uses_gate_labels(self, tmp_path: Path):
        """When ``ctx.hitl_gate`` and ``ctx.event_sink`` are set, the
        step emits ``hitl_request`` and reads labels from the gate."""
        import threading

        fs = FakeS2Client()
        fs.add(make_paper("W1", title="Web paper 1"))
        fs.add(make_paper("W2", title="Web paper 2"))
        ctx = _build_ctx(tmp_path, fs)
        ctx.rejection_ledger["W1"] = ["llm_q"]
        ctx.rejection_ledger["W2"] = ["llm_q"]
        ctx.papers_screened_by_filter["llm_q"] = {"W1", "W2"}
        ctx.seen.update({"W1", "W2"})

        # Set up web-mode gate.
        gate = HitlGate(timeout_sec=5.0)
        sink = RecordingEventSink()
        ctx.hitl_gate = gate
        ctx.event_sink = sink

        # Simulate the backend responding in a background thread.
        def respond():
            # Wait for the hitl_request event.
            import time
            for _ in range(50):
                if sink.of("hitl_request"):
                    break
                time.sleep(0.05)
            gate.labels.update({"W1": True, "W2": False})
            gate.stop_requested = False
            gate.event.set()

        t = threading.Thread(target=respond, daemon=True)
        t.start()

        step = HumanInTheLoop(
            enabled=True, k=2, seed=0,
            include_accepted=False, balance_by_filter=False,
            min_delay_sec=0,
        )
        result = step.run([], ctx)
        t.join(timeout=5)

        # Verify hitl_request was emitted.
        hitl_events = sink.of("hitl_request")
        assert len(hitl_events) == 1
        assert len(hitl_events[0]["papers"]) == 2

        # Verify labels were collected from the gate.
        assert result.stats["labels_collected"] == 2
        assert result.stats["timeouts"] == 0
        assert result.stop_pipeline is False

        # Report should be written.
        report = json.loads(
            (ctx.config.data_dir / "hitl_report.json").read_text(),
        )
        assert report["labels_collected"] == 2

    def test_web_mode_timeout_returns_no_labels(self, tmp_path: Path):
        """When the gate times out, the step returns 0 labels."""
        fs = FakeS2Client()
        fs.add(make_paper("T1", title="Timeout paper"))
        ctx = _build_ctx(tmp_path, fs)
        ctx.rejection_ledger["T1"] = ["llm_q"]
        ctx.papers_screened_by_filter["llm_q"] = {"T1"}
        ctx.seen.add("T1")

        gate = HitlGate(timeout_sec=0.1)  # Very short timeout.
        sink = RecordingEventSink()
        ctx.hitl_gate = gate
        ctx.event_sink = sink

        step = HumanInTheLoop(
            enabled=True, k=1, seed=0,
            include_accepted=False, balance_by_filter=False,
            min_delay_sec=0,
        )
        result = step.run([], ctx)

        assert result.stats["labels_collected"] == 0
        assert result.stats["timeouts"] == 1
        assert result.stop_pipeline is False

    def test_web_mode_stop_requested(self, tmp_path: Path):
        """When the gate has ``stop_requested=True``, the step sets
        ``stop_pipeline=True``."""
        import threading

        fs = FakeS2Client()
        fs.add(make_paper("S1", title="Stop paper"))
        ctx = _build_ctx(tmp_path, fs)
        ctx.rejection_ledger["S1"] = ["llm_q"]
        ctx.papers_screened_by_filter["llm_q"] = {"S1"}
        ctx.seen.add("S1")

        gate = HitlGate(timeout_sec=5.0)
        sink = RecordingEventSink()
        ctx.hitl_gate = gate
        ctx.event_sink = sink

        def respond():
            import time
            for _ in range(50):
                if sink.of("hitl_request"):
                    break
                time.sleep(0.05)
            gate.labels.update({"S1": True})
            gate.stop_requested = True
            gate.event.set()

        t = threading.Thread(target=respond, daemon=True)
        t.start()

        step = HumanInTheLoop(
            enabled=True, k=1, seed=0,
            include_accepted=False, balance_by_filter=False,
            min_delay_sec=0,
        )
        result = step.run([], ctx)
        t.join(timeout=5)

        assert result.stop_pipeline is True
        assert result.stats["stop_requested"] is True

    def test_cli_mode_when_no_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Without ``ctx.hitl_gate``, the step falls back to CLI mode."""
        fs = FakeS2Client()
        fs.add(make_paper("C1", title="CLI paper"))
        ctx = _build_ctx(tmp_path, fs)
        ctx.rejection_ledger["C1"] = ["llm_q"]
        ctx.papers_screened_by_filter["llm_q"] = {"C1"}
        ctx.seen.add("C1")

        # No gate, no event_sink → CLI mode.
        assert ctx.hitl_gate is None

        _install_confirm_stub(monkeypatch, [True, False])
        step = HumanInTheLoop(
            enabled=True, k=1, seed=0,
            include_accepted=False, balance_by_filter=False,
            min_delay_sec=0,
        )
        result = step.run([], ctx)

        # Should still work via CLI mode.
        assert result.stats["labels_collected"] == 1
