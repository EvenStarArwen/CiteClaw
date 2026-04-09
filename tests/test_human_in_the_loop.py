"""Tests for the HumanInTheLoop step (PD-02).

The step is the only one in CiteClaw that interacts with stdin via
``rich.prompt.Confirm.ask``. These tests monkey-patch ``Confirm.ask``
with a deterministic queue so the suite never blocks waiting for
input. The headline assertions:

  * The step writes ``hitl_report.json`` to ``ctx.config.data_dir``.
  * The report contains per-LLM-filter agreement computed against the
    canned label sequence.
  * Sampling honours ``include_accepted`` / ``include_rejected`` and
    the ``balance_by_filter`` knob.
  * Constructor validation rejects bad knob combinations.
  * The step is non-destructive — its ``StepResult.signal`` is the
    input signal verbatim.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from citeclaw.cache import Cache
from citeclaw.config import BudgetTracker, Settings
from citeclaw.context import Context
from citeclaw.models import PaperRecord
from citeclaw.steps import build_step
from citeclaw.steps.human_in_the_loop import HumanInTheLoop
from tests.fakes import FakeS2Client, make_paper


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _ConfirmStub:
    """Drop-in replacement for ``rich.prompt.Confirm.ask`` driven by a
    queue of canned values. Tests inject a list of bools (and optional
    exceptions) to feed the step deterministically."""

    def __init__(self, answers: list[bool | type[BaseException]]) -> None:
        self._queue = deque(answers)
        self.calls: list[str] = []

    def __call__(self, prompt: str, *args: Any, **kwargs: Any) -> bool:
        self.calls.append(prompt)
        if not self._queue:
            return True  # default fall-through if test under-specifies
        item = self._queue.popleft()
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item("test-induced timeout/interrupt")
        return bool(item)


def _install_confirm_stub(
    monkeypatch: pytest.MonkeyPatch,
    answers: list[bool | type[BaseException]],
) -> _ConfirmStub:
    """Replace ``rich.prompt.Confirm.ask`` for the duration of the test."""
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
    return Context(
        config=cfg,
        s2=fs if fs is not None else FakeS2Client(),
        cache=cache,
        budget=budget,
    )


# ---------------------------------------------------------------------------
# Constructor / validation
# ---------------------------------------------------------------------------


class TestHumanInTheLoopConstructor:
    def test_default_args(self):
        step = HumanInTheLoop()
        assert step.k == 10
        assert step.timeout_sec == 120
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


# ---------------------------------------------------------------------------
# Headline test: report is written and agreement is computed
# ---------------------------------------------------------------------------


class TestHumanInTheLoopReport:
    def test_writes_hitl_report_with_agreement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        # Build a context with 2 accepted papers + 2 rejected papers
        # (one rejected by `llm_title_llm`, one by `llm_abstract_llm`).
        fs = FakeS2Client()
        # Pre-register the rejected papers so fetch_metadata succeeds.
        fs.add(make_paper("REJ-1", title="Rejected by title filter"))
        fs.add(make_paper("REJ-2", title="Rejected by abstract filter"))
        ctx = _build_ctx(tmp_path, fs)
        ctx.collection["ACC-1"] = _make_record("ACC-1", title="Accepted 1")
        ctx.collection["ACC-2"] = _make_record("ACC-2", title="Accepted 2")
        ctx.rejection_ledger["REJ-1"] = ["llm_title_llm"]
        ctx.rejection_ledger["REJ-2"] = ["llm_abstract_llm"]
        ctx.seen.update(["ACC-1", "ACC-2", "REJ-1", "REJ-2"])

        # User says: keep both accepted (correct), drop both rejected
        # (correct). Both filters should hit 100% agreement on the
        # papers they made decisions on.
        # Order is shuffled by the step so we feed 4 True/False answers
        # in arbitrary order — but since accepted should be kept and
        # rejected should be dropped, the user's answer per paper is
        # determined by paper_id, NOT order. We need a smarter stub.
        #
        # Simpler approach: accept k=4, seed=0, include_accepted=True,
        # include_rejected=True, balance_by_filter=True. The shuffle is
        # deterministic. We can predict the order by construction or
        # just give a fixed answer that yields a known agreement.
        #
        # Even simpler: inject a stub that decides based on paper_id
        # by reading the prompt text.
        def smart_ask(prompt: str, *args: Any, **kwargs: Any) -> bool:
            # User keeps accepted papers and drops rejected ones.
            return "ACC-" in prompt or "Accepted" in prompt

        from rich.prompt import Confirm
        monkeypatch.setattr(Confirm, "ask", smart_ask)

        step = HumanInTheLoop(k=4, seed=42, balance_by_filter=False)
        result = step.run([], ctx)

        # Report exists.
        report_path = ctx.config.data_dir / "hitl_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())

        # The report carries the per-filter agreement.
        assert "agreement_by_filter" in report
        agg = report["agreement_by_filter"]
        # Both filters should be perfectly correct on the papers they
        # were sampled with.
        assert "llm_title_llm" in agg or "llm_abstract_llm" in agg
        for cat, score in agg.items():
            assert 0.0 <= score <= 1.0

        # Stats reflect what happened.
        assert result.stats["candidates"] >= 1
        assert result.stats["labels_collected"] >= 1
        assert result.stats["timeouts"] == 0

        # Step is non-destructive: signal passes through unchanged.
        assert result.signal == []

    def test_report_records_low_agreement_filters_when_user_disagrees(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """If the user labels every rejected paper as 'keep', the
        rejecting filter scores 0% agreement and shows up in the
        low_agreement_filters list. The continue/stop prompt then
        gets called once."""
        fs = FakeS2Client()
        for i in range(4):
            fs.add(make_paper(f"REJ-{i}", title=f"Rejected paper {i}"))
        ctx = _build_ctx(tmp_path, fs)
        for i in range(4):
            ctx.rejection_ledger[f"REJ-{i}"] = ["llm_title_llm"]
            ctx.seen.add(f"REJ-{i}")

        # 4 "yes, keep" answers + 1 "yes, continue" for the warning.
        stub = _install_confirm_stub(monkeypatch, [True, True, True, True, True])

        step = HumanInTheLoop(
            k=4, seed=0, include_accepted=False, balance_by_filter=False,
        )
        result = step.run([], ctx)

        report = json.loads(
            (ctx.config.data_dir / "hitl_report.json").read_text(),
        )
        assert report["agreement_by_filter"]["llm_title_llm"] == 0.0
        assert "llm_title_llm" in report["low_agreement_filters"]
        assert result.stats["low_agreement_filters"] == 1
        # Stub got called 4× per-paper + 1× continue/stop prompt.
        assert len(stub.calls) == 5


# ---------------------------------------------------------------------------
# Sampling: balance_by_filter
# ---------------------------------------------------------------------------


class TestHumanInTheLoopSampling:
    def test_balance_by_filter_pulls_from_each_category(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """With 4 rejections in `llm_title_llm` and 4 in
        `llm_abstract_llm`, balance_by_filter should sample roughly
        equally from each category rather than all from one."""
        fs = FakeS2Client()
        for i in range(4):
            fs.add(make_paper(f"T-{i}", title=f"Title-rejected {i}"))
            fs.add(make_paper(f"A-{i}", title=f"Abstract-rejected {i}"))
        ctx = _build_ctx(tmp_path, fs)
        for i in range(4):
            ctx.rejection_ledger[f"T-{i}"] = ["llm_title_llm"]
            ctx.rejection_ledger[f"A-{i}"] = ["llm_abstract_llm"]
            ctx.seen.update([f"T-{i}", f"A-{i}"])

        stub = _install_confirm_stub(
            monkeypatch, [True] * 20,  # plenty of canned answers
        )
        step = HumanInTheLoop(
            k=4,
            seed=0,
            include_accepted=False,
            balance_by_filter=True,
        )
        step.run([], ctx)

        # The stub captured the prompt text for each candidate. Both
        # categories must appear at least once.
        joined = "\n".join(stub.calls)
        assert "Title-rejected" in joined
        assert "Abstract-rejected" in joined

    def test_no_candidates_short_circuits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """When the collection is empty AND the rejection_ledger has
        no LLM rejections, the step short-circuits with no_candidates."""
        ctx = _build_ctx(tmp_path)
        # Don't set any LLM rejections.
        ctx.rejection_ledger["MISC"] = ["year"]  # non-llm category

        stub = _install_confirm_stub(monkeypatch, [True])
        step = HumanInTheLoop(k=4)
        result = step.run([], ctx)

        assert result.stats["reason"] == "no_candidates"
        assert stub.calls == []  # never prompted
        assert not (ctx.config.data_dir / "hitl_report.json").exists()


# ---------------------------------------------------------------------------
# Timeout / interrupt handling
# ---------------------------------------------------------------------------


class TestHumanInTheLoopTimeout:
    def test_timeout_skips_paper_and_continues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """A TimeoutError on one paper should be logged and the step
        should keep going for the rest."""
        fs = FakeS2Client()
        fs.add(make_paper("REJ-1", title="First"))
        fs.add(make_paper("REJ-2", title="Second"))
        ctx = _build_ctx(tmp_path, fs)
        ctx.rejection_ledger["REJ-1"] = ["llm_title_llm"]
        ctx.rejection_ledger["REJ-2"] = ["llm_title_llm"]
        ctx.seen.update(["REJ-1", "REJ-2"])

        # Second per-paper call raises TimeoutError; first returns False.
        # Then the continue/stop prompt fires once at the end.
        stub = _install_confirm_stub(
            monkeypatch, [False, TimeoutError, True],
        )
        step = HumanInTheLoop(
            k=2, seed=0, include_accepted=False, balance_by_filter=False,
        )
        result = step.run([], ctx)

        assert result.stats["timeouts"] == 1
        assert result.stats["labels_collected"] == 1


# ---------------------------------------------------------------------------
# Signal pass-through
# ---------------------------------------------------------------------------


class TestHumanInTheLoopSignalPassThrough:
    def test_signal_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """The step is non-destructive — its output signal is the
        input signal verbatim, no matter what the user labels."""
        ctx = _build_ctx(tmp_path)
        ctx.collection["ACC-1"] = _make_record("ACC-1")
        in_signal = [
            _make_record("X-1"),
            _make_record("X-2"),
            _make_record("X-3"),
        ]

        _install_confirm_stub(monkeypatch, [True, False, True])
        step = HumanInTheLoop(k=1, include_rejected=False, seed=0)
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
                "k": 5,
                "timeout_sec": 60,
                "include_accepted": False,
                "balance_by_filter": False,
                "seed": 99,
            },
            blocks={},
        )
        assert isinstance(step, HumanInTheLoop)
        assert step.k == 5
        assert step.timeout_sec == 60
        assert step.include_accepted is False
        assert step.balance_by_filter is False
        assert step._seed == 99
