"""Tests for S2 consecutive-failure auto-abort.

Drives :class:`S2Http` against a stub httpx client that always raises
``httpx.ConnectError`` so every retry burns through tenacity's six
attempts. After ``s2_max_consecutive_failures`` consecutive exhaustions,
the next failure is converted into an :class:`S2OutageError` rather
than a transport error, and the run can shut down cleanly.

The retry's ``wait_random_exponential(min=2, max=60)`` is bypassed by
patching ``time.sleep`` to a no-op so the test runs in milliseconds.
"""

from __future__ import annotations

import httpx
import pytest
from tenacity import RetryError

from citeclaw.clients.s2.http import S2Http
from citeclaw.config import BudgetTracker, Settings
from citeclaw.models import S2OutageError


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Skip backoff sleeps so the retry cascade finishes instantly."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "tenacity.wait.wait_random_exponential.__call__",
        lambda *_a, **_kw: 0,
    )


def _settings(threshold: int) -> Settings:
    return Settings(
        screening_model="stub",
        s2_rps=1000.0,  # disable throttle
        s2_max_consecutive_failures=threshold,
    )


class _AlwaysFailingHttpx:
    """Stand-in for httpx.Client that always raises ConnectError on get/post."""

    def get(self, *_args, **_kwargs):
        raise httpx.ConnectError("simulated outage")

    def post(self, *_args, **_kwargs):
        raise httpx.ConnectError("simulated outage")

    def close(self):
        pass


class _ConditionalHttpx:
    """Httpx stand-in: first N calls fail, then succeed."""

    def __init__(self, fail_first_n: int):
        self.calls = 0
        self.fail_first_n = fail_first_n

    def get(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls <= self.fail_first_n:
            raise httpx.ConnectError("simulated transient")

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"ok": True}

        return _Resp()

    def post(self, *_a, **_kw):
        return self.get()

    def close(self):
        pass


def test_outage_raises_after_threshold(monkeypatch):
    """Exactly ``s2_max_consecutive_failures`` exhaustions trigger S2OutageError."""
    cfg = _settings(threshold=3)
    budget = BudgetTracker()
    s2 = S2Http(cfg, budget)
    s2._http = _AlwaysFailingHttpx()

    # First two exhaustions surface the underlying httpx exception.
    for _ in range(2):
        with pytest.raises(RetryError):
            s2.get("/x", req_type="t")

    # The third one should escalate into S2OutageError.
    with pytest.raises(S2OutageError) as excinfo:
        s2.get("/x", req_type="t")
    assert "3 consecutive" in str(excinfo.value)


def test_success_resets_counter(monkeypatch):
    """A single successful call resets the consecutive-failure tally."""
    cfg = _settings(threshold=3)
    budget = BudgetTracker()
    s2 = S2Http(cfg, budget)

    # Two failures, then a success → counter resets to 0.
    s2._http = _AlwaysFailingHttpx()
    for _ in range(2):
        with pytest.raises(RetryError):
            s2.get("/x", req_type="t")
    assert s2._consecutive_failures == 2

    s2._http = _ConditionalHttpx(fail_first_n=0)
    assert s2.get("/y", req_type="t") == {"ok": True}
    assert s2._consecutive_failures == 0

    # Two more failures should not trigger outage (well below threshold of 3).
    s2._http = _AlwaysFailingHttpx()
    for _ in range(2):
        with pytest.raises(RetryError):
            s2.get("/x", req_type="t")


def test_outage_disabled_when_threshold_zero():
    """Setting s2_max_consecutive_failures=0 disables the auto-abort."""
    cfg = _settings(threshold=0)
    budget = BudgetTracker()
    s2 = S2Http(cfg, budget)
    s2._http = _AlwaysFailingHttpx()

    # Many exhaustions; none should escalate.
    for _ in range(10):
        with pytest.raises(RetryError):
            s2.get("/x", req_type="t")


def test_outage_tracked_across_methods(monkeypatch):
    """get / get_url / post all share the same consecutive-failure counter."""
    cfg = _settings(threshold=3)
    budget = BudgetTracker()
    s2 = S2Http(cfg, budget)
    s2._http = _AlwaysFailingHttpx()

    with pytest.raises(RetryError):
        s2.get("/x", req_type="t")
    with pytest.raises(RetryError):
        s2.get_url("https://x/y", req_type="t")
    with pytest.raises(S2OutageError):
        s2.post("https://x/z", json_body={"q": 1}, req_type="t")


def test_s2_outage_bypasses_generic_except_exception():
    """S2OutageError must not be caught by `except Exception` clauses."""
    err = S2OutageError("test outage")
    assert isinstance(err, BaseException)
    assert not isinstance(err, Exception)
