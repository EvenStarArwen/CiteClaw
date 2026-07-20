"""Pause / resume gate in the web run manager."""

from __future__ import annotations

import threading
import time

from web.live.backend.run_manager import _honor_pause, manager


class _RS:
    """Minimal RunState stand-in for the pause gate."""

    def __init__(self):
        self.paused = False
        self.stop_requested = False
        self.finalizing = False
        self.pause_wake = threading.Event()
        self.status = "running"


def test_honor_pause_releases_on_resume():
    rs = _RS()
    rs.paused = True

    def resume_soon():
        time.sleep(0.05)
        rs.paused = False
        rs.pause_wake.set()

    t = threading.Thread(target=resume_soon)
    t.start()
    _honor_pause(rs, lambda: None)  # blocks until resumed
    t.join()
    assert rs.paused is False


def test_honor_pause_releases_on_stop():
    rs = _RS()
    rs.paused = True

    def stop_soon():
        time.sleep(0.05)
        rs.stop_requested = True
        rs.pause_wake.set()

    t = threading.Thread(target=stop_soon)
    t.start()
    _honor_pause(rs, lambda: None)
    t.join()
    assert rs.stop_requested is True


def test_honor_pause_noop_when_not_paused():
    rs = _RS()  # paused is False

    def boom():
        raise AssertionError("heartbeat must not fire when not paused")

    _honor_pause(rs, boom)  # returns immediately, no heartbeat


def test_pause_resume_run_via_manager():
    rid = "pausetest_" + "abc123"
    rs = _RS()
    manager.runs[rid] = rs
    try:
        assert manager.pause_run(rid) is True
        assert rs.paused is True
        assert manager.resume_run(rid) is True
        assert rs.paused is False
        assert manager.pause_run("no-such-run") is False
        # a run already stopping cannot be paused
        rs.stop_requested = True
        assert manager.pause_run(rid) is False
    finally:
        manager.runs.pop(rid, None)
