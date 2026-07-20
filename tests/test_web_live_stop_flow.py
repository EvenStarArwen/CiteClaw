"""Stop-flow step semantics: the Finalize row must tell the truth.

Regression for the '"Finalizing…" topbar while the sidebar later says
Finalize was skipped' confusion — when finalize_partial writes artifacts
after a stop, the board shows Finalize as done(partial), the interrupted
step as interrupted, and truly-unreached steps as skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web"))

from live.backend.run_manager import StepProgress  # noqa: E402


def _steps():
    return [
        {"kind": "seed", "name": "Seed set", "localId": "SED-01",
         "cwName": "LoadSeeds", "status": "idle", "sub": "pending", "pct": 0},
        {"kind": "fwd", "name": "Forward", "localId": "FWD-02",
         "cwName": "ExpandForward", "status": "idle", "sub": "pending", "pct": 0},
        {"kind": "bwd", "name": "Backward", "localId": "BWD-03",
         "cwName": "ExpandBackward", "status": "idle", "sub": "pending", "pct": 0},
        {"kind": "sink", "name": "Finalize", "localId": "OUT-04",
         "cwName": "Finalize", "status": "idle", "sub": "pending", "pct": 0},
    ]


def _board_mid_run():
    b = StepProgress(_steps())
    b.start("LoadSeeds", "loading")
    b.end("LoadSeeds", 0, 2, 2)
    b.start("ExpandForward", "expanding")
    return b


def test_stop_marks_finalize_done_partial():
    b = _board_mid_run()
    b.note_stopping()
    assert b.steps[1]["sub"].startswith("stop requested")
    b.begin_finalizing()
    assert b.steps[3]["status"] == "active"
    assert "artifacts" in b.steps[3]["sub"]
    b.finish("stopped", None, finalized=True)
    by = {s["cwName"]: s for s in b.steps}
    assert by["ExpandForward"]["status"] == "skipped"
    assert by["ExpandForward"]["sub"] == "interrupted — run stopped here"
    assert by["ExpandBackward"]["status"] == "skipped"
    assert by["Finalize"]["status"] == "done"
    assert "after stop" in by["Finalize"]["sub"]
    assert b.snapshot()["overallPct"] == 100


def test_stop_without_finalize_keeps_skipped():
    b = _board_mid_run()
    b.finish("stopped", None, finalized=False)
    by = {s["cwName"]: s for s in b.steps}
    assert by["Finalize"]["status"] == "skipped"


def test_error_path_marks_finalize_partial():
    b = _board_mid_run()
    b.begin_finalizing()
    b.finish("error", None, finalized=True)
    by = {s["cwName"]: s for s in b.steps}
    assert by["Finalize"]["status"] == "done"
    assert "error" in by["Finalize"]["sub"]
    assert by["ExpandForward"]["status"] == "error"


def test_normal_completion_untouched():
    b = _board_mid_run()
    b.end("ExpandForward", 10, 5, 5)
    b.start("ExpandBackward", "x")
    b.end("ExpandBackward", 5, 3, 3)
    b.start("Finalize", "writing")
    b.end("Finalize", 0, 0, 0)
    b.finish("done", None)
    assert all(s["status"] == "done" for s in b.steps)
