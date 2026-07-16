"""Launch CiteClaw runs in a background thread and stream live events.

A ``WebFanoutSink`` (implementing ``citeclaw.event_sink.EventSink``) runs
inside the pipeline worker thread and forwards enriched events — step
progress, accepted papers, metric snapshots, live graph — to every
connected WebSocket via ``loop.call_soon_threadsafe``. Each run keeps a
full ordered event log so a browser that connects mid-run replays cleanly.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Any

from .snapshots import build_graph, build_metrics, paper_dict

# CiteClaw step name -> (design kind, friendly label, localId prefix)
_STEP_META = {
    "ResolveSeeds": ("seed", "Resolve seeds", "SED"),
    "LoadSeeds": ("seed", "Seed set", "SED"),
    "ExpandForward": ("fwd", "Forward screener", "FWD"),
    "ExpandBackward": ("bwd", "Backward screener", "BWD"),
    "ExpandBySearch": ("fwd", "Search agent", "SRC"),
    "ExpandBySemantics": ("fwd", "Semantic expand", "SEM"),
    "ExpandByAuthor": ("fwd", "Author expand", "AUT"),
    "ExpandByPDF": ("fwd", "PDF expand", "PDF"),
    "Rerank": ("rerank", "Diversified rerank", "RRK"),
    "ReScreen": ("rsc", "Rescreen", "RSC"),
    "Cluster": ("rerank", "Cluster", "CLU"),
    "HumanInTheLoop": ("rsc", "Human check", "HIL"),
    "Parallel": ("fwd", "Parallel branches", "PAR"),
    "MergeDuplicates": ("sink", "Merge duplicates", "MRG"),
    "Finalize": ("sink", "Finalize", "OUT"),
}


class _StopRun(Exception):
    """Raised inside the sink to abort a run at the next step/paper boundary."""


def _steps_meta_from_config(config) -> list[dict[str, Any]]:
    """Build the progress-step list from the translated pipeline.

    Mirrors the runner's auto-injection of ``MergeDuplicates`` before
    ``Finalize`` so indices line up with what actually executes.
    """
    raw = list(config.pipeline or [])
    names = [s.get("step") for s in raw]
    if "Finalize" in names and "MergeDuplicates" not in names:
        fi = names.index("Finalize")
        raw = raw[:fi] + [{"step": "MergeDuplicates"}] + raw[fi:]
    steps = []
    counters: dict[str, int] = {}
    for s in raw:
        name = s.get("step")
        kind, label, prefix = _STEP_META.get(name, ("fwd", name or "Step", "STP"))
        counters[prefix] = counters.get(prefix, 0) + 1
        steps.append({
            "kind": kind,
            "name": label,
            "localId": f"{prefix}-{len(steps) + 1:02d}",
            "cwName": name,
            "status": "idle",
            "sub": "pending",
            "pct": 0,
        })
    return steps


class StepProgress:
    """Tracks per-step status by CiteClaw step name (robust to injection)."""

    def __init__(self, steps: list[dict[str, Any]]):
        self.steps = steps
        self._by_name: dict[str, list[int]] = {}
        for i, s in enumerate(steps):
            self._by_name.setdefault(s["cwName"], []).append(i)
        self._cursor: dict[str, int] = {}
        self._active: int | None = None

    def _next_index(self, name: str) -> int:
        idxs = self._by_name.get(name)
        if not idxs:
            meta = _STEP_META.get(name, ("fwd", name, "STP"))
            i = len(self.steps)
            self.steps.append({
                "kind": meta[0], "name": meta[1],
                "localId": f"{meta[2]}-{i + 1:02d}", "cwName": name,
                "status": "idle", "sub": "pending", "pct": 0,
            })
            self._by_name.setdefault(name, []).append(i)
            return i
        c = self._cursor.get(name, 0)
        if c < len(idxs):
            self._cursor[name] = c + 1
            return idxs[c]
        return idxs[-1]

    def start(self, name: str, desc: str) -> None:
        i = self._next_index(name)
        self.steps[i]["status"] = "active"
        self.steps[i]["pct"] = 50
        self.steps[i]["sub"] = desc or "running…"
        self._active = i

    def end(self, name: str, in_c: int, out_c: int, delta: int) -> None:
        i = self._active if (self._active is not None
                             and self.steps[self._active]["cwName"] == name) else None
        if i is None:
            idxs = self._by_name.get(name)
            i = idxs[-1] if idxs else None
        if i is not None:
            self.steps[i]["status"] = "done"
            self.steps[i]["pct"] = 100
            self.steps[i]["sub"] = f"{in_c:,} in · {out_c:,} pass · +{delta:,}"

    def snapshot(self) -> dict[str, Any]:
        done = sum(1 for s in self.steps if s["status"] == "done")
        total = len(self.steps)
        cur = next((s["name"] for s in self.steps if s["status"] == "active"), None)
        return {
            "steps": self.steps,
            "done": done,
            "total": total,
            "current": cur,
            "overallPct": round(100 * done / total) if total else 0,
        }


class RunState:
    def __init__(self, run_id: str, config, steps_meta: list[dict[str, Any]]):
        self.run_id = run_id
        self.config = config
        self.status = "starting"  # starting|running|done|error|stopped
        self.events: list[dict[str, Any]] = []
        self.subscribers: set[asyncio.Queue] = set()
        self.lock = threading.Lock()
        self.ctx = None
        self.thread: threading.Thread | None = None
        self.stop_requested = False
        self.progress = StepProgress(steps_meta)
        self.error: str | None = None
        self.summary: dict[str, Any] | None = None


class WebFanoutSink:
    """EventSink that fans events out to WebSocket subscribers."""

    def __init__(self, mgr: "RunManager", rs: RunState):
        self.mgr = mgr
        self.rs = rs
        self._last_snap = 0.0

    def _check_stop(self) -> None:
        if self.rs.stop_requested:
            raise _StopRun()

    def step_start(self, idx: int, name: str, description: str) -> None:
        self._check_stop()
        self.rs.progress.start(name, description)
        self.mgr.broadcast(self.rs, {"type": "progress", "progress": self.rs.progress.snapshot()})
        self._snapshot(force=True)

    def paper_added(self, paper_id: str, source: str) -> None:
        ctx = self.rs.ctx
        p = ctx.collection.get(paper_id) if ctx else None
        if p is not None:
            self.mgr.broadcast(self.rs, {
                "type": "paper_added",
                "paper": paper_dict(p, seed_ids=ctx.seed_ids),
            })
        self._snapshot(force=False)
        self._check_stop()

    def step_end(self, idx: int, name: str, in_count: int, out_count: int,
                 delta_collection: int, stats: dict) -> None:
        self.rs.progress.end(name, in_count, out_count, delta_collection)
        self.mgr.broadcast(self.rs, {"type": "progress", "progress": self.rs.progress.snapshot()})
        self._snapshot(force=True)

    def paper_rejected(self, paper_id: str, category: str) -> None:
        pass

    def shape_table_update(self, rendered_shape: str) -> None:
        pass

    def hitl_request(self, run_id: str, papers: list[dict]) -> None:
        # HumanInTheLoop isn't wired interactively in this first version.
        pass

    def _snapshot(self, force: bool) -> None:
        now = time.monotonic()
        if not force and now - self._last_snap < 1.0:
            return
        self._last_snap = now
        ctx = self.rs.ctx
        if ctx is None:
            return
        try:
            self.mgr.broadcast(self.rs, {"type": "metrics", "metrics": build_metrics(ctx)})
            self.mgr.broadcast(self.rs, {"type": "graph", "graph": build_graph(ctx)})
        except Exception:
            pass  # never let a snapshot error crash the run


class RunManager:
    def __init__(self) -> None:
        self.runs: dict[str, RunState] = {}
        self.loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    # ---- event fan-out ------------------------------------------------
    def broadcast(self, rs: RunState, ev: dict[str, Any]) -> None:
        with rs.lock:
            rs.events.append(ev)
            subs = list(rs.subscribers)
        loop = self.loop
        if loop is None:
            return
        for q in subs:
            try:
                loop.call_soon_threadsafe(q.put_nowait, ev)
            except RuntimeError:
                pass

    def subscribe(self, rs: RunState) -> tuple[asyncio.Queue, list[dict[str, Any]]]:
        q: asyncio.Queue = asyncio.Queue()
        with rs.lock:
            backlog = list(rs.events)
            rs.subscribers.add(q)
        return q, backlog

    def unsubscribe(self, rs: RunState, q: asyncio.Queue) -> None:
        with rs.lock:
            rs.subscribers.discard(q)

    # ---- lifecycle ----------------------------------------------------
    def start_run(self, config) -> tuple[str, list[dict[str, Any]]]:
        run_id = uuid.uuid4().hex[:12]
        steps_meta = _steps_meta_from_config(config)
        rs = RunState(run_id, config, steps_meta)
        self.runs[run_id] = rs
        t = threading.Thread(target=self._run, args=(rs,), daemon=True, name=f"citeclaw-run-{run_id}")
        rs.thread = t
        t.start()
        return run_id, steps_meta

    def get(self, run_id: str) -> RunState | None:
        return self.runs.get(run_id)

    def stop_run(self, run_id: str) -> bool:
        rs = self.runs.get(run_id)
        if not rs:
            return False
        rs.stop_requested = True
        return True

    def _run(self, rs: RunState) -> None:
        from citeclaw.pipeline import build_context, finalize_partial, run_pipeline

        rs.status = "running"
        self.broadcast(rs, {"type": "hello", "run_id": rs.run_id,
                            "progress": rs.progress.snapshot()})
        s2 = cache = None
        try:
            ctx, s2, cache = build_context(rs.config)
            rs.ctx = ctx
            sink = WebFanoutSink(self, rs)
            try:
                run_pipeline(ctx, event_sink=sink)
                rs.status = "done"
            except _StopRun:
                try:
                    finalize_partial(ctx)
                except Exception:
                    pass
                rs.status = "stopped"
            except BaseException as e:  # noqa: BLE001 - surface any run error to UI
                try:
                    finalize_partial(ctx)
                except Exception:
                    pass
                rs.status = "error"
                rs.error = f"{type(e).__name__}: {e}"
                self.broadcast(rs, {"type": "error", "message": rs.error})
        except BaseException as e:  # noqa: BLE001 - setup failure (bad config, missing key)
            rs.status = "error"
            rs.error = f"{type(e).__name__}: {e}"
            self.broadcast(rs, {"type": "error", "message": rs.error})
        finally:
            for closer in (s2, cache):
                try:
                    if closer is not None:
                        closer.close()
                except Exception:
                    pass

        # final snapshots so the UI shows the completed graph + numbers
        if rs.ctx is not None:
            try:
                self.broadcast(rs, {"type": "metrics", "metrics": build_metrics(rs.ctx)})
                self.broadcast(rs, {"type": "graph", "graph": build_graph(rs.ctx)})
                rs.summary = {
                    "accepted": len(rs.ctx.collection),
                    "status": rs.status,
                    "data_dir": str(rs.config.data_dir),
                }
            except Exception:
                pass
        self.broadcast(rs, {"type": "done", "status": rs.status,
                            "summary": rs.summary, "error": rs.error})


# process-wide singleton
manager = RunManager()
