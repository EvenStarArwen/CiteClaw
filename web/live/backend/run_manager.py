"""Launch CiteClaw runs in a background thread and stream live events.

A ``WebFanoutSink`` (implementing ``citeclaw.event_sink.EventSink``) runs
inside the pipeline worker thread and forwards enriched events — step
progress, accepted papers, metric snapshots, live graph — to every
connected WebSocket via ``loop.call_soon_threadsafe``. Each run keeps a
full ordered event log so a browser that connects mid-run replays cleanly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from .snapshots import build_graph, build_metrics, paper_dict

# Keys whose values are secrets — redacted before the config is written to a
# downloadable file. ``*_env`` keys hold env-var NAMES (e.g. api_key_env), not
# the secrets themselves, so they are kept.
_SECRET_KEY_RE = re.compile(r"(api_key|secret|token|password)$", re.I)


def _scrub_secrets(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k)
            if ks.endswith("_env"):
                out[k] = v
            elif _SECRET_KEY_RE.search(ks) and isinstance(v, str) and v:
                out[k] = "***redacted***"
            else:
                out[k] = _scrub_secrets(v)
        return out
    if isinstance(obj, list):
        return [_scrub_secrets(x) for x in obj]
    return obj


def _dump_run_config(config: Any, path: Path) -> None:
    """Write the exact (secret-scrubbed) config a run used → pipeline_config.json."""
    try:
        data = config.model_dump(mode="json")  # pydantic Settings
    except Exception:  # noqa: BLE001 - fall back to a best-effort dict
        try:
            data = dict(config)
        except Exception:
            return
    try:
        path.write_text(json.dumps(_scrub_secrets(data), indent=2, default=str) + "\n",
                        encoding="utf-8")
    except Exception:  # noqa: BLE001 - a config dump must never break a run
        pass


class _ThreadLogFilter(logging.Filter):
    """Keep only this run thread's records (multi-tenant per-run file logs)."""

    def __init__(self, thread_name: str) -> None:
        super().__init__()
        self._tn = thread_name

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D102
        return record.threadName == self._tn

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


def _brief_hint(s: dict[str, Any]) -> str:
    """Short param hint for one step inside a Parallel branch."""
    name = s.get("step")
    if name == "ExpandForward":
        return f"≤{s.get('max_citations', 100)} citers per source"
    if name == "ExpandBackward":
        return "every reference, no cap"
    if name == "Rerank":
        d = s.get("diversity")
        extra = f", {d.get('type')}-diverse" if isinstance(d, dict) and d.get("type") else ""
        return f"top {s.get('k', 100)} by {s.get('metric', 'citation')}{extra}"
    return ""


def _filter_label(b: dict[str, Any]) -> str:
    """Compact human label for one filter block (recurses into composites)."""
    t = b.get("type")
    if t == "YearFilter":
        return f"Year {b.get('min', '…')}–{b.get('max', '…')}"
    if t == "CitationFilter":
        return f"Citation β={b.get('beta', '…')}"
    if t == "AbstractKeywordFilter":
        return "Abstract keywords"
    if t == "TitleKeywordFilter":
        return "Title keywords"
    if t == "VenueKeywordFilter":
        return "Venue keywords"
    if t == "SimilarityFilter":
        return f"Similarity ≥ {b.get('threshold', '…')}"
    if t == "LLMFilter":
        return f"LLM ({b.get('scope', 'title_abstract')})"
    if t == "Sequential":
        return " → ".join(_filter_label(x) for x in b.get("layers", []))
    if t == "Any":
        return "any of: " + " | ".join(_filter_label(x) for x in b.get("layers", []))
    if t == "Not":
        return "not " + _filter_label(b.get("layer") or {})
    return str(t or "?")


def _filter_hint(b: dict[str, Any]) -> str:
    """One phrase on what a single filter actually does — for the roadmap."""
    t = b.get("type")
    if t == "YearFilter":
        return f"keep papers published {b.get('min', '…')}–{b.get('max', '…')}"
    if t == "CitationFilter":
        return (f"drop papers under-cited for their age "
                f"(β={b.get('beta', '…')} citations/year baseline)")
    if t == "AbstractKeywordFilter":
        return "boolean keyword expression matched against the abstract — no LLM"
    if t == "TitleKeywordFilter":
        return "boolean keyword expression matched against the title — no LLM"
    if t == "VenueKeywordFilter":
        return "boolean keyword expression matched against the venue — no LLM"
    if t == "SimilarityFilter":
        return "reference / citation / embedding overlap with the source paper"
    if t == "LLMFilter":
        scope = str(b.get("scope", "title_abstract")).replace("_", " + ")
        return f"an LLM reads each paper's {scope} and votes keep / drop"
    if t in ("Sequential", "Any"):
        return "a group of sub-filters (see the composite label)"
    if t == "Not":
        return "passes exactly the papers the inner filter would reject"
    if t == "Route":
        return "routes each paper to a different sub-filter by venue / citations / year"
    return ""


def _filter_nodes(block: dict[str, Any] | None) -> list[dict[str, str]]:
    """One roadmap node per top-level screener layer.

    Matches how the runtime reports progress: a Sequential screener runs
    ``begin_phase`` once per direct layer (named ``<screener>.layerN``), so
    one node per top-level layer keeps the live highlight aligned with the
    filter actually running. Composite layers (Any / Route / nested
    Sequential) collapse to a single node described by their label.
    """
    if not block:
        return []
    layers = (block.get("layers") if block.get("type") == "Sequential" else [block]) or []
    return [{"label": _filter_label(b), "hint": _filter_hint(b)} for b in layers]


def _screen_stages(s: dict[str, Any]) -> list[dict[str, Any]]:
    """The screening cascade as individual roadmap nodes (filter A → B → C).

    ``screenIdx`` is the layer's position in the cascade; the front end reads
    the trailing ``.layerN`` of the live phase to light up the exact filter.
    """
    nodes = _filter_nodes(s.get("screener"))
    if not nodes:
        return [{"key": "__screen", "screen": True, "screenIdx": 0,
                 "label": "Screen candidates",
                 "hint": "no screener configured — every candidate passes"}]
    return [{"key": "__screen", "screen": True, "screenIdx": i,
             "label": f"Filter {i + 1} · {n['label']}", "hint": n["hint"]}
            for i, n in enumerate(nodes)]


def _add_stage(what: str) -> dict[str, Any]:
    """Terminal roadmap node — where survivors leave the step (input → output)."""
    return {"key": "__add", "terminal": True, "label": f"Add {what} to the collection",
            "hint": "accepted papers join the running collection and stream into the graph"}


def _step_road(s: dict[str, Any]) -> dict[str, Any]:
    """The step's internal ROADMAP: its stages in order, with live-match keys.

    ``key`` is the prefix the core uses for ``begin_phase`` descriptions, so
    the front end can highlight the stage that is running right now. The
    special key ``__screen`` matches "none of the others" — during screening
    the inner bar carries the individual FILTER names.
    """
    name = s.get("step")
    if name == "ExpandForward":
        cap = s.get("max_citations", 100)
        return {"loop": True,
                "blurb": "For every source paper in the signal: pull its citing papers, "
                         "enrich them, run them through the screening cascade, and add the "
                         "survivors to the collection.",
                "stages": [
                    {"key": "fetch citers", "label": "Fetch citing papers",
                     "hint": f"up to {cap} per source, page by page from Semantic Scholar"},
                    {"key": "fetch source refs", "label": "Fetch the source's references",
                     "hint": "needed for reference-overlap similarity"},
                    {"key": "enrich · batch", "label": "Enrich metadata",
                     "hint": "bulk-fill missing fields (venue, year, citations, authors)"},
                    {"key": "enrich · abstracts", "label": "Enrich abstracts",
                     "hint": "S2 first, OpenAlex fallback — needed before any text screening"},
                    *_screen_stages(s),
                    {"key": "fetch accepted refs", "label": "Fetch survivors' references",
                     "hint": "one batched call — their in-collection links appear in the graph right away"},
                    _add_stage("survivors"),
                ]}
    if name == "ExpandBackward":
        return {"loop": True,
                "blurb": "For every source paper in the signal: walk every reference it cites "
                         "(no cap), enrich them, screen them, and add the survivors to the "
                         "collection.",
                "stages": [
                    {"key": "fetch refs", "label": "Fetch references",
                     "hint": "all of them, page by page"},
                    {"key": "s2: resolve", "label": "Resolve OpenAlex fallback DOIs",
                     "hint": "only when S2 has no reference list for a source"},
                    {"key": "enrich · abstracts", "label": "Enrich abstracts",
                     "hint": "S2 first, OpenAlex fallback — needed before any text screening"},
                    *_screen_stages(s),
                    {"key": "fetch accepted refs", "label": "Fetch survivors' references",
                     "hint": "one batched call — their in-collection links appear in the graph right away"},
                    _add_stage("survivors"),
                ]}
    if name == "Parallel":
        n = len(s.get("branches") or [])
        branches = []
        for i, b in enumerate(s.get("branches") or [], 1):
            branches.append([{
                "key": f"branch {i}/{n} · {x.get('step')}",
                "label": _STEP_META.get(x.get("step"), ("", x.get("step") or "Step", ""))[1],
                "hint": _brief_hint(x),
            } for x in b])
        return {"loop": False,
                "blurb": "Each branch receives the same input; branches run one after "
                         "another and their outputs are merged (union) before the next step.",
                "stages": [], "branches": branches}
    if name == "Rerank":
        stages = [{"key": "compute", "label": f"Score every paper · {s.get('metric', 'citation')}",
                   "hint": "graph metric over the collected network"}]
        d = s.get("diversity")
        if d:
            algo = d.get("type") if isinstance(d, dict) else str(d)
            stages.append({"key": "cluster-diverse", "label": f"Cluster-diverse top-{s.get('k', 100)}",
                           "hint": f"{algo} clustering, floor-then-proportional allocation"})
        else:
            stages.append({"key": "rank top", "label": f"Keep top-{s.get('k', 100)}",
                           "hint": "plain sort + cut"})
        return {"loop": False,
                "blurb": "Non-destructive — only the forwarded signal is trimmed; the collection keeps everything.",
                "stages": stages}
    if name == "ExpandBySearch":
        it = (s.get("agent") or {}).get("max_iterations", 1)
        return {"loop": True,
                "blurb": "Experimental — the CLI search agent is a placeholder for now.",
                "stages": [
                    {"key": "search", "label": "LLM-designed Semantic Scholar search",
                     "hint": f"up to {it} query rounds"},
                    *_screen_stages(s),
                    _add_stage("survivors"),
                ]}
    if name == "ReScreen":
        return {"loop": False,
                "blurb": "Re-applies the screener to the whole collection (seeds exempt); "
                         "papers that now fail any filter are removed.",
                "stages": _screen_stages(s)}
    if name == "ResolveSeeds":
        return {"loop": False, "blurb": "Resolves title-only seeds to Semantic Scholar records.",
                "stages": [{"key": "resolve seed titles", "label": "Resolve seed titles",
                            "hint": "one S2 title match per seed"}]}
    if name == "LoadSeeds":
        return {"loop": False, "blurb": "Puts the seed papers into the collection.",
                "stages": [{"key": "fetch seed metadata", "label": "Fetch seed metadata",
                            "hint": "one S2 lookup per seed"}]}
    if name == "MergeDuplicates":
        return {"loop": False, "blurb": "Preprint ↔ published dedup before the artifacts are written.",
                "stages": [
                    {"key": "prefetch embeddings", "label": "Prefetch embeddings", "hint": "SPECTER2, for title similarity"},
                    {"key": "detect duplicate clusters", "label": "Detect duplicate clusters",
                     "hint": "DOI/arXiv links + title + embedding similarity"},
                    {"key": "merge clusters", "label": "Merge clusters", "hint": "the published version wins"},
                ]}
    if name == "Finalize":
        return {"loop": False, "blurb": "Writes every artifact of the run.",
                "stages": [
                    {"key": "enrich missing references", "label": "Fetch reference lists",
                     "hint": "one S2 fetch per accepted paper without cached references — reveals the links between accepted papers"},
                    {"key": "enrich missing abstracts", "label": "Enrich missing abstracts", "hint": ""},
                    {"key": "build output", "label": "Build the output collection", "hint": ""},
                    {"key": "write JSON", "label": "Write literature_collection.json", "hint": ""},
                    {"key": "write BibTeX", "label": "Write the .bib file", "hint": ""},
                    {"key": "write run_state", "label": "Write run_state.json", "hint": ""},
                    {"key": "write graphs", "label": "Write citation + collaboration graphs", "hint": "GraphML"},
                    {"key": "write rejections", "label": "Write the rejection ledger", "hint": ""},
                ]}
    return {"loop": False, "blurb": "", "stages": []}


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
            "road": _step_road(s),
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

    def note_stopping(self) -> None:
        """Mark the currently active step as winding down after a stop
        request, so the sidebar agrees with the topbar's 'Finalizing'."""
        for s in self.steps:
            if s["status"] == "active":
                s["sub"] = "stop requested — finishing current stage…"

    def begin_finalizing(self) -> None:
        """Light up the Finalize row while ``finalize_partial`` writes the
        artifacts of a stopped/errored run."""
        for s in reversed(self.steps):
            if s["cwName"] == "Finalize" and s["status"] != "done":
                s["status"] = "active"
                s["sub"] = "writing partial artifacts…"
                break

    def finish(self, status: str, note: str | None, *, finalized: bool = False) -> None:
        """Resolve every unfinished step when the run ends.

        Steps the pipeline never reached (paper/budget cap hit, user stop,
        error) are marked ``skipped`` with the reason, so the list never
        shows "pending" steps after a Finalize — which read as if the run
        had silently jumped over them. When ``finalized`` is True the
        Finalize row is marked done instead of skipped — ``finalize_partial``
        really did write the artifacts, and calling that 'skipped' while the
        topbar said 'Finalizing' confused everyone.
        """
        if finalized:
            for s in reversed(self.steps):
                if s["cwName"] == "Finalize":
                    if s["status"] != "done":
                        s["status"] = "done"
                        s["pct"] = 100
                        s["sub"] = ("partial — artifacts written after stop"
                                    if status == "stopped"
                                    else "partial — artifacts written despite the error")
                    break
        for s in self.steps:
            if s["status"] == "active":
                s["status"] = "error" if status == "error" else "skipped"
                s["sub"] = ("failed — see the live log" if status == "error"
                            else "interrupted — run stopped here")
            elif s["status"] == "idle":
                s["status"] = "skipped"
                s["sub"] = note or (
                    "skipped — run stopped before this step" if status == "stopped"
                    else "skipped — run ended before this step")

    def snapshot(self) -> dict[str, Any]:
        done = sum(1 for s in self.steps if s["status"] == "done")
        resolved = sum(1 for s in self.steps if s["status"] in ("done", "skipped", "error"))
        total = len(self.steps)
        cur = next((s["name"] for s in self.steps if s["status"] == "active"), None)
        # A finished run reads 100% even when steps were skipped by the cap.
        pct = 100 if total and resolved == total else (round(100 * done / total) if total else 0)
        return {
            "steps": self.steps,
            "done": done,
            "total": total,
            "current": cur,
            "overallPct": pct,
        }


class RunState:
    def __init__(self, run_id: str, config, steps_meta: list[dict[str, Any]]):
        self.run_id = run_id
        self.config = config
        self.status = "starting"  # starting|running|done|error|stopped
        self.events: list[dict[str, Any]] = []
        # High-churn streams are compacted for replay: only the latest
        # activity/metrics/graph snapshot and the last 100 log lines are
        # kept, so a browser reconnecting mid-run gets a bounded backlog.
        self.log_tail: deque[dict[str, Any]] = deque(maxlen=100)
        self.latest: dict[str, dict[str, Any]] = {}
        self.subscribers: set[asyncio.Queue] = set()
        self.lock = threading.Lock()
        self.ctx = None
        self.thread: threading.Thread | None = None
        self.stop_requested = False
        self.progress = StepProgress(steps_meta)
        self.error: str | None = None
        self.summary: dict[str, Any] | None = None
        # Set when the core announces an early stop (budget / paper cap) so
        # skipped steps can carry the actual reason.
        self.stop_note: str | None = None
        # Paper ids already streamed live by WebDashboard.paper_accepted —
        # the runner re-announces the same papers at step end (collection
        # diff), which must not duplicate rows in the accepted list.
        self.announced: set[str] = set()
        # True while finalize_partial writes artifacts after a stop — the
        # dashboard stop checks must not fire again mid-finalize.
        self.finalizing = False
        # Paper-cap gate: when the collection reaches max_papers_total the
        # run thread blocks (≤30s) on cap_event waiting for the user's
        # modal decision; cap_decision = {"action": "stop"|"raise", "max": N}.
        self.cap_prompted = False
        self.cap_event = threading.Event()
        self.cap_decision: dict[str, Any] | None = None
        # Pause gate: when paused the run thread blocks at the same hook sites
        # the stop check uses, staying responsive to Stop. pause_wake is
        # signalled on resume (or stop) to release the wait promptly.
        self.paused = False
        self.pause_wake = threading.Event()
        # Multi-tenant deployments stamp the owning session id here; the
        # single-user local server leaves it None.
        self.owner: str | None = None


def _honor_pause(rs, heartbeat) -> None:
    """Block the run thread while ``rs.paused``, staying responsive to Stop.

    Mirrors the paper-cap gate: waits on ``pause_wake`` (signalled by resume
    or stop) and sends a keepalive ``heartbeat()`` each second it stays held.
    Never raises — the caller re-checks ``stop_requested`` afterwards and
    raises ``_StopRun`` itself. Skipped once ``finalizing`` is set so a
    stop-then-finalize is never blocked at the pause gate.
    """
    while rs.paused and not rs.stop_requested and not rs.finalizing:
        if not rs.pause_wake.wait(timeout=1.0):
            heartbeat()


class WebDashboard:
    """DashboardLike shim — the CLI's two-level progress protocol, on the web.

    Core steps already narrate their inner life through the active dashboard
    (outer bar over source papers, inner phase bar per fetch/enrich/screen
    phase, S2 retry banners, notes). This shim forwards all of it to the
    browser as throttled ``activity`` events plus ``log`` lines, so the Run
    sidebar can show a real double progress bar and a liveness feed instead
    of going dark for the whole step.
    """

    def __init__(self, mgr: "RunManager", rs: RunState):
        self.mgr = mgr
        self.rs = rs
        self._outer: dict[str, Any] | None = None
        self._inner: dict[str, Any] | None = None
        self._retry: str | None = None
        self._seen = 0
        self._lane: str | None = None   # last "branch i/n · Step" phase seen
        # Per-lane state for Parallel steps: lane key -> {outer, state}.
        # The current lane's outer bar is stamped on every emit, so each
        # branch card in the step-detail page keeps its own source-papers
        # bar (live while running, frozen once the branch moves on).
        self._lanes: dict[str, dict[str, Any]] = {}
        self._last_emit = 0.0
        self._last_snap = 0.0
        self._last_retry_logged: str | None = None

    def _maybe_stop(self) -> None:
        # Mid-step stop/pause: steps drive these dashboard hooks from the run
        # thread between sources / acceptances, so acting here interrupts even
        # an hour-long step at a safe point. Never fires during the post-stop
        # finalize_partial (rs.finalizing), and only from main-thread call
        # sites — tick_inner may run on LLM worker threads.
        if self.rs.stop_requested and not self.rs.finalizing:
            raise _StopRun()
        if self.rs.paused and not self.rs.finalizing:
            _honor_pause(self.rs, lambda: self._emit(force=True))
            if self.rs.stop_requested and not self.rs.finalizing:
                raise _StopRun()

    # -- DashboardLike protocol ----------------------------------------
    def attach(self, ctx) -> None:
        # _expand_helpers reads ctx.dashboard (not the contextvar)
        ctx.dashboard = self

    def begin_run(self) -> None: ...

    def begin_step(self, idx: int, name: str, desc: str = "") -> None:
        self._outer = None
        self._inner = None
        self._retry = None
        self._seen = 0
        self._lane = None
        self._lanes = {}
        self._emit(force=True)

    def end_step(self, *, candidates: int | None = None) -> None:
        self._outer = None
        self._inner = None
        self._retry = None
        self._lane = None
        self._lanes = {}
        self._emit(force=True)

    def enable_outer_bar(self, total: int, *, description: str = "source papers") -> None:
        self._outer = {"desc": str(description), "total": int(total), "done": 0}
        self._emit(force=True)

    def advance_outer(self, n: int = 1) -> None:
        if self._outer:
            self._outer["done"] += n
        self._emit()
        self._live_snapshot()
        self._maybe_stop()

    def begin_phase(self, description: str, total) -> None:
        d = str(description)
        # Parallel announces "branch i/n · Step" before running each sub-step,
        # which then overwrites the phase with its own — latch the lane so the
        # step-detail page can highlight which branch is live throughout.
        if d.startswith("branch "):
            if self._lane and self._lane in self._lanes:
                prev = self._lanes[self._lane]
                prev["state"] = "done"
                # Stamp the finishing lane's final bar here — its last
                # advance_outer may have been throttled out of _emit.
                if self._outer:
                    prev["outer"] = dict(self._outer)
            self._lane = d
            self._lanes[d] = {"outer": None, "state": "run"}
            # The previous sub-step's outer bar must not bleed into the new
            # lane — the incoming sub-step enables its own.
            self._outer = None
        self._inner = {"desc": d,
                       "total": int(total) if total else None, "done": 0}
        self._emit(force=True)
        self._log("PHASE", d + (f" · {int(total):,} items" if total else ""))
        self._maybe_stop()

    def retotal_phase(self, total) -> None:
        if self._inner:
            self._inner["total"] = int(total) if total else None
        self._emit(force=True)

    def tick_inner(self, n: int = 1) -> None:
        if self._inner:
            self._inner["done"] += n
        self._emit()
        # Long single phases (e.g. Finalize's per-paper reference fetch)
        # mutate the collection's edge data without adding papers — refresh
        # the graph/metrics on the same throttle so the network visibly
        # densifies instead of jumping at step end.
        self._live_snapshot()

    def complete_phase(self) -> None:
        if self._inner and self._inner.get("total"):
            self._inner["done"] = self._inner["total"]
        self._emit(force=True)

    def note_candidates_seen(self, n: int = 1) -> None:
        self._seen += n
        self._emit()

    def paper_accepted(self, paper, *, saturation=None) -> None:
        # Steps announce every acceptance the moment it happens — stream it
        # so the accepted list and the network grow per source paper instead
        # of one giant flush when the (possibly Parallel) step returns.
        pid = getattr(paper, "paper_id", None)
        ctx = self.rs.ctx
        if pid and ctx is not None:
            self.rs.announced.add(pid)
            try:
                self.mgr.broadcast(self.rs, {
                    "type": "paper_added",
                    "paper": paper_dict(paper, seed_ids=ctx.seed_ids),
                })
            except Exception:
                pass  # never let a stream hiccup crash the run
        self._live_snapshot(min_gap=1.0)
        self._maybe_stop()
        self._check_cap()

    def _check_cap(self) -> None:
        """Hard paper cap: prompt the user the moment the limit is hit.

        Broadcasts ``cap_reached`` (the UI shows a modal with a 30s
        countdown), then BLOCKS the run thread until the user answers or
        the deadline passes. "raise" bumps ``max_papers_total`` and
        continues; "stop" or a timeout stops the run — finalize_partial
        keeps everything found so far.
        """
        rs = self.rs
        ctx = rs.ctx
        if ctx is None or rs.finalizing or rs.cap_prompted:
            return
        cap = int(getattr(ctx.config, "max_papers_total", 0) or 0)
        if cap <= 0 or len(ctx.collection) < cap:
            return
        rs.cap_prompted = True
        rs.cap_event.clear()
        rs.cap_decision = None
        n = len(ctx.collection)
        self.mgr.broadcast(rs, {"type": "cap_reached", "cap": cap,
                                "accepted": n, "timeout_s": 30})
        self._log("WARN", f"Paper cap reached ({n} ≥ {cap}) — run paused, waiting for your decision (30s)")
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not rs.stop_requested:
            if rs.cap_event.wait(timeout=1.0):
                break
            self._emit(force=True)  # heartbeat while the run is held
        self.mgr.broadcast(rs, {"type": "cap_resolved"})
        decision = rs.cap_decision or {}
        if decision.get("action") == "raise" and not rs.stop_requested:
            try:
                new_max = int(decision.get("max") or 0)
            except (TypeError, ValueError):
                new_max = 0
            if new_max > len(ctx.collection):
                ctx.config.max_papers_total = new_max
                rs.cap_prompted = False  # re-arm at the new threshold
                self._log("NOTE", f"Paper cap raised to {new_max} — continuing")
                return
            self._log("WARN", f"New cap {new_max} not above current count — stopping")
        # explicit stop, timeout, or unusable raise value
        self._log("WARN", "Stopping at the paper cap — finalizing current results")
        rs.stop_requested = True  # backstop: swallowed raises still stop at the next hook
        raise _StopRun()

    def set_retry_status(self, msg: str) -> None:
        msg = str(msg)
        self._retry = msg
        self._emit(force=True)
        # log each DISTINCT retry banner (attempt counter changes count)
        if msg != self._last_retry_logged:
            self._last_retry_logged = msg
            self._log("RETRY", msg)

    def clear_retry_status(self) -> None:
        if self._retry is not None:
            self._retry = None
            self._emit(force=True)

    def warn(self, msg: str) -> None:
        m = str(msg)
        # The core announces cap/budget stops via warn("Budget/cap reached —
        # stopping early"); remember it so skipped steps can say WHY.
        if "stopping early" in m.lower():
            self.rs.stop_note = "skipped — " + m
        self._log("WARN", m)

    def note(self, msg: str) -> None:
        self._log("NOTE", str(msg))

    def finalize(self) -> None: ...

    # -- plumbing -------------------------------------------------------
    def _log(self, tag: str, msg: str) -> None:
        self.mgr.broadcast(self.rs, {"type": "log", "log": {
            "t": time.strftime("%H:%M:%S"), "tag": tag, "msg": msg[:240]}})

    def _emit(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_emit < 0.25:
            return
        self._last_emit = now
        # Keep the current lane's outer bar in step — sub-steps drive
        # enable_outer_bar/advance_outer on the shared slot, and this stamp
        # is what lets a finished branch keep its frozen bar afterwards.
        if self._lane and self._lane in self._lanes:
            self._lanes[self._lane]["outer"] = dict(self._outer) if self._outer else None
        self.mgr.broadcast(self.rs, {"type": "activity", "activity": {
            "outer": dict(self._outer) if self._outer else None,
            "inner": dict(self._inner) if self._inner else None,
            "retry": self._retry,
            "seen": self._seen,
            "lane": self._lane,
            "lanes": {
                k: {"outer": dict(v["outer"]) if v.get("outer") else None,
                    "state": v.get("state", "run")}
                for k, v in self._lanes.items()
            } if self._lanes else None,
        }})

    def _live_snapshot(self, min_gap: float = 2.5) -> None:
        """Throttled mid-step metrics + graph broadcast.

        The event sink only snapshots at top-level step boundaries; this is
        the mid-step channel driven by dashboard traffic (acceptances, outer
        advances, inner ticks) so a long step — especially a Parallel block —
        streams instead of going dark.
        """
        now = time.monotonic()
        if now - self._last_snap < min_gap:
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


class _LogBridge(logging.Handler):
    """Forward citeclaw log records to the browser's live log.

    Captures the same chatter the CLI's file log sees (every S2 request /
    retry at DEBUG, LLM batch warnings, step notes) so the user can watch
    each tiny API step. Rate-limited so a pathological burst can't swamp
    the WebSocket.
    """

    def __init__(self, mgr: "RunManager", rs: RunState,
                 only_thread: str | None = None):
        super().__init__(level=logging.DEBUG)
        self.mgr = mgr
        self.rs = rs
        # Multi-tenant servers scope each bridge to its own run thread —
        # the "citeclaw" logger is process-global, so without this filter
        # concurrent runs would stream each other's log lines.
        self.only_thread = only_thread
        self._window_start = 0.0
        self._window_count = 0
        self._dropped = 0

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        try:
            if self.only_thread and record.threadName != self.only_thread:
                return
            now = time.monotonic()
            if now - self._window_start > 2.0:
                if self._dropped:
                    self._send("LOG", f"… {self._dropped} more log lines suppressed")
                self._window_start = now
                self._window_count = 0
                self._dropped = 0
            if self._window_count >= 30:
                self._dropped += 1
                return
            self._window_count += 1

            name = record.name
            if record.levelno >= logging.ERROR:
                tag = "ERR"
            elif record.levelno >= logging.WARNING:
                tag = "WARN"
            elif name.startswith("citeclaw.s2"):
                tag = "S2"
            elif "llm" in name or "screening" in name:
                tag = "LLM"
            else:
                tag = "LOG"
            self._send(tag, record.getMessage())
        except Exception:  # noqa: BLE001 — logging must never break the run
            pass

    def _send(self, tag: str, msg: str) -> None:
        self.mgr.broadcast(self.rs, {"type": "log", "log": {
            "t": time.strftime("%H:%M:%S"), "tag": tag, "msg": str(msg)[:240]}})


class WebFanoutSink:
    """EventSink that fans events out to WebSocket subscribers."""

    def __init__(self, mgr: "RunManager", rs: RunState):
        self.mgr = mgr
        self.rs = rs
        self._last_snap = 0.0

    def _check_stop(self) -> None:
        if self.rs.stop_requested:
            raise _StopRun()
        if self.rs.paused and not self.rs.finalizing:
            _honor_pause(self.rs, lambda: self._snapshot(force=True))
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
        # Skip papers WebDashboard.paper_accepted already streamed mid-step —
        # this runner-side event is the step-end collection diff, which
        # re-announces every acceptance.
        if p is not None and paper_id not in self.rs.announced:
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
            t = ev.get("type")
            if t in ("activity", "metrics", "graph"):
                rs.latest[t] = ev            # replay keeps only the newest
            elif t == "log":
                rs.log_tail.append(ev)       # replay keeps the last 200
            else:
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
            backlog = list(rs.events) + list(rs.log_tail)
            for key in ("metrics", "graph", "activity"):
                if key in rs.latest:
                    backlog.append(rs.latest[key])
            rs.subscribers.add(q)
        return q, backlog

    def unsubscribe(self, rs: RunState, q: asyncio.Queue) -> None:
        with rs.lock:
            rs.subscribers.discard(q)

    # ---- lifecycle ----------------------------------------------------
    # Multi-tenant subclasses flip this so each run's log bridge only
    # forwards records from its own run thread (see _LogBridge).
    scope_logs_to_thread = False

    def start_run(self, config, owner: str | None = None,
                  run_id: str | None = None) -> tuple[str, list[dict[str, Any]]]:
        run_id = run_id or uuid.uuid4().hex[:12]
        steps_meta = _steps_meta_from_config(config)
        rs = RunState(run_id, config, steps_meta)
        rs.owner = owner
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
        rs.cap_event.set()   # unblock a run held at the cap prompt
        rs.pause_wake.set()  # unblock a run held at the pause gate
        try:
            rs.progress.note_stopping()
            self.broadcast(rs, {"type": "progress", "progress": rs.progress.snapshot()})
        except Exception:
            pass
        return True

    def cap_decide(self, run_id: str, action: str, new_max: int | None) -> bool:
        """Answer a pending cap prompt: action 'stop' or 'raise' (+ new_max)."""
        rs = self.runs.get(run_id)
        if not rs:
            return False
        rs.cap_decision = {"action": action, "max": new_max}
        rs.cap_event.set()
        return True

    def pause_run(self, run_id: str) -> bool:
        """Hold the run at the next safe hook (staying responsive to Stop)."""
        rs = self.runs.get(run_id)
        if not rs:
            return False
        if rs.status not in ("running", "starting") or rs.stop_requested:
            return False
        rs.paused = True
        rs.pause_wake.clear()
        return True

    def resume_run(self, run_id: str) -> bool:
        """Release a paused run."""
        rs = self.runs.get(run_id)
        if not rs:
            return False
        rs.paused = False
        rs.pause_wake.set()
        return True

    def _run(self, rs: RunState) -> None:
        from citeclaw.pipeline import build_context, finalize_partial, run_pipeline

        rs.status = "running"
        self.broadcast(rs, {"type": "hello", "run_id": rs.run_id,
                            "started_at": time.time(),
                            "progress": rs.progress.snapshot()})
        # Fine-grained visibility: the dashboard shim narrates within-step
        # phases (run_pipeline installs it as ctx.dashboard + the active
        # contextvar dashboard); the log bridge relays every citeclaw log
        # line (S2 calls, retries, LLM warnings) to the browser's live log.
        dash = WebDashboard(self, rs)
        cc_logger = logging.getLogger("citeclaw")
        prior_level = cc_logger.level
        bridge = _LogBridge(self, rs,
                            only_thread=(threading.current_thread().name
                                         if self.scope_logs_to_thread else None))
        cc_logger.addHandler(bridge)
        # let DEBUG records reach OUR handler; console handlers keep their
        # own (higher) levels, so the terminal stays quiet
        if prior_level == logging.NOTSET or prior_level > logging.DEBUG:
            cc_logger.setLevel(logging.DEBUG)
        # Per-run file log + the exact config used, written into the run dir so
        # the downloadable bundle is self-contained (the pipeline itself only
        # writes its result artifacts). Thread-scoped on multi-tenant servers so
        # one tenant's file never picks up another's lines.
        run_log_handler = None
        try:
            data_dir = Path(str(rs.config.data_dir))
            data_dir.mkdir(parents=True, exist_ok=True)
            _dump_run_config(rs.config, data_dir / "pipeline_config.json")
            run_log_handler = logging.FileHandler(data_dir / "citeclaw.log", encoding="utf-8")
            run_log_handler.setLevel(logging.DEBUG)
            run_log_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
            if self.scope_logs_to_thread:
                run_log_handler.addFilter(_ThreadLogFilter(threading.current_thread().name))
            cc_logger.addHandler(run_log_handler)
        except Exception:  # noqa: BLE001 - artifacts are best-effort; never block a run
            run_log_handler = None
        s2 = cache = None
        try:
            ctx, s2, cache = build_context(rs.config)
            rs.ctx = ctx
            sink = WebFanoutSink(self, rs)
            try:
                run_pipeline(ctx, event_sink=sink, dashboard=dash)
                rs.status = "done"
            except _StopRun:
                rs.finalizing = True
                rs.progress.begin_finalizing()
                self.broadcast(rs, {"type": "progress", "progress": rs.progress.snapshot()})
                try:
                    finalize_partial(ctx)
                except Exception:
                    pass
                rs.status = "stopped"
            except BaseException as e:  # noqa: BLE001 - surface any run error to UI
                rs.finalizing = True
                rs.progress.begin_finalizing()
                self.broadcast(rs, {"type": "progress", "progress": rs.progress.snapshot()})
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
            try:
                cc_logger.removeHandler(bridge)
                if run_log_handler is not None:
                    try:
                        run_log_handler.flush()
                        run_log_handler.close()
                    finally:
                        cc_logger.removeHandler(run_log_handler)
                cc_logger.setLevel(prior_level)
            except Exception:
                pass
            for closer in (s2, cache):
                try:
                    if closer is not None:
                        closer.close()
                except Exception:
                    pass

        # Resolve unreached steps (skipped/interrupted, with the reason) so
        # the list never ends with "pending" rows after Finalize already ran.
        try:
            rs.progress.finish(rs.status, rs.stop_note, finalized=rs.finalizing)
            self.broadcast(rs, {"type": "progress", "progress": rs.progress.snapshot()})
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
        try:
            self.on_run_end(rs)
        except Exception:  # noqa: BLE001 - bookkeeping must not mask run status
            pass

    def on_run_end(self, rs: RunState) -> None:
        """Hook for subclasses (quota release, artifact sync). No-op here."""


# process-wide singleton
manager = RunManager()
