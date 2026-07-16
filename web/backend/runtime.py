"""In-process run manager and pipeline-to-WebSocket bridge.

The first Web UI is intentionally local and single-run. CiteClaw's pipeline is
synchronous, so each run executes in a background thread while FastAPI remains
responsive. Events are retained in an append-only in-memory log; WebSocket
clients can connect late and replay from event zero without thread/async queue
coordination.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from citeclaw.config import Settings, _normalize_yaml
from citeclaw.context import HitlGate
from citeclaw.event_sink import EventSink
from citeclaw.network import build_citation_graph
from citeclaw.pipeline import build_context, run_pipeline

from catalog import SUPPORTED_MODEL, SUPPORTED_REASONING_EFFORT


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "runs" / "web"
TERMINAL_STATUSES = frozenset({"completed", "failed"})


class RunConflictError(RuntimeError):
    """Raised when a second run is requested while one is active."""


class UnsupportedRunConfiguration(ValueError):
    """Raised before network access for unsupported first-release options."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_error(exc: BaseException) -> str:
    text = str(exc).strip() or type(exc).__name__
    return text[:2_000]


def _paper_payload(paper: Any, *, seed: bool = False) -> dict[str, Any]:
    authors: list[str] = []
    for author in getattr(paper, "authors", []) or []:
        if isinstance(author, dict):
            name = author.get("name") or author.get("authorName")
            if name:
                authors.append(str(name))
        elif author:
            authors.append(str(author))
    return {
        "paper_id": paper.paper_id,
        "title": paper.title or "Untitled paper",
        "abstract": paper.abstract or "",
        "authors": authors,
        "year": paper.year,
        "venue": paper.venue or "",
        "citation_count": paper.citation_count or 0,
        "depth": paper.depth,
        "source": paper.source,
        "seed": seed or paper.source == "seed",
        "external_ids": dict(getattr(paper, "external_ids", {}) or {}),
    }


def graph_payload(ctx: Any) -> dict[str, Any]:
    """Build the compact citation graph consumed by sigma.js."""
    papers = dict(ctx.collection)
    nodes = [_paper_payload(paper, seed=pid in ctx.seed_ids) for pid, paper in papers.items()]
    graph = build_citation_graph(papers)
    ids = list(graph.vs["paper_id"]) if graph.vcount() else []
    edges = [{"source": ids[source], "target": ids[target]} for source, target in graph.get_edgelist()]
    return {"nodes": nodes, "edges": edges}


@dataclass
class RunSession:
    run_id: str
    config_name: str
    output_dir: Path
    config_snapshot: dict[str, Any]
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    status: str = "queued"
    error: str | None = None
    current_step: dict[str, Any] | None = None
    completed_steps: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    ctx: Any = None
    hitl_gate: HitlGate | None = None
    result: dict[str, Any] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def emit(self, event_type: str, **payload: Any) -> None:
        with self._lock:
            self.events.append(
                {
                    "seq": len(self.events),
                    "type": event_type,
                    "at": utc_now(),
                    **payload,
                }
            )

    def events_since(self, index: int) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            return list(self.events[index:]), len(self.events)

    def metrics(self) -> dict[str, Any]:
        ctx = self.ctx
        if ctx is None:
            return {
                "accepted": 0,
                "rejected": 0,
                "seen": 0,
                "llm_tokens": 0,
                "llm_input_tokens": 0,
                "llm_output_tokens": 0,
                "llm_reasoning_tokens": 0,
                "llm_calls": 0,
                "cost_usd": 0.0,
                "s2_requests": 0,
                "s2_cache_hits": 0,
                "rejection_counts": {},
                "elapsed_sec": 0,
            }
        budget = ctx.budget
        cost = budget.total_cost_usd()
        if cost == 0 and budget.llm_total_tokens:
            cost = budget.cost_estimate(ctx.config.screening_model)
        elapsed = 0
        if self.started_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.completed_at) if self.completed_at else datetime.now(UTC)
            elapsed = max(0, int((end - start).total_seconds()))
        return {
            "accepted": len(ctx.collection),
            "rejected": len(ctx.rejected),
            "seen": len(ctx.seen),
            "llm_tokens": budget.llm_total_tokens,
            "llm_input_tokens": budget.llm_input_tokens,
            "llm_output_tokens": budget.llm_output_tokens,
            "llm_reasoning_tokens": budget.llm_reasoning_tokens,
            "llm_calls": budget.llm_calls,
            "cost_usd": round(cost, 6),
            "s2_requests": budget.s2_requests,
            "s2_cache_hits": budget.s2_cache_hits,
            "rejection_counts": dict(ctx.rejection_counts),
            "elapsed_sec": elapsed,
        }

    def snapshot(self, *, include_graph: bool = False) -> dict[str, Any]:
        with self._lock:
            data: dict[str, Any] = {
                "run_id": self.run_id,
                "config_name": self.config_name,
                "output_dir": str(self.output_dir),
                "created_at": self.created_at,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "status": self.status,
                "error": self.error,
                "current_step": self.current_step,
                "completed_steps": list(self.completed_steps),
                "metrics": self.metrics(),
            }
        if include_graph and self.ctx is not None:
            data["graph"] = graph_payload(self.ctx)
        return data


class WebEventSink(EventSink):
    def __init__(self, session: RunSession) -> None:
        self.session = session

    def step_start(self, idx: int, name: str, description: str) -> None:
        payload = {"idx": idx, "name": name, "description": description}
        self.session.current_step = payload
        self.session.emit("step_start", **payload)

    def step_end(
        self,
        idx: int,
        name: str,
        in_count: int,
        out_count: int,
        delta_collection: int,
        stats: dict[str, Any],
    ) -> None:
        payload = {
            "idx": idx,
            "name": name,
            "in_count": in_count,
            "out_count": out_count,
            "delta_collection": delta_collection,
            "stats": dict(stats),
        }
        self.session.completed_steps.append(payload)
        self.session.current_step = None
        self.session.emit("step_end", **payload)
        self.session.emit("metrics", metrics=self.session.metrics())
        try:
            self.session.emit("graph_snapshot", graph=graph_payload(self.session.ctx))
        except Exception as exc:  # graph rendering must never abort a run
            self.session.emit("log", level="WARNING", message=f"Graph snapshot skipped: {exc}")

    def paper_added(self, paper_id: str, source: str) -> None:
        paper = self.session.ctx.collection.get(paper_id)
        payload: dict[str, Any] = {"paper_id": paper_id, "source": source}
        if paper is not None:
            payload["paper"] = _paper_payload(
                paper,
                seed=paper_id in self.session.ctx.seed_ids,
            )
        self.session.emit("paper_added", **payload)

    def paper_rejected(self, paper_id: str, category: str) -> None:
        self.session.emit("paper_rejected", paper_id=paper_id, category=category)

    def shape_table_update(self, rendered_shape: str) -> None:
        self.session.emit("shape_table_update", rendered_shape=rendered_shape)

    def hitl_request(self, run_id: str, papers: list[dict[str, Any]]) -> None:
        self.session.emit("hitl_request", run_id=run_id, papers=list(papers))


class _RunLogHandler(logging.Handler):
    def __init__(self, session: RunSession, secrets: list[str]) -> None:
        super().__init__(logging.INFO)
        self.session = session
        self.secrets = [secret for secret in secrets if secret]

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            for secret in self.secrets:
                message = message.replace(secret, "[redacted]")
            self.session.emit("log", level=record.levelname, message=message[:4_000])
        except Exception:
            pass


def _find_unsupported_values(config: Any) -> tuple[set[str], set[str]]:
    models: set[str] = set()
    efforts: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"screening_model", "search_model", "model", "pdf_model"} and isinstance(item, str) and item:
                    models.add(item)
                if key == "reasoning_effort" and isinstance(item, str) and item:
                    efforts.add(item)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(config)
    return models - {SUPPORTED_MODEL}, efforts - {SUPPORTED_REASONING_EFFORT}


def prepare_config(
    config_yaml: str,
    credentials: dict[str, str],
    *,
    output_dir: Path,
) -> tuple[dict[str, Any], Settings]:
    raw = yaml.safe_load(config_yaml) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must contain a top-level mapping.")
    raw = _normalize_yaml(raw)
    unsupported_models, unsupported_efforts = _find_unsupported_values(raw)
    selected = str(raw.get("screening_model") or "")
    if selected != SUPPORTED_MODEL or unsupported_models:
        found = sorted({selected, *unsupported_models} - {""})
        raise UnsupportedRunConfiguration(
            f"Model not supported in this first release: {', '.join(found) or '<empty>'}. Select {SUPPORTED_MODEL}."
        )
    effort = str(raw.get("reasoning_effort") or "")
    if effort != SUPPORTED_REASONING_EFFORT or unsupported_efforts:
        found = sorted({effort, *unsupported_efforts} - {""})
        raise UnsupportedRunConfiguration(
            f"Reasoning effort not supported in this first release: {', '.join(found) or '<empty>'}. "
            f"Select {SUPPORTED_REASONING_EFFORT}."
        )

    s2_key = (
        credentials.get("s2_api_key")
        or os.environ.get("CITECLAW_S2_API_KEY")
        or os.environ.get("S2_API_KEY")
        or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        or ""
    )
    gemini_key = (
        credentials.get("gemini_api_key")
        or os.environ.get("CITECLAW_GEMINI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or ""
    )
    openai_key = (
        credentials.get("openai_api_key")
        or os.environ.get("CITECLAW_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    if not s2_key:
        raise ValueError("Semantic Scholar API key is required for a live run.")
    if not gemini_key:
        raise ValueError(f"Gemini API key is required for {SUPPORTED_MODEL}.")

    snapshot = dict(raw)
    snapshot["data_dir"] = str(output_dir)
    values = {
        **snapshot,
        "s2_api_key": s2_key,
        "gemini_api_key": gemini_key,
        "openai_api_key": openai_key,
    }
    return snapshot, Settings(**values)


class RunManager:
    def __init__(self) -> None:
        self._sessions: dict[str, RunSession] = {}
        self._lock = threading.RLock()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            sessions = list(self._sessions.values())
        return [session.snapshot() for session in reversed(sessions)]

    def get(self, run_id: str) -> RunSession | None:
        with self._lock:
            return self._sessions.get(run_id)

    def start(
        self,
        *,
        config_yaml: str,
        config_name: str,
        credentials: dict[str, str],
    ) -> RunSession:
        with self._lock:
            if any(s.status not in TERMINAL_STATUSES for s in self._sessions.values()):
                raise RunConflictError("A CiteClaw run is already active.")

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        output_dir = RUNS_ROOT / run_id
        snapshot, settings = prepare_config(config_yaml, credentials, output_dir=output_dir)
        session = RunSession(
            run_id=run_id,
            config_name=config_name,
            output_dir=output_dir,
            config_snapshot=snapshot,
        )
        with self._lock:
            self._sessions[run_id] = session

        thread = threading.Thread(
            target=self._run,
            args=(
                session,
                settings,
                [
                    *credentials.values(),
                    settings.s2_api_key,
                    settings.gemini_api_key,
                    settings.openai_api_key,
                ],
            ),
            name=f"citeclaw-web-{run_id}",
            daemon=True,
        )
        thread.start()
        return session

    def _run(self, session: RunSession, settings: Settings, secrets: list[str]) -> None:
        session.status = "running"
        session.started_at = utc_now()
        session.output_dir.mkdir(parents=True, exist_ok=True)
        (session.output_dir / "config.yaml").write_text(
            yaml.safe_dump(session.config_snapshot, sort_keys=False),
            encoding="utf-8",
        )
        session.emit("run_started", snapshot=session.snapshot())

        logger = logging.getLogger("citeclaw")
        handler = _RunLogHandler(session, secrets)
        handler.setFormatter(logging.Formatter("%(name)s · %(message)s"))
        logger.addHandler(handler)
        telemetry_stop = threading.Event()

        def telemetry() -> None:
            while not telemetry_stop.wait(1.0):
                session.emit("metrics", metrics=session.metrics())

        telemetry_thread = threading.Thread(target=telemetry, daemon=True)
        final_status = "completed"
        try:
            ctx, _, _ = build_context(settings)
            session.ctx = ctx
            session.hitl_gate = HitlGate()
            ctx.hitl_gate = session.hitl_gate
            ctx.run_id = session.run_id
            sink = WebEventSink(session)
            ctx.event_sink = sink
            telemetry_thread.start()
            session.result = run_pipeline(ctx, event_sink=sink)
        except BaseException as exc:  # S2OutageError intentionally inherits BaseException
            final_status = "failed"
            session.error = _safe_error(exc)
            session.emit(
                "log",
                level="ERROR",
                message="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            )
        finally:
            telemetry_stop.set()
            if telemetry_thread.is_alive():
                telemetry_thread.join(timeout=2)
            logger.removeHandler(handler)
            handler.close()
            session.completed_at = utc_now()
            final_graph = {"nodes": [], "edges": []}
            if session.ctx is not None:
                try:
                    final_graph = graph_payload(session.ctx)
                except Exception as exc:
                    session.emit("log", level="WARNING", message=f"Final graph unavailable: {exc}")
            session.status = final_status
            session.emit(
                "run_complete",
                status=session.status,
                error=session.error,
                metrics=session.metrics(),
                graph=final_graph,
                output_dir=str(session.output_dir),
            )
            status_path = session.output_dir / "web_run.json"
            status_path.write_text(
                json.dumps(session.snapshot(), indent=2, default=str) + "\n",
                encoding="utf-8",
            )


run_manager = RunManager()
