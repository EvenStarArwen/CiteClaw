"""Walk citeclaw's built pipeline into a render-friendly node tree.

Each Step in ``Settings.pipeline_built`` becomes a :class:`StepNode` with
its display-relevant params already extracted. ``Parallel`` steps get
nested branches (list of list of StepNode); every other step has no
children. Renderers consume the StepNode list — they never have to poke
into Step internals themselves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepNode:
    idx: int
    name: str
    params: list[tuple[str, str]] = field(default_factory=list)
    branches: list[list["StepNode"]] = field(default_factory=list)


def _params_for(step: Any) -> list[tuple[str, str]]:
    """Pick the few attributes worth showing per step type.

    The convention is: ``screener=yes`` is implied for every step that
    accepts a screener — we omit it. Only parameters that *differ from
    defaults* or *change behaviour observably* are surfaced.
    """
    out: list[tuple[str, str]] = []
    n = step.name

    def add(k: str, v: Any) -> None:
        if v is None or v == "" or v is False:
            return
        if isinstance(v, dict):
            if v:
                out.append((k, ", ".join(f"{kk}={vv}" for kk, vv in v.items())))
        elif isinstance(v, list):
            if v:
                out.append((k, f"[{len(v)} items]"))
        else:
            out.append((k, str(v)))

    if n == "ExpandForward":
        add("max_citations", getattr(step, "max_citations", None))
    elif n == "ExpandBackward":
        if getattr(step, "pdf_references", False):
            parser = getattr(step, "parser", None)
            add("pdf_refs", parser or "true")
    elif n == "ExpandBySearch":
        agent = getattr(step, "agent", {}) or {}
        if "reasoning_effort" in agent:
            add("reasoning", agent["reasoning_effort"])
        if "max_papers_per_iteration" in agent:
            add("max_papers", agent["max_papers_per_iteration"])
    elif n == "ExpandBySemantics":
        add("mode", getattr(step, "mode", None))
        if getattr(step, "mode", "") == "per_paper":
            add("recs", getattr(step, "recs_per_paper", None))
        else:
            add("limit", getattr(step, "limit", None))
    elif n == "ExpandByAuthor":
        add("top_k_authors", getattr(step, "top_k_authors", None))
        add("author_metric", getattr(step, "author_metric", None))
    elif n == "ExpandByPDF":
        add("max_papers", getattr(step, "max_papers", None))
        add("model", getattr(step, "model", None))
    elif n == "Rerank":
        add("metric", getattr(step, "metric", None))
        add("k", getattr(step, "k", None))
        div = getattr(step, "diversity", None)
        if div:
            add("diversity", "yes")
    elif n == "Cluster":
        add("store_as", getattr(step, "store_as", None))
        algo = getattr(step, "algorithm", None)
        if isinstance(algo, dict):
            add("algorithm", algo.get("type"))
    elif n == "HumanInTheLoop":
        if getattr(step, "enabled", False):
            add("enabled", "yes")
        add("k", getattr(step, "k", None))
    elif n == "Parallel":
        add("branches", len(getattr(step, "branches", []) or []))

    return out


def extract(steps: list, start_idx: int = 1) -> list[StepNode]:
    """Turn a built pipeline (list of Step objects) into StepNode tree."""
    nodes: list[StepNode] = []
    for i, step in enumerate(steps, start=start_idx):
        node = StepNode(
            idx=i,
            name=step.name,
            params=_params_for(step),
        )
        if step.name == "Parallel":
            for branch in getattr(step, "branches", []) or []:
                node.branches.append(extract(branch, start_idx=1))
        nodes.append(node)
    return nodes
