"""Step registry + builder."""

from __future__ import annotations

from typing import Any, Callable

from citeclaw.steps.base import BaseStep, StepResult  # noqa: F401
from citeclaw.steps.cluster import Cluster
from citeclaw.steps.expand_backward import ExpandBackward
from citeclaw.steps.expand_by_search import ExpandBySearch
from citeclaw.steps.expand_forward import ExpandForward
from citeclaw.steps.finalize import Finalize
from citeclaw.steps.load_seeds import LoadSeeds
from citeclaw.steps.merge_duplicates import MergeDuplicates
from citeclaw.steps.parallel import Parallel
from citeclaw.steps.rerank import Rerank
from citeclaw.steps.rescreen import ReScreen


def _resolve(name_or_dict: Any, blocks: dict):
    if isinstance(name_or_dict, str):
        if name_or_dict not in blocks:
            raise KeyError(f"Screener block {name_or_dict!r} not defined")
        return blocks[name_or_dict]
    from citeclaw.filters.builder import build_blocks
    # inline anonymous block
    return build_blocks({"_anon": name_or_dict})["_anon"]


def _build_load_seeds(d: dict, blocks: dict) -> BaseStep:
    return LoadSeeds(file=d.get("file"))


def _build_expand_forward(d: dict, blocks: dict) -> BaseStep:
    return ExpandForward(
        max_citations=int(d.get("max_citations", 100)),
        screener=_resolve(d["screener"], blocks) if "screener" in d else None,
    )


def _build_expand_backward(d: dict, blocks: dict) -> BaseStep:
    return ExpandBackward(
        screener=_resolve(d["screener"], blocks) if "screener" in d else None,
    )


def _build_expand_by_search(d: dict, blocks: dict) -> BaseStep:
    """Build an ``ExpandBySearch`` step from its YAML dict.

    The ``agent:`` sub-dict is forwarded into ``AgentConfig`` kwargs
    so users can override any combination of iteration cap, token cap,
    target_count, model, reasoning_effort, etc. without learning a
    second schema.
    """
    from citeclaw.agents.iterative_search import AgentConfig

    agent_raw = d.get("agent") or {}
    if isinstance(agent_raw, AgentConfig):
        agent_cfg = agent_raw
    elif isinstance(agent_raw, dict):
        agent_cfg = AgentConfig(**agent_raw)
    else:
        raise ValueError(
            "ExpandBySearch.agent must be a mapping (or omitted to use defaults)"
        )
    return ExpandBySearch(
        agent=agent_cfg,
        screener=_resolve(d["screener"], blocks) if "screener" in d else None,
        topic_description=d.get("topic_description"),
        max_anchor_papers=int(d.get("max_anchor_papers", 20)),
        apply_local_query_args=d.get("apply_local_query_args"),
    )


def _build_rerank(d: dict, blocks: dict) -> BaseStep:
    return Rerank(
        metric=d.get("metric", "citation"),
        k=int(d.get("k", 100)),
        diversity=d.get("diversity"),  # None | str | dict
    )


def _build_rescreen(d: dict, blocks: dict) -> BaseStep:
    return ReScreen(
        screener=_resolve(d["screener"], blocks) if "screener" in d else None,
    )


def _build_finalize(d: dict, blocks: dict) -> BaseStep:
    return Finalize()


def _build_parallel(d: dict, blocks: dict) -> BaseStep:
    branches_raw = d.get("branches", [])
    branches = [[build_step(s, blocks) for s in branch] for branch in branches_raw]
    return Parallel(branches=branches)


def _build_merge_duplicates(d: dict, blocks: dict) -> BaseStep:
    return MergeDuplicates(
        title_threshold=float(d.get("title_threshold", 0.95)),
        semantic_threshold=float(d.get("semantic_threshold", 0.98)),
        year_window=int(d.get("year_window", 1)),
        use_embeddings=bool(d.get("use_embeddings", True)),
    )


def _build_cluster(d: dict, blocks: dict) -> BaseStep:
    return Cluster(
        store_as=d.get("store_as", ""),
        algorithm=d.get("algorithm"),
        naming=d.get("naming"),
        drop_noise=bool(d.get("drop_noise", False)),
    )


STEP_REGISTRY: dict[str, Callable[[dict, dict], BaseStep]] = {
    "LoadSeeds":       _build_load_seeds,
    "ExpandForward":   _build_expand_forward,
    "ExpandBackward":  _build_expand_backward,
    "ExpandBySearch":  _build_expand_by_search,
    "Rerank":          _build_rerank,
    "ReScreen":        _build_rescreen,
    "Finalize":        _build_finalize,
    "Parallel":        _build_parallel,
    "MergeDuplicates": _build_merge_duplicates,
    "Cluster":         _build_cluster,
}


def build_step(d: dict, blocks: dict | None = None) -> BaseStep:
    blocks = blocks or {}
    name = d.get("step")
    factory = STEP_REGISTRY.get(name)
    if factory is None:
        raise ValueError(f"Unknown step: {name!r}")
    return factory(d, blocks)
