"""Step registry + builder for the YAML pipeline schema.

:func:`build_step` is the single entry point used by
:mod:`citeclaw.config` to translate a step dict from a YAML
``pipeline:`` list into a concrete :class:`BaseStep` instance.
:data:`STEP_REGISTRY` maps the canonical step-name strings (the
YAML ``step:`` field) to small ``_build_<name>(d, blocks)`` factory
functions that read the per-step kwargs and forward them to the
constructor.

Adding a new step is a 4-step recipe:

1. Implement the step class in ``citeclaw/steps/<name>.py``
   exposing ``name`` and a ``run(signal, ctx) -> StepResult``.
2. Import it at the top of this module.
3. Add a ``_build_<name>(d, blocks)`` factory that pulls the
   step-specific kwargs from ``d`` (the YAML dict) and forwards
   them.
4. Register it in :data:`STEP_REGISTRY` under the user-facing name.

The ``_optional_screener`` / ``_resolve`` helpers handle the common
"screener: <named-block-or-inline-dict>" YAML shape so individual
step factories don't have to replicate the dispatch.
"""

from __future__ import annotations

from typing import Any, Callable

from citeclaw.steps.base import BaseStep, StepResult  # noqa: F401
from citeclaw.steps.cluster import Cluster
from citeclaw.steps.expand_backward import ExpandBackward
from citeclaw.steps.expand_by_author import ExpandByAuthor
from citeclaw.steps.expand_by_search import ExpandBySearch
from citeclaw.steps.expand_by_pdf import ExpandByPDF
from citeclaw.steps.expand_by_semantics import ExpandBySemantics
from citeclaw.steps.expand_forward import ExpandForward
from citeclaw.steps.finalize import Finalize
from citeclaw.steps.human_in_the_loop import HumanInTheLoop
from citeclaw.steps.load_seeds import LoadSeeds
from citeclaw.steps.merge_duplicates import MergeDuplicates
from citeclaw.steps.parallel import Parallel
from citeclaw.steps.rerank import Rerank
from citeclaw.steps.rescreen import ReScreen
from citeclaw.steps.resolve_seeds import ResolveSeeds


def _resolve(name_or_dict: Any, blocks: dict):
    """Resolve a screener spec to a built filter block.

    Accepts either a string (looked up in ``blocks``, the named
    blocks dict from the YAML ``blocks:`` section) or an inline
    block dict (built lazily via
    :func:`citeclaw.filters.builder.build_blocks` under an anonymous
    name). Raises :class:`KeyError` on unknown named refs so a YAML
    typo fails fast.
    """
    if isinstance(name_or_dict, str):
        if name_or_dict not in blocks:
            raise KeyError(f"Screener block {name_or_dict!r} not defined")
        return blocks[name_or_dict]
    from citeclaw.filters.builder import build_blocks
    # inline anonymous block
    return build_blocks({"_anon": name_or_dict})["_anon"]


def _optional_screener(d: dict, blocks: dict):
    """Resolve ``d['screener']`` if present (named ref or inline dict), else None."""
    if "screener" not in d:
        return None
    return _resolve(d["screener"], blocks)


def _build_load_seeds(d: dict, blocks: dict) -> BaseStep:
    return LoadSeeds(file=d.get("file"))


def _build_resolve_seeds(d: dict, blocks: dict) -> BaseStep:
    return ResolveSeeds(
        include_siblings=bool(d.get("include_siblings", False)),
    )


def _build_expand_forward(d: dict, blocks: dict) -> BaseStep:
    return ExpandForward(
        max_citations=int(d.get("max_citations", 100)),
        screener=_optional_screener(d, blocks),
    )


def _build_expand_backward(d: dict, blocks: dict) -> BaseStep:
    return ExpandBackward(
        screener=_optional_screener(d, blocks),
        pdf_references=bool(d.get("pdf_references", False)),
        pdf_model=d.get("pdf_model"),
        headless=bool(d.get("headless", True)),
        openalex_references=bool(d.get("openalex_references", True)),
    )


def _build_expand_by_search(d: dict, blocks: dict) -> BaseStep:
    """Build an ``ExpandBySearch`` step from its YAML dict.

    The ``agent:`` sub-dict is passed straight through as a raw mapping;
    the step shell only validates that it is a mapping (or omitted) and
    leaves schema enforcement to whoever wires up the agent backend.
    """
    agent_raw = d.get("agent") or {}
    if not isinstance(agent_raw, dict):
        raise ValueError(
            "ExpandBySearch.agent must be a mapping (or omitted to use defaults)"
        )
    return ExpandBySearch(
        agent=agent_raw,
        screener=_optional_screener(d, blocks),
        topic_description=d.get("topic_description"),
        max_anchor_papers=int(d.get("max_anchor_papers", 20)),
        apply_local_query_args=d.get("apply_local_query_args"),
    )


def _build_expand_by_pdf(d: dict, blocks: dict) -> BaseStep:
    """Build an ``ExpandByPDF`` step from its YAML dict."""
    return ExpandByPDF(
        screener=_optional_screener(d, blocks),
        topic_description=d.get("topic_description"),
        model=d.get("model"),
        reasoning_effort=d.get("reasoning_effort", "high"),
        max_papers=int(d["max_papers"]) if "max_papers" in d else None,
        max_input_chars=int(d.get("max_input_chars", 24_000)),
        headless=bool(d.get("headless", True)),
    )


def _build_expand_by_semantics(d: dict, blocks: dict) -> BaseStep:
    """Build an ``ExpandBySemantics`` step from its YAML dict."""
    return ExpandBySemantics(
        screener=_optional_screener(d, blocks),
        max_anchor_papers=int(d.get("max_anchor_papers", 10)),
        limit=int(d.get("limit", 100)),
        use_rejected_as_negatives=bool(d.get("use_rejected_as_negatives", False)),
    )


def _build_expand_by_author(d: dict, blocks: dict) -> BaseStep:
    """Build an ``ExpandByAuthor`` step from its YAML dict."""
    return ExpandByAuthor(
        screener=_optional_screener(d, blocks),
        top_k_authors=int(d.get("top_k_authors", 10)),
        author_metric=str(d.get("author_metric", "degree_in_collab_graph")),
        papers_per_author=int(d.get("papers_per_author", 50)),
    )


def _build_human_in_the_loop(d: dict, blocks: dict) -> BaseStep:
    """Build a ``HumanInTheLoop`` step from its YAML dict.

    The PD-02 v2 schema:

      - ``enabled`` (default ``False``) — opt-in flag so unattended
        runs never block on input.
      - ``min_delay_sec`` (default 180s = 3 min) — minimum elapsed time
        from pipeline start before HITL takes input. If less time has
        passed when the step is reached, it sleeps the remainder.
      - ``first_prompt_timeout_sec`` (default 60s = 1 min) — wallclock
        deadline for the FIRST prompt. If no response in that window,
        the step bails with no labels and the pipeline keeps going.
      - ``k`` (default 10) — number of papers to label.
      - ``include_accepted`` / ``include_rejected`` — sampling pools.
      - ``balance_by_filter`` — round-robin from each LLM rejection
        bucket so per-filter agreement gets a fair sample.
      - ``seed`` — RNG seed for deterministic test fixtures.
    """
    return HumanInTheLoop(
        enabled=bool(d.get("enabled", False)),
        min_delay_sec=int(d.get("min_delay_sec", 180)),
        first_prompt_timeout_sec=int(d.get("first_prompt_timeout_sec", 60)),
        k=int(d.get("k", 10)),
        include_accepted=bool(d.get("include_accepted", True)),
        include_rejected=bool(d.get("include_rejected", True)),
        balance_by_filter=bool(d.get("balance_by_filter", True)),
        seed=d.get("seed"),
    )


def _build_rerank(d: dict, blocks: dict) -> BaseStep:
    return Rerank(
        metric=d.get("metric", "citation"),
        k=int(d.get("k", 100)),
        diversity=d.get("diversity"),  # None | str | dict
    )


def _build_rescreen(d: dict, blocks: dict) -> BaseStep:
    return ReScreen(
        screener=_optional_screener(d, blocks),
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
    "LoadSeeds":         _build_load_seeds,
    "ResolveSeeds":      _build_resolve_seeds,
    "ExpandForward":     _build_expand_forward,
    "ExpandBackward":    _build_expand_backward,
    "ExpandByPDF":       _build_expand_by_pdf,
    "ExpandBySearch":    _build_expand_by_search,
    "ExpandBySemantics": _build_expand_by_semantics,
    "ExpandByAuthor":    _build_expand_by_author,
    "HumanInTheLoop":    _build_human_in_the_loop,
    "Rerank":            _build_rerank,
    "ReScreen":          _build_rescreen,
    "Finalize":          _build_finalize,
    "Parallel":          _build_parallel,
    "MergeDuplicates":   _build_merge_duplicates,
    "Cluster":           _build_cluster,
}


def build_step(d: dict, blocks: dict | None = None) -> BaseStep:
    """Translate a YAML step dict into a concrete :class:`BaseStep`.

    ``d["step"]`` selects the factory from :data:`STEP_REGISTRY`;
    ``blocks`` is the dict of named filter blocks produced by
    :func:`citeclaw.filters.builder.build_blocks`. Raises
    :class:`ValueError` on an unknown step name so YAML typos fail
    fast rather than silently dropping a pipeline stage.
    """
    blocks = blocks or {}
    name = d.get("step")
    factory = STEP_REGISTRY.get(name)
    if factory is None:
        raise ValueError(f"Unknown step: {name!r}")
    return factory(d, blocks)
