"""High-level orchestration for the ``citeclaw mainpath`` subcommand.

:func:`run_mpa` reads a GraphML file, optionally collapses/transforms
cyclic parts, computes arc weights, extracts the main-path subgraph
using the chosen search variant, annotates it with ``mp_weight`` /
``mp_role`` attributes, writes the subgraph back as GraphML, and emits
a sibling JSON summary — returning a :class:`MainPathResult` for
programmatic callers.

All three pipeline axes (weight, search, cycle) are dispatched by
string key against the module-level registries
(:data:`~citeclaw.mainpath.weights.WEIGHT_REGISTRY`,
:data:`~citeclaw.mainpath.search.SEARCH_REGISTRY`,
:data:`~citeclaw.mainpath.cycles.CYCLE_REGISTRY`). Unknown keys raise
``ValueError`` with a list of valid options.

Defaults follow the consensus of the MPA literature:

* ``weight="spc"`` — Batagelj (2003) flow conservation + theoretical
  endorsement from Price & Evans (2025).
* ``search="key-route"`` — Liu, Lu & Ho (2019) recommend it as "the
  most relevant" variant because it guarantees the single
  highest-weighted arc lands on the output.
* ``cycle="shrink"`` — Liu, Lu & Ho (2019)'s "family" recommendation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import igraph as ig

from citeclaw.mainpath.base import MainPathResult
from citeclaw.mainpath.cycles import CYCLE_REGISTRY
from citeclaw.mainpath.search import SEARCH_REGISTRY
from citeclaw.mainpath.weights import WEIGHT_REGISTRY

log = logging.getLogger("citeclaw.mainpath.runner")


def _check_registry_key(name: str, key: str, registry: dict) -> None:
    """Raise a helpful ``ValueError`` if ``key`` isn't registered."""
    if key not in registry:
        raise ValueError(
            f"Unknown {name} {key!r} — valid options: {sorted(registry)}",
        )


def _annotate_subgraph(
    sub: ig.Graph,
    edge_weights: list[int],
    dag: ig.Graph,
    on_path_edges: set[int],
) -> dict[tuple[str, str], float]:
    """Fill ``mp_weight`` on sub edges and ``mp_role`` on sub vertices.

    Returns a dict mapping ``(src_pid, tgt_pid)`` to the original DAG
    arc weight — convenient for JSON summary rendering without
    re-traversing igraph objects.

    Assumes ``sub`` was produced via ``dag.subgraph_edges(on_path,
    delete_vertices=True)``, so sub edges correspond 1-to-1 to the
    selected DAG edges in insertion order.
    """
    on_path_sorted = sorted(on_path_edges)
    ew_dict: dict[tuple[str, str], float] = {}
    for sub_e_idx, dag_e_idx in enumerate(on_path_sorted):
        w = float(edge_weights[dag_e_idx])
        sub.es[sub_e_idx]["mp_weight"] = w
        src_pid = dag.vs[dag.es[dag_e_idx].source]["paper_id"]
        tgt_pid = dag.vs[dag.es[dag_e_idx].target]["paper_id"]
        ew_dict[(src_pid, tgt_pid)] = w

    for v in sub.vs:
        idx = v.index
        if sub.indegree(idx) == 0:
            v["mp_role"] = "source"
        elif sub.outdegree(idx) == 0:
            v["mp_role"] = "sink"
        else:
            v["mp_role"] = "intermediate"
    return ew_dict


def _paper_order(sub: ig.Graph) -> list[str]:
    """Papers on the main path in chronological (topological) order."""
    if sub.vcount() == 0:
        return []
    try:
        order = sub.topological_sorting(mode="out")
    except ig.InternalError:
        # Defensive: non-DAG subgraph shouldn't happen post-cycle-policy,
        # but if it does fall back to vertex index order.
        log.warning("Main-path subgraph is not a DAG; falling back to index order")
        order = list(range(sub.vcount()))
    return [sub.vs[v]["paper_id"] for v in order]


def _write_summary_json(
    summary_path: Path,
    result: MainPathResult,
    input_graph_path: Path,
) -> None:
    """Emit a compact JSON summary next to the output GraphML."""
    sub = result.subgraph
    papers_in_order: list[dict] = []
    pid_to_v = {v["paper_id"]: v for v in sub.vs}
    for pid in result.paper_order:
        v = pid_to_v.get(pid)
        if v is None:
            continue
        attrs = v.attributes()
        entry: dict = {"paper_id": pid, "role": attrs.get("mp_role", "")}
        title = attrs.get("title")
        if title:
            entry["title"] = title
        year = attrs.get("year")
        if year is not None:
            entry["year"] = int(year)
        venue = attrs.get("venue")
        if venue:
            entry["venue"] = venue
        papers_in_order.append(entry)

    summary = {
        "input_graph": str(input_graph_path),
        "weight_variant": result.weight_variant,
        "search_variant": result.search_variant,
        "cycle_policy": result.cycle_policy,
        "stats": dict(result.stats),
        "cycle_trace": asdict(result.cycle_trace),
        "papers": papers_in_order,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def run_mpa(
    *,
    graph_path: Path,
    output_path: Path,
    weight: str = "spc",
    search: str = "key-route",
    cycle: str = "shrink",
    key_routes: int = 1,
    tolerance: float = 0.0,
) -> MainPathResult:
    """Execute the main-path pipeline on a CiteClaw GraphML file.

    Writes two files next to ``output_path``:

    * ``output_path`` itself — the main-path subgraph as GraphML with
      ``mp_weight`` edge attr + ``mp_role`` vertex attr added.
    * ``output_path.with_suffix(".json")`` — a JSON summary listing
      papers in chronological order plus provenance / stats.

    Raises ``ValueError`` for unknown ``weight`` / ``search`` /
    ``cycle`` keys, or if the post-cycle-policy graph still isn't a
    DAG (bug / unsupported cycle shape).

    Numeric knobs are ignored by variants that don't use them —
    ``key_routes`` is for ``search="key-route"`` only, ``tolerance``
    is for ``search="multi-local"`` only.
    """
    _check_registry_key("weight", weight, WEIGHT_REGISTRY)
    _check_registry_key("search", search, SEARCH_REGISTRY)
    _check_registry_key("cycle", cycle, CYCLE_REGISTRY)

    g = ig.Graph.Read_GraphML(str(graph_path))
    log.info(
        "Loaded %s: %d nodes, %d edges",
        graph_path, g.vcount(), g.ecount(),
    )

    cycle_fn = CYCLE_REGISTRY[cycle]
    dag, trace = cycle_fn(g)
    if not dag.is_dag():
        raise RuntimeError(
            f"cycle policy {cycle!r} did not produce a DAG — "
            f"likely a pathological input graph",
        )

    weight_fn = WEIGHT_REGISTRY[weight]
    edge_weights = weight_fn(dag)
    max_w = max(edge_weights) if edge_weights else 0
    n_positive = sum(1 for w in edge_weights if w > 0)
    log.info(
        "Weights %s: max=%d, %d/%d arcs carry positive weight",
        weight, max_w, n_positive, len(edge_weights),
    )

    search_fn = SEARCH_REGISTRY[search]
    on_path_edges = search_fn(
        dag, edge_weights, k=key_routes, tolerance=tolerance,
    )
    log.info(
        "Search %s: %d arcs selected (k=%d, tolerance=%g)",
        search, len(on_path_edges), key_routes, tolerance,
    )

    if not on_path_edges:
        log.warning(
            "Main path is empty — check that the input graph has "
            "at least one path from an in-degree-0 vertex to an "
            "out-degree-0 vertex",
        )
        sub = ig.Graph(n=0, directed=True)
        edge_weights_dict: dict[tuple[str, str], float] = {}
        order: list[str] = []
    else:
        sub = dag.subgraph_edges(
            list(on_path_edges), delete_vertices=True,
        )
        edge_weights_dict = _annotate_subgraph(
            sub, edge_weights, dag, on_path_edges,
        )
        order = _paper_order(sub)

    stats = {
        "n_input_nodes": g.vcount(),
        "n_input_edges": g.ecount(),
        "n_dag_nodes": dag.vcount(),
        "n_dag_edges": dag.ecount(),
        "n_mainpath_nodes": sub.vcount(),
        "n_mainpath_edges": sub.ecount(),
        "max_weight": int(max_w) if isinstance(max_w, (int, float)) else max_w,
        "n_positive_weight_arcs": n_positive,
        "key_routes": key_routes if search == "key-route" else None,
        "tolerance": tolerance if search == "multi-local" else None,
    }

    result = MainPathResult(
        subgraph=sub,
        edge_weights=edge_weights_dict,
        paper_order=order,
        weight_variant=weight,
        search_variant=search,
        cycle_policy=cycle,
        cycle_trace=trace,
        stats=stats,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if sub.vcount() > 0:
        sub.write_graphml(str(output_path))
    else:
        output_path.write_text(
            '<?xml version="1.0"?>\n<graphml><graph edgedefault="directed"/></graphml>\n',
        )
    summary_path = output_path.with_suffix(".json")
    _write_summary_json(summary_path, result, graph_path)
    log.info(
        "Wrote main-path subgraph: %s (%d nodes, %d edges) + %s",
        output_path, sub.vcount(), sub.ecount(), summary_path.name,
    )
    return result
