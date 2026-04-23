"""MainPathResult + CyclePolicyTrace — immutable output shapes for the
main-path analysis subcommand.

The MPA pipeline is:

    GraphML → (cycle policy) → (weight) → (search) → subgraph + order

:class:`MainPathResult` is the runner's single return value; it carries
both the extracted subgraph (an ``igraph.Graph``) and enough provenance
(the three variant names, per-edge weights, cycle-handler trace, free-
form stats) that callers can render a JSON summary without re-running
anything.

:class:`CyclePolicyTrace` records what the cycle handler did — how
many non-trivial SCCs it found, which paper ids were collapsed into
each supernode (shrink-family only), and node/edge counts before and
after. The runner embeds a compact projection of this into
:attr:`MainPathResult.stats` for the JSON summary and DEBUG-logs the
full mapping.

Neither class is a Protocol — the MPA pipeline composes concrete
functions from module-level registries (:data:`WEIGHT_REGISTRY`,
:data:`SEARCH_REGISTRY`, :data:`CYCLE_REGISTRY` in
:mod:`citeclaw.mainpath`), not plugin classes — so there's no
structural type the builder needs to enforce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import igraph as ig


@dataclass(frozen=True)
class CyclePolicyTrace:
    """What the cycle handler did — used for logging and run summary.

    ``scc_sizes`` lists the sizes of every non-trivial SCC (size ≥ 2)
    that was found in the input. ``supernode_members`` maps the
    representative paper_id of each collapsed SCC to the list of its
    member paper_ids; empty for the preprint-transform policy which
    never collapses.

    ``n_nodes_*`` / ``n_edges_*`` fields document the graph-size
    change induced by the policy, so the run summary can surface
    "shrink collapsed 8 SCCs; 456 → 412 nodes, 2585 → 2108 edges".
    """

    policy: str
    scc_sizes: list[int] = field(default_factory=list)
    supernode_members: dict[str, list[str]] = field(default_factory=dict)
    n_nodes_before: int = 0
    n_nodes_after: int = 0
    n_edges_before: int = 0
    n_edges_after: int = 0


@dataclass(frozen=True)
class MainPathResult:
    """Immutable output of one :func:`citeclaw.mainpath.run_mpa` call.

    * ``subgraph`` — the extracted main-path network as a fresh
      ``igraph.Graph``. Edges are exactly the ones selected by the
      search; vertices are exactly the endpoints of those edges.
      Original node attributes from the input GraphML are preserved
      (``paper_id``, ``title``, ``year``, …); new attributes added:
      ``mp_role`` (``"source"`` / ``"sink"`` / ``"intermediate"``)
      on vertices, ``mp_weight`` (float) on edges.

    * ``edge_weights`` — ``{(src_paper_id, tgt_paper_id): float}`` for
      every edge in the *post-cycle-handler* DAG, not just the main
      path. Available to callers who want to overlay the full weight
      field on the original graph (e.g. for an interactive viewer).

    * ``paper_order`` — papers on the main path in topological order
      (oldest first). Duplicates removed. Intended for quick text or
      JSON summary rendering.

    * ``weight_variant`` / ``search_variant`` / ``cycle_policy`` —
      provenance strings matching the registry keys, so re-running
      with the same three strings produces the same output.

    * ``cycle_trace`` — see :class:`CyclePolicyTrace`.

    * ``stats`` — free-form diagnostic dict with input/output node +
      edge counts, the max / mean observed weight, and any other
      per-search diagnostics. Runner defensively copies this before
      forwarding to the JSON summary writer.
    """

    subgraph: "ig.Graph"
    edge_weights: dict[tuple[str, str], float]
    paper_order: list[str]
    weight_variant: str
    search_variant: str
    cycle_policy: str
    cycle_trace: CyclePolicyTrace
    stats: dict[str, Any] = field(default_factory=dict)
