"""cluster_diverse_top_k — stratified top-K selection with floor-then-proportional allocator.

The allocator is cluster-source-agnostic. The ``cfg`` argument can be either:

  * ``{"cluster": "<store_as>"}`` — reuse a cluster previously stored in
    ``ctx.clusters`` by an upstream :class:`~citeclaw.steps.cluster.Cluster`
    step. This is the recommended pattern: cluster once, reuse anywhere.
  * ``{"type": "walktrap", "n_communities": 3}`` (or other clusterer kwargs) —
    build a clusterer inline via :func:`~citeclaw.cluster.build_clusterer` and
    run it just for this Rerank invocation.
  * ``"walktrap"`` (a bare string name) — build a clusterer inline with no
    extra kwargs.
"""

from __future__ import annotations

from citeclaw.cluster import build_clusterer
from citeclaw.models import PaperRecord


def cluster_diverse_top_k(
    signal: list[PaperRecord],
    scores: dict[str, float],
    ctx,
    k: int,
    cfg: dict | str,
) -> list[PaperRecord]:
    """Top-K with stratified selection: >=1 per cluster (floor), then proportional."""
    if isinstance(cfg, dict) and "cluster" in cfg:
        name = cfg["cluster"]
        result = ctx.clusters.get(name)
        if result is None:
            raise ValueError(
                f"Rerank diversity references unknown cluster {name!r}; "
                f"add a Cluster step earlier in the pipeline that stores it"
            )
        membership = result.membership
    else:
        clusterer = build_clusterer(cfg)
        result = clusterer.cluster(signal, ctx)
        membership = result.membership

    if not membership:
        # Fallback: plain top-k
        ranked = sorted(signal, key=lambda p: scores.get(p.paper_id, 0), reverse=True)
        return ranked[:k]

    by_comm: dict[int, list[PaperRecord]] = {}
    for p in signal:
        c = membership.get(p.paper_id)
        if c is None or c == -1:
            # Skip noise / unassigned papers — they don't get a guaranteed
            # slot. They can still resurface in the leftover-fill loop.
            continue
        by_comm.setdefault(c, []).append(p)
    for c in by_comm:
        by_comm[c].sort(key=lambda p: scores.get(p.paper_id, 0), reverse=True)

    if not by_comm:
        # Every paper landed in -1 (noise). Fall back to plain top-k.
        ranked = sorted(signal, key=lambda p: scores.get(p.paper_id, 0), reverse=True)
        return ranked[:k]

    # Allocation in two phases:
    #   Phase A — reserve 1 slot per cluster (the floor that the user wants).
    #   Phase B — distribute the leftover (k - n_clusters) slots proportionally
    #             to cluster size, again capped by cluster size.
    # If k < n_clusters we can't give every cluster a slot; in that case we keep
    # the largest k clusters (by member count) and give each exactly 1.
    # Any slack from clusters smaller than their proportional share is mopped
    # up by the leftover-fill loop at the bottom.
    sorted_comms = sorted(by_comm.items(), key=lambda x: len(x[1]), reverse=True)
    n_comms = len(sorted_comms)

    slots: dict[int, int] = {}
    if k <= n_comms:
        # Not enough slots for every cluster — pick the k largest, one slot each.
        for c, _ in sorted_comms[:k]:
            slots[c] = 1
    else:
        # Floor: 1 slot per cluster.
        for c, _ in sorted_comms:
            slots[c] = 1
        # Distribute the surplus proportionally to size, capped by member count.
        surplus = k - n_comms
        total = sum(len(m) for _, m in sorted_comms)
        for c, members in sorted_comms:
            if surplus <= 0:
                break
            extra = round(surplus * len(members) / total) if total else 0
            extra = min(extra, len(members) - slots[c], surplus)
            if extra < 0:
                extra = 0
            slots[c] += extra
            surplus -= extra
        # Any rounding remainder gets handed to the largest clusters that
        # still have headroom (largest-remainder method).
        for c, members in sorted_comms:
            if surplus <= 0:
                break
            headroom = len(members) - slots[c]
            if headroom > 0:
                add = min(headroom, surplus)
                slots[c] += add
                surplus -= add

    selected: list[PaperRecord] = []
    for c, n in slots.items():
        selected.extend(by_comm[c][:n])

    # Final leftover fill: if some clusters couldn't fill their allocation
    # (or were entirely shut out by k < n_clusters), mop up the highest-scored
    # remaining papers across ALL clusters to reach k. Noise papers (-1) are
    # also eligible here so they aren't permanently locked out by clustering.
    if len(selected) < k:
        chosen_ids = {p.paper_id for p in selected}
        leftover = sorted(
            [p for p in signal if p.paper_id not in chosen_ids],
            key=lambda p: scores.get(p.paper_id, 0),
            reverse=True,
        )
        selected.extend(leftover[: k - len(selected)])

    return selected
