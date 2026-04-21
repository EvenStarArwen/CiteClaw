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


def _largest_remainder_allocate(
    n_slots: int, weights: list[int], caps: list[int],
) -> list[int]:
    """Apportion ``n_slots`` among items proportional to ``weights``,
    respecting per-item ``caps``, using Hamilton's largest-remainder method.

    The previous ``round()``-based distribution accumulated rounding drift
    and also drained the running ``surplus`` variable in-loop — later
    clusters systematically got less than their proportional share. This
    helper fixes both: compute ideal allocations up-front, floor them,
    then distribute the remainder by descending fractional part.

    Slots that can't be assigned because all cap-respecting items are
    full are returned unassigned (the caller's leftover-fill path mops
    them up).
    """
    n = len(weights)
    assert len(caps) == n, "weights and caps must be same length"
    if n == 0 or n_slots <= 0:
        return [0] * n
    total_weight = sum(weights)
    if total_weight <= 0:
        return [0] * n

    ideals = [w * n_slots / total_weight for w in weights]
    alloc = [min(int(ideals[i]), caps[i]) for i in range(n)]

    remaining = n_slots - sum(alloc)
    # Indices sorted by descending fractional remainder; ties broken
    # deterministically by input order (stable sort).
    order = sorted(range(n), key=lambda i: -(ideals[i] - int(ideals[i])))
    # Redistribute remaining slots one at a time in remainder order. Loop
    # until either the surplus is spent or every item is capped — the
    # outer ``while`` is what handles the over-capped-item overflow case
    # (first pass hits a cap, remaining slots cascade to uncapped items).
    while remaining > 0:
        progress = False
        for i in order:
            if remaining <= 0:
                break
            if alloc[i] < caps[i]:
                alloc[i] += 1
                remaining -= 1
                progress = True
        if not progress:
            break
    return alloc


def _plain_top_k(
    signal: list[PaperRecord], scores: dict[str, float], k: int,
) -> list[PaperRecord]:
    """Plain top-K fallback when there's no usable cluster signal.

    Used when (a) ``membership`` is empty altogether, or (b) every
    paper landed in the noise bucket. Papers without a score default
    to 0 so the sort is total-order-stable.
    """
    ranked = sorted(signal, key=lambda p: scores.get(p.paper_id, 0), reverse=True)
    return ranked[:k]


def _resolve_membership(
    cfg: dict | str, signal: list[PaperRecord], ctx,
) -> dict[str, int]:
    """Pick the right ``{paper_id -> cluster_id}`` source for ``cfg``.

    ``cfg`` may be ``{"cluster": "<store_as>"}`` (look up in
    ``ctx.clusters``) or any other shape that
    :func:`citeclaw.cluster.build_clusterer` accepts (build inline +
    run on the signal). Raises :class:`ValueError` when the named
    cluster doesn't exist so a config typo fails fast rather than
    silently degrading to plain top-K.
    """
    if isinstance(cfg, dict) and "cluster" in cfg:
        name = cfg["cluster"]
        result = ctx.clusters.get(name)
        if result is None:
            raise ValueError(
                f"Rerank diversity references unknown cluster {name!r}; "
                f"add a Cluster step earlier in the pipeline that stores it"
            )
        return result.membership
    clusterer = build_clusterer(cfg)
    return clusterer.cluster(signal, ctx).membership


def cluster_diverse_top_k(
    signal: list[PaperRecord],
    scores: dict[str, float],
    ctx,
    k: int,
    cfg: dict | str,
) -> list[PaperRecord]:
    """Top-K with stratified selection: >=1 per cluster, then proportional.

    Algorithm:

    1. Resolve a ``{paper_id -> cluster_id}`` map from ``cfg`` (either
       a previously-stored cluster reference or an inline build).
    2. If no usable cluster signal exists → plain top-K fallback.
    3. Group signal papers by cluster id (skipping noise = -1) and
       sort each group descending by score.
    4. Allocate slots:
       - **k <= n_clusters**: pick the k largest clusters, one slot each.
         Smaller clusters and noise papers fall through to the
         leftover-fill at the bottom.
       - **k > n_clusters**: floor 1 slot per cluster, then distribute
         the surplus ``(k - n_clusters)`` proportionally to cluster
         size via Hamilton's largest-remainder method (capped by
         per-cluster size).
    5. Take the top-N papers from each cluster per its allocation.
    6. Leftover fill: if step 5 produced fewer than k papers (some
       clusters didn't fill their quota or were shut out by step 4a),
       mop up the highest-scored remaining papers across the whole
       signal — noise papers (-1) are eligible here so they aren't
       permanently locked out.
    """
    membership = _resolve_membership(cfg, signal, ctx)

    if not membership:
        return _plain_top_k(signal, scores, k)

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
        return _plain_top_k(signal, scores, k)

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
        # Distribute the surplus proportionally via Hamilton's method
        # (largest-remainder). Weights = cluster sizes; caps = cluster size
        # minus the 1 slot already reserved.
        surplus = k - n_comms
        weights = [len(m) for _, m in sorted_comms]
        caps = [len(m) - 1 for _, m in sorted_comms]
        extras = _largest_remainder_allocate(surplus, weights, caps)
        for (c, _), extra in zip(sorted_comms, extras):
            slots[c] += extra

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
