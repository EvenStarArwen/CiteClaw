"""ExpandBySearch V4.1 — anchor grounding + worker-owned facets.

Design principles:

1. **Supervisor is plan-only.** It decides sub-topic count using the
   anchor-shared facet test — same facets → 1 sub-topic, different
   facets → decompose — but it emits just ``{id, description}`` per
   sub-topic. No facet skeleton, no query sketches. Parent topic is
   re-shown every react turn so ``NOT`` scope stays enforced.

2. **Anchor-discovery agent** runs between ``set_strategy`` and
   worker dispatch for each sub-topic. Writes a precise S2 query
   from the description, takes the top 15 by citation, marks each
   on-topic / off-topic, hands 5-15 confirmed anchors to the worker
   as real domain vocabulary. See
   :mod:`citeclaw.agents.v3.anchor_discovery`.

3. **Worker owns facets.** On iter 0 the worker reads the anchors
   and designs its own facets in :func:`~state.QueryPlan` form.
   From iter 1 on it picks up to two transformations from a closed
   set of eight ops (``add/remove_or_alternative``,
   ``tighten/loosen_term``, ``swap_operator``,
   ``add_facet``, ``remove_facet``, ``add_exclusion``) and the
   system mutates the plan in place. See
   :mod:`citeclaw.agents.v3.transformations`.

4. **Anchor coverage** runs every iteration against the auto-
   injected anchor titles; ``present >= 80%`` + clean clusters
   triggers ``satisfied=true`` early termination. Total count + the
   query tree are informational context inside ``diagnose_plan``,
   not standalone signals. See
   :mod:`citeclaw.agents.v3.anchor_coverage`.

5. **Direction-neutral refinement.** Each iteration combines up to
   one widening op (pearl growing / loosen) and one narrowing op
   (prune / tighten). Ops are symmetric; no "refine toward
   precision — never the reverse" language.
"""
