"""ExpandBySearch V4 — facet skeleton + anchor grounding + structured ops.

Design principles (superseding V3's tutorial-style worker):

1. **Supervisor writes facet skeletons**, not just descriptions.
   Decomposition decision is operationalised: same facets across
   candidates → one sub-topic; structurally different facets →
   decompose. See :mod:`citeclaw.agents.v3.state` (Facet /
   FacetSkeleton) and :mod:`citeclaw.prompts.search_agent_v3`.

2. **Anchor-discovery agent** runs between ``set_strategy`` and
   worker dispatch for each sub-topic. Writes a precise S2 query,
   takes the top 15 by citation, marks each on-topic / off-topic,
   hands 5-15 confirmed anchors to the worker as real domain
   vocabulary. See :mod:`citeclaw.agents.v3.anchor_discovery`.

3. **Worker loop** starts with an amendment turn on the supervisor's
   skeleton (worker can add/remove a facet or reshape seeds), then
   emits an initial :class:`~state.QueryPlan` grounded in skeleton +
   anchors, then picks up to two closed-set transformations per
   iteration. See :mod:`citeclaw.agents.v3.transformations`.

4. **Anchor coverage** runs every iteration against the auto-
   injected anchor titles; ``present >= 80%`` + clean clusters
   triggers ``satisfied=true`` early termination. Total count + the
   query tree are demoted from standalone signals to context inside
   diagnose_plan. See :mod:`citeclaw.agents.v3.anchor_coverage`.

5. **Direction-neutral refinement.** Each iteration combines up to
   one widening op (pearl growing / loosen) and one narrowing op
   (prune / tighten). No "refine toward precision — never the
   reverse" language; ops are symmetric.
"""
