"""Cluster step — run a clusterer over the signal and store the result.

The standalone "lego block" for clustering. A ``Cluster`` step takes a
clusterer config (graph-based or embedding-based — the registry is in
:mod:`citeclaw.cluster`), runs it once over the current signal, and stores
the resulting :class:`~citeclaw.cluster.base.ClusterResult` in
``ctx.clusters[store_as]``. Downstream steps reference the cluster by
name.

The step is signal-pass-through by default — it enriches ``ctx`` but
leaves the signal unchanged. Set ``drop_noise: true`` to filter out papers
that landed in cluster ``-1`` (e.g. HDBSCAN noise points or papers without
embeddings).

Example::

    pipeline:
      - step: Cluster
        store_as: forward_topics
        algorithm:
          type: topic_model
          min_cluster_size: 5
        naming:
          mode: both
          n_keywords: 10
          n_representative: 5

      - step: Rerank
        metric: pagerank
        k: 20
        diversity:
          cluster: forward_topics
"""

from __future__ import annotations

import logging
from typing import Any

from citeclaw.cluster import build_clusterer
from citeclaw.cluster.representation import (
    extract_keywords_ctfidf,
    name_topics_via_llm,
    select_representative_papers,
)
from citeclaw.models import PaperRecord
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.cluster")


class Cluster:
    """Run a clusterer over the signal and store the result on ``ctx``.

    Standalone "lego block" for clustering — see module docstring for
    the YAML config recipe and the ``Rerank`` cross-reference pattern.
    """

    name = "Cluster"

    def __init__(
        self,
        *,
        store_as: str,
        algorithm: dict | str,
        naming: dict | None = None,
        drop_noise: bool = False,
    ) -> None:
        """Configure the cluster step.

        Parameters
        ----------
        store_as:
            Required key under which the resulting `ClusterResult` is
            stored in `ctx.clusters` for downstream steps to look up.
            Empty / missing raises `ValueError` immediately so YAML
            typos fail fast.
        algorithm:
            Dict (or bare string) routed through
            :func:`citeclaw.cluster.build_clusterer`. Required —
            empty raises `ValueError`.
        naming:
            Optional ``{mode: none|tfidf|llm|both, n_keywords, n_representative,
            model, reasoning_effort}`` dict. Invalid `mode` value
            raises `ValueError`. Defaults to no naming.
        drop_noise:
            When True, papers in cluster -1 (noise / unembeddable) are
            dropped from the returned signal. The cluster result still
            stores their membership, just under -1.
        """
        if not store_as:
            raise ValueError("Cluster step requires a non-empty 'store_as'")
        if not algorithm:
            raise ValueError("Cluster step requires an 'algorithm' spec")
        self.store_as = store_as
        self.algorithm = algorithm
        self.naming: dict[str, Any] = dict(naming or {})
        self.drop_noise = drop_noise

        mode = self.naming.get("mode", "none")
        if mode not in ("none", "tfidf", "llm", "both"):
            raise ValueError(
                f"Cluster.naming.mode must be one of "
                f"'none' | 'tfidf' | 'llm' | 'both', got {mode!r}"
            )

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        """Cluster → optional naming → store → optional noise-drop.

        Four-step flow: (1) build the clusterer + run on the signal;
        (2) optionally enrich `result.metadata` with c-TF-IDF keywords
        and/or LLM-named labels (driven by ``naming.mode``); (3) store
        the `ClusterResult` under `store_as` (logging at WARNING on
        key collision so the user notices); (4) optionally drop
        noise-bucket papers from the returned signal.

        The embedding-fetch path for LLM naming logs at WARNING +
        skips on failure (no LLM names attached, but the cluster
        result + keywords still go to ctx).
        """
        if not signal:
            return StepResult(
                signal=[],
                in_count=0,
                stats={"store_as": self.store_as, "n_clusters": 0, "n_noise": 0},
            )

        dash = ctx.dashboard
        dash.note_candidates_seen(len(signal))

        # 1. Build clusterer + run.
        algo_name = self.algorithm.get("type") if isinstance(self.algorithm, dict) else self.algorithm
        dash.begin_phase(f"clustering · {algo_name}", total=1)
        clusterer = build_clusterer(self.algorithm)
        result = clusterer.cluster(signal, ctx)
        dash.tick_inner(1)

        # 2. Naming pipeline (optional, mode-driven).
        mode = self.naming.get("mode", "none")
        n_keywords = int(self.naming.get("n_keywords", 10))
        n_repr = int(self.naming.get("n_representative", 5))

        if mode in ("tfidf", "both") and result.metadata:
            dash.begin_phase("extract keywords · c-TF-IDF", total=1)
            keywords = extract_keywords_ctfidf(
                result.membership,
                ctx.collection,
                n_keywords=n_keywords,
            )
            for cid, kws in keywords.items():
                if cid in result.metadata:
                    result.metadata[cid].keywords = kws
            dash.tick_inner(1)

        if mode in ("llm", "both") and result.metadata:
            dash.begin_phase("fetch SPECTER2 embeddings", total=1)
            # Need representative papers (centroid-closest in embedding space).
            try:
                ids = list(result.membership.keys())
                embeddings = ctx.s2.fetch_embeddings_batch(ids)
            except Exception as exc:
                log.warning(
                    "cluster step %r: embedding fetch failed (%s); "
                    "skipping LLM naming",
                    self.store_as, exc,
                )
                embeddings = {}
            dash.tick_inner(1)

            if embeddings:
                reps = select_representative_papers(
                    result.membership, embeddings, n=n_repr,
                )
                for cid, paper_ids in reps.items():
                    if cid in result.metadata:
                        result.metadata[cid].representative_papers = paper_ids

                # Build the LLM client honouring per-step model overrides.
                from citeclaw.clients.llm import build_llm_client
                model_override = self.naming.get("model")
                reasoning_override = self.naming.get("reasoning_effort")
                client = build_llm_client(
                    ctx.config, ctx.budget,
                    model=model_override,
                    reasoning_effort=reasoning_override,
                    cache=getattr(ctx, "cache", None),
                )
                dash.begin_phase("name topics · LLM", total=max(1, len(result.metadata)))
                name_topics_via_llm(
                    result.metadata,
                    ctx.collection,
                    client=client,
                    max_workers=max(1, ctx.config.llm_concurrency),
                )
                dash.tick_inner(len(result.metadata))

        # 3. Store on context (warn on key collision so users notice).
        if self.store_as in ctx.clusters:
            log.warning(
                "Cluster step overwriting existing ctx.clusters[%r]",
                self.store_as,
            )
        ctx.clusters[self.store_as] = result

        # 4. Optionally drop noise papers from the signal.
        out_signal = signal
        n_noise = sum(
            1 for p in signal if result.membership.get(p.paper_id, -1) == -1
        )
        if self.drop_noise:
            out_signal = [
                p for p in signal
                if result.membership.get(p.paper_id, -1) != -1
            ]

        n_clusters = len({c for c in result.membership.values() if c != -1})
        named = mode != "none"
        return StepResult(
            signal=out_signal,
            in_count=len(signal),
            stats={
                "store_as": self.store_as,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "named": named,
                "drop_noise": self.drop_noise,
            },
        )
