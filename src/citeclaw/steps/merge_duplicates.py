"""``MergeDuplicates`` step — fold preprint↔published duplicates.

Destructive step that operates on ``ctx.collection``. Designed to
run after expansion has added candidates and before Rerank /
Finalize so double-counted preprints don't contaminate downstream
scoring.

:func:`citeclaw.pipeline._ensure_merge_duplicates` auto-injects a
default-config ``MergeDuplicates`` before ``Finalize`` when the
user's YAML doesn't list one explicitly; adding it to the YAML is
only necessary if you want dedup to run earlier in the pipeline
(e.g. before a Rerank).

Detection signals (all optional, composable):

* title Jaccard similarity above ``title_threshold``
* SPECTER2 embedding cosine above ``semantic_threshold`` (only when
  ``use_embeddings=True`` and the S2 fetch succeeds; falls back to
  title+author signals on failure, WARNING-logged)
* year within ``year_window`` of the canonical

Survivors land back in ``ctx.collection`` under their canonical id.
Absorbed papers are recorded in ``ctx.alias_map`` so downstream code
(including the step's own signal rewrite) can redirect references.
The per-run count of folded duplicates is added to
``ctx.rejection_counts["merged_duplicate"]`` so the dashboard shows
the dedup footprint alongside other rejection categories.
"""

from __future__ import annotations

import logging

from citeclaw.dedup import detect_duplicate_clusters, merge_cluster
from citeclaw.models import PaperRecord
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.merge_duplicates")


class MergeDuplicates:
    """Fold preprint↔published duplicates in ``ctx.collection``."""

    name = "MergeDuplicates"

    def __init__(
        self,
        *,
        title_threshold: float = 0.95,
        semantic_threshold: float = 0.98,
        year_window: int = 1,
        use_embeddings: bool = True,
    ) -> None:
        """Configure detection thresholds.

        Parameters
        ----------
        title_threshold:
            Jaccard-over-token-sets cutoff for title-based duplicate
            detection. 0.95 is strict enough to avoid cross-paper
            false positives while catching preprint↔published pairs
            where one side has a trailing colon / version marker.
        semantic_threshold:
            SPECTER2 cosine cutoff (only used when
            ``use_embeddings=True`` AND the prefetch succeeds).
            Intentionally high (0.98) — lower values start collapsing
            same-topic-different-paper pairs.
        year_window:
            Allowable year difference between canonical and absorbed
            paper. 1 catches preprint→published same-year and +/-1
            publication-date pairs.
        use_embeddings:
            When False, skips the SPECTER2 prefetch entirely and uses
            title + author signals only. Useful for runs where S2
            embedding fetches are slow or unavailable.
        """
        self.title_threshold = title_threshold
        self.semantic_threshold = semantic_threshold
        self.year_window = year_window
        self.use_embeddings = use_embeddings

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        """Detect + merge duplicate clusters, rewrite the signal.

        Four-step flow: (1) short-circuit for < 2 papers (nothing to
        merge); (2) optionally prefetch SPECTER2 embeddings (cache-
        warm is free; failure logs WARNING and falls back to
        title+author-only signals); (3) detect clusters via
        :func:`citeclaw.dedup.detect_duplicate_clusters` and merge
        each via :func:`citeclaw.dedup.merge_cluster`; (4) rewrite
        the incoming signal so absorbed records are replaced by
        their canonical form, deduped, and stale references dropped.
        """
        collection = ctx.collection
        if len(collection) < 2:
            return StepResult(
                signal=list(signal), in_count=len(signal),
                stats={"clusters": 0, "merged": 0},
            )

        dash = ctx.dashboard
        dash.note_candidates_seen(len(collection))

        # Prefetch embeddings if the user asked for the semantic signal.
        # The fetch is cache-first — warm runs pay nothing, cold runs pay
        # one batched S2 call.
        embeddings: dict[str, list[float] | None] | None = None
        if self.use_embeddings:
            dash.begin_phase("prefetch embeddings", total=1)
            try:
                embeddings = ctx.s2.fetch_embeddings_batch(list(collection.keys()))
            except Exception as exc:
                log.warning("embedding prefetch failed: %s — falling back to "
                            "title + author signals only", exc)
                embeddings = None
            dash.tick_inner(1)

        dash.begin_phase("detect duplicate clusters", total=1)
        clusters = detect_duplicate_clusters(
            collection,
            title_threshold=self.title_threshold,
            semantic_threshold=self.semantic_threshold,
            year_window=self.year_window,
            embeddings=embeddings,
        )
        dash.tick_inner(1)

        dash.begin_phase("merge clusters", total=max(1, len(clusters)))
        merged_count = 0
        canonical_ids: set[str] = set()
        for cluster in clusters:
            canonical_id = merge_cluster(
                collection, cluster, alias_map=ctx.alias_map,
            )
            if canonical_id is not None:
                canonical_ids.add(canonical_id)
                merged_count += len(cluster) - 1
            dash.tick_inner(1)

        # Rewrite the incoming signal so absorbed records are replaced by
        # their canonical form, deduped, and stale references are dropped.
        new_signal: list[PaperRecord] = []
        seen_ids: set[str] = set()
        for rec in signal:
            canonical_id = ctx.alias_map.get(rec.paper_id, rec.paper_id)
            if canonical_id in seen_ids:
                continue
            if canonical_id not in collection:
                continue  # was absorbed and its canonical is also gone
            seen_ids.add(canonical_id)
            new_signal.append(collection[canonical_id])

        if merged_count:
            ctx.rejection_counts["merged_duplicate"] = (
                ctx.rejection_counts.get("merged_duplicate", 0) + merged_count
            )

        log.info(
            "MergeDuplicates: %d clusters, %d records folded into canonicals",
            len(clusters), merged_count,
        )
        return StepResult(
            signal=new_signal, in_count=len(signal),
            stats={"clusters": len(clusters), "merged": merged_count},
        )
