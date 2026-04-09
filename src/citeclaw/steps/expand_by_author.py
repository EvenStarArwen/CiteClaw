"""ExpandByAuthor — author-graph traversal expansion step.

This step asks: "of all the authors who appear in the input signal,
which ones are most worth pulling more papers from, and what else have
they written?" It picks the top-K authors by a configurable metric
(h-index, citation count, or co-authorship degree within the signal),
fetches each one's recent papers via S2, and funnels them through the
screener.

Composable at the same level as ``ExpandForward`` /
``ExpandBackward`` / ``ExpandBySearch`` / ``ExpandBySemantics``. Like
the rest of the ``ExpandBy*`` family it has no source paper, so the
``FilterContext`` carries ``source=None`` and the screener cascade
must tolerate source-less mode (audited in PC-05).

Run loop (matches the roadmap spec):

  1. Idempotency check via fingerprint over (step name, sorted signal
     IDs, author cap, metric, papers-per-author).
  2. Collect distinct author IDs from ``p.authors`` across the signal
     (skip authors that lack an ``authorId`` — name-only fallbacks
     can't be queried by S2).
  3. ``ctx.s2.fetch_authors_batch(author_ids)`` → metadata for h-index
     / citation-count ranking.
  4. Rank by ``author_metric``:
     - ``h_index`` / ``citation_count`` — straight pull from S2 metadata.
     - ``degree_in_collab_graph`` — build an inline collaboration graph
       over the signal via :func:`citeclaw.author_graph.build_author_graph`
       and rank by node degree.
  5. Take the top ``top_k_authors``.
  6. For each chosen author, ``ctx.s2.fetch_author_papers(author_id,
     limit=papers_per_author)`` to pull their paper list.
  7. Flatten + dedup against ``ctx.seen``, hydrate via
     ``ctx.s2.enrich_batch``, fill abstracts via
     ``ctx.s2.enrich_with_abstracts``, stamp ``source="author"``.
  8. Source-less ``FilterContext``, ``apply_block`` the screener,
     ``record_rejections``.
  9. Survivors → ``ctx.collection``; mark fingerprint.
"""

from __future__ import annotations

import logging
from typing import Any

from citeclaw.models import PaperRecord
from citeclaw.steps._expand_helpers import (
    check_already_searched,
    fingerprint_signal,
    screen_expand_candidates,
)
from citeclaw.steps.base import StepResult

log = logging.getLogger("citeclaw.steps.expand_by_author")


_VALID_METRICS = {"h_index", "citation_count", "degree_in_collab_graph"}


class ExpandByAuthor:
    """Author-graph traversal expansion step.

    Picks top-K authors from the input signal by the configured
    metric, then pulls each one's papers from S2 and screens them.
    """

    name = "ExpandByAuthor"

    def __init__(
        self,
        *,
        screener: Any = None,
        top_k_authors: int = 10,
        author_metric: str = "degree_in_collab_graph",
        papers_per_author: int = 50,
    ) -> None:
        if author_metric not in _VALID_METRICS:
            raise ValueError(
                f"ExpandByAuthor.author_metric must be one of "
                f"{sorted(_VALID_METRICS)}; got {author_metric!r}"
            )
        self.screener = screener
        self.top_k_authors = top_k_authors
        self.author_metric = author_metric
        self.papers_per_author = papers_per_author

    def _collect_author_ids(self, signal: list[PaperRecord]) -> list[str]:
        """Distinct authorIds across the signal, in first-seen order.

        Skips author entries that lack an ``authorId`` — name-only
        fallbacks can't be looked up via S2's author batch endpoint.
        """
        seen: set[str] = set()
        out: list[str] = []
        for paper in signal:
            for author in paper.authors or []:
                if not isinstance(author, dict):
                    continue
                aid = author.get("authorId")
                if not isinstance(aid, str) or not aid:
                    continue
                if aid in seen:
                    continue
                seen.add(aid)
                out.append(aid)
        return out

    def _score_authors(
        self,
        author_ids: list[str],
        author_meta: dict[str, dict[str, Any]],
        signal: list[PaperRecord],
    ) -> dict[str, float]:
        """Compute the per-author score for the configured metric."""
        if self.author_metric == "h_index":
            return {
                aid: float((author_meta.get(aid) or {}).get("hIndex") or 0)
                for aid in author_ids
            }
        if self.author_metric == "citation_count":
            return {
                aid: float((author_meta.get(aid) or {}).get("citationCount") or 0)
                for aid in author_ids
            }
        # degree_in_collab_graph — build an inline graph over the signal.
        from citeclaw.author_graph import build_author_graph

        signal_collection = {p.paper_id: p for p in signal if p.paper_id}
        try:
            graph = build_author_graph(signal_collection, author_details=author_meta)
        except Exception as exc:  # noqa: BLE001 — fail soft to zero scores
            log.warning("ExpandByAuthor: build_author_graph failed: %s", exc)
            return {aid: 0.0 for aid in author_ids}
        # graph.vs["name"] holds the author key (authorId or "name:..." fallback).
        names = list(graph.vs["name"]) if graph.vcount() else []
        degrees = list(graph.degree())
        deg_by_key = dict(zip(names, degrees))
        return {aid: float(deg_by_key.get(aid, 0)) for aid in author_ids}

    def run(self, signal: list[PaperRecord], ctx) -> StepResult:
        if self.screener is None:
            return StepResult(
                signal=[],
                in_count=len(signal),
                stats={"reason": "no screener"},
            )

        # 1. Idempotency.
        fp = fingerprint_signal(
            self.name, signal,
            top_k_authors=self.top_k_authors,
            author_metric=self.author_metric,
            papers_per_author=self.papers_per_author,
        )
        if (skip := check_already_searched(self.name, fp, ctx, len(signal))):
            return skip

        # 2. Collect distinct authorIds.
        author_ids = self._collect_author_ids(signal)
        if not author_ids:
            return StepResult(
                signal=[],
                in_count=len(signal),
                stats={"reason": "no_authors"},
            )

        # 3. Fetch author metadata in one batch.
        try:
            author_meta = ctx.s2.fetch_authors_batch(author_ids)
        except Exception as exc:  # noqa: BLE001 — surface the error
            log.warning("ExpandByAuthor: fetch_authors_batch failed: %s", exc)
            return StepResult(
                signal=[],
                in_count=len(signal),
                stats={"reason": "fetch_authors_failed", "error": str(exc)[:120]},
            )

        # 4. Score + rank.
        scores = self._score_authors(author_ids, author_meta or {}, signal)
        # Sort by score descending, then authorId for stable ordering.
        ranked = sorted(author_ids, key=lambda aid: (-scores.get(aid, 0.0), aid))
        chosen = ranked[: self.top_k_authors]

        # 5-6. Retriever — fetch each author's papers and flatten.
        raw_papers: list[dict[str, Any]] = []
        for aid in chosen:
            try:
                papers = ctx.s2.fetch_author_papers(
                    aid, limit=self.papers_per_author,
                )
            except Exception as exc:  # noqa: BLE001 — keep going on individual failures
                log.warning(
                    "ExpandByAuthor: fetch_author_papers(%s) failed: %s", aid, exc,
                )
                continue
            if papers:
                raw_papers.extend(papers)

        # 7. Shared screening pipeline (hydrate → enrich → dedup →
        # screen → commit). Handles paperId extraction + deduplication
        # internally so we can pass the unfiltered raw_papers list.
        screened = screen_expand_candidates(
            raw_hits=raw_papers,
            source_label="author",
            screener=self.screener,
            ctx=ctx,
        )

        # 8. Mark fingerprint so re-runs are no-ops.
        ctx.searched_signals.add(fp)

        return StepResult(
            signal=screened.passed,
            in_count=len(screened.hydrated),
            stats={
                **screened.base_stats,
                "distinct_authors": len(author_ids),
                "chosen_authors": len(chosen),
                "raw_paper_count": len(raw_papers),
                "metric": self.author_metric,
            },
        )
