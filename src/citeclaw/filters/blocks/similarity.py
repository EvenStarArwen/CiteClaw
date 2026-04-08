"""SimilarityFilter — pass if max(normalized score across measures) >= threshold."""

from __future__ import annotations

from citeclaw.filters.base import PASS, FilterContext, FilterOutcome
from citeclaw.models import PaperRecord


class SimilarityFilter:
    """Pass if max(normalized similarity score across measures) >= threshold.

    Decision logic, in order:

    1. Each measure is asked for a score; ``None`` means "this measure has
       no data for this paper" (e.g. SemanticSim couldn't fetch the
       embedding from S2).
    2. The filter takes ``max`` over the measures that DID return a score.
       If at least one measure has data, ``on_no_data`` is **never**
       consulted — the threshold check decides.
    3. ``on_no_data`` only kicks in when **every single measure** returned
       ``None``. This is the "we know nothing about this paper" path:
       ``'pass'`` lets the paper through to the next filter, ``'reject'``
       drops it. Default is ``'pass'`` so missing-data papers don't get
       silently nuked by a similarity check that couldn't run.

    In practice this means a noisy CitSim can't drag down a confident
    SemanticSim signal, and a paper with one missing measure still gets
    a fair shot via the others.
    """

    def __init__(
        self,
        name: str = "similarity",
        *,
        threshold: float = 0.10,
        measures: list,
        on_no_data: str = "pass",
    ) -> None:
        self.name = name
        self.threshold = threshold
        self.measures = measures
        if on_no_data not in ("pass", "reject"):
            raise ValueError(
                f"on_no_data must be 'pass' or 'reject', got {on_no_data!r}"
            )
        self.on_no_data = on_no_data

    def check(self, paper: PaperRecord, fctx: FilterContext) -> FilterOutcome:
        # Collect scores from measures that have data; skip the ones that
        # returned None ("no signal"). ``scored`` is empty iff every
        # measure returned None — that's the only path that consults
        # ``on_no_data``.
        scored: list[tuple[str, float]] = []
        for m in self.measures:
            s = m.compute(paper, fctx)
            if s is not None:
                scored.append((m.name, s))
        if not scored:
            # All measures lack data for this paper.
            if self.on_no_data == "pass":
                return PASS
            return FilterOutcome(False, "no similarity data", "similarity_no_data")
        # At least one measure has data — take the max and threshold-check.
        max_name, max_score = max(scored, key=lambda x: x[1])
        if max_score >= self.threshold:
            return PASS
        return FilterOutcome(
            False,
            f"max sim {max_name}={max_score:.3f} < {self.threshold:.3f}",
            "similarity",
        )

    def check_batch(
        self, papers: list[PaperRecord], fctx: FilterContext,
    ) -> list[FilterOutcome]:
        """Batched dispatch: measures may implement ``prefetch(papers, fctx)``
        to bulk-warm any caches (e.g. SemanticSim pulls all embeddings in one
        S2 batch call). Then falls through to per-paper ``check``."""
        for m in self.measures:
            prefetch = getattr(m, "prefetch", None)
            if callable(prefetch):
                try:
                    prefetch(papers, fctx)
                except Exception:
                    pass  # non-fatal; per-paper compute will fall back
        return [self.check(p, fctx) for p in papers]
