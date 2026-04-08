"""SemanticSimMeasure — cosine similarity over SPECTER2 (or other) embeddings.

The default backend is ``"s2"``, which uses Semantic Scholar's precomputed
SPECTER2 vectors (fetched via :class:`SemanticScholarClient.fetch_embedding` and
cached in ``cache.db``). Other backends (``"voyage"``, ``"local"``) route through
:mod:`citeclaw.clients.embeddings` and currently raise ``NotImplementedError``
until a real backend is wired in.
"""

from __future__ import annotations

import logging
import math

from citeclaw.filters.base import FilterContext
from citeclaw.models import PaperRecord

log = logging.getLogger("citeclaw.filters.measures.semantic_sim")


def _cosine(a: list[float], b: list[float]) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return None
    return dot / (math.sqrt(na) * math.sqrt(nb))


class SemanticSimMeasure:
    """Cosine similarity over paper embeddings.

    Args:
        name: display name (used in rejection categories).
        embedder: embedding backend. ``"s2"`` (default) uses Semantic Scholar's
            SPECTER2 vectors via ``ctx.s2.fetch_embedding``. Any other string
            routes through :func:`citeclaw.clients.embeddings.build_embedder`,
            which currently raises ``NotImplementedError`` for ``voyage:*`` and
            ``local:*``.
    """

    def __init__(self, name: str = "semantic_sim", *, embedder: str | dict = "s2") -> None:
        self.name = name
        self._embedder_spec = embedder
        self._is_s2 = isinstance(embedder, str) and embedder.lower() == "s2"
        self._embedder = None  # lazy for non-S2 backends

    # -------- optional batch hook (called by SimilarityFilter.check_batch) --------

    def prefetch(self, papers: list[PaperRecord], fctx: FilterContext) -> None:
        """Bulk-fetch embeddings into cache for all papers in this batch.

        Only meaningful for the S2 backend; for other backends this is a noop
        (text-based embedders have no precomputed per-paper cache to warm).
        """
        if not self._is_s2:
            return
        ids: list[str] = [p.paper_id for p in papers if p.paper_id]
        if fctx.source is not None and fctx.source.paper_id:
            ids.append(fctx.source.paper_id)
        if not ids:
            return
        try:
            fctx.ctx.s2.fetch_embeddings_batch(ids)
        except Exception as exc:
            log.warning("semantic_sim prefetch failed: %s", exc)

    # -------- per-paper compute --------

    def compute(self, paper: PaperRecord, fctx: FilterContext) -> float | None:
        if fctx.source is None:
            return None
        if self._is_s2:
            return self._compute_s2(paper, fctx)
        return self._compute_external(paper, fctx)

    def _compute_s2(self, paper: PaperRecord, fctx: FilterContext) -> float | None:
        try:
            src_emb = fctx.ctx.s2.fetch_embedding(fctx.source.paper_id)
        except Exception as exc:
            log.debug("semantic_sim: source embed fetch failed: %s", exc)
            return None
        if not src_emb:
            return None
        try:
            cand_emb = fctx.ctx.s2.fetch_embedding(paper.paper_id)
        except Exception as exc:
            log.debug("semantic_sim: candidate embed fetch failed: %s", exc)
            return None
        if not cand_emb:
            return None
        cos = _cosine(src_emb, cand_emb)
        if cos is None:
            return None
        # SPECTER2 cosines cluster in [0, 1] but can dip below 0 for
        # unrelated pairs; clamp negative values to 0.
        return max(0.0, cos)

    def _compute_external(self, paper: PaperRecord, fctx: FilterContext) -> float | None:
        if self._embedder is None:
            from citeclaw.clients.embeddings import build_embedder
            self._embedder = build_embedder(self._embedder_spec)
        src_text = (fctx.source.title or "") + "\n" + (fctx.source.abstract or "")
        cand_text = (paper.title or "") + "\n" + (paper.abstract or "")
        if not src_text.strip() or not cand_text.strip():
            return None
        embeds = self._embedder.embed([src_text, cand_text])  # raises NotImplementedError today
        cos = _cosine(embeds[0], embeds[1])
        if cos is None:
            return None
        return max(0.0, cos)
