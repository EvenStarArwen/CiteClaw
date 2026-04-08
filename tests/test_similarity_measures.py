"""Tests for similarity measures (RefSim, CitSim, SemanticSim)."""

from __future__ import annotations

import math

from citeclaw.filters.base import FilterContext
from citeclaw.filters.measures.cit_sim import CitSimMeasure
from citeclaw.filters.measures.ref_sim import RefSimMeasure
from citeclaw.filters.measures.semantic_sim import SemanticSimMeasure, _cosine
from citeclaw.models import PaperRecord


def _fctx(ctx, **kw):
    return FilterContext(ctx=ctx, **kw)


# ---------------------------------------------------------------------------
# RefSimMeasure
# ---------------------------------------------------------------------------


class TestRefSimMeasure:
    def test_no_source_refs_is_none(self, ctx):
        m = RefSimMeasure()
        out = m.compute(PaperRecord(paper_id="CITER1"), _fctx(ctx, source_refs=None))
        assert out is None

    def test_jaccard_like(self, ctx):
        # SEED's refs are REF1, REF2; CITER1's refs are SEED, REF3
        # Overlap: 0; min(2, 2) = 2 → 0/2 = 0.0
        m = RefSimMeasure()
        out = m.compute(
            PaperRecord(paper_id="CITER1"),
            _fctx(ctx, source_refs={"REF1", "REF2"}),
        )
        assert out == 0.0

    def test_perfect_overlap(self, ctx):
        m = RefSimMeasure()
        out = m.compute(
            PaperRecord(paper_id="SEED"),
            _fctx(ctx, source_refs={"REF1", "REF2"}),
        )
        # SEED's refs = {REF1, REF2}, source_refs = {REF1, REF2}; overlap = 2
        # min(2, 2) = 2 → 2/2 = 1.0
        assert out == 1.0

    def test_partial_overlap(self, ctx):
        """CITER2 refs = {SEED, CITER1}, source refs = {SEED}; overlap 1/1 = 1.0."""
        m = RefSimMeasure()
        out = m.compute(
            PaperRecord(paper_id="CITER2"),
            _fctx(ctx, source_refs={"SEED"}),
        )
        assert out == 1.0

    def test_empty_candidate_refs(self, ctx):
        """REF1 has no references → measure returns None."""
        m = RefSimMeasure()
        out = m.compute(
            PaperRecord(paper_id="REF1"),
            _fctx(ctx, source_refs={"X", "Y"}),
        )
        assert out is None

    def test_fetch_failure_returns_none(self, ctx):
        class FailingS2:
            def fetch_reference_ids(self, pid):
                raise RuntimeError("api down")

        ctx.s2 = FailingS2()
        m = RefSimMeasure()
        assert m.compute(
            PaperRecord(paper_id="p"), _fctx(ctx, source_refs={"X"})
        ) is None


# ---------------------------------------------------------------------------
# CitSimMeasure
# ---------------------------------------------------------------------------


class TestCitSimMeasure:
    def test_heavy_citation_shortcut(self, ctx):
        m = CitSimMeasure(pass_if_cited_at_least=100)
        out = m.compute(
            PaperRecord(paper_id="x", citation_count=500),
            _fctx(ctx, source_citers={"C1"}),
        )
        assert out == 1.0

    def test_no_source_citers_returns_none(self, ctx):
        m = CitSimMeasure(pass_if_cited_at_least=10**9)
        out = m.compute(
            PaperRecord(paper_id="SEED", citation_count=50),
            _fctx(ctx, source_citers=None),
        )
        assert out is None

    def test_overlap_jaccard(self, ctx):
        """SEED's citers = {CITER1, CITER2}; CITER1's citers = {CITER2}.
        Overlap 1; min(2,1)=1 → 1.0."""
        m = CitSimMeasure(pass_if_cited_at_least=10**9)
        out = m.compute(
            PaperRecord(paper_id="CITER1", citation_count=1),
            _fctx(ctx, source_citers={"CITER1", "CITER2"}),
        )
        assert out == 1.0

    def test_fetch_failure_returns_none(self, ctx):
        class FailingS2:
            def fetch_citation_ids_and_counts(self, pid):
                raise RuntimeError("boom")

        ctx.s2 = FailingS2()
        m = CitSimMeasure(pass_if_cited_at_least=10**9)
        assert m.compute(
            PaperRecord(paper_id="p", citation_count=0),
            _fctx(ctx, source_citers={"C1"}),
        ) is None


# ---------------------------------------------------------------------------
# SemanticSimMeasure — cosine + S2 backend via FakeS2
# ---------------------------------------------------------------------------


class TestCosine:
    def test_identical_vectors(self):
        assert _cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == 1.0

    def test_orthogonal(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_length_mismatch(self):
        assert _cosine([1.0], [1.0, 0.0]) is None

    def test_zero_vector(self):
        assert _cosine([0.0, 0.0], [1.0, 0.0]) is None

    def test_known_value(self):
        v = _cosine([1.0, 1.0], [1.0, 0.0])
        assert v is not None
        assert math.isclose(v, 1 / math.sqrt(2), rel_tol=1e-9)


class TestSemanticSimMeasure:
    def test_source_missing_returns_none(self, ctx):
        m = SemanticSimMeasure()
        out = m.compute(PaperRecord(paper_id="SEED"), _fctx(ctx, source=None))
        assert out is None

    def test_s2_happy_path(self, ctx):
        """SEED embedding [1,0,0], CITER1 [0.9, 0.1, 0]; cosine ~ 0.994."""
        m = SemanticSimMeasure()
        out = m.compute(
            PaperRecord(paper_id="CITER1"),
            _fctx(ctx, source=PaperRecord(paper_id="SEED")),
        )
        assert out is not None
        assert 0.99 < out < 1.0

    def test_missing_candidate_embedding_returns_none(self, ctx):
        """REF1 has no embedding in the fake corpus → None."""
        m = SemanticSimMeasure()
        out = m.compute(
            PaperRecord(paper_id="REF1"),
            _fctx(ctx, source=PaperRecord(paper_id="SEED")),
        )
        assert out is None

    def test_prefetch_is_noop_for_s2_without_papers(self, ctx):
        m = SemanticSimMeasure()
        m.prefetch([], _fctx(ctx))  # empty list must not crash

    def test_prefetch_warms_via_s2_batch(self, ctx):
        m = SemanticSimMeasure()
        m.prefetch(
            [PaperRecord(paper_id="SEED"), PaperRecord(paper_id="CITER1")],
            _fctx(ctx, source=PaperRecord(paper_id="SEED")),
        )
        assert ctx.s2.calls.get("fetch_embeddings_batch", 0) == 1

    def test_prefetch_swallows_s2_errors(self, ctx, monkeypatch):
        m = SemanticSimMeasure()
        def raise_err(*a, **kw): raise RuntimeError("boom")
        monkeypatch.setattr(ctx.s2, "fetch_embeddings_batch", raise_err)
        # No exception leaks out.
        m.prefetch([PaperRecord(paper_id="SEED")], _fctx(ctx))

    def test_external_backend_raises(self):
        """Non-S2 backends route through build_embedder, whose placeholder
        classes raise NotImplementedError on .embed()."""
        m = SemanticSimMeasure(embedder="voyage:voyage-3")

        class _Ctx:
            class s2:
                pass

        fctx = FilterContext(
            ctx=_Ctx(),
            source=PaperRecord(paper_id="s", title="t", abstract="a"),
        )
        import pytest
        with pytest.raises(NotImplementedError):
            m.compute(PaperRecord(paper_id="c", title="t2", abstract="a2"), fctx)

    def test_negative_cosine_clamped_to_zero(self):
        """SPECTER2 cosines can dip negative — the measure clamps to [0,1]."""
        m = SemanticSimMeasure()

        class FakeS2:
            def fetch_embedding(self, pid):
                return [1.0, 0.0] if pid == "a" else [-1.0, 0.0]

        class _Ctx:
            s2 = FakeS2()

        fctx = FilterContext(ctx=_Ctx(), source=PaperRecord(paper_id="a"))
        out = m.compute(PaperRecord(paper_id="b"), fctx)
        assert out == 0.0
