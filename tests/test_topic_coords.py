"""Tests for the topic map's 2D coordinates.

Two layers: the pure-Python ``normalize_to_box`` box-fitting (runs
everywhere) and the real UMAP projection ``project_2d`` plus the
``TopicModelClusterer`` integration (needs the ``topic_model`` extras).

The invariant the UI depends on: coordinates are finite, 2-dimensional,
inside the padded 1000-unit box, and identical across runs of the same
corpus — a topic map that reshuffles on every run destroys the user's
spatial memory of their own literature.
"""

from __future__ import annotations

import pytest

from citeclaw.cluster.base import ClusterResult
from citeclaw.cluster.topic_model import (
    COORD_BOX,
    COORD_PAD,
    TopicModelClusterer,
    normalize_to_box,
    project_2d,
)
from citeclaw.models import PaperRecord
from tests.fakes import FakeS2Client

try:
    import hdbscan  # noqa: F401
    import numpy  # noqa: F401
    import umap  # noqa: F401

    _HAS_TOPIC_EXTRAS = True
except ImportError:
    _HAS_TOPIC_EXTRAS = False

LO = COORD_PAD
HI = COORD_BOX - COORD_PAD


def _in_box(pt: tuple[float, float]) -> bool:
    return LO - 1e-6 <= pt[0] <= HI + 1e-6 and LO - 1e-6 <= pt[1] <= HI + 1e-6


class TestNormalizeToBox:
    def test_longer_axis_fills_the_padded_box(self):
        pts = [(0.0, 0.0), (10.0, 4.0), (5.0, 2.0)]
        out = normalize_to_box(pts)
        xs = [p[0] for p in out]
        assert min(xs) == pytest.approx(LO)
        assert max(xs) == pytest.approx(HI)

    def test_shorter_axis_is_centred_not_stretched(self):
        pts = [(0.0, 0.0), (10.0, 4.0), (5.0, 2.0)]
        out = normalize_to_box(pts)
        ys = [p[1] for p in out]
        # y span is 4/10 of x span, so it must occupy 40% of the box …
        assert (max(ys) - min(ys)) == pytest.approx(0.4 * (HI - LO), abs=0.01)
        # … centred on 500, exactly as the run37 fixture is.
        assert (max(ys) + min(ys)) / 2 == pytest.approx(COORD_BOX / 2, abs=0.01)

    def test_aspect_ratio_is_preserved(self):
        pts = [(0.0, 0.0), (8.0, 2.0), (4.0, 1.0), (1.0, 1.5)]
        out = normalize_to_box(pts)
        src = (max(p[0] for p in pts) - min(p[0] for p in pts)) / (
            max(p[1] for p in pts) - min(p[1] for p in pts)
        )
        dst = (max(p[0] for p in out) - min(p[0] for p in out)) / (
            max(p[1] for p in out) - min(p[1] for p in out)
        )
        assert dst == pytest.approx(src, rel=1e-3)

    def test_all_points_stay_inside_the_box(self):
        pts = [(float(i), float(i * i % 17)) for i in range(50)]
        assert all(_in_box(p) for p in normalize_to_box(pts))

    def test_single_point_collapses_to_centre(self):
        assert normalize_to_box([(3.0, 9.0)]) == [(500.0, 500.0)]

    def test_all_identical_points_collapse_to_centre(self):
        out = normalize_to_box([(2.0, 2.0)] * 5)
        assert out == [(500.0, 500.0)] * 5

    def test_empty_input(self):
        assert normalize_to_box([]) == []

    def test_rounded_to_two_decimals(self):
        out = normalize_to_box([(0.0, 0.0), (3.0, 7.0), (1.0, 1.0)])
        for x, y in out:
            assert round(x, 2) == x
            assert round(y, 2) == y

    def test_matches_the_fixture_coordinate_envelope(self):
        """A wide, shallow cloud must reproduce run37's [25, 975] x-range."""
        pts = [(float(i), float(i % 9)) for i in range(200)]
        out = normalize_to_box(pts)
        xs = [p[0] for p in out]
        assert min(xs) == pytest.approx(25.0)
        assert max(xs) == pytest.approx(975.0)


@pytest.mark.skipif(not _HAS_TOPIC_EXTRAS, reason="topic_model extras not installed")
class TestProject2D:
    @staticmethod
    def _vectors(n_per: int = 12) -> tuple[list[str], list[list[float]]]:
        """Three well-separated blobs in 8-dim space, seeded deterministically."""
        import random

        rng = random.Random(0)
        ids: list[str] = []
        vecs: list[list[float]] = []
        for c in range(3):
            base = [0.0] * 8
            base[c] = 1.0
            for i in range(n_per):
                ids.append(f"c{c}p{i}")
                vecs.append([b + rng.uniform(-0.05, 0.05) for b in base])
        return ids, vecs

    def test_returns_two_finite_coordinates_per_paper(self):
        ids, vecs = self._vectors()
        coords = project_2d(ids, vecs, random_state=42)
        assert set(coords) == set(ids)
        for pt in coords.values():
            assert len(pt) == 2
            assert all(isinstance(v, float) for v in pt)
            assert all(v == v and abs(v) != float("inf") for v in pt)

    def test_coordinates_land_inside_the_padded_box(self):
        ids, vecs = self._vectors()
        coords = project_2d(ids, vecs, random_state=42)
        assert all(_in_box(pt) for pt in coords.values())

    def test_deterministic_for_a_pinned_random_state(self):
        ids, vecs = self._vectors()
        first = project_2d(ids, vecs, random_state=42)
        second = project_2d(ids, vecs, random_state=42)
        assert first == second

    def test_separated_blobs_stay_separated_in_2d(self):
        """The projection has to carry structure, not just produce numbers."""
        ids, vecs = self._vectors()
        coords = project_2d(ids, vecs, random_state=42)

        def centroid(prefix: str) -> tuple[float, float]:
            pts = [v for k, v in coords.items() if k.startswith(prefix)]
            return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)

        def spread(prefix: str) -> float:
            pts = [v for k, v in coords.items() if k.startswith(prefix)]
            cx, cy = centroid(prefix)
            return max(((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5 for p in pts)

        c0, c1 = centroid("c0"), centroid("c1")
        gap = ((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2) ** 0.5
        assert gap > spread("c0") + spread("c1")

    def test_too_few_points_returns_empty(self):
        assert project_2d(["a", "b"], [[1.0, 0.0], [0.0, 1.0]]) == {}

    def test_missing_extras_returns_empty_not_raise(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "umap", None)
        ids, vecs = self._vectors(4)
        assert project_2d(ids, vecs) == {}


@pytest.mark.skipif(not _HAS_TOPIC_EXTRAS, reason="topic_model extras not installed")
class TestTopicModelEmitsCoords:
    def _ctx(self, n_per: int = 12):
        import random

        rng = random.Random(1)
        client = FakeS2Client()
        papers: list[PaperRecord] = []
        for c in range(3):
            base = [0.0] * 8
            base[c] = 1.0
            for i in range(n_per):
                pid = f"c{c}p{i}"
                papers.append(PaperRecord(paper_id=pid, title=f"title-{pid}"))
                client._embeddings[pid] = [
                    b + rng.uniform(-0.05, 0.05) for b in base
                ]

        class _Ctx:
            s2 = client

        return papers, _Ctx()

    def test_coords_are_emitted_by_default(self):
        papers, ctx = self._ctx()
        result = TopicModelClusterer(min_cluster_size=4).cluster(papers, ctx)
        assert isinstance(result, ClusterResult)
        assert set(result.coords_2d) == {p.paper_id for p in papers}
        assert all(_in_box(pt) for pt in result.coords_2d.values())

    def test_emit_coords_2d_false_skips_the_extra_umap(self):
        papers, ctx = self._ctx()
        result = TopicModelClusterer(
            min_cluster_size=4, emit_coords_2d=False,
        ).cluster(papers, ctx)
        assert result.coords_2d == {}
        # …and membership is unaffected by the knob.
        with_coords = TopicModelClusterer(min_cluster_size=4).cluster(papers, ctx)
        assert result.membership == with_coords.membership

    def test_papers_without_embeddings_get_no_coordinate(self):
        papers, ctx = self._ctx()
        orphan = PaperRecord(paper_id="no-vector", title="orphan")
        papers.append(orphan)
        result = TopicModelClusterer(min_cluster_size=4).cluster(papers, ctx)
        assert "no-vector" not in result.coords_2d
        assert result.membership["no-vector"] == -1

    def test_clustering_is_unchanged_run_to_run(self):
        papers, ctx = self._ctx()
        a = TopicModelClusterer(min_cluster_size=4).cluster(papers, ctx)
        b = TopicModelClusterer(min_cluster_size=4).cluster(papers, ctx)
        assert a.membership == b.membership
        assert a.coords_2d == b.coords_2d
