"""Tests for the group artifact writer (``citeclaw.output.groups``).

Everything here is driven off hand-built ``ClusterResult`` objects, so
the tests pin the *file contract* — column names, ordering, separators,
empty-cell semantics — independently of whether UMAP / leidenalg are
installed. The shapes are checked against the design fixture's headers
so a rename on either side is caught here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from citeclaw.cluster.base import ClusterMetadata, ClusterResult
from citeclaw.context import Context
from citeclaw.models import PaperRecord
from citeclaw.output.groups import GROUP_COLUMNS, METHOD_COLUMNS, write_group_artifacts

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "web" / "design-reference" / "uploads" / "run37"


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _header(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as fh:
        return next(csv.reader(fh))


def _populate(ctx: Context, pids: list[str]) -> None:
    for pid in pids:
        ctx.collection[pid] = PaperRecord(paper_id=pid, title=f"Title of {pid}")


def _topic_result() -> ClusterResult:
    """Two topics (3 + 2 papers) plus one noise paper, with coordinates."""
    return ClusterResult(
        membership={"p1": 0, "p2": 0, "p3": 0, "p4": 1, "p5": 1, "p6": -1},
        metadata={
            0: ClusterMetadata(
                label="Robotic Chemistry",
                summary="Self-driving labs.",
                size=3,
                keywords=["robots", "synthesis"],
                representative_papers=["p1", "p2"],
            ),
            1: ClusterMetadata(
                label="Literature Screening",
                summary="Sifting abstracts.",
                size=2,
                keywords=["screening"],
                representative_papers=["p4"],
            ),
        },
        algorithm="topic_model",
        coords_2d={
            "p1": (100.0, 200.0),
            "p2": (200.0, 200.0),
            "p3": (300.0, 200.0),
            "p4": (800.0, 900.0),
            "p5": (900.0, 900.0),
            "p6": (500.0, 500.0),
        },
    )


def _leiden_result() -> ClusterResult:
    return ClusterResult(
        membership={"p1": 0, "p2": 0, "p3": 0, "p4": 0, "p5": 1, "p6": 1},
        metadata={
            0: ClusterMetadata(label="Core", size=4, keywords=["a", "b"]),
            1: ClusterMetadata(label="Fringe", size=2),
        },
        algorithm="leiden",
        quality={"modularity": 0.4321, "n_communities": 2.0},
    )


@pytest.fixture
def wired(ctx: Context) -> Context:
    _populate(ctx, ["p1", "p2", "p3", "p4", "p5", "p6"])
    ctx.clusters["topics"] = _topic_result()
    ctx.clusters["communities"] = _leiden_result()
    return ctx


class TestNoClusters:
    def test_writes_nothing_when_no_cluster_ran(self, ctx: Context):
        assert write_group_artifacts(ctx) == []
        assert not list(Path(ctx.config.data_dir).glob("*groups*"))

    def test_ignores_empty_cluster_results(self, ctx: Context):
        ctx.clusters["empty"] = ClusterResult(membership={}, algorithm="leiden")
        assert write_group_artifacts(ctx) == []


class TestFileSet:
    def test_writes_the_expected_files(self, wired: Context):
        written = write_group_artifacts(wired)
        assert {p.name for p in written} == {
            "topics.csv",
            "communities_leiden.csv",
            "paper_groups.csv",
            "community_methods.csv",
            "groups.json",
        }

    def test_iteration_runs_get_an_expN_suffix(self, wired: Context):
        wired.iteration = 3
        written = write_group_artifacts(wired)
        assert all(".exp3." in p.name for p in written)

    def test_explicit_suffix_overrides(self, wired: Context):
        written = write_group_artifacts(wired, suffix=".regen")
        assert all(".regen." in p.name for p in written)


class TestPaperGroups:
    def test_columns_and_row_count(self, wired: Context):
        path = Path(wired.config.data_dir) / "paper_groups.csv"
        write_group_artifacts(wired)
        assert _header(path) == [
            "id", "x", "y", "topic", "topic_label", "community_leiden",
        ]
        assert len(_read(path)) == 6

    def test_coordinates_and_group_ids_per_paper(self, wired: Context):
        write_group_artifacts(wired)
        rows = {
            r["id"]: r
            for r in _read(Path(wired.config.data_dir) / "paper_groups.csv")
        }
        assert rows["p1"]["x"] == "100.0" and rows["p1"]["y"] == "200.0"
        assert rows["p1"]["topic"] == "0"
        assert rows["p1"]["topic_label"] == "Robotic Chemistry"
        assert rows["p1"]["community_leiden"] == "0"
        assert rows["p5"]["community_leiden"] == "1"

    def test_noise_paper_keeps_its_minus_one_and_no_label(self, wired: Context):
        write_group_artifacts(wired)
        rows = {
            r["id"]: r
            for r in _read(Path(wired.config.data_dir) / "paper_groups.csv")
        }
        assert rows["p6"]["topic"] == "-1"
        assert rows["p6"]["topic_label"] == ""

    def test_missing_coordinates_are_blank_not_zero(self, ctx: Context):
        """A paper with no embedding must not be planted at (0, 0)."""
        _populate(ctx, ["p1", "p2", "p3"])
        ctx.clusters["communities"] = ClusterResult(
            membership={"p1": 0, "p2": 0, "p3": 1},
            algorithm="leiden",
            quality={"modularity": 0.1},
        )
        write_group_artifacts(ctx)
        rows = _read(Path(ctx.config.data_dir) / "paper_groups.csv")
        assert all(r["x"] == "" and r["y"] == "" for r in rows)

    def test_rows_sorted_by_id_for_stable_diffs(self, wired: Context):
        write_group_artifacts(wired)
        ids = [
            r["id"] for r in _read(Path(wired.config.data_dir) / "paper_groups.csv")
        ]
        assert ids == sorted(ids)


class TestGroupTables:
    def test_topic_table_columns_match_the_fixture(self, wired: Context):
        write_group_artifacts(wired)
        assert _header(Path(wired.config.data_dir) / "topics.csv") == [
            "topic_id", *GROUP_COLUMNS,
        ]

    def test_community_table_columns_match_the_fixture(self, wired: Context):
        write_group_artifacts(wired)
        assert _header(
            Path(wired.config.data_dir) / "communities_leiden.csv"
        ) == ["community_id", *GROUP_COLUMNS]

    @pytest.mark.skipif(
        not (FIXTURE_DIR / "topics.csv").exists(),
        reason="design-reference fixture not present in this checkout",
    )
    def test_headers_are_byte_identical_to_the_design_fixture(self, wired: Context):
        write_group_artifacts(wired)
        data_dir = Path(wired.config.data_dir)
        assert _header(data_dir / "topics.csv") == _header(FIXTURE_DIR / "topics.csv")
        assert _header(data_dir / "communities_leiden.csv") == _header(
            FIXTURE_DIR / "communities_leiden.csv"
        )
        assert _header(data_dir / "community_methods.csv") == _header(
            FIXTURE_DIR / "community_methods.csv"
        )

    def test_noise_bucket_is_not_a_topic_row(self, wired: Context):
        write_group_artifacts(wired)
        rows = _read(Path(wired.config.data_dir) / "topics.csv")
        assert [r["topic_id"] for r in rows] == ["0", "1"]

    def test_rows_ordered_largest_group_first(self, ctx: Context):
        _populate(ctx, ["a", "b", "c", "d"])
        ctx.clusters["c"] = ClusterResult(
            # cluster 5 has 1 paper, cluster 9 has 3 — 9 must come first.
            membership={"a": 5, "b": 9, "c": 9, "d": 9},
            algorithm="leiden",
        )
        write_group_artifacts(ctx)
        rows = _read(Path(ctx.config.data_dir) / "communities_leiden.csv")
        assert [r["community_id"] for r in rows] == ["9", "5"]
        assert [r["size"] for r in rows] == ["3", "1"]

    def test_centroid_is_the_mean_of_member_coordinates(self, wired: Context):
        write_group_artifacts(wired)
        rows = {
            r["topic_id"]: r
            for r in _read(Path(wired.config.data_dir) / "topics.csv")
        }
        # topic 0 = p1/p2/p3 at x 100/200/300, y all 200
        assert rows["0"]["centroid_x"] == "200.0"
        assert rows["0"]["centroid_y"] == "200.0"
        # topic 1 = p4/p5 at x 800/900, y all 900
        assert rows["1"]["centroid_x"] == "850.0"
        assert rows["1"]["centroid_y"] == "900.0"

    def test_community_centroids_use_the_topic_map_coordinates(self, wired: Context):
        """Communities have no embedding of their own; they borrow the map."""
        write_group_artifacts(wired)
        rows = {
            r["community_id"]: r
            for r in _read(Path(wired.config.data_dir) / "communities_leiden.csv")
        }
        # community 0 = p1..p4 at x 100/200/300/800 -> mean 350
        assert rows["0"]["centroid_x"] == "350.0"

    def test_centroid_blank_when_no_coordinates_exist(self, ctx: Context):
        _populate(ctx, ["a", "b"])
        ctx.clusters["c"] = ClusterResult(
            membership={"a": 0, "b": 0}, algorithm="leiden",
        )
        write_group_artifacts(ctx)
        row = _read(Path(ctx.config.data_dir) / "communities_leiden.csv")[0]
        assert row["centroid_x"] == "" and row["centroid_y"] == ""

    def test_multi_value_cells_use_the_fixture_separators(self, wired: Context):
        write_group_artifacts(wired)
        row = _read(Path(wired.config.data_dir) / "topics.csv")[0]
        assert row["keywords"] == "robots; synthesis"
        assert row["representative_paper_ids"] == "p1; p2"
        assert row["representative_paper_titles"] == "Title of p1 | Title of p2"

    def test_label_and_description_come_from_metadata(self, wired: Context):
        write_group_artifacts(wired)
        row = _read(Path(wired.config.data_dir) / "topics.csv")[0]
        assert row["label"] == "Robotic Chemistry"
        assert row["description"] == "Self-driving labs."

    def test_size_is_recomputed_not_trusted_from_metadata(self, ctx: Context):
        """Metadata sizes are filled by the naming pipeline, which may not run."""
        _populate(ctx, ["a", "b", "c"])
        ctx.clusters["c"] = ClusterResult(
            membership={"a": 0, "b": 0, "c": 0},
            metadata={0: ClusterMetadata(label="x", size=99)},
            algorithm="leiden",
        )
        write_group_artifacts(ctx)
        assert _read(Path(ctx.config.data_dir) / "communities_leiden.csv")[0][
            "size"
        ] == "3"


class TestCommunityMethods:
    def test_columns(self, wired: Context):
        write_group_artifacts(wired)
        assert _header(
            Path(wired.config.data_dir) / "community_methods.csv"
        ) == METHOD_COLUMNS

    def test_shape_and_quality_columns(self, wired: Context):
        write_group_artifacts(wired)
        row = _read(Path(wired.config.data_dir) / "community_methods.csv")[0]
        assert row["method"] == "leiden"
        assert row["n_communities"] == "2"
        assert row["modularity"] == "0.4321"
        assert row["largest_community"] == "4"
        assert row["largest_fraction"] == "0.6667"
        assert row["singletons"] == "0"
        assert row["median_size"] == "3"

    def test_agreement_against_the_topic_partition_is_present(self, wired: Context):
        write_group_artifacts(wired)
        row = _read(Path(wired.config.data_dir) / "community_methods.csv")[0]
        assert row["nmi_vs_topic_model"] != ""
        assert 0.0 <= float(row["nmi_vs_topic_model"]) <= 1.0
        assert row["adjusted_rand_vs_topic_model"] != ""

    def test_agreement_blank_without_a_topic_model(self, ctx: Context):
        _populate(ctx, ["a", "b", "c"])
        ctx.clusters["c"] = ClusterResult(
            membership={"a": 0, "b": 0, "c": 1}, algorithm="leiden",
        )
        write_group_artifacts(ctx)
        row = _read(Path(ctx.config.data_dir) / "community_methods.csv")[0]
        assert row["nmi_vs_topic_model"] == ""
        assert row["adjusted_rand_vs_topic_model"] == ""

    def test_modularity_blank_when_the_clusterer_did_not_report_one(
        self, ctx: Context,
    ):
        _populate(ctx, ["a", "b"])
        ctx.clusters["c"] = ClusterResult(
            membership={"a": 0, "b": 1}, algorithm="walktrap",
        )
        write_group_artifacts(ctx)
        row = _read(Path(ctx.config.data_dir) / "community_methods.csv")[0]
        assert row["modularity"] == ""


class TestMultiplePartitions:
    def test_several_graph_algorithms_each_get_a_file_and_a_column(
        self, ctx: Context,
    ):
        _populate(ctx, ["a", "b", "c", "d"])
        ctx.clusters["l"] = ClusterResult(
            membership={"a": 0, "b": 0, "c": 1, "d": 1}, algorithm="leiden",
        )
        ctx.clusters["w"] = ClusterResult(
            membership={"a": 0, "b": 1, "c": 1, "d": 1}, algorithm="walktrap",
        )
        written = {p.name for p in write_group_artifacts(ctx)}
        assert "communities_leiden.csv" in written
        assert "communities_walktrap.csv" in written
        header = _header(Path(ctx.config.data_dir) / "paper_groups.csv")
        assert "community_leiden" in header and "community_walktrap" in header
        assert len(_read(Path(ctx.config.data_dir) / "community_methods.csv")) == 2

    def test_same_algorithm_twice_is_disambiguated_by_store_as(self, ctx: Context):
        _populate(ctx, ["a", "b"])
        ctx.clusters["first"] = ClusterResult(
            membership={"a": 0, "b": 1}, algorithm="leiden",
        )
        ctx.clusters["second"] = ClusterResult(
            membership={"a": 0, "b": 0}, algorithm="leiden",
        )
        written = {p.name for p in write_group_artifacts(ctx)}
        assert "communities_leiden.csv" in written
        assert "communities_leiden_second.csv" in written


@pytest.mark.skipif(
    not (FIXTURE_DIR / "graph.json").exists(),
    reason="design-reference fixture not present in this checkout",
)
class TestRun37EndToEnd:
    """Full path at fixture scale: 354 real nodes → Leiden → 2D → CSVs.

    Coordinates here come from UMAP over each paper's adjacency row (a
    real 354-dimensional feature vector derived from the real citation
    graph) rather than SPECTER2, which isn't checked in. That exercises
    the same ``project_2d`` code the pipeline uses, at the same scale,
    without needing network access or a cached embedding store.
    """

    @staticmethod
    def _load():
        import igraph as ig

        data = json.loads((FIXTURE_DIR / "graph.json").read_text(encoding="utf-8"))
        pids = [n["pid"] for n in data["nodes"]]
        edges = [(e[0], e[1]) for e in data["edges"]]
        g = ig.Graph(n=len(pids), edges=edges, directed=True)
        return g, pids, data

    def _build_ctx(self, ctx: Context) -> Context:
        from citeclaw.cluster.leiden import LeidenClusterer

        _g, pids, data = self._load()
        refs: dict[str, list[str]] = {pid: [] for pid in pids}
        for src, dst, *_ in data["edges"]:
            refs[pids[src]].append(pids[dst])
        papers = [
            PaperRecord(paper_id=pid, title=f"Title of {pid}", references=refs[pid])
            for pid in pids
        ]
        for p in papers:
            ctx.collection[p.paper_id] = p
        ctx.clusters["communities"] = LeidenClusterer(seed=42).cluster(papers, ctx)
        return ctx

    def test_communities_are_non_degenerate(self, ctx: Context):
        ctx = self._build_ctx(ctx)
        write_group_artifacts(ctx)
        rows = _read(Path(ctx.config.data_dir) / "communities_leiden.csv")
        sizes = [int(r["size"]) for r in rows]
        assert sum(sizes) == 354
        assert 2 <= len(sizes) <= 60          # neither one blob nor 354 singletons
        assert max(sizes) < 0.5 * 354         # no dominant super-community
        assert sum(1 for s in sizes if s >= 5) >= 3
        method = _read(Path(ctx.config.data_dir) / "community_methods.csv")[0]
        assert float(method["modularity"]) > 0.3
        assert int(method["n_communities"]) == len(sizes)

    def test_every_paper_lands_in_exactly_one_community(self, ctx: Context):
        ctx = self._build_ctx(ctx)
        write_group_artifacts(ctx)
        rows = _read(Path(ctx.config.data_dir) / "paper_groups.csv")
        assert len(rows) == 354
        assert len({r["id"] for r in rows}) == 354
        assert all(r["community_leiden"] != "" for r in rows)
        n_communities = len(
            _read(Path(ctx.config.data_dir) / "communities_leiden.csv")
        )
        ids = {int(r["community_leiden"]) for r in rows}
        assert ids == set(range(n_communities))   # contiguous, ranked from 0

    def test_coordinates_are_2d_and_finite_for_all_354_papers(self, ctx: Context):
        pytest.importorskip("umap", reason="topic_model extras not installed")
        from citeclaw.cluster.topic_model import COORD_BOX, COORD_PAD, project_2d

        g, pids, _ = self._load()
        adjacency = [
            [float(v) for v in row]
            for row in g.as_undirected(mode="collapse").get_adjacency()
        ]
        coords = project_2d(pids, adjacency, metric="cosine", random_state=42)
        assert len(coords) == 354
        for pid, pt in coords.items():
            assert len(pt) == 2
            x, y = pt
            assert x == x and y == y                       # not NaN
            assert abs(x) != float("inf") and abs(y) != float("inf")
            assert COORD_PAD - 1e-6 <= x <= COORD_BOX - COORD_PAD + 1e-6
            assert COORD_PAD - 1e-6 <= y <= COORD_BOX - COORD_PAD + 1e-6

        ctx = self._build_ctx(ctx)
        ctx.clusters["communities"].coords_2d = coords
        write_group_artifacts(ctx)
        rows = _read(Path(ctx.config.data_dir) / "paper_groups.csv")
        assert all(r["x"] != "" and r["y"] != "" for r in rows)
        assert all(
            abs(float(r["x"])) < 1e9 and abs(float(r["y"])) < 1e9 for r in rows
        )


@pytest.mark.skipif(
    not (FIXTURE_DIR / "accepted.csv").exists(),
    reason="design-reference fixture not present in this checkout",
)
class TestReproducesTheFixtureTables:
    """Feed run37's own labels + coordinates in; expect run37's tables out.

    This is the artifact-level counterpart to the algorithm-level golden
    test in ``test_cluster_leiden.py``. It isolates the *writer*: given
    the partition and coordinates the design fixture was built from, do
    we emit the same rows? Every column except the two float ties below
    matches character-for-character.
    """

    @staticmethod
    def _run(ctx: Context) -> Path:
        with (FIXTURE_DIR / "accepted.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            ctx.collection[r["id"]] = PaperRecord(
                paper_id=r["id"], title=r["title"],
            )
        coords = {r["id"]: (float(r["x"]), float(r["y"])) for r in rows}
        ctx.clusters["topics"] = ClusterResult(
            membership={r["id"]: int(r["topic"]) for r in rows},
            algorithm="topic_model",
            coords_2d=coords,
        )
        ctx.clusters["communities"] = ClusterResult(
            membership={r["id"]: int(r["community_leiden"]) for r in rows},
            algorithm="leiden",
            quality={"modularity": 0.5558},
        )
        write_group_artifacts(ctx)
        return Path(ctx.config.data_dir)

    def test_community_methods_row_is_identical(self, ctx: Context):
        out = self._run(ctx)
        mine = _read(out / "community_methods.csv")
        theirs = [
            r
            for r in _read(FIXTURE_DIR / "community_methods.csv")
            if r["method"] == "leiden"
        ]
        assert mine == theirs

    @pytest.mark.parametrize(
        "filename, id_column",
        [
            ("communities_leiden.csv", "community_id"),
            ("topics.csv", "topic_id"),
        ],
    )
    def test_group_tables_match_on_id_size_and_centroid(
        self, ctx: Context, filename: str, id_column: str,
    ):
        out = self._run(ctx)
        mine = _read(out / filename)
        theirs = _read(FIXTURE_DIR / filename)
        assert len(mine) == len(theirs)
        for a, b in zip(mine, theirs):
            assert a[id_column] == b[id_column]      # same row order (size desc)
            assert a["size"] == b["size"]
            # Two of the 31 centroids in the fixture sit exactly on a
            # round-half tie (…905 / …805) where the design tool and
            # CPython's round() disagree in the last digit, so compare
            # numerically at 2dp rather than as strings.
            assert float(a["centroid_x"]) == pytest.approx(
                float(b["centroid_x"]), abs=0.011,
            )
            assert float(a["centroid_y"]) == pytest.approx(
                float(b["centroid_y"]), abs=0.011,
            )

    def test_float_cells_keep_the_fixture_s_trailing_zero_style(self, ctx: Context):
        """``707.0`` not ``707``; ``444.9`` not ``444.90``."""
        out = self._run(ctx)
        cells = [r["x"] for r in _read(out / "paper_groups.csv") if r["x"]]
        assert len(cells) == 354
        for cell in cells:
            assert "." in cell, f"{cell!r} lost its decimal point"
            assert len(cell.split(".")[1]) <= 2, f"{cell!r} has >2 decimals"
        # The fixture contains both styles; so must we.
        assert any(c.endswith(".0") for c in cells)
        assert any(len(c.split(".")[1]) == 2 for c in cells)


class TestGroupsIndex:
    def test_index_summarises_what_ran(self, wired: Context):
        write_group_artifacts(wired)
        index = json.loads(
            (Path(wired.config.data_dir) / "groups.json").read_text(encoding="utf-8")
        )
        assert index["topics"]["n_topics"] == 2
        assert index["topics"]["n_noise"] == 1
        assert index["communities"][0]["algorithm"] == "leiden"
        assert index["communities"][0]["n_communities"] == 2
        assert index["communities"][0]["quality"]["modularity"] == 0.4321
        assert index["papers"]["n_papers"] == 6
        assert index["coords_2d_papers"] == 6
        assert index["coord_space"] == {
            "box": 1000.0, "pad": 25.0, "origin": "top_left",
        }
