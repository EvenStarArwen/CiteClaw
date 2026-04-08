"""Tests for :mod:`citeclaw.output` — JSON / BibTeX / GraphML writers."""

from __future__ import annotations

import json
from pathlib import Path

import igraph as ig

from citeclaw.author_graph import build_author_graph, export_author_graphml
from citeclaw.config import BudgetTracker
from citeclaw.models import PaperRecord
from citeclaw.output import (
    build_output,
    export_graphml,
    with_iteration_suffix,
    write_bibtex,
    write_json,
    write_run_state,
)


def _paper(pid, **kw):
    defaults = dict(title=f"Title {pid}", year=2022, venue="Nature", citation_count=10)
    defaults.update(kw)
    return PaperRecord(paper_id=pid, **defaults)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


class TestBuildOutput:
    def test_summary_shape(self):
        collection = {
            "A": _paper("A", citation_count=100, depth=0, source="seed"),
            "B": _paper("B", citation_count=50, depth=1, source="forward"),
        }
        rejected = {"R1", "R2"}
        seen = {"A", "B", "R1", "R2", "R3"}
        budget = BudgetTracker()
        out = build_output(collection, rejected, seen, budget)
        assert out["summary"]["total_accepted"] == 2
        assert out["summary"]["total_rejected"] == 2
        assert out["summary"]["total_seen"] == 5
        assert "depth_distribution" in out["summary"]
        assert "source_distribution" in out["summary"]
        assert "budget" in out["summary"]

    def test_papers_sorted_by_citation_count_desc(self):
        collection = {
            "A": _paper("A", citation_count=5),
            "B": _paper("B", citation_count=50),
            "C": _paper("C", citation_count=None),  # None sorts to bottom
        }
        out = build_output(collection, set(), set(), BudgetTracker())
        ordered_ids = [p["paper_id"] for p in out["papers"]]
        assert ordered_ids[0] == "B"
        assert ordered_ids[1] == "A"
        assert ordered_ids[2] == "C"

    def test_paper_fields_serialized(self):
        collection = {"A": _paper("A", authors=[{"authorId": "X", "name": "Alice"}])}
        out = build_output(collection, set(), set(), BudgetTracker())
        p = out["papers"][0]
        for field in [
            "paper_id", "title", "abstract", "year", "venue",
            "citation_count", "references", "depth", "source",
            "authors", "expanded", "pdf_url",
        ]:
            assert field in p


class TestWriteJson:
    def test_roundtrip(self, tmp_path: Path):
        data = {"hello": "world", "nums": [1, 2, 3]}
        out_path = tmp_path / "nested" / "out.json"
        write_json(data, out_path)
        assert out_path.exists()
        assert json.loads(out_path.read_text()) == data

    def test_default_stringifies_unknowns(self, tmp_path: Path):
        data = {"path": tmp_path}  # Path is not JSON-native
        out_path = tmp_path / "out.json"
        write_json(data, out_path)
        assert out_path.exists()


class TestWithIterationSuffix:
    def test_iteration_one_unchanged(self, tmp_path):
        p = tmp_path / "literature_collection.json"
        assert with_iteration_suffix(p, 1) == p

    def test_iteration_two(self, tmp_path):
        p = tmp_path / "literature_collection.json"
        out = with_iteration_suffix(p, 2)
        assert out.name == "literature_collection.exp2.json"

    def test_iteration_five(self, tmp_path):
        p = tmp_path / "run_state.json"
        out = with_iteration_suffix(p, 5)
        assert out.name == "run_state.exp5.json"


class TestWriteRunState:
    def test_fields_present(self, tmp_path: Path):
        collection = {"A": _paper("A")}
        rejected = {"R1"}
        seen = {"A", "R1"}
        path = tmp_path / "run_state.json"
        write_run_state(
            collection, rejected, seen, queue_ids=["Q1"],
            budget=BudgetTracker(), path=path, iteration=3,
            parent_dir="/tmp/foo", new_seed_ids=["SEED1"],
        )
        data = json.loads(path.read_text())
        assert data["iteration"] == 3
        assert data["parent_dir"] == "/tmp/foo"
        assert data["new_seed_ids"] == ["SEED1"]
        assert data["collection_ids"] == ["A"]
        assert data["queue_ids"] == ["Q1"]
        assert "A" in data["seen_ids"]
        assert "R1" in data["rejected_ids"]
        assert "budget" in data


# ---------------------------------------------------------------------------
# BibTeX output
# ---------------------------------------------------------------------------


class TestWriteBibtex:
    def test_basic_entry(self, tmp_path: Path):
        papers = [
            _paper("DOI:10.1/abc", title="Hello World", year=2020, venue="Nature", citation_count=42),
        ]
        path = tmp_path / "refs.bib"
        write_bibtex(papers, path)
        text = path.read_text()
        assert "@article{DOI101abc" in text
        assert "title     = {Hello World}" in text
        assert "year      = {2020}" in text
        assert "journal   = {Nature}" in text
        assert "S2: DOI:10.1/abc" in text

    def test_braces_in_title_escaped(self, tmp_path: Path):
        papers = [_paper("A", title="{Braces} everywhere}")]
        path = tmp_path / "refs.bib"
        write_bibtex(papers, path)
        text = path.read_text()
        assert "\\{Braces\\}" in text

    def test_empty_papers(self, tmp_path: Path):
        path = tmp_path / "refs.bib"
        write_bibtex([], path)
        assert path.exists()
        assert path.read_text().strip() == ""

    def test_missing_year_and_venue(self, tmp_path: Path):
        papers = [_paper("A", year=None, venue=None)]
        path = tmp_path / "refs.bib"
        write_bibtex(papers, path)
        text = path.read_text()
        assert "year      = {}" in text


# ---------------------------------------------------------------------------
# GraphML export
# ---------------------------------------------------------------------------


class TestExportGraphml:
    def test_basic_export(self, tmp_path: Path):
        collection = {
            "A": _paper("A", references=[]),
            "B": _paper("B", references=["A"]),
            "C": _paper("C", references=["A", "B"]),
        }
        path = tmp_path / "graph.graphml"
        export_graphml(collection, path)
        assert path.exists()
        g = ig.Graph.Read_GraphML(str(path))
        assert g.vcount() == 3
        assert g.ecount() == 3

    def test_edge_weight_and_similarities_present(self, tmp_path: Path):
        """Every edge should carry ref_similarity, cit_similarity,
        semantic_similarity, and a combined weight attribute."""
        collection = {
            "A": _paper("A", references=[]),
            "B": _paper("B", references=["A"]),
        }
        path = tmp_path / "graph.graphml"
        export_graphml(collection, path)
        g = ig.Graph.Read_GraphML(str(path))
        assert g.ecount() == 1
        edge_attrs = set(g.es.attributes())
        assert {"ref_similarity", "cit_similarity", "semantic_similarity", "weight"}.issubset(edge_attrs)
        # With no s2 prefetch, semantic_similarity is 0.0 for all edges.
        assert g.es[0]["semantic_similarity"] == 0.0
        # weight is the max of the three, floored at _MIN_SIM.
        from citeclaw.output.graphml_writer import _MIN_SIM
        assert g.es[0]["weight"] >= _MIN_SIM
        expected = max(
            g.es[0]["ref_similarity"],
            g.es[0]["cit_similarity"],
            g.es[0]["semantic_similarity"],
            _MIN_SIM,
        )
        assert abs(g.es[0]["weight"] - expected) < 1e-9

    def test_semantic_similarity_from_s2_embeddings(self, tmp_path: Path, fake_s2):
        """When an s2 client is passed in, export_graphml should prefetch
        embeddings and compute semantic_similarity per edge."""
        # The chain corpus has SEED [1,0,0], CITER1 [0.9,0.1,0], CITER2 [0.7,0.3,0]
        # with CITER1 referencing SEED and CITER2 referencing SEED + CITER1.
        from citeclaw.clients.s2.converters import paper_to_record
        collection = {
            pid: paper_to_record(fake_s2._papers[pid])
            for pid in ("SEED", "CITER1", "CITER2")
        }
        path = tmp_path / "graph.graphml"
        export_graphml(collection, path, s2=fake_s2)
        g = ig.Graph.Read_GraphML(str(path))
        # Each connected edge should have a non-zero semantic similarity.
        sem_values = g.es["semantic_similarity"]
        assert len(sem_values) == g.ecount()
        assert all(s > 0.0 for s in sem_values), sem_values
        # And s2.fetch_embeddings_batch was called exactly once.
        assert fake_s2.calls.get("fetch_embeddings_batch", 0) == 1

    def test_semantic_similarity_zero_on_missing_embedding(self, tmp_path: Path, fake_s2):
        """A paper without an embedding in the fake corpus yields
        semantic_similarity=0.0 for any edge touching it."""
        from citeclaw.clients.s2.converters import paper_to_record
        # REF1 has no embedding in build_chain_corpus; SEED does.
        collection = {
            "SEED": paper_to_record(fake_s2._papers["SEED"]),
            "REF1": paper_to_record(fake_s2._papers["REF1"]),
        }
        # Manually wire SEED -> REF1 edge by having SEED reference REF1 (it does).
        path = tmp_path / "graph.graphml"
        export_graphml(collection, path, s2=fake_s2)
        g = ig.Graph.Read_GraphML(str(path))
        assert g.ecount() == 1
        assert g.es[0]["semantic_similarity"] == 0.0

    def test_s2_prefetch_failure_is_tolerated(self, tmp_path: Path, fake_s2, monkeypatch):
        """If the embedding prefetch raises, export_graphml logs a warning
        and writes the graph with semantic_similarity=0."""
        def boom(*a, **kw):
            raise RuntimeError("embeddings down")
        monkeypatch.setattr(fake_s2, "fetch_embeddings_batch", boom)
        collection = {
            "A": _paper("A"),
            "B": _paper("B", references=["A"]),
        }
        path = tmp_path / "graph.graphml"
        export_graphml(collection, path, s2=fake_s2)
        g = ig.Graph.Read_GraphML(str(path))
        assert g.ecount() == 1
        assert g.es[0]["semantic_similarity"] == 0.0

    def test_with_metadata_and_edge_meta(self, tmp_path: Path):
        collection = {
            "A": _paper("A"),
            "B": _paper("B", references=["A"]),
        }
        edge_meta = {
            ("A", "B"): {
                "contexts": ["quoted text 1", "quoted text 2"],
                "intents": ["methodology"],
                "is_influential": True,
            }
        }
        path = tmp_path / "graph.graphml"
        export_graphml(
            collection, path,
            metadata={"citeclaw_iteration": 2},
            edge_meta=edge_meta,
        )
        g = ig.Graph.Read_GraphML(str(path))
        assert g.ecount() == 1
        # Edge metadata should survive the round-trip.
        assert "contexts" in g.es.attributes()
        assert "intents" in g.es.attributes()
        assert "is_influential" in g.es.attributes()
        assert "quoted text 1" in g.es[0]["contexts"]
        assert g.es[0]["intents"] == "methodology"
        assert g.es[0]["is_influential"] == "true"

    def test_single_node(self, tmp_path: Path):
        collection = {"A": _paper("A")}
        path = tmp_path / "graph.graphml"
        export_graphml(collection, path)
        g = ig.Graph.Read_GraphML(str(path))
        assert g.vcount() == 1
        assert g.ecount() == 0

    def test_author_columns_present(self, tmp_path: Path):
        collection = {
            "A": _paper(
                "A",
                authors=[
                    {"authorId": "au1", "name": "Alice"},
                    {"authorId": "au2", "name": "Bob"},
                ],
            ),
        }
        path = tmp_path / "graph.graphml"
        export_graphml(collection, path)
        g = ig.Graph.Read_GraphML(str(path))
        assert "authors" in g.vs.attributes()
        assert g.vs[0]["authors"] == "Alice; Bob"
        assert g.vs[0]["author_ids"] == "au1,au2"

    def test_cluster_node_attributes(self, tmp_path: Path):
        """Passing ``clusters={...}`` writes cluster_<name> node attributes,
        plus cluster_<name>_label when at least one cluster has a label."""
        from citeclaw.cluster.base import ClusterMetadata, ClusterResult

        collection = {
            "A": _paper("A"),
            "B": _paper("B", references=["A"]),
            "C": _paper("C", references=["A"]),
        }
        cluster_result = ClusterResult(
            membership={"A": 0, "B": 0, "C": 1},
            metadata={
                0: ClusterMetadata(label="ml topic", size=2),
                1: ClusterMetadata(label="bio topic", size=1),
            },
            algorithm="precomputed",
        )
        path = tmp_path / "graph.graphml"
        export_graphml(collection, path, clusters={"topics": cluster_result})
        g = ig.Graph.Read_GraphML(str(path))
        # The integer cluster id and label both surface as node attributes.
        assert "cluster_topics" in g.vs.attributes()
        assert "cluster_topics_label" in g.vs.attributes()
        # Cluster ids round-trip as int.
        ids = {v["paper_id"]: v["cluster_topics"] for v in g.vs}
        assert ids["A"] == 0
        assert ids["B"] == 0
        assert ids["C"] == 1
        labels = {v["paper_id"]: v["cluster_topics_label"] for v in g.vs}
        assert labels["A"] == "ml topic"
        assert labels["C"] == "bio topic"

    def test_cluster_attribute_no_labels(self, tmp_path: Path):
        """When no cluster has a label, only cluster_<name> is written
        (no parallel _label attribute)."""
        from citeclaw.cluster.base import ClusterResult

        collection = {"A": _paper("A"), "B": _paper("B")}
        cluster_result = ClusterResult(
            membership={"A": 0, "B": 1},
            algorithm="x",
        )
        path = tmp_path / "graph.graphml"
        export_graphml(collection, path, clusters={"x": cluster_result})
        g = ig.Graph.Read_GraphML(str(path))
        assert "cluster_x" in g.vs.attributes()
        assert "cluster_x_label" not in g.vs.attributes()

    def test_papers_outside_membership_default_to_minus_one(self, tmp_path: Path):
        """A paper present in the collection but absent from the cluster's
        membership map gets cluster_id = -1 (noise / unassigned)."""
        from citeclaw.cluster.base import ClusterResult

        collection = {"A": _paper("A"), "B": _paper("B")}
        # Only A is in the membership.
        cluster_result = ClusterResult(membership={"A": 0}, algorithm="x")
        path = tmp_path / "graph.graphml"
        export_graphml(collection, path, clusters={"x": cluster_result})
        g = ig.Graph.Read_GraphML(str(path))
        ids = {v["paper_id"]: v["cluster_x"] for v in g.vs}
        assert ids["A"] == 0
        assert ids["B"] == -1


# ---------------------------------------------------------------------------
# Author collaboration graph
# ---------------------------------------------------------------------------


class TestAuthorGraph:
    def test_empty_collection(self, tmp_path: Path):
        g = build_author_graph({})
        assert g.vcount() == 0
        assert g.ecount() == 0

    def test_co_authorship_edge(self):
        """Alice + Bob co-author paper A → one edge."""
        collection = {
            "A": _paper(
                "A", year=2020,
                authors=[
                    {"authorId": "au1", "name": "Alice"},
                    {"authorId": "au2", "name": "Bob"},
                ],
            )
        }
        g = build_author_graph(collection)
        assert g.vcount() == 2
        assert g.ecount() == 1
        # Strength of a 2-author paper = 1/2
        assert abs(g.es[0]["strength"] - 0.5) < 1e-9
        assert g.es[0]["first_year"] == 2020
        assert g.es[0]["last_year"] == 2020
        assert g.es[0]["duration"] == 0
        assert g.es[0]["n_collaborations"] == 1

    def test_multiple_papers_accumulate(self):
        auth1 = {"authorId": "A1", "name": "Alice"}
        auth2 = {"authorId": "A2", "name": "Bob"}
        collection = {
            "P1": _paper("P1", year=2018, authors=[auth1, auth2]),
            "P2": _paper("P2", year=2022, authors=[auth1, auth2]),
        }
        g = build_author_graph(collection)
        assert g.ecount() == 1
        e = g.es[0]
        assert e["n_collaborations"] == 2
        assert e["first_year"] == 2018
        assert e["last_year"] == 2022
        assert e["duration"] == 4

    def test_author_details_populate_attrs(self):
        auth = {"authorId": "A1", "name": "Alice"}
        collection = {"P": _paper("P", authors=[auth])}
        details = {
            "A1": {
                "name": "Alice Adams",
                "citationCount": 999,
                "hIndex": 15,
                "paperCount": 42,
                "affiliations": ["Exeter", "JIC"],
            }
        }
        g = build_author_graph(collection, details)
        assert g.vs[0]["name"] == "Alice"
        assert g.vs[0]["total_citation"] == 999
        assert g.vs[0]["h_index"] == 15
        assert g.vs[0]["paper_count_s2"] == 42
        assert "Exeter" in g.vs[0]["affiliation"]

    def test_fallback_to_name_key(self):
        """Authors with no authorId should still be included via a name: key."""
        collection = {
            "P": _paper(
                "P",
                authors=[{"name": "Solo Author"}],
            )
        }
        g = build_author_graph(collection)
        assert g.vcount() == 1
        assert g.vs[0]["name"] == "Solo Author"
        assert g.vs[0]["author_id"].startswith("name:")

    def test_export_roundtrip(self, tmp_path: Path):
        collection = {
            "P": _paper(
                "P", year=2020,
                authors=[
                    {"authorId": "A1", "name": "Alice"},
                    {"authorId": "A2", "name": "Bob"},
                ],
            )
        }
        path = tmp_path / "collab.graphml"
        export_author_graphml(collection, {}, path, metadata={"citeclaw_iteration": 1})
        assert path.exists()
        g = ig.Graph.Read_GraphML(str(path))
        assert g.vcount() == 2
        assert g.ecount() == 1
