"""Tests for :mod:`citeclaw.dedup` and the ``MergeDuplicates`` pipeline step.

Covers:
- Jaro-Winkler similarity math
- Title normalisation
- Canonical selection (peer-reviewed preferred)
- Cluster detection across all four signals (external IDs, title
  similarity, author + year, semantic cosine)
- Destructive merge: alias list, reference union, metadata fill-in
- Alias remap rewrites references in unrelated records
- ``MergeDuplicates`` step integration with ``ctx.alias_map``
- End-to-end: graph export respects the alias map
"""

from __future__ import annotations

import igraph as ig

from citeclaw.dedup import (
    detect_duplicate_clusters,
    jaro_similarity,
    jaro_winkler_similarity,
    merge_cluster,
    normalize_title,
    pick_canonical,
)
from citeclaw.models import PaperRecord
from citeclaw.output import export_graphml
from citeclaw.steps.merge_duplicates import MergeDuplicates


def _paper(
    paper_id: str,
    *,
    title: str = "",
    year: int | None = 2022,
    venue: str | None = "Nature",
    citation_count: int | None = 10,
    references: list[str] | None = None,
    authors: list[dict] | None = None,
    external_ids: dict[str, str] | None = None,
    abstract: str | None = "some abstract",
) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title or paper_id,
        year=year,
        venue=venue,
        citation_count=citation_count,
        references=references or [],
        authors=authors or [{"authorId": "au1", "name": "Alice"}],
        external_ids=external_ids or {},
        abstract=abstract,
    )


# ---------------------------------------------------------------------------
# Jaro / Jaro-Winkler
# ---------------------------------------------------------------------------


class TestJaroWinkler:
    def test_identical(self):
        assert jaro_similarity("abc", "abc") == 1.0
        assert jaro_winkler_similarity("abc", "abc") == 1.0

    def test_empty(self):
        assert jaro_similarity("", "") == 1.0
        assert jaro_similarity("abc", "") == 0.0
        assert jaro_similarity("", "abc") == 0.0

    def test_no_match(self):
        assert jaro_similarity("abc", "xyz") == 0.0

    def test_close_match_passes_095(self):
        # Preprint vs published with a minor title tweak
        a = "attention is all you need"
        b = "attention is all you need."
        assert jaro_winkler_similarity(a, b) >= 0.95

    def test_prefix_boost(self):
        # Two strings sharing "martha" prefix should score higher than raw Jaro.
        a = "martha"
        b = "marhta"
        jw = jaro_winkler_similarity(a, b)
        j = jaro_similarity(a, b)
        assert jw > j
        # Well-known example: JW('MARTHA', 'MARHTA') ≈ 0.961
        assert 0.955 < jw < 0.965

    def test_dissimilar_below_095(self):
        a = "bert pre training of deep bidirectional transformers"
        b = "gpt language models are few shot learners"
        assert jaro_winkler_similarity(a, b) < 0.95


class TestNormalizeTitle:
    def test_basic(self):
        assert normalize_title("Hello, World!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_title("  Hello  \n  World  ") == "hello world"

    def test_empty(self):
        assert normalize_title("") == ""
        assert normalize_title("   ") == ""

    def test_strips_punctuation(self):
        assert normalize_title("A Paper: Subtitle!") == "a paper subtitle"


# ---------------------------------------------------------------------------
# Canonical selection
# ---------------------------------------------------------------------------


class TestPickCanonical:
    def test_prefers_peer_reviewed(self):
        cluster = [
            _paper("arXiv-v1", venue="arXiv", citation_count=50),
            _paper("NeurIPS", venue="NeurIPS", citation_count=50),
        ]
        chosen = pick_canonical(cluster)
        assert chosen.paper_id == "NeurIPS"

    def test_prefers_higher_citations_within_same_venue_class(self):
        cluster = [
            _paper("a", venue="arXiv", citation_count=10),
            _paper("b", venue="arXiv", citation_count=500),
        ]
        assert pick_canonical(cluster).paper_id == "b"

    def test_deterministic_tiebreak_by_paper_id(self):
        cluster = [
            _paper("z", venue="ICML", citation_count=100, abstract="x"),
            _paper("a", venue="ICML", citation_count=100, abstract="x"),
        ]
        assert pick_canonical(cluster).paper_id == "a"

    def test_prefers_longer_abstract_when_venue_and_cit_tied(self):
        cluster = [
            _paper("a", venue="ICML", citation_count=10, abstract="short"),
            _paper("b", venue="ICML", citation_count=10, abstract="a much longer abstract"),
        ]
        assert pick_canonical(cluster).paper_id == "b"

    def test_all_preprints_still_picks_one(self):
        cluster = [
            _paper("a", venue="arXiv", citation_count=5),
            _paper("b", venue="bioRxiv", citation_count=20),
        ]
        # Both are preprints — higher citation wins.
        assert pick_canonical(cluster).paper_id == "b"


# ---------------------------------------------------------------------------
# Cluster detection
# ---------------------------------------------------------------------------


class TestDetectClusters:
    def test_shared_external_id_doi(self):
        """Same DOI in externalIds → same cluster, even if titles diverge."""
        coll = {
            "arx": _paper(
                "arx", title="Preprint Title",
                venue="arXiv", external_ids={"DOI": "10.1/same"},
            ),
            "neurips": _paper(
                "neurips", title="Published Title",
                venue="NeurIPS", external_ids={"DOI": "10.1/same"},
            ),
            "other": _paper("other", title="Unrelated"),
        }
        clusters = detect_duplicate_clusters(coll)
        assert len(clusters) == 1
        assert set(clusters[0]) == {"arx", "neurips"}

    def test_shared_arxiv_id(self):
        coll = {
            "a": _paper("a", external_ids={"ArXiv": "2301.00001"}, venue="arXiv"),
            "b": _paper("b", external_ids={"ArXiv": "2301.00001"}, venue="ICML"),
        }
        clusters = detect_duplicate_clusters(coll)
        assert len(clusters) == 1

    def test_title_similarity_same_author_year(self):
        """High Jaro-Winkler + same first-author + same year = merge."""
        author = [{"authorId": "au1", "name": "Alice Adams"}]
        coll = {
            "arx": _paper(
                "arx", title="Attention is all you need",
                year=2017, venue="arXiv", authors=author,
            ),
            "neurips": _paper(
                "neurips", title="Attention is all you need.",
                year=2017, venue="NeurIPS", authors=author,
            ),
        }
        clusters = detect_duplicate_clusters(coll)
        assert len(clusters) == 1
        assert set(clusters[0]) == {"arx", "neurips"}

    def test_title_similarity_different_author_no_merge(self):
        """Same title but different first authors → not a duplicate."""
        coll = {
            "a": _paper(
                "a", title="Attention is all you need",
                authors=[{"authorId": "au1", "name": "Alice"}],
            ),
            "b": _paper(
                "b", title="Attention is all you need",
                authors=[{"authorId": "au99", "name": "Zoe"}],
            ),
        }
        clusters = detect_duplicate_clusters(coll)
        assert clusters == []

    def test_title_similarity_different_year_outside_window(self):
        coll = {
            "a": _paper("a", title="Exact title", year=2018),
            "b": _paper("b", title="Exact title", year=2023),
        }
        clusters = detect_duplicate_clusters(coll, year_window=1)
        assert clusters == []

    def test_title_similarity_year_window_tolerant(self):
        coll = {
            "a": _paper("a", title="Exact title", year=2022),
            "b": _paper("b", title="Exact title", year=2023),
        }
        clusters = detect_duplicate_clusters(coll, year_window=1)
        assert len(clusters) == 1

    def test_semantic_cosine_triggers_merge(self):
        """Near-1 cosine on SPECTER2 alone should merge."""
        coll = {
            "a": _paper("a", title="Totally unrelated title one"),
            "b": _paper("b", title="Totally different text two"),
        }
        # Force both to score above the 0.98 semantic threshold.
        embeddings = {
            "a": [1.0, 0.0, 0.0],
            "b": [0.999, 0.01, 0.0],
        }
        clusters = detect_duplicate_clusters(
            coll, embeddings=embeddings, semantic_threshold=0.98,
        )
        assert len(clusters) == 1

    def test_semantic_low_cosine_no_merge(self):
        coll = {
            "a": _paper("a", title="First title"),
            "b": _paper("b", title="Second title"),
        }
        embeddings = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        clusters = detect_duplicate_clusters(
            coll, embeddings=embeddings, semantic_threshold=0.98,
        )
        assert clusters == []

    def test_no_false_positive_on_unrelated_papers(self):
        coll = {
            "a": _paper("a", title="BERT: Pre-training of deep bidirectional transformers"),
            "b": _paper("b", title="GPT: Language models are few shot learners"),
        }
        clusters = detect_duplicate_clusters(coll)
        assert clusters == []

    def test_three_way_cluster_via_transitivity(self):
        """a↔b share DOI, b↔c share title+author → all three fold together."""
        coll = {
            "a": _paper("a", external_ids={"DOI": "10.1/same"}, title="Paper",
                        venue="arXiv"),
            "b": _paper("b", external_ids={"DOI": "10.1/same"}, title="Paper",
                        venue="ICML"),
            "c": _paper("c", title="Paper", venue="ICML",
                        authors=[{"authorId": "au1", "name": "Alice"}]),
        }
        clusters = detect_duplicate_clusters(coll)
        # All three should be in one cluster
        assert len(clusters) == 1
        assert set(clusters[0]) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# merge_cluster
# ---------------------------------------------------------------------------


class TestMergeCluster:
    def test_canonical_absorbs_aliases(self):
        coll = {
            "arx": _paper("arx", venue="arXiv", abstract=""),
            "neurips": _paper("neurips", venue="NeurIPS", abstract="full"),
        }
        alias_map: dict[str, str] = {}
        canonical = merge_cluster(coll, ["arx", "neurips"], alias_map=alias_map)
        assert canonical == "neurips"
        assert "arx" in coll["neurips"].aliases
        assert "arx" not in coll  # absorbed and removed
        assert alias_map == {"arx": "neurips"}

    def test_merge_fills_missing_metadata(self):
        coll = {
            "arx": _paper(
                "arx", venue="arXiv", abstract="long arXiv abstract",
                citation_count=20, year=2020,
            ),
            "neurips": _paper(
                "neurips", venue="NeurIPS",
                abstract=None, citation_count=None, year=None,
            ),
        }
        merge_cluster(coll, ["arx", "neurips"], alias_map={})
        canonical = coll["neurips"]
        assert canonical.abstract == "long arXiv abstract"
        assert canonical.year == 2020
        # Higher citation_count wins.
        assert canonical.citation_count == 20

    def test_merge_unions_references(self):
        coll = {
            "arx": _paper("arx", venue="arXiv", references=["R1", "R2"]),
            "neurips": _paper(
                "neurips", venue="NeurIPS", references=["R2", "R3"],
            ),
        }
        merge_cluster(coll, ["arx", "neurips"], alias_map={})
        canonical = coll["neurips"]
        # Order: NeurIPS's originals first, then new ones from arXiv
        assert canonical.references == ["R2", "R3", "R1"]

    def test_merge_rewrites_references_in_other_records(self):
        """If paper X references the absorbed preprint ID, that reference
        must be rewritten to the canonical ID so graph edges are remapped."""
        coll = {
            "arx": _paper("arx", venue="arXiv"),
            "neurips": _paper("neurips", venue="NeurIPS"),
            "X": _paper("X", references=["arx", "other"]),
        }
        merge_cluster(coll, ["arx", "neurips"], alias_map={})
        assert coll["X"].references == ["neurips", "other"]

    def test_merge_cluster_removes_duplicates_after_rewrite(self):
        """If record X cites BOTH the preprint and the canonical, the
        rewritten list should dedupe to a single reference."""
        coll = {
            "arx": _paper("arx", venue="arXiv"),
            "neurips": _paper("neurips", venue="NeurIPS"),
            "X": _paper("X", references=["neurips", "arx"]),
        }
        merge_cluster(coll, ["arx", "neurips"], alias_map={})
        assert coll["X"].references == ["neurips"]

    def test_merge_transitive_aliases(self):
        """A duplicate record that was itself a merge target should pass
        its aliases into the canonical."""
        coll = {
            "a": _paper("a", venue="arXiv", abstract="first"),
            "b": _paper("b", venue="bioRxiv", abstract="second"),
            "c": _paper("c", venue="NeurIPS", abstract="third"),
        }
        # Pretend b has already absorbed x in an earlier merge.
        coll["b"].aliases = ["x"]
        merge_cluster(coll, ["a", "b", "c"], alias_map={})
        canonical = coll["c"]
        assert set(canonical.aliases) >= {"a", "b", "x"}

    def test_singleton_cluster_is_noop(self):
        coll = {"x": _paper("x")}
        result = merge_cluster(coll, ["x"], alias_map={})
        assert result == "x"
        assert "x" in coll


# ---------------------------------------------------------------------------
# MergeDuplicates step
# ---------------------------------------------------------------------------


class TestMergeDuplicatesStep:
    def test_runs_on_empty_collection(self, ctx):
        ctx.collection = {}
        step = MergeDuplicates(use_embeddings=False)
        result = step.run([], ctx)
        assert result.stats == {"clusters": 0, "merged": 0}

    def test_step_merges_and_updates_alias_map(self, ctx):
        ctx.collection = {
            "arx": _paper("arx", venue="arXiv", external_ids={"DOI": "10.1/a"}),
            "neurips": _paper("neurips", venue="NeurIPS", external_ids={"DOI": "10.1/a"}),
            "other": _paper("other", title="Different", authors=[{"authorId": "au9", "name": "Bob"}]),
        }
        step = MergeDuplicates(use_embeddings=False)
        result = step.run(list(ctx.collection.values()), ctx)
        assert result.stats["clusters"] == 1
        assert result.stats["merged"] == 1
        assert "arx" not in ctx.collection
        assert "neurips" in ctx.collection
        assert ctx.alias_map == {"arx": "neurips"}
        # rejection_counts also tracked
        assert ctx.rejection_counts["merged_duplicate"] == 1

    def test_step_rewrites_signal_to_canonicals(self, ctx):
        arx = _paper("arx", venue="arXiv", external_ids={"DOI": "10.1/a"})
        neurips = _paper("neurips", venue="NeurIPS", external_ids={"DOI": "10.1/a"})
        other = _paper("other", title="Different", authors=[{"authorId": "au9", "name": "Bob"}])
        ctx.collection = {"arx": arx, "neurips": neurips, "other": other}
        # Signal still contains the arx (preprint) — it should come back as
        # the canonical neurips record.
        step = MergeDuplicates(use_embeddings=False)
        result = step.run([arx, neurips, other], ctx)
        ids = [p.paper_id for p in result.signal]
        assert "arx" not in ids
        assert set(ids) == {"neurips", "other"}

    def test_embedding_prefetch_failure_falls_back(self, ctx, monkeypatch):
        """An S2 prefetch failure should log a warning but not crash the
        step — detection proceeds without the semantic signal."""
        def boom(*a, **kw):
            raise RuntimeError("embeddings down")

        monkeypatch.setattr(ctx.s2, "fetch_embeddings_batch", boom)
        ctx.collection = {
            "arx": _paper("arx", external_ids={"DOI": "10.1/a"}, venue="arXiv"),
            "neurips": _paper("neurips", external_ids={"DOI": "10.1/a"}, venue="NeurIPS"),
        }
        step = MergeDuplicates(use_embeddings=True)
        result = step.run(list(ctx.collection.values()), ctx)
        assert result.stats["clusters"] == 1


# ---------------------------------------------------------------------------
# Graph export respects alias_map
# ---------------------------------------------------------------------------


class TestGraphExportWithAliasMap:
    def test_reference_rewrite_before_export(self, ctx, tmp_path):
        """After MergeDuplicates, references in remaining records point at
        the canonical ID so export_graphml produces one edge, not two."""
        arx = _paper("arx", venue="arXiv", external_ids={"DOI": "10.1/a"})
        neurips = _paper("neurips", venue="NeurIPS", external_ids={"DOI": "10.1/a"})
        # X references the preprint. After merge, its reference should be
        # rewritten to point at the canonical.
        x = _paper("X", references=["arx"], authors=[{"authorId": "au9", "name": "Bob"}])
        ctx.collection = {"arx": arx, "neurips": neurips, "X": x}
        MergeDuplicates(use_embeddings=False).run(list(ctx.collection.values()), ctx)
        path = tmp_path / "graph.graphml"
        export_graphml(ctx.collection, path)
        g = ig.Graph.Read_GraphML(str(path))
        # Only two nodes (neurips + X); one edge neurips→X.
        assert g.vcount() == 2
        assert g.ecount() == 1
