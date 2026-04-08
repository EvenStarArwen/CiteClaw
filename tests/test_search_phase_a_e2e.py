"""Phase A end-to-end smoke test for the new search surface against
:class:`tests.fakes.FakeS2Client`.

PA-01..PA-05 added six new methods to the real
``SemanticScholarClient`` (``search_bulk``, ``search_match``,
``search_relevance``, ``fetch_recommendations``,
``fetch_recommendations_for_paper``, ``fetch_author_papers``). Phase B
and C tests will lean on the offline fake to canned responses for the
meta-LLM agent and the ``ExpandBy*`` step family. This file is the
contract test that proves the fake's surface for the four methods PA-10
extends — ``search_bulk`` / ``search_match`` / ``fetch_recommendations``
/ ``fetch_author_papers`` — behaves the way downstream test code will
assume: registration → canned data, no registration → empty payload in
the same shape the real client returns.

These tests intentionally exercise the fake itself, not the real
client; the real client's wire-shape is already covered by
``tests/test_s2_search_api.py``.
"""

from __future__ import annotations

import pytest

from tests.fakes import FakeS2Client, make_paper


@pytest.fixture
def client() -> FakeS2Client:
    return FakeS2Client()


# ---------------------------------------------------------------------------
# search_bulk
# ---------------------------------------------------------------------------


class TestFakeSearchBulk:
    def test_returns_canned_data_for_registered_query(self, client: FakeS2Client):
        client.register_search_bulk(
            "transformers",
            [
                make_paper("p1", title="Attention Is All You Need"),
                make_paper("p2", title="Reformer"),
            ],
        )

        result = client.search_bulk("transformers")

        assert result["total"] == 2
        assert result["token"] is None
        assert [p["paperId"] for p in result["data"]] == ["p1", "p2"]
        assert result["data"][0]["title"] == "Attention Is All You Need"

    def test_unregistered_query_returns_empty_payload(self, client: FakeS2Client):
        result = client.search_bulk("never registered")

        assert result == {"data": [], "total": 0, "token": None}

    def test_limit_truncates_data_but_total_reflects_full_corpus(
        self, client: FakeS2Client,
    ):
        client.register_search_bulk(
            "q", [make_paper(f"p{i}") for i in range(20)],
        )

        result = client.search_bulk("q", limit=5)

        assert len(result["data"]) == 5
        assert [p["paperId"] for p in result["data"]] == ["p0", "p1", "p2", "p3", "p4"]
        # The real client surfaces ``total`` as the unbounded match count;
        # the fake mirrors that so callers can detect when more pages exist.
        assert result["total"] == 20

    def test_filters_sort_token_accepted_but_ignored(self, client: FakeS2Client):
        client.register_search_bulk("q", [make_paper("p1")])

        result = client.search_bulk(
            "q",
            filters={"year": "2020-2025", "venue": "Nature"},
            sort="citationCount:desc",
            token="page-2",
        )

        assert len(result["data"]) == 1
        assert result["data"][0]["paperId"] == "p1"

    def test_returned_papers_are_independent_copies(self, client: FakeS2Client):
        client.register_search_bulk("q", [make_paper("p1", title="Original")])

        first = client.search_bulk("q")
        first["data"][0]["title"] = "MUTATED"

        second = client.search_bulk("q")
        assert second["data"][0]["title"] == "Original"

    def test_records_call_count(self, client: FakeS2Client):
        client.search_bulk("q1")
        client.search_bulk("q2")
        client.search_bulk("q1")

        assert client.calls.get("search_bulk") == 3


# ---------------------------------------------------------------------------
# search_match
# ---------------------------------------------------------------------------


class TestFakeSearchMatch:
    def test_returns_canned_paper_for_registered_title(self, client: FakeS2Client):
        canned = make_paper("AIAYN", title="Attention Is All You Need")
        client.register_search_match("Attention Is All You Need", canned)

        result = client.search_match("Attention Is All You Need")

        assert result is not None
        assert result["paperId"] == "AIAYN"
        assert result["title"] == "Attention Is All You Need"

    def test_unregistered_title_returns_none(self, client: FakeS2Client):
        assert client.search_match("nonexistent title") is None

    def test_register_with_none_clears_previous(self, client: FakeS2Client):
        client.register_search_match("title", make_paper("p1"))
        assert client.search_match("title") is not None

        client.register_search_match("title", None)
        assert client.search_match("title") is None

    def test_returned_paper_is_independent_copy(self, client: FakeS2Client):
        client.register_search_match("t", make_paper("p1", title="Original"))

        first = client.search_match("t")
        assert first is not None
        first["title"] = "MUTATED"

        second = client.search_match("t")
        assert second is not None
        assert second["title"] == "Original"

    def test_records_call_count(self, client: FakeS2Client):
        client.search_match("a")
        client.search_match("b")

        assert client.calls.get("search_match") == 2


# ---------------------------------------------------------------------------
# fetch_recommendations
# ---------------------------------------------------------------------------


class TestFakeFetchRecommendations:
    def test_returns_canned_for_registered_anchor_set(self, client: FakeS2Client):
        client.register_recommendations(
            ["seed-1", "seed-2"],
            [
                make_paper("rec-1"),
                make_paper("rec-2"),
                make_paper("rec-3"),
            ],
        )

        result = client.fetch_recommendations(["seed-1", "seed-2"])

        assert [p["paperId"] for p in result] == ["rec-1", "rec-2", "rec-3"]

    def test_anchor_order_does_not_matter(self, client: FakeS2Client):
        client.register_recommendations(
            ["seed-1", "seed-2"], [make_paper("rec-1")],
        )

        result = client.fetch_recommendations(["seed-2", "seed-1"])

        assert len(result) == 1
        assert result[0]["paperId"] == "rec-1"

    def test_unregistered_anchors_return_empty_list(self, client: FakeS2Client):
        assert client.fetch_recommendations(["unknown-1", "unknown-2"]) == []

    def test_limit_truncates_result(self, client: FakeS2Client):
        client.register_recommendations(
            ["a"], [make_paper(f"rec-{i}") for i in range(50)],
        )

        result = client.fetch_recommendations(["a"], limit=10)

        assert len(result) == 10
        assert [p["paperId"] for p in result] == [f"rec-{i}" for i in range(10)]

    def test_negative_ids_and_fields_accepted_but_ignored(self, client: FakeS2Client):
        client.register_recommendations(["a"], [make_paper("rec-1")])

        result = client.fetch_recommendations(
            ["a"],
            negative_ids=["bad-1", "bad-2"],
            fields="paperId,title,year,abstract",
        )

        assert len(result) == 1
        assert result[0]["paperId"] == "rec-1"

    def test_returned_papers_are_independent_copies(self, client: FakeS2Client):
        client.register_recommendations(
            ["a"], [make_paper("p1", title="Original")],
        )

        first = client.fetch_recommendations(["a"])
        first[0]["title"] = "MUTATED"

        second = client.fetch_recommendations(["a"])
        assert second[0]["title"] == "Original"

    def test_records_call_count(self, client: FakeS2Client):
        client.fetch_recommendations(["x"])
        client.fetch_recommendations(["y"])

        assert client.calls.get("fetch_recommendations") == 2


# ---------------------------------------------------------------------------
# fetch_author_papers
# ---------------------------------------------------------------------------


class TestFakeFetchAuthorPapers:
    def test_returns_canned_for_registered_author(self, client: FakeS2Client):
        client.register_author_papers(
            "author-1",
            [
                make_paper("p1"),
                make_paper("p2"),
                make_paper("p3"),
            ],
        )

        result = client.fetch_author_papers("author-1")

        assert [p["paperId"] for p in result] == ["p1", "p2", "p3"]

    def test_unregistered_author_returns_empty_list(self, client: FakeS2Client):
        assert client.fetch_author_papers("unknown-author") == []

    def test_limit_truncates_result(self, client: FakeS2Client):
        client.register_author_papers(
            "a1", [make_paper(f"p{i}") for i in range(50)],
        )

        result = client.fetch_author_papers("a1", limit=5)

        assert len(result) == 5
        assert [p["paperId"] for p in result] == [f"p{i}" for i in range(5)]

    def test_fields_accepted_but_ignored(self, client: FakeS2Client):
        client.register_author_papers("a1", [make_paper("p1")])

        result = client.fetch_author_papers(
            "a1", fields="paperId,title,abstract,year,venue,citationCount",
        )

        assert len(result) == 1
        assert result[0]["paperId"] == "p1"

    def test_returned_papers_are_independent_copies(self, client: FakeS2Client):
        client.register_author_papers(
            "a1", [make_paper("p1", title="Original")],
        )

        first = client.fetch_author_papers("a1")
        first[0]["title"] = "MUTATED"

        second = client.fetch_author_papers("a1")
        assert second[0]["title"] == "Original"

    def test_records_call_count(self, client: FakeS2Client):
        client.fetch_author_papers("a1")
        client.fetch_author_papers("a2")
        client.fetch_author_papers("a1")

        assert client.calls.get("fetch_author_papers") == 3


# ---------------------------------------------------------------------------
# Cross-method integration: a single FakeS2Client can serve all four
# surfaces simultaneously without state leaking between them.
# ---------------------------------------------------------------------------


class TestFakeSurfaceIntegration:
    def test_all_four_surfaces_share_one_client(self, client: FakeS2Client):
        client.register_search_bulk("ml", [make_paper("bulk-1")])
        client.register_search_match("Famous Paper", make_paper("match-1"))
        client.register_recommendations(["anchor"], [make_paper("rec-1")])
        client.register_author_papers("au-1", [make_paper("auth-1")])

        bulk = client.search_bulk("ml")
        match = client.search_match("Famous Paper")
        recs = client.fetch_recommendations(["anchor"])
        auth = client.fetch_author_papers("au-1")

        assert bulk["data"][0]["paperId"] == "bulk-1"
        assert match is not None and match["paperId"] == "match-1"
        assert recs[0]["paperId"] == "rec-1"
        assert auth[0]["paperId"] == "auth-1"

        # Each method bumped only its own counter.
        assert client.calls == {
            "search_bulk": 1,
            "search_match": 1,
            "fetch_recommendations": 1,
            "fetch_author_papers": 1,
        }
