"""Tests for ``citeclaw.steps.expand_by_author.ExpandByAuthor``."""

from __future__ import annotations

import pytest

from citeclaw.context import Context
from citeclaw.filters.builder import build_blocks
from citeclaw.models import PaperRecord
from citeclaw.steps.expand_by_author import ExpandByAuthor
from tests.fakes import FakeS2Client, make_paper


def _author(aid: str, *, h: int = 0, citations: int = 0, name: str | None = None) -> dict:
    return {
        "authorId": aid,
        "name": name or f"Author {aid}",
        "hIndex": h,
        "citationCount": citations,
        "paperCount": 1,
        "affiliations": [],
    }


def _paper_with_authors(pid: str, authors: list[dict], *, year: int = 2024) -> dict:
    return make_paper(pid, year=year, authors=authors)


def _basic_screener():
    blocks = build_blocks({
        "y": {"type": "YearFilter", "min": 2020},
        "screen": {"type": "Sequential", "layers": ["y"]},
    })
    return blocks["screen"]


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="author_metric"):
            ExpandByAuthor(author_metric="bogus")

    @pytest.mark.parametrize("metric", ["h_index", "citation_count", "degree_in_collab_graph"])
    def test_valid_metric_accepted(self, metric: str):
        s = ExpandByAuthor(author_metric=metric)
        assert s.author_metric == metric


# ---------------------------------------------------------------------------
# Run-loop early-exit guards
# ---------------------------------------------------------------------------


class TestEarlyExits:
    def test_no_screener_returns_empty(self, ctx: Context):
        signal = [PaperRecord(paper_id="p1")]
        result = ExpandByAuthor().run(signal, ctx)
        assert result.signal == []
        assert result.stats == {"reason": "no screener"}

    def test_no_authors_in_signal_returns_empty(self, ctx: Context):
        signal = [PaperRecord(paper_id="p1", authors=[])]
        result = ExpandByAuthor(screener=_basic_screener()).run(signal, ctx)
        assert result.signal == []
        assert result.stats["reason"] == "no_authors"

    def test_authors_without_authorid_skipped(self, ctx: Context):
        signal = [PaperRecord(paper_id="p1", authors=[{"name": "Solo"}])]
        result = ExpandByAuthor(screener=_basic_screener()).run(signal, ctx)
        assert result.stats["reason"] == "no_authors"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_picks_top_authors_and_screens_their_papers(
        self, ctx: Context, fake_s2: FakeS2Client,
    ):
        # Signal: two papers each authored by a different person.
        signal_a = _paper_with_authors("seed_a", [{"authorId": "A1", "name": "Alice"}])
        signal_b = _paper_with_authors("seed_b", [{"authorId": "A2", "name": "Bob"}])
        fake_s2.add(signal_a)
        fake_s2.add(signal_b)

        # Bob has higher h-index than Alice, so he wins under h_index.
        fake_s2.add_author("A1", _author("A1", h=5))
        fake_s2.add_author("A2", _author("A2", h=20))

        # Each author has one extra paper not yet in collection.
        bob_paper = make_paper("bob_solo", title="Bob's solo work", year=2023)
        fake_s2.add(bob_paper)
        fake_s2.register_author_papers("A2", [bob_paper])
        # Alice would be ignored under top_k_authors=1.

        signal = [
            PaperRecord(paper_id="seed_a", authors=signal_a["authors"]),
            PaperRecord(paper_id="seed_b", authors=signal_b["authors"]),
        ]
        step = ExpandByAuthor(
            screener=_basic_screener(),
            top_k_authors=1,
            author_metric="h_index",
            papers_per_author=10,
        )
        result = step.run(signal, ctx)

        assert result.stats["distinct_authors"] == 2
        assert result.stats["chosen_authors"] == 1
        assert result.stats["raw_paper_count"] == 1
        assert result.stats["accepted"] == 1
        assert result.stats["metric"] == "h_index"
        assert "bob_solo" in ctx.collection

    def test_idempotent_second_run(self, ctx: Context, fake_s2: FakeS2Client):
        seed = _paper_with_authors("seed", [{"authorId": "A1", "name": "Alice"}])
        fake_s2.add(seed)
        fake_s2.add_author("A1", _author("A1", h=5))
        alice_paper = make_paper("alice_p", year=2024)
        fake_s2.add(alice_paper)
        fake_s2.register_author_papers("A1", [alice_paper])

        signal = [PaperRecord(paper_id="seed", authors=seed["authors"])]
        step = ExpandByAuthor(screener=_basic_screener(), top_k_authors=1)
        first = step.run(signal, ctx)
        assert first.stats["accepted"] == 1

        # Second invocation with the same signal should short-circuit.
        second = step.run(signal, ctx)
        assert second.stats["reason"] == "already_searched"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_fetch_authors_failure_returns_stats_with_reason(
        self, ctx: Context, fake_s2: FakeS2Client, monkeypatch: pytest.MonkeyPatch,
    ):
        seed = _paper_with_authors("seed", [{"authorId": "A1", "name": "Alice"}])
        fake_s2.add(seed)

        def boom(_ids):
            raise RuntimeError("S2 down")

        monkeypatch.setattr(fake_s2, "fetch_authors_batch", boom)
        signal = [PaperRecord(paper_id="seed", authors=seed["authors"])]
        result = ExpandByAuthor(screener=_basic_screener()).run(signal, ctx)
        assert result.stats["reason"] == "fetch_authors_failed"
        assert "S2 down" in result.stats["error"]

    def test_per_author_fetch_failure_continues(
        self, ctx: Context, fake_s2: FakeS2Client, monkeypatch: pytest.MonkeyPatch,
    ):
        seed_a = _paper_with_authors("seed_a", [{"authorId": "A1", "name": "Alice"}])
        seed_b = _paper_with_authors("seed_b", [{"authorId": "A2", "name": "Bob"}])
        fake_s2.add(seed_a)
        fake_s2.add(seed_b)
        fake_s2.add_author("A1", _author("A1", h=5))
        fake_s2.add_author("A2", _author("A2", h=10))

        # Bob has higher h-index; A2 fetch will fail. A1 succeeds.
        alice_paper = make_paper("alice_p", year=2024)
        fake_s2.add(alice_paper)
        fake_s2.register_author_papers("A1", [alice_paper])

        original = fake_s2.fetch_author_papers

        def fetch(aid, **kw):
            if aid == "A2":
                raise RuntimeError("A2 broken")
            return original(aid, **kw)

        monkeypatch.setattr(fake_s2, "fetch_author_papers", fetch)
        signal = [
            PaperRecord(paper_id="seed_a", authors=seed_a["authors"]),
            PaperRecord(paper_id="seed_b", authors=seed_b["authors"]),
        ]
        # top_k=2 chooses both; A2 fails individually but A1 still produces.
        result = ExpandByAuthor(screener=_basic_screener(), top_k_authors=2).run(signal, ctx)
        assert result.stats["accepted"] == 1
        assert "alice_p" in ctx.collection
