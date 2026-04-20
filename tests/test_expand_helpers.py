"""Tests for ``citeclaw.steps._expand_helpers``.

Covers the three exported helpers (``fingerprint_signal``,
``check_already_searched``, ``screen_expand_candidates``) end-to-end:
hydrate → enrich → optional post-hydrate trim → seen-dedup → screener
→ ctx.collection commit. All five ``ExpandBy*`` steps share this code,
so the unit tests here pin behaviour the steps rely on.
"""

from __future__ import annotations

from typing import Any

import pytest

from citeclaw.context import Context
from citeclaw.filters.atoms.year import YearFilter
from citeclaw.filters.builder import build_blocks
from citeclaw.models import PaperRecord
from citeclaw.steps._expand_helpers import (
    ExpandScreenResult,
    check_already_searched,
    fingerprint_signal,
    screen_expand_candidates,
)
from tests.fakes import FakeS2Client, make_paper


# ---------------------------------------------------------------------------
# fingerprint_signal
# ---------------------------------------------------------------------------


class TestFingerprintSignal:
    def test_stable_for_same_inputs(self):
        sig = [PaperRecord(paper_id="p1"), PaperRecord(paper_id="p2")]
        a = fingerprint_signal("ExpandBySearch", sig, k=3)
        b = fingerprint_signal("ExpandBySearch", sig, k=3)
        assert a == b
        assert len(a) == 64

    def test_order_insensitive(self):
        s1 = [PaperRecord(paper_id="p1"), PaperRecord(paper_id="p2")]
        s2 = [PaperRecord(paper_id="p2"), PaperRecord(paper_id="p1")]
        assert fingerprint_signal("X", s1) == fingerprint_signal("X", s2)

    def test_step_name_changes_fingerprint(self):
        sig = [PaperRecord(paper_id="p1")]
        assert fingerprint_signal("A", sig) != fingerprint_signal("B", sig)

    def test_extra_kwargs_change_fingerprint(self):
        sig = [PaperRecord(paper_id="p1")]
        assert fingerprint_signal("X", sig, k=3) != fingerprint_signal("X", sig, k=4)

    def test_skips_records_without_paper_id(self):
        sig = [PaperRecord(paper_id="p1"), PaperRecord(paper_id="")]
        bare = [PaperRecord(paper_id="p1")]
        assert fingerprint_signal("X", sig) == fingerprint_signal("X", bare)


# ---------------------------------------------------------------------------
# check_already_searched
# ---------------------------------------------------------------------------


class TestCheckAlreadySearched:
    def test_returns_none_on_fresh_fingerprint(self, ctx: Context):
        result = check_already_searched("ExpandBySearch", "deadbeef" * 8, ctx, signal_len=5)
        assert result is None

    def test_returns_noop_step_result_when_seen(self, ctx: Context):
        fp = "cafebabe" * 8
        ctx.searched_signals.add(fp)
        result = check_already_searched("ExpandBySearch", fp, ctx, signal_len=7)
        assert result is not None
        assert result.signal == []
        assert result.in_count == 7
        assert result.stats["reason"] == "already_searched"
        assert result.stats["fingerprint"].startswith("cafebabe")


# ---------------------------------------------------------------------------
# screen_expand_candidates — full pipeline
# ---------------------------------------------------------------------------


def _make_screener(min_year: int = 2020):
    """Tiny YearFilter wrapped as a one-layer Sequential block."""
    blocks = build_blocks({
        "y": {"type": "YearFilter", "min": min_year},
        "screen": {"type": "Sequential", "layers": ["y"]},
    })
    return blocks["screen"]


def _seed_corpus(fake: FakeS2Client) -> None:
    fake.add(make_paper("p_old", title="Old", year=2010, citation_count=5))
    fake.add(make_paper("p_new", title="New", year=2024, citation_count=20))
    fake.add(make_paper("p_other", title="Other", year=2023, citation_count=15))


class TestScreenExpandCandidatesEmpty:
    def test_empty_raw_hits_returns_empty(self, ctx: Context):
        screener = _make_screener()
        result = screen_expand_candidates(
            raw_hits=[], source_label="search", screener=screener, ctx=ctx,
        )
        assert isinstance(result, ExpandScreenResult)
        assert result.hydrated == []
        assert result.passed == []
        assert result.base_stats["raw_hits"] == 0
        assert result.base_stats["accepted"] == 0
        assert ctx.collection == {}

    def test_none_raw_hits_returns_empty(self, ctx: Context):
        screener = _make_screener()
        result = screen_expand_candidates(
            raw_hits=None, source_label="search", screener=screener, ctx=ctx,
        )
        assert result.base_stats["raw_hits"] == 0


class TestScreenExpandCandidatesPipeline:
    def test_screener_filters_by_year(self, ctx: Context, fake_s2: FakeS2Client):
        _seed_corpus(fake_s2)
        screener = _make_screener(min_year=2020)

        result = screen_expand_candidates(
            raw_hits=[
                {"paperId": "p_old"},     # rejected
                {"paperId": "p_new"},     # accepted
                {"paperId": "p_other"},   # accepted
            ],
            source_label="semantic",
            screener=screener,
            ctx=ctx,
        )

        passed_ids = {p.paper_id for p in result.passed}
        assert passed_ids == {"p_new", "p_other"}
        assert result.base_stats["raw_hits"] == 3
        assert result.base_stats["hydrated"] == 3
        assert result.base_stats["novel"] == 3
        assert result.base_stats["accepted"] == 2
        assert result.base_stats["rejected"] == 1

    def test_source_label_stamped_on_survivors(self, ctx: Context, fake_s2: FakeS2Client):
        _seed_corpus(fake_s2)
        screener = _make_screener()
        result = screen_expand_candidates(
            raw_hits=[{"paperId": "p_new"}],
            source_label="author",
            screener=screener,
            ctx=ctx,
        )
        assert result.passed[0].source == "author"

    def test_survivors_added_to_collection_with_accept_verdict(self, ctx: Context, fake_s2: FakeS2Client):
        _seed_corpus(fake_s2)
        screener = _make_screener()
        screen_expand_candidates(
            raw_hits=[{"paperId": "p_new"}],
            source_label="search",
            screener=screener,
            ctx=ctx,
        )
        assert "p_new" in ctx.collection
        assert ctx.collection["p_new"].llm_verdict == "accept"

    def test_already_seen_papers_skipped(self, ctx: Context, fake_s2: FakeS2Client):
        _seed_corpus(fake_s2)
        ctx.seen.add("p_new")
        screener = _make_screener()
        result = screen_expand_candidates(
            raw_hits=[{"paperId": "p_new"}, {"paperId": "p_other"}],
            source_label="search",
            screener=screener,
            ctx=ctx,
        )
        # p_new is filtered at the dedup stage so only p_other reaches the screener.
        novel_ids = {p.paper_id for p in result.novel}
        assert novel_ids == {"p_other"}
        assert result.base_stats["novel"] == 1

    def test_unknown_paper_id_silently_dropped(self, ctx: Context, fake_s2: FakeS2Client):
        _seed_corpus(fake_s2)
        screener = _make_screener()
        result = screen_expand_candidates(
            raw_hits=[{"paperId": "p_new"}, {"paperId": "p_unknown"}],
            source_label="search",
            screener=screener,
            ctx=ctx,
        )
        assert result.base_stats["raw_hits"] == 2
        assert result.base_stats["hydrated"] == 1

    def test_hits_without_paper_id_skipped(self, ctx: Context, fake_s2: FakeS2Client):
        _seed_corpus(fake_s2)
        screener = _make_screener()
        result = screen_expand_candidates(
            raw_hits=[{"paperId": "p_new"}, {}, {"other_key": "x"}],
            source_label="search",
            screener=screener,
            ctx=ctx,
        )
        assert result.base_stats["hydrated"] == 1

    def test_paperid_attribute_form_also_works(self, ctx: Context, fake_s2: FakeS2Client):
        """``raw_hits`` may carry attr-style objects (PaperRecord-shaped)."""
        _seed_corpus(fake_s2)
        screener = _make_screener()

        class Hit:
            paper_id = "p_new"

        result = screen_expand_candidates(
            raw_hits=[Hit()],
            source_label="search",
            screener=screener,
            ctx=ctx,
        )
        assert result.base_stats["accepted"] == 1


class TestScreenExpandPostHydrate:
    def test_post_hydrate_trim_applied(self, ctx: Context, fake_s2: FakeS2Client):
        _seed_corpus(fake_s2)
        screener = _make_screener()

        def keep_first(records: list[PaperRecord]) -> list[PaperRecord]:
            return records[:1]

        result = screen_expand_candidates(
            raw_hits=[{"paperId": "p_new"}, {"paperId": "p_other"}],
            source_label="search",
            screener=screener,
            ctx=ctx,
            post_hydrate_fn=keep_first,
        )
        assert result.base_stats["hydrated"] == 1

    def test_post_hydrate_failure_falls_back_to_untrimmed(
        self, ctx: Context, fake_s2: FakeS2Client,
    ):
        _seed_corpus(fake_s2)
        screener = _make_screener()

        def boom(records: list[PaperRecord]) -> list[PaperRecord]:
            raise RuntimeError("post-hydrate exploded")

        result = screen_expand_candidates(
            raw_hits=[{"paperId": "p_new"}],
            source_label="search",
            screener=screener,
            ctx=ctx,
            post_hydrate_fn=boom,
        )
        assert result.base_stats["accepted"] == 1


class TestScreenExpandRejections:
    def test_rejection_ledger_populated(self, ctx: Context, fake_s2: FakeS2Client):
        _seed_corpus(fake_s2)
        screener = _make_screener(min_year=2020)
        screen_expand_candidates(
            raw_hits=[{"paperId": "p_old"}],
            source_label="search",
            screener=screener,
            ctx=ctx,
        )
        assert "p_old" in ctx.rejection_ledger
        assert "year" in ctx.rejection_ledger["p_old"]
        assert ctx.rejection_counts["year"] == 1

    def test_all_rejected_keeps_collection_empty(self, ctx: Context, fake_s2: FakeS2Client):
        _seed_corpus(fake_s2)
        screener = _make_screener(min_year=2030)
        result = screen_expand_candidates(
            raw_hits=[{"paperId": "p_new"}, {"paperId": "p_other"}],
            source_label="search",
            screener=screener,
            ctx=ctx,
        )
        assert result.passed == []
        assert ctx.collection == {}
