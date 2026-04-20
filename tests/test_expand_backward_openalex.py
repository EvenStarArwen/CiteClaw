"""Tests for ExpandBackward's OpenAlex reference fallback.

When S2 has no references AND the source paper has a DOI in
external_ids, the step consults OpenAlex. Covered here:

  * Fallback triggers only when S2 returns empty.
  * Fallback skipped when paper has no DOI.
  * Fallback skipped when ``openalex_references=False``.
  * ``ref_dois`` from OpenAlex are each resolved via ``s2.fetch_metadata``.
  * Failures (OpenAlex down, unresolvable DOI) don't crash.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from citeclaw.cache import Cache
from citeclaw.budget import BudgetTracker
from citeclaw.config import Settings
from citeclaw.context import Context
from citeclaw.models import PaperRecord
from citeclaw.steps.expand_backward import ExpandBackward
from tests.fakes import FakeS2Client, make_paper


@pytest.fixture
def ctx(tmp_path: Path) -> Context:
    cfg = Settings(
        screening_model="stub",
        s2_api_key="dummy",
        data_dir=tmp_path,
    )
    cache = Cache(tmp_path / "cache.db")
    s2 = FakeS2Client()
    budget = BudgetTracker()
    c = Context(config=cfg, s2=s2, cache=cache, budget=budget)
    yield c
    cache.close()


def _make_screener_accept_all():
    """Build a screener block that accepts every paper."""
    from citeclaw.filters.builder import build_blocks
    return build_blocks({
        "accept_all": {"type": "YearFilter", "min": 1900, "max": 2100},
    })["accept_all"]


def test_openalex_fallback_triggers_when_s2_refs_empty(ctx):
    """S2 returns empty → OpenAlex is consulted → each returned DOI
    is resolved via fetch_metadata.

    The fake S2 stores papers under their paperId, so we key the
    resolvable papers by the ``DOI:...`` form the step will pass in.
    """
    ctx.s2.add(make_paper("src", title="Source", year=2024))
    ctx.s2.add(make_paper("DOI:10.1/ref1", title="Ref 1", year=2020, citation_count=50))
    ctx.s2.add(make_paper("DOI:10.1/ref2", title="Ref 2", year=2021, citation_count=30))

    step = ExpandBackward(
        screener=_make_screener_accept_all(),
        openalex_references=True,
    )
    source_rec = PaperRecord(
        paper_id="src", title="Source",
        external_ids={"DOI": "10.1/source"},
    )

    with patch(
        "citeclaw.clients.openalex.OpenAlexClient"
    ) as MockClient:
        MockClient.return_value.fetch_references_by_doi.return_value = [
            "10.1/ref1", "10.1/ref2",
        ]
        result = step.run([source_rec], ctx)

    assert result.stats.get("openalex_fallback_used", 0) == 1
    accepted_ids = {p.paper_id for p in result.signal}
    assert accepted_ids == {"DOI:10.1/ref1", "DOI:10.1/ref2"}


def test_openalex_fallback_skipped_when_no_doi(ctx):
    """No DOI → OpenAlex never consulted."""
    source_rec = PaperRecord(paper_id="src", title="Source", external_ids={})
    ctx.s2.add(make_paper("src", title="Source"))

    step = ExpandBackward(
        screener=_make_screener_accept_all(),
        openalex_references=True,
    )
    with patch(
        "citeclaw.clients.openalex.OpenAlexClient"
    ) as MockClient:
        step.run([source_rec], ctx)
        MockClient.assert_not_called()


def test_openalex_fallback_skipped_when_disabled(ctx):
    """openalex_references=False → OpenAlex never consulted."""
    source_rec = PaperRecord(
        paper_id="src", title="Source",
        external_ids={"DOI": "10.1/source"},
    )
    ctx.s2.add(make_paper("src", title="Source"))

    step = ExpandBackward(
        screener=_make_screener_accept_all(),
        openalex_references=False,
    )
    with patch(
        "citeclaw.clients.openalex.OpenAlexClient"
    ) as MockClient:
        step.run([source_rec], ctx)
        MockClient.assert_not_called()


def test_openalex_fallback_handles_client_failure(ctx):
    """OpenAlex raising mid-lookup must not crash the step."""
    source_rec = PaperRecord(
        paper_id="src", title="Source",
        external_ids={"DOI": "10.1/source"},
    )
    ctx.s2.add(make_paper("src", title="Source"))

    step = ExpandBackward(
        screener=_make_screener_accept_all(),
        openalex_references=True,
    )
    with patch(
        "citeclaw.clients.openalex.OpenAlexClient"
    ) as MockClient:
        MockClient.return_value.fetch_references_by_doi.side_effect = RuntimeError("boom")
        # Should complete without raising.
        result = step.run([source_rec], ctx)

    # No refs resolved because OpenAlex failed. Step still completes.
    assert result.signal == []
    assert result.stats.get("openalex_fallback_used", 0) == 0


def test_openalex_fallback_skips_unresolvable_dois(ctx):
    """OpenAlex returns DOIs but some fail to resolve via S2 — skip
    those silently; the ones that do resolve still make it through."""
    source_rec = PaperRecord(
        paper_id="src", title="Source",
        external_ids={"DOI": "10.1/source"},
    )
    ctx.s2.add(make_paper("src", title="Source"))
    # Only ref1 is resolvable; the other two DOIs aren't in the fake.
    ctx.s2.add(make_paper("DOI:10.1/ref1", title="Ref 1", year=2020))

    step = ExpandBackward(
        screener=_make_screener_accept_all(),
        openalex_references=True,
    )
    with patch(
        "citeclaw.clients.openalex.OpenAlexClient"
    ) as MockClient:
        MockClient.return_value.fetch_references_by_doi.return_value = [
            "10.1/ref1", "10.1/missing2", "10.1/missing3",
        ]
        result = step.run([source_rec], ctx)

    assert {p.paper_id for p in result.signal} == {"DOI:10.1/ref1"}
    assert result.stats.get("openalex_fallback_used", 0) == 1
