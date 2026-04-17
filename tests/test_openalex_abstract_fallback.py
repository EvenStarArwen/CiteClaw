"""Integration test: OpenAlex abstract fallback inside enrich_with_abstracts.

When S2 returns no abstract AND the paper has a DOI in ``external_ids``,
:meth:`SemanticScholarClient.enrich_with_abstracts` should fall through
to OpenAlex. A successful OpenAlex lookup populates
``rec.abstract`` and write-throughs to the metadata cache so a re-run
doesn't re-query.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from citeclaw.cache import Cache
from citeclaw.clients.s2 import SemanticScholarClient
from citeclaw.config import BudgetTracker, Settings
from citeclaw.models import PaperRecord


@pytest.fixture
def s2_client(tmp_path: Path) -> SemanticScholarClient:
    cfg = Settings(
        screening_model="stub",
        s2_api_key="dummy",
        data_dir=tmp_path,
    )
    cache = Cache(tmp_path / "cache.db")
    budget = BudgetTracker()
    client = SemanticScholarClient(cfg, cache, budget)
    yield client
    client.close()
    cache.close()


def test_openalex_called_when_s2_returns_no_abstract(s2_client, monkeypatch):
    """S2 miss + DOI present → OpenAlex is consulted; abstract fills in."""
    rec = PaperRecord(
        paper_id="p1", title="x",
        external_ids={"DOI": "10.1038/nature12345"},
    )

    # Stub the S2 batch fetch to return no abstract.
    monkeypatch.setattr(
        s2_client, "_batch_fetch",
        lambda ids, fields: [{"paperId": "p1", "title": "x"}],
    )

    with patch(
        "citeclaw.clients.openalex.OpenAlexClient"
    ) as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.fetch_abstract_by_doi.return_value = (
            "Reconstructed abstract from OpenAlex"
        )
        s2_client.enrich_with_abstracts([rec])

    assert rec.abstract == "Reconstructed abstract from OpenAlex"
    mock_instance.fetch_abstract_by_doi.assert_called_once_with(
        "10.1038/nature12345"
    )


def test_openalex_not_called_when_no_doi(s2_client, monkeypatch):
    """No DOI on the record → skip OpenAlex entirely."""
    rec = PaperRecord(paper_id="p1", title="x", external_ids={})

    monkeypatch.setattr(
        s2_client, "_batch_fetch",
        lambda ids, fields: [{"paperId": "p1", "title": "x"}],
    )

    with patch(
        "citeclaw.clients.openalex.OpenAlexClient"
    ) as MockClient:
        s2_client.enrich_with_abstracts([rec])
        MockClient.assert_not_called()

    assert rec.abstract is None


def test_openalex_write_through_to_cache(s2_client, monkeypatch):
    """A successful OpenAlex fetch writes the abstract into the metadata
    cache so a re-run doesn't re-query."""
    rec = PaperRecord(
        paper_id="p1",
        external_ids={"DOI": "10.1038/abc"},
    )
    monkeypatch.setattr(
        s2_client, "_batch_fetch",
        lambda ids, fields: [{"paperId": "p1", "title": "x"}],
    )

    with patch(
        "citeclaw.clients.openalex.OpenAlexClient"
    ) as MockClient:
        MockClient.return_value.fetch_abstract_by_doi.return_value = "Oabs"
        s2_client.enrich_with_abstracts([rec])

    cached = s2_client._cache.get_metadata("p1")
    assert cached is not None
    assert cached.get("abstract") == "Oabs"


def test_openalex_failure_swallowed(s2_client, monkeypatch):
    """If OpenAlex raises, enrich_with_abstracts must NOT crash — the
    record just stays un-enriched."""
    rec = PaperRecord(
        paper_id="p1",
        external_ids={"DOI": "10.1038/abc"},
    )
    monkeypatch.setattr(
        s2_client, "_batch_fetch",
        lambda ids, fields: [{"paperId": "p1", "title": "x"}],
    )

    with patch(
        "citeclaw.clients.openalex.OpenAlexClient"
    ) as MockClient:
        MockClient.return_value.fetch_abstract_by_doi.side_effect = RuntimeError("oops")
        s2_client.enrich_with_abstracts([rec])  # must not raise

    assert rec.abstract is None


def test_openalex_not_called_when_s2_returned_abstract(s2_client, monkeypatch):
    """S2 returned an abstract → OpenAlex is never reached."""
    rec = PaperRecord(
        paper_id="p1",
        external_ids={"DOI": "10.1038/abc"},
    )
    monkeypatch.setattr(
        s2_client, "_batch_fetch",
        lambda ids, fields: [{"paperId": "p1", "title": "x", "abstract": "S2 gave one"}],
    )

    with patch(
        "citeclaw.clients.openalex.OpenAlexClient"
    ) as MockClient:
        s2_client.enrich_with_abstracts([rec])
        MockClient.assert_not_called()

    assert rec.abstract == "S2 gave one"
