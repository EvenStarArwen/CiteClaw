"""Tests for :mod:`citeclaw.clients.embeddings` — factory + placeholder classes."""

from __future__ import annotations

import pytest

from citeclaw.clients.embeddings import LocalEmbedder, VoyageEmbedder, build_embedder


class TestBuildEmbedder:
    def test_string_voyage(self):
        e = build_embedder("voyage:voyage-3")
        assert isinstance(e, VoyageEmbedder)
        assert e.model == "voyage-3"
        assert e.name == "voyage:voyage-3"

    def test_string_local(self):
        e = build_embedder("local:bge-small")
        assert isinstance(e, LocalEmbedder)
        assert e.model == "bge-small"

    def test_string_voyage_without_colon_uses_default_model(self):
        e = build_embedder("voyage")
        assert isinstance(e, VoyageEmbedder)
        assert e.model == "voyage-3"

    def test_string_unknown_kind(self):
        with pytest.raises(ValueError):
            build_embedder("unknown:foo")

    def test_dict_voyage(self):
        e = build_embedder({"type": "voyage", "model": "voyage-2"})
        assert isinstance(e, VoyageEmbedder)
        assert e.model == "voyage-2"

    def test_dict_local(self):
        e = build_embedder({"type": "local", "model": "bge-base"})
        assert isinstance(e, LocalEmbedder)

    def test_dict_unknown_type(self):
        with pytest.raises(ValueError):
            build_embedder({"type": "nonesuch"})

    def test_bad_spec(self):
        with pytest.raises(ValueError):
            build_embedder(42)


class TestPlaceholders:
    def test_voyage_embed_raises(self):
        e = VoyageEmbedder()
        with pytest.raises(NotImplementedError, match="Voyage"):
            e.embed(["hello"])

    def test_local_embed_raises(self):
        e = LocalEmbedder()
        with pytest.raises(NotImplementedError, match="Local"):
            e.embed(["hello"])
