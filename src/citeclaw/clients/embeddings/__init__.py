"""Embedding clients — Protocol + concrete providers + factory."""

from __future__ import annotations

from citeclaw.clients.embeddings.base import EmbeddingClient
from citeclaw.clients.embeddings.factory import build_embedder
from citeclaw.clients.embeddings.local import LocalEmbedder
from citeclaw.clients.embeddings.voyage import VoyageEmbedder

__all__ = [
    "EmbeddingClient",
    "VoyageEmbedder",
    "LocalEmbedder",
    "build_embedder",
]
