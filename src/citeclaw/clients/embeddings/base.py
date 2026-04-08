"""EmbeddingClient Protocol — uniform .embed(texts) -> list[list[float]]."""

from __future__ import annotations

from typing import Protocol


class EmbeddingClient(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...
