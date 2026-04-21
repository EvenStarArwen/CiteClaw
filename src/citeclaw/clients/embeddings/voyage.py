"""Voyage AI embedder — placeholder until the Voyage SDK is wired up.

Constructed by :func:`citeclaw.clients.embeddings.factory.build_embedder`
when the YAML spec resolves to ``kind="voyage"``. The real
``voyageai`` SDK call is intentionally not implemented yet — calling
:meth:`VoyageEmbedder.embed` raises :class:`NotImplementedError` with
an actionable message rather than silently returning empty vectors.
``api_key`` is accepted (for forward compatibility) but stored only;
the future implementation will read it when constructing the SDK
client.
"""

from __future__ import annotations


class VoyageEmbedder:
    """Placeholder :class:`EmbeddingClient` for Voyage AI's hosted models."""

    #: Default model used when the YAML spec omits one (kept in sync
    #: with the factory's :data:`_REGISTRY` table — same string in both
    #: places until a future round consolidates them).
    DEFAULT_MODEL: str = "voyage-3"

    def __init__(self, *, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        self.name = f"voyage:{model}"
        self.model = model
        self._api_key = api_key

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Raise :class:`NotImplementedError` — placeholder, not wired."""
        raise NotImplementedError(
            f"VoyageEmbedder({self.model!r}).embed() is a placeholder. "
            "Install the Voyage SDK and wire the call here."
        )
