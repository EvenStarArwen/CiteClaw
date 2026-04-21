"""Local sentence-transformers embedder — placeholder until the SDK is wired up.

Constructed by :func:`citeclaw.clients.embeddings.factory.build_embedder`
when the YAML spec resolves to ``kind="local"``. The real
``sentence_transformers`` call is intentionally not implemented yet —
calling :meth:`LocalEmbedder.embed` raises :class:`NotImplementedError`
with an actionable message rather than silently returning empty vectors.
The default model is HuggingFace's ``BAAI/bge-base-en-v1.5``; future
wiring should download it lazily on first call.
"""

from __future__ import annotations


class LocalEmbedder:
    """Placeholder :class:`EmbeddingClient` for local sentence-transformers models."""

    #: Default model used when the YAML spec omits one (kept in sync
    #: with the factory's :data:`_REGISTRY` table — same string in both
    #: places until a future round consolidates them).
    DEFAULT_MODEL: str = "BAAI/bge-base-en-v1.5"

    def __init__(self, *, model: str = DEFAULT_MODEL) -> None:
        self.name = f"local:{model}"
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Raise :class:`NotImplementedError` — placeholder, not wired."""
        raise NotImplementedError(
            f"LocalEmbedder({self.model!r}).embed() is a placeholder. "
            "Install sentence-transformers and wire the call here."
        )
