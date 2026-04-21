"""EmbeddingClient Protocol — uniform ``.embed(texts) -> list[list[float]]``.

Concrete implementations live in sibling modules
(:mod:`citeclaw.clients.embeddings.voyage`,
:mod:`citeclaw.clients.embeddings.local`) and are constructed via
:func:`citeclaw.clients.embeddings.factory.build_embedder`. The two
shipped clients are placeholders that raise ``NotImplementedError`` on
``.embed()`` — wired into the Protocol so the
:class:`citeclaw.filters.measures.semantic_sim.SemanticSim` measure can
already route through them once an SDK call is filled in. The default
``embedder: s2`` path on ``SemanticSim`` skips this Protocol entirely.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    """Provider-agnostic embedding model.

    Implementations must expose a ``name`` attribute (used for cache
    keying and dashboard display) and an :meth:`embed` method that
    converts a batch of texts into vectors of a fixed per-instance
    dimensionality.
    """

    name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Returns one vector per input text in the same order; every vector
        in a single call has the same dimensionality (provider- and
        model-specific). An empty input list returns an empty output
        list — callers should not rely on a side-effectful warm-up call.
        Implementations may raise the underlying SDK's exception on
        network / quota failures; the caller is responsible for retry
        policy.
        """
        ...
