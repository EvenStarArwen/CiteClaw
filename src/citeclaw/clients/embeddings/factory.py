"""``build_embedder`` — turn a YAML spec into a concrete :class:`EmbeddingClient`.

Two shorthand forms are accepted, both intended for inline use under
``SimilarityFilter`` ``measures:`` blocks:

    embedder: "voyage:voyage-3"
    embedder: {type: voyage, model: voyage-3, api_key: ...}

The ``:`` in the string form separates the registered ``kind`` (the
left-hand side, used to pick a class) from the model identifier (the
right-hand side, passed through as ``model=``). When omitted, the
registry's per-kind default model is used.
"""

from __future__ import annotations

from typing import Any

from citeclaw.clients.embeddings.local import LocalEmbedder
from citeclaw.clients.embeddings.voyage import VoyageEmbedder

# kind → (concrete class, default model name when the spec omits one).
# Adding a new embedder is one entry here plus a class with the same
# ``__init__(*, model=..., **kwargs)`` shape as the existing two.
_REGISTRY: dict[str, tuple[type, str]] = {
    "voyage": (VoyageEmbedder, "voyage-3"),
    "local": (LocalEmbedder, "BAAI/bge-base-en-v1.5"),
}


def build_embedder(spec: Any):
    """Build an :class:`EmbeddingClient` from a string or dict spec.

    See the module docstring for the accepted shorthand forms. Raises
    :class:`ValueError` on an unknown ``kind`` / ``type`` or on any
    spec that is neither a string nor a dict. The error wording differs
    slightly between the two paths (``"kind"`` vs ``"type"``) so the
    message points back at the YAML field the user actually wrote.
    """
    if isinstance(spec, str):
        kind, _, model = spec.partition(":")
        if kind not in _REGISTRY:
            raise ValueError(f"Unknown embedder kind {kind!r}")
        cls, default_model = _REGISTRY[kind]
        return cls(model=model or default_model)
    if isinstance(spec, dict):
        kind = spec.get("type")
        if kind not in _REGISTRY:
            raise ValueError(f"Unknown embedder type {kind!r}")
        cls, _ = _REGISTRY[kind]
        kwargs = {k: v for k, v in spec.items() if k != "type"}
        return cls(**kwargs)
    raise ValueError(f"Bad embedder spec: {spec!r}")
