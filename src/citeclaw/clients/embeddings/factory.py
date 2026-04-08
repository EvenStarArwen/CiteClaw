"""build_embedder — string-or-dict spec -> EmbeddingClient."""

from __future__ import annotations

from citeclaw.clients.embeddings.local import LocalEmbedder
from citeclaw.clients.embeddings.voyage import VoyageEmbedder


def build_embedder(spec):
    """Build an EmbeddingClient from a spec.

    Accepts either:
      - a string like ``'voyage:voyage-3'`` or ``'local:bge-base'``;
      - a dict like ``{'type': 'voyage', 'model': 'voyage-3', 'api_key': '...'}``.
    """
    if isinstance(spec, str):
        if ":" in spec:
            kind, model = spec.split(":", 1)
        else:
            kind, model = spec, None
        if kind == "voyage":
            return VoyageEmbedder(model=model or "voyage-3")
        if kind == "local":
            return LocalEmbedder(model=model or "BAAI/bge-base-en-v1.5")
        raise ValueError(f"Unknown embedder kind {kind!r}")
    if isinstance(spec, dict):
        kind = spec.get("type")
        kwargs = {k: v for k, v in spec.items() if k != "type"}
        if kind == "voyage":
            return VoyageEmbedder(**kwargs)
        if kind == "local":
            return LocalEmbedder(**kwargs)
        raise ValueError(f"Unknown embedder type {kind!r}")
    raise ValueError(f"Bad embedder spec: {spec!r}")
