"""LocalEmbedder — placeholder; raises NotImplementedError on .embed()."""

from __future__ import annotations


class LocalEmbedder:
    def __init__(self, *, model: str = "BAAI/bge-base-en-v1.5") -> None:
        self.name = f"local:{model}"
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            f"LocalEmbedder({self.model!r}).embed() is a placeholder. "
            "Install sentence-transformers and wire the call here."
        )
