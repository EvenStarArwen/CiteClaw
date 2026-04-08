"""VoyageEmbedder — placeholder; raises NotImplementedError on .embed()."""

from __future__ import annotations


class VoyageEmbedder:
    def __init__(self, *, model: str = "voyage-3", api_key: str | None = None) -> None:
        self.name = f"voyage:{model}"
        self.model = model
        self._api_key = api_key

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            f"VoyageEmbedder({self.model!r}).embed() is a placeholder. "
            "Install the Voyage SDK and wire the call here."
        )
