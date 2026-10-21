"""Embeddings from the Ollama server."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from ollama import AsyncClient


class OllamaEmbeddingEngine:
    """Turn text into embeddings with an Ollama embedding model."""

    def __init__(self, client: AsyncClient, model: str) -> None:
        """Initialize the engine."""
        self.client = client
        self.model = model

    async def async_embed(self, texts: Sequence[str]) -> list[NDArray[np.float32]]:
        """Return one embedding per text."""
        response = await self.client.embed(model=self.model, input=list(texts))
        return [np.asarray(vector, dtype=np.float32) for vector in response.embeddings]
