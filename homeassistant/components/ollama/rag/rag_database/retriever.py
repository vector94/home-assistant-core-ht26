"""Find the memories that best match a new message."""

import numpy as np
from numpy.typing import NDArray

from .models import MemoryItem


def find_similar(
    query: NDArray[np.float32],
    items: list[MemoryItem],
    top_k: int,
    min_score: float,
) -> list[tuple[float, MemoryItem]]:
    """Return up to top_k items scoring at least min_score, best first."""
    # Embeddings of a different size come from another model and can't be compared.
    usable = [item for item in items if item.embedding.shape == query.shape]
    if not usable:
        return []
    matrix = np.stack([item.embedding for item in usable])
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    scores = np.divide(
        matrix @ query,
        norms,
        out=np.zeros(len(usable), dtype=np.float32),
        where=norms > 0,
    )
    best = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), usable[i]) for i in best if scores[i] >= min_score]
