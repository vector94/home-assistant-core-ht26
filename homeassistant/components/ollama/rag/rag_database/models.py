"""Data objects for the Ollama chat memory."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class MemoryItem:
    """A remembered exchange from an earlier conversation."""

    id: int
    user_id: str
    conversation_id: str
    content: str
    embedding: NDArray[np.float32]
