"""Long-term chat memory for Ollama conversation agents."""

from collections.abc import Callable
import logging
from pathlib import Path
import sqlite3

import httpx
import numpy as np
from numpy.typing import NDArray
import ollama

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import STORAGE_DIR

from ..const import MEMORY_MIN_SCORE, MEMORY_TOP_K
from .rag_database.embedding import OllamaEmbeddingEngine
from .rag_database.retriever import find_similar
from .rag_database.storage import SQLiteStorage
from .rag_utilities.generate_prompt import build_memory_prompt

_LOGGER = logging.getLogger(__name__)

MEMORY_FILE_PREFIX = "ollama_memory_"
RECALCULATE_BATCH_SIZE = 32
EMBEDDING_ERRORS = (
    ollama.RequestError,
    ollama.ResponseError,
    httpx.HTTPError,
    ConnectionError,
)


def memory_file(hass: HomeAssistant, subentry_id: str) -> str:
    """Return the memory database path of a conversation agent."""
    return hass.config.path(STORAGE_DIR, f"{MEMORY_FILE_PREFIX}{subentry_id}.db")


def remove_orphaned_memories(hass: HomeAssistant, subentry_ids: set[str]) -> None:
    """Delete the memory files of conversation agents that no longer exist."""
    for path in Path(hass.config.path(STORAGE_DIR)).glob(f"{MEMORY_FILE_PREFIX}*.db"):
        if path.stem.removeprefix(MEMORY_FILE_PREFIX) not in subentry_ids:
            _LOGGER.debug("Removing the memory of a deleted agent: %s", path.name)
            path.unlink(missing_ok=True)


class ChatMemory:
    """What one conversation agent remembers about each user."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: ollama.AsyncClient,
        subentry_id: str,
        model: str,
        limit: int,
        history_size: int = 0,
    ) -> None:
        """Initialize the memory.

        history_size is how many exchanges of the current conversation the AI
        still sees in its chat history, 0 meaning all of them.
        """
        self._hass = hass
        self._model = model
        self._limit = limit
        self._history_size = history_size
        self._storage = SQLiteStorage(memory_file(hass, subentry_id))
        self._embedding = OllamaEmbeddingEngine(client, model)

    async def async_setup(self) -> bool:
        """Create the database and apply the message limit, return success."""
        try:
            await self._hass.async_add_executor_job(self._storage.setup, self._limit)
        except sqlite3.Error:
            _LOGGER.exception("Could not open the memory database")
            return False
        return True

    async def async_build_prompt(
        self, user_id: str, conversation_id: str, text: str
    ) -> str | None:
        """Return the memory text to add to the AI's instructions."""
        try:
            items = await self._hass.async_add_executor_job(
                self._storage.fetch, user_id, self._model
            )
        except sqlite3.Error:
            _LOGGER.exception("Could not read the memory database")
            return None
        # Skip the exchanges the AI still sees in the current chat history.
        current = [item for item in items if item.conversation_id == conversation_id]
        visible = current if self._history_size < 1 else current[-self._history_size :]
        visible_ids = {item.id for item in visible}
        candidates = [item for item in items if item.id not in visible_ids]
        memories: list[str] = []
        if candidates and (query := await self._async_embed(text)) is not None:
            scored = find_similar(query, candidates, MEMORY_TOP_K, MEMORY_MIN_SCORE)
            _LOGGER.debug(
                "Memory scores for %r: %s",
                text,
                [(round(score, 3), item.content[:60]) for score, item in scored],
            )
            memories = [item.content for _, item in scored]
        return build_memory_prompt(memories, len(items), self._limit)

    async def async_remember(
        self, user_id: str, conversation_id: str, content: str
    ) -> None:
        """Save an exchange, forgetting the user's oldest beyond the limit."""
        if (vector := await self._async_embed(content)) is None:
            return
        try:
            await self._hass.async_add_executor_job(
                self._storage.add,
                user_id,
                conversation_id,
                content,
                self._model,
                vector,
                self._limit,
            )
        except sqlite3.Error:
            _LOGGER.exception("Could not save to the memory database")

    async def async_count_stale(self) -> int:
        """Return how many memories were embedded with another model."""
        try:
            return await self._hass.async_add_executor_job(
                self._storage.count_stale, self._model
            )
        except sqlite3.Error:
            _LOGGER.exception("Could not read the memory database")
            return 0

    async def async_recalculate(
        self, progress: Callable[[float], None] | None = None
    ) -> int:
        """Re-embed memories made with another model, return how many."""
        stale = await self._hass.async_add_executor_job(
            self._storage.fetch_stale, self._model
        )
        for start in range(0, len(stale), RECALCULATE_BATCH_SIZE):
            batch = stale[start : start + RECALCULATE_BATCH_SIZE]
            vectors = await self._embedding.async_embed(
                [content for _, content in batch]
            )
            await self._hass.async_add_executor_job(
                self._storage.update_embeddings,
                self._model,
                [
                    (row_id, vector)
                    for (row_id, _), vector in zip(batch, vectors, strict=True)
                ],
            )
            if progress:
                progress((start + len(batch)) / len(stale))
        return len(stale)

    async def async_update_leftovers(self) -> None:
        """Re-embed memories still made with an older model, logging failures."""
        try:
            if updated := await self.async_recalculate():
                _LOGGER.info("Updated %s memories to model %s", updated, self._model)
        except (*EMBEDDING_ERRORS, sqlite3.Error) as err:
            _LOGGER.warning("Could not update memories to %s: %s", self._model, err)

    async def _async_embed(self, text: str) -> NDArray[np.float32] | None:
        """Embed one text, returning None if the embedding model fails."""
        try:
            (vector,) = await self._embedding.async_embed([text])
        except EMBEDDING_ERRORS as err:
            _LOGGER.warning("Could not embed with %s: %s", self._model, err)
            return None
        return vector
