"""SQLite storage for the Ollama chat memory.

Every method touches the disk, so call them from an executor job.
"""

from collections.abc import Iterator
from contextlib import closing, contextmanager
import os
import sqlite3

import numpy as np
from numpy.typing import NDArray

from .models import MemoryItem

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    content TEXT NOT NULL,
    model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS memories_by_user ON memories (user_id, id);
"""


def _to_blob(vector: NDArray[np.float32]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


class SQLiteStorage:
    """The saved memories of one conversation agent."""

    def __init__(self, db_path: str) -> None:
        """Initialize the storage."""
        self.db_path = db_path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection that commits on success and always closes."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def setup(self, limit: int) -> None:
        """Create the database if needed and apply the message limit."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        if self.trim(limit):
            self.vacuum()

    def add(
        self,
        user_id: str,
        conversation_id: str,
        content: str,
        model: str,
        embedding: NDArray[np.float32],
        limit: int,
    ) -> None:
        """Save a memory and forget the user's oldest ones beyond the limit."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memories"
                " (user_id, conversation_id, content, model, embedding)"
                " VALUES (?, ?, ?, ?, ?)",
                (user_id, conversation_id, content, model, _to_blob(embedding)),
            )
            conn.execute(
                "DELETE FROM memories WHERE user_id = ? AND id NOT IN"
                " (SELECT id FROM memories WHERE user_id = ? ORDER BY id DESC LIMIT ?)",
                (user_id, user_id, limit),
            )

    def fetch(self, user_id: str, model: str) -> list[MemoryItem]:
        """Return a user's memories that were embedded with the given model."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, user_id, conversation_id, content, embedding"
                " FROM memories WHERE user_id = ? AND model = ? ORDER BY id",
                (user_id, model),
            ).fetchall()
        return [
            MemoryItem(
                row_id,
                row_user_id,
                conversation_id,
                content,
                np.frombuffer(embedding, dtype=np.float32),
            )
            for row_id, row_user_id, conversation_id, content, embedding in rows
        ]

    def trim(self, limit: int) -> int:
        """Forget every user's oldest memories beyond the limit, return how many."""
        with self._connect() as conn:
            return conn.execute(
                "DELETE FROM memories WHERE id IN (SELECT id FROM (SELECT id,"
                " ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY id DESC) AS position"
                " FROM memories) WHERE position > ?)",
                (limit,),
            ).rowcount

    def count_stale(self, model: str) -> int:
        """Return how many memories were embedded with another model."""
        if not os.path.exists(self.db_path):
            return 0
        with self._connect() as conn:
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE model != ?", (model,)
            ).fetchone()
        return int(count)

    def fetch_stale(self, model: str) -> list[tuple[int, str]]:
        """Return the id and text of memories embedded with another model."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT id, content FROM memories WHERE model != ? ORDER BY id",
                (model,),
            ).fetchall()

    def update_embeddings(
        self, model: str, embeddings: list[tuple[int, NDArray[np.float32]]]
    ) -> None:
        """Store new embeddings made with the given model."""
        with self._connect() as conn:
            conn.executemany(
                "UPDATE memories SET model = ?, embedding = ? WHERE id = ?",
                [(model, _to_blob(vector), row_id) for row_id, vector in embeddings],
            )

    def vacuum(self) -> None:
        """Shrink the file after many memories were deleted."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("VACUUM")
