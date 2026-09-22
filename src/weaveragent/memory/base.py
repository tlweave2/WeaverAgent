"""The memory store interface."""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..types import Message


@dataclass(slots=True)
class MemoryRecord:
    """A single remembered item.

    ``kind`` separates conversation turns (``"message"``) from durable notes an
    agent writes for itself (``"fact"``), so recalling history does not drag in
    every stored fact and vice versa.
    """

    content: str
    kind: str = "message"
    session_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_message(self) -> Message | None:
        """Rebuild the original :class:`Message` if this record holds one."""
        payload = self.metadata.get("message")
        return Message.from_dict(payload) if isinstance(payload, dict) else None


class MemoryStore(ABC):
    """Storage for an agent's conversation history and durable facts.

    Implementations must be safe to use across agent runs;
    :class:`~weaveragent.memory.sqlite.SQLiteMemory` additionally survives
    process restarts.
    """

    @abstractmethod
    def add(self, record: MemoryRecord) -> MemoryRecord:
        """Persist one record and return it."""

    @abstractmethod
    def recent(
        self, *, session_id: str = "default", limit: int = 50, kind: str | None = "message"
    ) -> list[MemoryRecord]:
        """Return up to ``limit`` records, oldest first."""

    @abstractmethod
    def search(
        self, query: str, *, session_id: str | None = "default", limit: int = 10
    ) -> list[MemoryRecord]:
        """Return records matching ``query``, most relevant first."""

    @abstractmethod
    def clear(self, *, session_id: str | None = None) -> int:
        """Delete records for one session, or all sessions when ``session_id`` is None.

        Returns the number of records deleted.
        """

    @abstractmethod
    def sessions(self) -> list[str]:
        """List the session ids that hold at least one record."""

    # -- convenience built on the primitives above ---------------------------

    def add_message(self, message: Message, *, session_id: str = "default") -> MemoryRecord:
        """Store a conversation turn, keeping its tool calls and results intact."""
        return self.add(
            MemoryRecord(
                content=message.content,
                kind="message",
                session_id=session_id,
                metadata={"message": message.to_dict(), "role": message.role.value},
            )
        )

    def add_fact(
        self, content: str, *, session_id: str = "default", **metadata: Any
    ) -> MemoryRecord:
        """Store a durable note, outside the conversation transcript."""
        return self.add(
            MemoryRecord(content=content, kind="fact", session_id=session_id, metadata=metadata)
        )

    def history(self, *, session_id: str = "default", limit: int = 50) -> list[Message]:
        """Rebuild the conversation as :class:`Message` objects, oldest first."""
        records = self.recent(session_id=session_id, limit=limit, kind="message")
        return [m for m in (r.to_message() for r in records) if m is not None]

    def close(self) -> None:  # noqa: B027 - optional hook, not every backend has resources
        """Release any backing resources. Safe to call more than once."""
