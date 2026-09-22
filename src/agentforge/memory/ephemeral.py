"""In-process memory, discarded when the process exits."""

from __future__ import annotations

import threading

from .base import MemoryRecord, MemoryStore


class EphemeralMemory(MemoryStore):
    """Keeps records in a list. Fast, unpersisted, and the default.

    Args:
        max_records: Optional cap per session. When exceeded, the oldest
            records are dropped, which bounds memory in long-running agents.
    """

    def __init__(self, *, max_records: int | None = None) -> None:
        self._records: list[MemoryRecord] = []
        self._max_records = max_records
        self._lock = threading.Lock()

    def add(self, record: MemoryRecord) -> MemoryRecord:
        with self._lock:
            self._records.append(record)
            if self._max_records is not None:
                same_session = [r for r in self._records if r.session_id == record.session_id]
                excess = len(same_session) - self._max_records
                for stale in same_session[:excess] if excess > 0 else []:
                    self._records.remove(stale)
        return record

    def recent(
        self, *, session_id: str = "default", limit: int = 50, kind: str | None = "message"
    ) -> list[MemoryRecord]:
        with self._lock:
            matches = [
                r
                for r in self._records
                if r.session_id == session_id and (kind is None or r.kind == kind)
            ]
        return matches[-limit:] if limit else matches

    def search(
        self, query: str, *, session_id: str | None = "default", limit: int = 10
    ) -> list[MemoryRecord]:
        """Rank records by how many of the query's terms they contain.

        This is a substring match, not a semantic one -- enough for recall over
        a single agent's history, and it needs no embedding model. Swap in a
        vector store by subclassing :class:`MemoryStore`.
        """
        terms = [t for t in query.lower().split() if t]
        if not terms:
            return []

        with self._lock:
            candidates = [
                r for r in self._records if session_id is None or r.session_id == session_id
            ]

        scored = []
        for record in candidates:
            haystack = record.content.lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, record.created_at, record))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [record for _, _, record in scored[:limit]]

    def clear(self, *, session_id: str | None = None) -> int:
        with self._lock:
            if session_id is None:
                count = len(self._records)
                self._records.clear()
                return count
            keep = [r for r in self._records if r.session_id != session_id]
            count = len(self._records) - len(keep)
            self._records = keep
            return count

    def sessions(self) -> list[str]:
        with self._lock:
            return sorted({r.session_id for r in self._records})

    def __len__(self) -> int:
        return len(self._records)
