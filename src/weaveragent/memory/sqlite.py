"""Persistent memory backed by SQLite."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from ..errors import MemoryError_
from .base import MemoryRecord, MemoryStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    content     TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_created ON records (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_kind ON records (kind);
"""

# FTS5 gives real relevance ranking, but it is a compile-time option and not
# present in every Python build, so it is created opportunistically and the
# store falls back to LIKE matching when it is missing.
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS records_fts
    USING fts5(content, id UNINDEXED, tokenize='porter unicode61');
"""


class SQLiteMemory(MemoryStore):
    """Durable memory in a single SQLite file.

    Survives process restarts, which is what separates it from
    :class:`~weaveragent.memory.ephemeral.EphemeralMemory`. Pass
    ``path=":memory:"`` for a throwaway database with the same semantics.

    Args:
        path: Database file. Parent directories are created as needed.
    """

    def __init__(self, path: str | Path = "weaveragent_memory.sqlite3") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self.path = str(Path(self.path).expanduser())

        # check_same_thread=False plus an explicit lock: agents may be driven
        # from a worker thread, and serializing writes here is simpler than
        # threading a connection through every call.
        self._lock = threading.Lock()
        try:
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
        except sqlite3.Error as exc:
            raise MemoryError_(f"cannot open memory database at {self.path!r}: {exc}") from exc

        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            try:
                self._conn.executescript(_FTS_SCHEMA)
                self._fts = True
            except sqlite3.OperationalError:
                self._fts = False
            self._conn.commit()

    # -- writes ---------------------------------------------------------------

    def add(self, record: MemoryRecord) -> MemoryRecord:
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO records "
                    "(id, session_id, kind, content, metadata, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        record.id,
                        record.session_id,
                        record.kind,
                        record.content,
                        json.dumps(record.metadata, default=str),
                        record.created_at,
                    ),
                )
                if self._fts and record.content:
                    self._conn.execute(
                        "INSERT INTO records_fts (content, id) VALUES (?, ?)",
                        (record.content, record.id),
                    )
                self._conn.commit()
            except sqlite3.Error as exc:
                self._conn.rollback()
                raise MemoryError_(f"failed to store record: {exc}") from exc
        return record

    # -- reads ----------------------------------------------------------------

    def recent(
        self, *, session_id: str = "default", limit: int = 50, kind: str | None = "message"
    ) -> list[MemoryRecord]:
        sql = "SELECT * FROM records WHERE session_id = ?"
        params: list[Any] = [session_id]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        # Newest-first with LIMIT, then reversed: the alternative would scan
        # the whole session to take the tail.
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_to_record(row) for row in reversed(rows)]

    def search(
        self, query: str, *, session_id: str | None = "default", limit: int = 10
    ) -> list[MemoryRecord]:
        if not query.strip():
            return []
        if self._fts:
            try:
                return self._search_fts(query, session_id, limit)
            except sqlite3.OperationalError:
                # FTS5 rejects some unescaped query syntax; LIKE always works.
                pass
        return self._search_like(query, session_id, limit)

    def _search_fts(self, query: str, session_id: str | None, limit: int) -> list[MemoryRecord]:
        # Quote each term so operators in user text are treated as literals.
        match = " OR ".join(f'"{term}"' for term in query.split() if term)
        if not match:
            return []

        sql = (
            "SELECT r.* FROM records_fts f JOIN records r ON r.id = f.id WHERE records_fts MATCH ?"
        )
        params: list[Any] = [match]
        if session_id is not None:
            sql += " AND r.session_id = ?"
            params.append(session_id)
        sql += " ORDER BY bm25(records_fts) LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_to_record(row) for row in rows]

    def _search_like(self, query: str, session_id: str | None, limit: int) -> list[MemoryRecord]:
        terms = [t for t in query.lower().split() if t]
        if not terms:
            return []

        clauses = " OR ".join(["LOWER(content) LIKE ?"] * len(terms))
        sql = f"SELECT * FROM records WHERE ({clauses})"
        params: list[Any] = [f"%{t}%" for t in terms]
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_to_record(row) for row in rows]

    # -- maintenance ----------------------------------------------------------

    def clear(self, *, session_id: str | None = None) -> int:
        with self._lock:
            if session_id is None:
                count = self._conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
                self._conn.execute("DELETE FROM records")
                if self._fts:
                    self._conn.execute("DELETE FROM records_fts")
            else:
                count = self._conn.execute(
                    "SELECT COUNT(*) FROM records WHERE session_id = ?", (session_id,)
                ).fetchone()[0]
                if self._fts:
                    self._conn.execute(
                        "DELETE FROM records_fts WHERE id IN "
                        "(SELECT id FROM records WHERE session_id = ?)",
                        (session_id,),
                    )
                self._conn.execute("DELETE FROM records WHERE session_id = ?", (session_id,))
            self._conn.commit()
        return int(count)

    def sessions(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT session_id FROM records ORDER BY session_id"
            ).fetchall()
        return [row["session_id"] for row in rows]

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    def __enter__(self) -> SQLiteMemory:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM records").fetchone()[0])


def _to_record(row: sqlite3.Row) -> MemoryRecord:
    try:
        metadata = json.loads(row["metadata"])
    except (json.JSONDecodeError, TypeError):
        metadata = {}
    return MemoryRecord(
        id=row["id"],
        session_id=row["session_id"],
        kind=row["kind"],
        content=row["content"],
        metadata=metadata if isinstance(metadata, dict) else {},
        created_at=row["created_at"],
    )
