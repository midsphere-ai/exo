"""SQLite-backed memory store with JSON metadata indexes and soft deletes."""

from __future__ import annotations

import json
from typing import Any

import aiosqlite  # pyright: ignore[reportMissingImports]

from orbiter.memory.base import (  # pyright: ignore[reportMissingImports]
    MemoryCategory,
    MemoryItem,
    MemoryMetadata,
    MemoryStatus,
)

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS memory_items (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'accepted',
    metadata    TEXT NOT NULL DEFAULT '{}',
    extra_json  TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    deleted     INTEGER NOT NULL DEFAULT 0,
    version     INTEGER NOT NULL DEFAULT 1
)"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_items(memory_type) WHERE deleted = 0",
    "CREATE INDEX IF NOT EXISTS idx_status ON memory_items(status) WHERE deleted = 0",
    "CREATE INDEX IF NOT EXISTS idx_created_at ON memory_items(created_at) WHERE deleted = 0",
]


class SQLiteMemoryStore:
    """SQLite-backed persistent memory store.

    Implements the MemoryStore protocol with JSON indexes for metadata
    fields, soft deletes, and a version field for optimistic concurrency.

    Use ``async with SQLiteMemoryStore(path) as store:`` or call
    ``await store.init()`` / ``await store.close()`` manually.
    """

    __slots__ = ("_db", "_initialized", "db_path")

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._initialized = False

    # -- lifecycle ------------------------------------------------------------

    async def init(self) -> None:
        """Open the database and create tables if needed."""
        if self._db is not None:
            return
        self._db = await aiosqlite.connect(self.db_path)
        db = self._db
        assert db is not None  # always set after connect()
        db.row_factory = aiosqlite.Row
        await db.execute(_CREATE_TABLE)
        for idx_sql in _CREATE_INDEXES:
            await db.execute(idx_sql)
        await db.commit()
        self._initialized = True

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            self._initialized = False

    async def __aenter__(self) -> SQLiteMemoryStore:
        await self.init()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def _ensure_init(self) -> aiosqlite.Connection:
        if self._db is None:
            msg = "Store not initialized — call init() or use 'async with'"
            raise RuntimeError(msg)
        return self._db

    # -- MemoryStore protocol -------------------------------------------------

    async def add(self, item: MemoryItem) -> None:
        """Persist a memory item (upsert — bumps version on conflict)."""
        db = self._ensure_init()
        extra = _extra_fields(item)
        await db.execute(
            """\
            INSERT INTO memory_items
                (id, content, memory_type, status, metadata, extra_json,
                 created_at, updated_at, deleted, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 1)
            ON CONFLICT(id) DO UPDATE SET
                content    = excluded.content,
                status     = excluded.status,
                metadata   = excluded.metadata,
                extra_json = excluded.extra_json,
                updated_at = excluded.updated_at,
                version    = version + 1,
                deleted    = 0
            """,
            (
                item.id,
                item.content,
                item.memory_type,
                item.status.value,
                item.metadata.model_dump_json(),
                json.dumps(extra),
                item.created_at,
                item.updated_at,
            ),
        )
        await db.commit()

    async def get(self, item_id: str) -> MemoryItem | None:
        """Retrieve a non-deleted memory item by ID."""
        db = self._ensure_init()
        cursor = await db.execute(
            "SELECT * FROM memory_items WHERE id = ? AND deleted = 0",
            (item_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_item(row)

    async def search(
        self,
        *,
        query: str = "",
        metadata: MemoryMetadata | None = None,
        memory_type: str | None = None,
        category: MemoryCategory | None = None,
        status: MemoryStatus | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """Search memory items with optional filters."""
        db = self._ensure_init()
        clauses: list[str] = ["deleted = 0"]
        params: list[Any] = []

        if memory_type:
            clauses.append("memory_type = ?")
            params.append(memory_type)
        if category is not None:
            clauses.append("json_extract(extra_json, '$.category') = ?")
            params.append(category.value)
        if status:
            clauses.append("status = ?")
            params.append(status.value)
        if query:
            clauses.append("content LIKE ?")
            params.append(f"%{query}%")
        if metadata:
            if metadata.user_id:
                clauses.append("json_extract(metadata, '$.user_id') = ?")
                params.append(metadata.user_id)
            if metadata.session_id:
                clauses.append("json_extract(metadata, '$.session_id') = ?")
                params.append(metadata.session_id)
            if metadata.task_id:
                clauses.append("json_extract(metadata, '$.task_id') = ?")
                params.append(metadata.task_id)
            if metadata.agent_id:
                clauses.append("json_extract(metadata, '$.agent_id') = ?")
                params.append(metadata.agent_id)

        where = " AND ".join(clauses)
        sql = f"SELECT * FROM memory_items WHERE {where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_item(r) for r in rows]

    async def clear(
        self,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> int:
        """Soft-delete memory items matching the filter. Returns count."""
        db = self._ensure_init()
        if metadata is None:
            cursor = await db.execute("UPDATE memory_items SET deleted = 1 WHERE deleted = 0")
            await db.commit()
            return cursor.rowcount

        clauses: list[str] = ["deleted = 0"]
        params: list[Any] = []
        if metadata.user_id:
            clauses.append("json_extract(metadata, '$.user_id') = ?")
            params.append(metadata.user_id)
        if metadata.session_id:
            clauses.append("json_extract(metadata, '$.session_id') = ?")
            params.append(metadata.session_id)
        if metadata.task_id:
            clauses.append("json_extract(metadata, '$.task_id') = ?")
            params.append(metadata.task_id)
        if metadata.agent_id:
            clauses.append("json_extract(metadata, '$.agent_id') = ?")
            params.append(metadata.agent_id)

        where = " AND ".join(clauses)
        cursor = await db.execute(
            f"UPDATE memory_items SET deleted = 1 WHERE {where}",
            params,
        )
        await db.commit()
        return cursor.rowcount

    # -- extras ---------------------------------------------------------------

    async def count(self, *, include_deleted: bool = False) -> int:
        """Return the number of stored items."""
        db = self._ensure_init()
        where = "" if include_deleted else " WHERE deleted = 0"
        cursor = await db.execute(f"SELECT COUNT(*) FROM memory_items{where}")
        row = await cursor.fetchone()
        return row[0] if row else 0  # type: ignore[index]

    def __repr__(self) -> str:
        return f"SQLiteMemoryStore(db_path={self.db_path!r})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extra_fields(item: MemoryItem) -> dict[str, Any]:
    """Extract subclass-specific fields into a JSON dict."""
    data: dict[str, Any] = {}
    if item.category is not None:
        data["category"] = item.category.value
    if hasattr(item, "tool_calls"):
        data["tool_calls"] = item.tool_calls  # type: ignore[attr-defined]
    if hasattr(item, "tool_call_id"):
        data["tool_call_id"] = item.tool_call_id  # type: ignore[attr-defined]
    if hasattr(item, "tool_name"):
        data["tool_name"] = item.tool_name  # type: ignore[attr-defined]
    if hasattr(item, "is_error"):
        data["is_error"] = item.is_error  # type: ignore[attr-defined]
    return data


def _row_to_item(row: Any) -> MemoryItem:
    """Reconstruct a MemoryItem from a database row."""
    meta_dict = json.loads(row["metadata"])
    extra = json.loads(row["extra_json"])

    kwargs: dict[str, Any] = {
        "id": row["id"],
        "content": row["content"],
        "memory_type": row["memory_type"],
        "status": MemoryStatus(row["status"]),
        "metadata": MemoryMetadata(**meta_dict),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }

    if "category" in extra:
        kwargs["category"] = MemoryCategory(extra["category"])

    # Dispatch to subtype based on memory_type
    from orbiter.memory.base import (  # pyright: ignore[reportMissingImports]
        AIMemory,
        HumanMemory,
        SystemMemory,
        ToolMemory,
    )

    memory_type = row["memory_type"]
    if memory_type == "system":
        return SystemMemory(**kwargs)
    if memory_type == "human":
        return HumanMemory(**kwargs)
    if memory_type == "ai":
        kwargs["tool_calls"] = extra.get("tool_calls", [])
        return AIMemory(**kwargs)
    if memory_type == "tool":
        kwargs["tool_call_id"] = extra.get("tool_call_id", "")
        kwargs["tool_name"] = extra.get("tool_name", "")
        kwargs["is_error"] = extra.get("is_error", False)
        return ToolMemory(**kwargs)
    return MemoryItem(**kwargs)
