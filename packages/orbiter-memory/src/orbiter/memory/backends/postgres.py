"""Postgres-backed memory store using asyncpg."""

from __future__ import annotations

import json
from typing import Any

import asyncpg  # pyright: ignore[reportMissingImports]

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
    metadata    JSONB NOT NULL DEFAULT '{}',
    extra_json  JSONB NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    deleted     SMALLINT NOT NULL DEFAULT 0,
    version     INTEGER NOT NULL DEFAULT 1
)"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_items(memory_type) WHERE deleted = 0",
    "CREATE INDEX IF NOT EXISTS idx_status ON memory_items(status) WHERE deleted = 0",
    "CREATE INDEX IF NOT EXISTS idx_created_at ON memory_items(created_at) WHERE deleted = 0",
    "CREATE INDEX IF NOT EXISTS idx_metadata_user ON memory_items((metadata->>'user_id')) WHERE deleted = 0",
    "CREATE INDEX IF NOT EXISTS idx_metadata_session ON memory_items((metadata->>'session_id')) WHERE deleted = 0",
]


class PostgresMemoryStore:
    """Postgres-backed persistent memory store.

    Implements the MemoryStore protocol with JSONB indexes for metadata
    fields, soft deletes, and a version field for optimistic concurrency.

    Use ``async with PostgresMemoryStore(dsn) as store:`` or call
    ``await store.init()`` / ``await store.close()`` manually.
    """

    __slots__ = ("_initialized", "_pool", "dsn")

    def __init__(self, dsn: str = "postgresql://localhost/orbiter") -> None:
        self.dsn = dsn
        self._pool: asyncpg.Pool | None = None  # type: ignore[type-arg]
        self._initialized = False

    # -- lifecycle ------------------------------------------------------------

    async def init(self) -> None:
        """Open the connection pool and create tables if needed."""
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(self.dsn)
        pool = self._pool
        assert pool is not None
        async with pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE)
            for idx_sql in _CREATE_INDEXES:
                await conn.execute(idx_sql)
        self._initialized = True

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._initialized = False

    async def __aenter__(self) -> PostgresMemoryStore:
        await self.init()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def _ensure_init(self) -> asyncpg.Pool:  # type: ignore[type-arg]
        if self._pool is None:
            msg = "Store not initialized — call init() or use 'async with'"
            raise RuntimeError(msg)
        return self._pool

    # -- MemoryStore protocol -------------------------------------------------

    async def add(self, item: MemoryItem) -> None:
        """Persist a memory item (upsert — bumps version on conflict)."""
        pool = self._ensure_init()
        extra = _extra_fields(item)
        async with pool.acquire() as conn:
            await conn.execute(
                """\
                INSERT INTO memory_items
                    (id, content, memory_type, status, metadata, extra_json,
                     created_at, updated_at, deleted, version)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8, 0, 1)
                ON CONFLICT (id) DO UPDATE SET
                    content    = EXCLUDED.content,
                    status     = EXCLUDED.status,
                    metadata   = EXCLUDED.metadata,
                    extra_json = EXCLUDED.extra_json,
                    updated_at = EXCLUDED.updated_at,
                    version    = memory_items.version + 1,
                    deleted    = 0
                """,
                item.id,
                item.content,
                item.memory_type,
                item.status.value,
                item.metadata.model_dump_json(),
                json.dumps(extra),
                item.created_at,
                item.updated_at,
            )

    async def get(self, item_id: str) -> MemoryItem | None:
        """Retrieve a non-deleted memory item by ID."""
        pool = self._ensure_init()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM memory_items WHERE id = $1 AND deleted = 0",
                item_id,
            )
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
        pool = self._ensure_init()
        clauses: list[str] = ["deleted = 0"]
        params: list[Any] = []
        idx = 1

        if memory_type:
            clauses.append(f"memory_type = ${idx}")
            params.append(memory_type)
            idx += 1
        if category is not None:
            clauses.append(f"extra_json->>'category' = ${idx}")
            params.append(category.value)
            idx += 1
        if status:
            clauses.append(f"status = ${idx}")
            params.append(status.value)
            idx += 1
        if query:
            clauses.append(f"content ILIKE ${idx}")
            params.append(f"%{query}%")
            idx += 1
        if metadata:
            if metadata.user_id:
                clauses.append(f"metadata->>'user_id' = ${idx}")
                params.append(metadata.user_id)
                idx += 1
            if metadata.session_id:
                clauses.append(f"metadata->>'session_id' = ${idx}")
                params.append(metadata.session_id)
                idx += 1
            if metadata.task_id:
                clauses.append(f"metadata->>'task_id' = ${idx}")
                params.append(metadata.task_id)
                idx += 1
            if metadata.agent_id:
                clauses.append(f"metadata->>'agent_id' = ${idx}")
                params.append(metadata.agent_id)
                idx += 1

        where = " AND ".join(clauses)
        sql = f"SELECT * FROM memory_items WHERE {where} ORDER BY created_at DESC LIMIT ${idx}"
        params.append(limit)

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [_row_to_item(r) for r in rows]

    async def clear(
        self,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> int:
        """Soft-delete memory items matching the filter. Returns count."""
        pool = self._ensure_init()
        async with pool.acquire() as conn:
            if metadata is None:
                result = await conn.execute("UPDATE memory_items SET deleted = 1 WHERE deleted = 0")
                return _parse_rowcount(result)

            clauses: list[str] = ["deleted = 0"]
            params: list[Any] = []
            idx = 1
            if metadata.user_id:
                clauses.append(f"metadata->>'user_id' = ${idx}")
                params.append(metadata.user_id)
                idx += 1
            if metadata.session_id:
                clauses.append(f"metadata->>'session_id' = ${idx}")
                params.append(metadata.session_id)
                idx += 1
            if metadata.task_id:
                clauses.append(f"metadata->>'task_id' = ${idx}")
                params.append(metadata.task_id)
                idx += 1
            if metadata.agent_id:
                clauses.append(f"metadata->>'agent_id' = ${idx}")
                params.append(metadata.agent_id)
                idx += 1

            where = " AND ".join(clauses)
            result = await conn.execute(
                f"UPDATE memory_items SET deleted = 1 WHERE {where}",
                *params,
            )
            return _parse_rowcount(result)

    # -- extras ---------------------------------------------------------------

    async def count(self, *, include_deleted: bool = False) -> int:
        """Return the number of stored items."""
        pool = self._ensure_init()
        where = "" if include_deleted else " WHERE deleted = 0"
        async with pool.acquire() as conn:
            row = await conn.fetchval(f"SELECT COUNT(*) FROM memory_items{where}")
        return row or 0

    def __repr__(self) -> str:
        return f"PostgresMemoryStore(dsn={self.dsn!r})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_rowcount(result: str) -> int:
    """Parse the row count from asyncpg's UPDATE command status string."""
    # asyncpg returns e.g. "UPDATE 3"
    parts = result.split()
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            return 0
    return 0


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


def _row_to_item(row: asyncpg.Record) -> MemoryItem:
    """Reconstruct a MemoryItem from a database row."""
    meta_raw = row["metadata"]
    meta_dict = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw

    extra_raw = row["extra_json"]
    extra = json.loads(extra_raw) if isinstance(extra_raw, str) else extra_raw

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
