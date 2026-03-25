"""Cursor-based pagination utility for list endpoints."""

from __future__ import annotations

import base64
from typing import Any

from pydantic import BaseModel


class PaginationMeta(BaseModel):
    """Pagination metadata included in paginated responses."""

    next_cursor: str | None
    has_more: bool
    total: int


class PaginatedResponse(BaseModel):
    """Envelope for paginated list responses."""

    data: list[Any]
    pagination: PaginationMeta


def decode_cursor(cursor: str) -> tuple[str, str]:
    """Decode a base64 cursor into (created_at, id) tuple.

    Raises ``ValueError`` on malformed cursors.
    """
    try:
        decoded = base64.urlsafe_b64decode(cursor.encode()).decode()
        created_at, row_id = decoded.split("|", 1)
    except Exception as exc:
        raise ValueError("Invalid cursor") from exc
    return created_at, row_id


def encode_cursor(created_at: str, row_id: str) -> str:
    """Encode (created_at, id) into a base64 cursor string."""
    raw = f"{created_at}|{row_id}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


async def paginate(
    db: Any,
    *,
    table: str,
    conditions: list[str],
    params: list[Any],
    cursor: str | None = None,
    limit: int = 20,
    row_mapper: Any = dict,
) -> PaginatedResponse:
    """Execute a paginated query and return a ``PaginatedResponse``.

    Parameters
    ----------
    db:
        An open ``aiosqlite.Connection``.
    table:
        The SQL table name.
    conditions:
        List of SQL WHERE fragments (e.g. ``["user_id = ?"]``).
    params:
        Corresponding bind parameters for *conditions*.
    cursor:
        Optional cursor from a previous response.
    limit:
        Page size (clamped to 1-100).
    row_mapper:
        Callable to transform each ``aiosqlite.Row`` into a dict.
    """
    limit = max(1, min(limit, 100))

    # Build WHERE clause -------------------------------------------------
    where_parts = list(conditions)
    query_params: list[Any] = list(params)

    if cursor:
        created_at, row_id = decode_cursor(cursor)
        where_parts.append("(created_at < ? OR (created_at = ? AND id < ?))")
        query_params.extend([created_at, created_at, row_id])

    where_sql = " AND ".join(where_parts) if where_parts else "1=1"

    # Total count (ignoring cursor) --------------------------------------
    count_cursor = await db.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {' AND '.join(conditions) if conditions else '1=1'}",
        params,
    )
    total = (await count_cursor.fetchone())[0]

    # Fetch one extra row to detect has_more -----------------------------
    data_cursor = await db.execute(
        f"SELECT * FROM {table} WHERE {where_sql} ORDER BY created_at DESC, id DESC LIMIT ?",
        [*query_params, limit + 1],
    )
    rows = await data_cursor.fetchall()

    has_more = len(rows) > limit
    rows = rows[:limit]

    next_cursor: str | None = None
    if has_more and rows:
        last = dict(rows[-1])
        next_cursor = encode_cursor(last["created_at"], last["id"])

    return PaginatedResponse(
        data=[row_mapper(r) for r in rows],
        pagination=PaginationMeta(
            next_cursor=next_cursor,
            has_more=has_more,
            total=total,
        ),
    )
