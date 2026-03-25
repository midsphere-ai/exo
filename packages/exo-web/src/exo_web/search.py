"""Full-text search across entities using SQLite FTS5."""

from __future__ import annotations

from typing import Any

import aiosqlite

# Entity type configuration: maps entity type to (fts_table, source_table, fts_columns)
_ENTITY_CONFIG: dict[str, tuple[str, str, list[str]]] = {
    "agents": ("agents_fts", "agents", ["name", "description", "instructions"]),
    "workflows": ("workflows_fts", "workflows", ["name", "description"]),
    "tools": ("tools_fts", "tools", ["name", "description"]),
    "threads": (
        "thread_messages_fts",
        "thread_messages",
        ["content"],
    ),
}

VALID_ENTITY_TYPES = set(_ENTITY_CONFIG.keys())


async def search_entities(
    db: aiosqlite.Connection,
    query: str,
    entity_types: list[str] | None = None,
    limit: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    """Search across entity types using FTS5.

    Args:
        db: Active aiosqlite connection (with Row factory).
        query: Search query string. Supports FTS5 syntax including prefix
            matching with ``*`` (e.g. ``"chat*"`` matches ``"chatbot"``).
        entity_types: Which entity types to search. Defaults to all types.
        limit: Maximum results per entity type.

    Returns:
        Dict keyed by entity type, each containing a list of result dicts
        with ``id``, ``name`` (or relevant field), ``type``, ``snippet``,
        and ``rank`` (relevance score).
    """
    if not query or not query.strip():
        return {}

    types = entity_types or list(_ENTITY_CONFIG.keys())
    # Filter out invalid types
    types = [t for t in types if t in _ENTITY_CONFIG]

    results: dict[str, list[dict[str, Any]]] = {}

    for entity_type in types:
        fts_table, source_table, fts_columns = _ENTITY_CONFIG[entity_type]
        rows = await _search_entity(
            db, fts_table, source_table, fts_columns, entity_type, query, limit
        )
        results[entity_type] = rows

    return results


async def _search_entity(
    db: aiosqlite.Connection,
    fts_table: str,
    source_table: str,
    fts_columns: list[str],
    entity_type: str,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Search a single entity type and return formatted results."""
    # Build snippet column for the first text column in the FTS table.
    # snippet(table, col_index, before, after, ellipsis, max_tokens)
    snippet_col = f"snippet({fts_table}, 0, '<b>', '</b>', '...', 32)"

    if entity_type == "threads":
        # For threads, join through to the thread to get an ID and name-like field
        sql = f"""
            SELECT
                t.id,
                t.first_message_preview AS name,
                '{entity_type}' AS type,
                {snippet_col} AS snippet,
                f.rank
            FROM {fts_table} f
            JOIN {source_table} s ON s.rowid = f.rowid
            JOIN threads t ON t.id = s.thread_id
            WHERE {fts_table} MATCH ?
            ORDER BY f.rank
            LIMIT ?
        """
    else:
        sql = f"""
            SELECT
                s.id,
                s.name,
                '{entity_type}' AS type,
                {snippet_col} AS snippet,
                f.rank
            FROM {fts_table} f
            JOIN {source_table} s ON s.rowid = f.rowid
            WHERE {fts_table} MATCH ?
            ORDER BY f.rank
            LIMIT ?
        """

    cursor = await db.execute(sql, (query, limit))
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]
