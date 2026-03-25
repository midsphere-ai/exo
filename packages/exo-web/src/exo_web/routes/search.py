"""Global search endpoint (Cmd+K) — cross-entity search using FTS5."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user
from exo_web.search import search_entities

router = APIRouter(prefix="/api/v1/search", tags=["search"])

# URL path templates per entity type
_URL_TEMPLATES: dict[str, str] = {
    "agents": "/agents/{id}",
    "workflows": "/workflows/{id}",
    "tools": "/tools/{id}",
    "knowledge_bases": "/knowledge-bases/{id}",
}

# Entity types that the global search supports
_GLOBAL_SEARCH_TYPES = {"agents", "workflows", "tools", "knowledge_bases"}

_DEFAULT_PER_TYPE_LIMIT = 5


def _add_url(result: dict[str, Any]) -> dict[str, Any]:
    """Add ``url`` field to a search result based on its type."""
    entity_type = result.get("type", "")
    template = _URL_TEMPLATES.get(entity_type, "")
    result["url"] = template.format(id=result.get("id", ""))
    return result


async def _search_knowledge_bases(
    db: Any,
    query: str,
    user_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Search knowledge_bases using LIKE (no FTS5 table available)."""
    like_param = f"%{query}%"
    cursor = await db.execute(
        """
        SELECT id, name, 'knowledge_bases' AS type, description AS snippet
        FROM knowledge_bases
        WHERE user_id = ? AND (name LIKE ? OR description LIKE ?)
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, like_param, like_param, limit),
    )
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def _get_recent_items(
    db: Any,
    user_id: str,
    types: list[str],
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    """Return recent items per entity type for empty queries."""
    results: dict[str, list[dict[str, Any]]] = {}

    queries: dict[str, str] = {
        "agents": """
            SELECT id, name, 'agents' AS type, description AS snippet
            FROM agents WHERE user_id = ?
            ORDER BY created_at DESC LIMIT ?
        """,
        "workflows": """
            SELECT id, name, 'workflows' AS type, description AS snippet
            FROM workflows WHERE user_id = ?
            ORDER BY created_at DESC LIMIT ?
        """,
        "tools": """
            SELECT id, name, 'tools' AS type, description AS snippet
            FROM tools WHERE user_id = ?
            ORDER BY created_at DESC LIMIT ?
        """,
        "knowledge_bases": """
            SELECT id, name, 'knowledge_bases' AS type, description AS snippet
            FROM knowledge_bases WHERE user_id = ?
            ORDER BY created_at DESC LIMIT ?
        """,
    }

    for entity_type in types:
        sql = queries.get(entity_type)
        if not sql:
            continue
        cursor = await db.execute(sql, (user_id, limit))
        rows = await cursor.fetchall()
        results[entity_type] = [_add_url(dict(row)) for row in rows]

    return results


@router.get("")
async def global_search(
    q: str = Query("", description="Search query"),
    types: str = Query(
        "agents,workflows,tools,knowledge_bases",
        description="Comma-separated entity types to search",
    ),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Search across agents, workflows, tools, and knowledge bases.

    Empty query returns recent items instead of search results.
    """
    requested_types = [t.strip() for t in types.split(",") if t.strip()]
    # Filter to valid types only
    requested_types = [t for t in requested_types if t in _GLOBAL_SEARCH_TYPES]
    if not requested_types:
        requested_types = list(_GLOBAL_SEARCH_TYPES)

    user_id: str = user["id"]

    async with get_db() as db:
        # Empty query → return recent items
        if not q or not q.strip():
            return await _get_recent_items(db, user_id, requested_types, _DEFAULT_PER_TYPE_LIMIT)

        results: dict[str, list[dict[str, Any]]] = {}

        # FTS5-backed types (agents, workflows, tools)
        fts_types = [t for t in requested_types if t != "knowledge_bases"]
        if fts_types:
            fts_results = await search_entities(
                db, q, entity_types=fts_types, limit=_DEFAULT_PER_TYPE_LIMIT
            )
            for entity_type, rows in fts_results.items():
                results[entity_type] = [_add_url(row) for row in rows]

        # knowledge_bases: LIKE fallback
        if "knowledge_bases" in requested_types:
            kb_rows = await _search_knowledge_bases(db, q, user_id, _DEFAULT_PER_TYPE_LIMIT)
            results["knowledge_bases"] = [_add_url(row) for row in kb_rows]

    return results
