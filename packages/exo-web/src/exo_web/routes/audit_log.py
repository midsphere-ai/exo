"""Audit log endpoints — admin-only access to security event history."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.pagination import PaginatedResponse, paginate
from exo_web.routes.auth import require_role

router = APIRouter(prefix="/api/v1/audit-log", tags=["audit-log"])


class AuditLogEntry(BaseModel):
    id: str = Field(description="Unique identifier")
    user_id: str = Field(description="Owning user identifier")
    action: str = Field(description="Action")
    entity_type: str | None = Field(description="Entity type")
    entity_id: str | None = Field(description="Entity id")
    details: dict[str, Any] | None = Field(description="Details")
    ip_address: str | None = Field(description="Ip address")
    created_at: str = Field(description="ISO 8601 creation timestamp")


def _map_row(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite Row to a dict, parsing details_json."""
    d = dict(row)
    raw = d.pop("details_json", None)
    d["details"] = json.loads(raw) if raw else None
    return d


@router.get("/filters")
async def audit_log_filters(
    _user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> dict[str, list[str]]:
    """Return distinct filter values for the audit log UI (admin only)."""
    async with get_db() as db:
        users_cur = await db.execute(
            "SELECT DISTINCT al.user_id, u.email FROM audit_log al LEFT JOIN users u ON u.id = al.user_id ORDER BY u.email"
        )
        users = [{"id": r[0], "email": r[1] or r[0]} for r in await users_cur.fetchall()]

        actions_cur = await db.execute("SELECT DISTINCT action FROM audit_log ORDER BY action")
        actions = [r[0] for r in await actions_cur.fetchall()]

        types_cur = await db.execute(
            "SELECT DISTINCT entity_type FROM audit_log WHERE entity_type IS NOT NULL ORDER BY entity_type"
        )
        entity_types = [r[0] for r in await types_cur.fetchall()]

    return {"users": users, "actions": actions, "entity_types": entity_types}


@router.get("")
async def list_audit_log(
    user_id: str | None = None,
    action: str | None = None,
    entity_type: str | None = None,
    date_from: str | None = Query(None, description="ISO date lower bound (inclusive)"),
    date_to: str | None = Query(None, description="ISO date upper bound (inclusive)"),
    cursor: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    _user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> PaginatedResponse:
    """Return paginated audit log entries with optional filters (admin only)."""
    conditions: list[str] = []
    params: list[Any] = []

    if user_id:
        conditions.append("user_id = ?")
        params.append(user_id)
    if action:
        conditions.append("action = ?")
        params.append(action)
    if entity_type:
        conditions.append("entity_type = ?")
        params.append(entity_type)
    if date_from:
        conditions.append("created_at >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("created_at <= datetime(?, '+1 day')")
        params.append(date_to)

    async with get_db() as db:
        return await paginate(
            db,
            table="audit_log",
            conditions=conditions,
            params=params,
            cursor=cursor,
            limit=limit,
            row_mapper=_map_row,
        )
