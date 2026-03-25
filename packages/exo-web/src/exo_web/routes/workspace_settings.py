"""Workspace settings endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/v1/settings/workspace", tags=["workspace-settings"])


class WorkspaceSettingUpdate(BaseModel):
    value: str = Field(description="Value")


@router.get("")
async def get_workspace_settings(
    _user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, str]:
    """Return all workspace settings as a key-value map."""
    async with get_db() as db:
        cursor = await db.execute("SELECT key, value FROM workspace_settings")
        rows = await cursor.fetchall()
    return {row["key"]: row["value"] for row in rows}


@router.put("/{key}")
async def update_workspace_setting(
    key: str,
    body: WorkspaceSettingUpdate,
    _user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, str]:
    """Update a single workspace setting."""
    async with get_db() as db:
        await db.execute(
            "INSERT INTO workspace_settings (key, value, updated_at) VALUES (?, ?, datetime('now')) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, body.value),
        )
        await db.commit()
    return {"key": key, "value": body.value}
