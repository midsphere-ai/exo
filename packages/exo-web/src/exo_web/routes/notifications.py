"""Notifications REST API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.pagination import PaginatedResponse, paginate
from exo_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/v1/notifications", tags=["notifications"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class NotificationResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    type: str = Field(description="Type")
    title: str = Field(description="Title")
    message: str = Field(description="Message")
    entity_type: str | None = Field(None, description="Entity type")
    entity_id: str | None = Field(None, description="Entity id")
    read: bool = Field(description="Read")
    created_at: str = Field(description="ISO 8601 creation timestamp")


class UnreadCountResponse(BaseModel):
    count: int = Field(description="Count")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_notification_row(row: Any) -> dict[str, Any]:
    """Convert a DB row to a dict with bool read field."""
    r = dict(row)
    r["read"] = bool(r.get("read", 0))
    r.pop("user_id", None)
    return r


# ---------------------------------------------------------------------------
# GET /api/notifications/unread-count — unread notification count
# ---------------------------------------------------------------------------


@router.get("/unread-count", response_model=UnreadCountResponse)
async def get_unread_count(
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return the number of unread notifications for the current user."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM notifications WHERE user_id = ? AND read = 0",
            (user["id"],),
        )
        row = await cursor.fetchone()
    return {"count": row[0]}


# ---------------------------------------------------------------------------
# POST /api/notifications/read-all — mark all as read
# ---------------------------------------------------------------------------


@router.post("/read-all", response_model=dict[str, int])
async def mark_all_read(
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, int]:
    """Mark all unread notifications as read for the current user."""
    async with get_db() as db:
        cursor = await db.execute(
            "UPDATE notifications SET read = 1 WHERE user_id = ? AND read = 0",
            (user["id"],),
        )
        await db.commit()
    return {"updated": cursor.rowcount}


# ---------------------------------------------------------------------------
# GET /api/notifications — list notifications (paginated)
# ---------------------------------------------------------------------------


@router.get("", response_model=PaginatedResponse)
async def list_notifications(
    cursor: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> PaginatedResponse:
    """List notifications for the current user, newest first."""
    async with get_db() as db:
        return await paginate(
            db,
            table="notifications",
            conditions=["user_id = ?"],
            params=[user["id"]],
            cursor=cursor,
            limit=limit,
            row_mapper=_parse_notification_row,
        )


# ---------------------------------------------------------------------------
# PUT /api/notifications/:id/read — mark single notification as read
# ---------------------------------------------------------------------------


@router.put("/{notification_id}/read", response_model=NotificationResponse)
async def mark_notification_read(
    notification_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Mark a single notification as read."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM notifications WHERE id = ? AND user_id = ?",
            (notification_id, user["id"]),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Notification not found")

        await db.execute(
            "UPDATE notifications SET read = 1 WHERE id = ?",
            (notification_id,),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM notifications WHERE id = ?", (notification_id,))
        row = await cursor.fetchone()

    return _parse_notification_row(row)
