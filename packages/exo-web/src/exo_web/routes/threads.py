"""Thread management routes for conversation organization."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/v1/threads", tags=["threads"])


class ThreadCreate(BaseModel):
    agent_id: str = Field(min_length=1, description="Associated agent identifier")


class ThreadResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    agent_id: str = Field(description="Associated agent identifier")
    user_id: str = Field(description="Owning user identifier")
    first_message_preview: str = Field(description="Preview of the first message")
    message_count: int = Field(description="Number of messages in thread")
    total_tokens: int = Field(description="Total token count")
    total_cost: float = Field(description="Total cost in USD")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


class ThreadMessageResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    thread_id: str = Field(description="Associated thread identifier")
    role: str = Field(description="Message role (system, user, assistant)")
    content: str = Field(description="Text content")
    tool_calls_json: str | None = Field(None, description="JSON array of tool call objects")
    usage_json: str | None = Field(None, description="JSON token usage statistics")
    created_at: str = Field(description="ISO 8601 creation timestamp")


@router.get("/search")
async def search_threads(
    q: str = Query(min_length=1),
    user: dict = Depends(get_current_user),  # noqa: B008
) -> list[ThreadResponse]:
    """Search across threads by message content."""
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT DISTINCT t.* FROM threads t
            JOIN thread_messages tm ON tm.thread_id = t.id
            WHERE t.user_id = ? AND tm.content LIKE ?
            ORDER BY t.updated_at DESC
            """,
            (user["id"], f"%{q}%"),
        )
        rows = await cursor.fetchall()
    return [ThreadResponse(**dict(r)) for r in rows]


@router.get("")
async def list_threads(
    agent_id: str | None = None,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> list[ThreadResponse]:
    """List threads, optionally filtered by agent_id."""
    async with get_db() as db:
        if agent_id:
            cursor = await db.execute(
                "SELECT * FROM threads WHERE user_id = ? AND agent_id = ? ORDER BY updated_at DESC",
                (user["id"], agent_id),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM threads WHERE user_id = ? ORDER BY updated_at DESC",
                (user["id"],),
            )
        rows = await cursor.fetchall()
    return [ThreadResponse(**dict(r)) for r in rows]


@router.post("", status_code=201)
async def create_thread(
    body: ThreadCreate,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> ThreadResponse:
    """Create a new thread."""
    thread_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        # Verify the agent exists
        cursor = await db.execute("SELECT id FROM agents WHERE id = ?", (body.agent_id,))
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Agent not found")

        await db.execute(
            """
            INSERT INTO threads (id, agent_id, user_id, first_message_preview, message_count, total_tokens, total_cost, created_at, updated_at)
            VALUES (?, ?, ?, '', 0, 0, 0.0, ?, ?)
            """,
            (thread_id, body.agent_id, user["id"], now, now),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM threads WHERE id = ?", (thread_id,))
        row = await cursor.fetchone()
    return ThreadResponse(**dict(row))  # type: ignore[arg-type]


@router.get("/{thread_id}")
async def get_thread(
    thread_id: str,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> ThreadResponse:
    """Get a single thread with its metadata."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM threads WHERE id = ? AND user_id = ?",
            (thread_id, user["id"]),
        )
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Thread not found")
    return ThreadResponse(**dict(row))


@router.delete("/{thread_id}", status_code=204)
async def delete_thread(
    thread_id: str,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a thread and all its messages."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM threads WHERE id = ? AND user_id = ?",
            (thread_id, user["id"]),
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Thread not found")

        await db.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        await db.commit()


@router.get("/{thread_id}/messages")
async def list_thread_messages(
    thread_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(get_current_user),  # noqa: B008
) -> list[ThreadMessageResponse]:
    """List messages in a thread with pagination."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM threads WHERE id = ? AND user_id = ?",
            (thread_id, user["id"]),
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Thread not found")

        cursor = await db.execute(
            "SELECT * FROM thread_messages WHERE thread_id = ? ORDER BY created_at ASC LIMIT ? OFFSET ?",
            (thread_id, limit, offset),
        )
        rows = await cursor.fetchall()
    return [ThreadMessageResponse(**dict(r)) for r in rows]
