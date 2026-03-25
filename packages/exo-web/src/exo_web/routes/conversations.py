"""Conversation persistence routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user
from exo_web.sanitize import sanitize_html
from exo_web.services.memory import memory_service

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


class ConversationResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    agent_id: str = Field(description="Associated agent identifier")
    user_id: str = Field(description="Owning user identifier")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


class MessageResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    conversation_id: str = Field(description="Associated conversation identifier")
    role: str = Field(description="Message role (system, user, assistant)")
    content: str = Field(description="Text content")
    tool_calls_json: str | None = Field(None, description="JSON array of tool call objects")
    usage_json: str | None = Field(None, description="JSON token usage statistics")
    created_at: str = Field(description="ISO 8601 creation timestamp")


class MessageEditRequest(BaseModel):
    content: str = Field(min_length=1, description="Text content")


class MessageEditResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    original_content: str = Field(description="Original content")
    edited_at: str = Field(description="Edited at")


@router.get("")
async def list_conversations(
    agent_id: str | None = None,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> list[ConversationResponse]:
    """List conversations, optionally filtered by agent_id."""
    async with get_db() as db:
        if agent_id:
            cursor = await db.execute(
                "SELECT * FROM conversations WHERE user_id = ? AND agent_id = ? ORDER BY updated_at DESC",
                (user["id"], agent_id),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
                (user["id"],),
            )
        rows = await cursor.fetchall()
    return [ConversationResponse(**dict(r)) for r in rows]


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> ConversationResponse:
    """Get a single conversation."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user["id"]),
        )
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationResponse(**dict(row))


@router.get("/{conversation_id}/messages")
async def list_messages(
    conversation_id: str,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> list[MessageResponse]:
    """List messages in a conversation, ordered chronologically."""
    # Verify conversation belongs to user
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user["id"]),
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Conversation not found")

        cursor = await db.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
    return [MessageResponse(**dict(r)) for r in rows]


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> dict[str, str]:
    """Delete a conversation and all its messages."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user["id"]),
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get agent_id before deleting so we can clear memory
        cursor = await db.execute(
            "SELECT agent_id FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        conv_row = await cursor.fetchone()
        await db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        await db.commit()
    # Clear associated agent memory
    if conv_row:
        await memory_service.clear_memory(conv_row["agent_id"], conversation_id)
    return {"status": "deleted"}


async def _verify_message_ownership(
    db: object, conversation_id: str, message_id: str, user_id: str
) -> dict | None:
    """Verify message belongs to user's conversation. Returns message row or None."""
    cursor = await db.execute(  # type: ignore[union-attr]
        "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user_id),
    )
    if not await cursor.fetchone():
        return None
    cursor = await db.execute(  # type: ignore[union-attr]
        "SELECT * FROM messages WHERE id = ? AND conversation_id = ?",
        (message_id, conversation_id),
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


@router.put("/{conversation_id}/messages/{message_id}")
async def edit_message(
    conversation_id: str,
    message_id: str,
    body: MessageEditRequest,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> MessageResponse:
    """Edit a message's content, saving the original in edit history."""
    async with get_db() as db:
        msg = await _verify_message_ownership(db, conversation_id, message_id, user["id"])
        if not msg:
            raise HTTPException(status_code=404, detail="Message not found")

        # Save original content to edit history
        edit_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO message_edits (id, message_id, original_content) VALUES (?, ?, ?)",
            (edit_id, message_id, msg["content"]),
        )

        # Update message content
        new_content = sanitize_html(body.content)
        await db.execute(
            "UPDATE messages SET content = ? WHERE id = ?",
            (new_content, message_id),
        )
        await db.execute(
            "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
            (conversation_id,),
        )
        await db.commit()

        # Return updated message
        cursor = await db.execute("SELECT * FROM messages WHERE id = ?", (message_id,))
        row = await cursor.fetchone()
    return MessageResponse(**dict(row))  # type: ignore[arg-type]


@router.get("/{conversation_id}/messages/{message_id}/edits")
async def list_message_edits(
    conversation_id: str,
    message_id: str,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> list[MessageEditResponse]:
    """List edit history for a message (oldest first)."""
    async with get_db() as db:
        msg = await _verify_message_ownership(db, conversation_id, message_id, user["id"])
        if not msg:
            raise HTTPException(status_code=404, detail="Message not found")

        cursor = await db.execute(
            "SELECT * FROM message_edits WHERE message_id = ? ORDER BY edited_at ASC",
            (message_id,),
        )
        rows = await cursor.fetchall()
    return [MessageEditResponse(**dict(r)) for r in rows]


@router.delete("/{conversation_id}/messages/{message_id}")
async def delete_message(
    conversation_id: str,
    message_id: str,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> dict[str, str]:
    """Delete a message from a conversation."""
    async with get_db() as db:
        msg = await _verify_message_ownership(db, conversation_id, message_id, user["id"])
        if not msg:
            raise HTTPException(status_code=404, detail="Message not found")

        await db.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        await db.execute(
            "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
            (conversation_id,),
        )
        await db.commit()
    return {"status": "deleted"}


@router.post("/{conversation_id}/messages/{message_id}/replay")
async def replay_from_message(
    conversation_id: str,
    message_id: str,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> list[MessageResponse]:
    """Delete all messages after the given message and return the remaining messages.

    This prepares the conversation for replay: the client should then re-send
    the edited message to the agent to get a new response.
    """
    async with get_db() as db:
        msg = await _verify_message_ownership(db, conversation_id, message_id, user["id"])
        if not msg:
            raise HTTPException(status_code=404, detail="Message not found")

        # Delete all messages created after this one
        await db.execute(
            "DELETE FROM messages WHERE conversation_id = ? AND created_at > ?",
            (conversation_id, msg["created_at"]),
        )
        await db.execute(
            "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
            (conversation_id,),
        )
        await db.commit()

        # Return remaining messages
        cursor = await db.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
    return [MessageResponse(**dict(r)) for r in rows]
