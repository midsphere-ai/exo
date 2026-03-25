"""Notification service — create and push notifications."""

from __future__ import annotations

import uuid
from typing import Any

from exo_web.database import get_db


async def create_notification(
    user_id: str,
    type: str,
    title: str,
    message: str,
    entity_type: str | None = None,
    entity_id: str | None = None,
) -> dict[str, Any]:
    """Create a notification and push it via WebSocket.

    Parameters
    ----------
    user_id:
        Target user for the notification.
    type:
        Notification category (e.g. ``"approval"``, ``"alert"``, ``"budget"``).
    title:
        Short human-readable title.
    message:
        Notification body text.
    entity_type:
        Optional entity kind (e.g. ``"workflow"``, ``"agent"``).
    entity_id:
        Optional entity identifier for navigation.

    Returns
    -------
    dict
        The created notification row as a dict.
    """
    notif_id = str(uuid.uuid4())

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO notifications (id, user_id, type, title, message, entity_type, entity_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (notif_id, user_id, type, title, message, entity_type, entity_id),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM notifications WHERE id = ?", (notif_id,))
        row = await cursor.fetchone()

    notif = dict(row)

    # Push via WebSocket to the user's notifications channel
    from exo_web.websocket import manager

    await manager.broadcast_to_user(
        user_id,
        "notifications",
        {
            "type": "notification_created",
            "id": notif["id"],
            "notification_type": notif["type"],
            "title": notif["title"],
            "message": notif["message"],
            "entity_type": notif["entity_type"],
            "entity_id": notif["entity_id"],
            "created_at": notif["created_at"],
        },
    )

    return notif
