"""Multiplexed WebSocket manager for Exo Web.

Provides a single ``ws://api/ws`` endpoint with channel-based routing.
All real-time features (chat, execution, logs, sandbox, notifications, system)
share one connection per client.

Message envelope format::

    {
        "channel": "chat" | "execution" | "logs" | "sandbox" | "notifications" | "system",
        "type": "<message-type>",
        "payload": { ... }
    }
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from exo_web.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

VALID_CHANNELS = frozenset({"chat", "execution", "logs", "sandbox", "notifications", "system"})

_HEARTBEAT_INTERVAL = 30  # seconds
_HEARTBEAT_TIMEOUT = 10  # seconds


# ---------------------------------------------------------------------------
# Connection registry
# ---------------------------------------------------------------------------


class _Connection:
    """Tracks a single WebSocket connection and its subscribed channels."""

    __slots__ = ("_send_lock", "channels", "user_id", "ws")

    def __init__(self, ws: WebSocket, user_id: str) -> None:
        self.ws = ws
        self.user_id = user_id
        self.channels: set[str] = set()
        self._send_lock = asyncio.Lock()

    async def send(self, message: dict[str, Any]) -> bool:
        """Send a JSON message. Returns False if the connection is broken."""
        try:
            async with self._send_lock:
                await self.ws.send_json(message)
            return True
        except Exception:
            return False


class WebSocketManager:
    """Manages all active WebSocket connections with channel-based routing."""

    def __init__(self) -> None:
        self._connections: dict[WebSocket, _Connection] = {}
        self._user_connections: dict[str, set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    # -- lifecycle -----------------------------------------------------------

    async def connect(self, ws: WebSocket, user_id: str) -> _Connection:
        """Register a new connection."""
        conn = _Connection(ws, user_id)
        async with self._lock:
            self._connections[ws] = conn
            self._user_connections.setdefault(user_id, set()).add(ws)
        logger.info("WS connected: user=%s total=%d", user_id, len(self._connections))
        return conn

    async def disconnect(self, ws: WebSocket) -> None:
        """Remove a connection from the registry."""
        async with self._lock:
            conn = self._connections.pop(ws, None)
            if conn is not None:
                user_set = self._user_connections.get(conn.user_id)
                if user_set is not None:
                    user_set.discard(ws)
                    if not user_set:
                        del self._user_connections[conn.user_id]
                logger.info(
                    "WS disconnected: user=%s total=%d",
                    conn.user_id,
                    len(self._connections),
                )

    # -- channel management --------------------------------------------------

    def subscribe(self, ws: WebSocket, channel: str) -> bool:
        """Subscribe a connection to a channel. Returns True on success."""
        conn = self._connections.get(ws)
        if conn is None or channel not in VALID_CHANNELS:
            return False
        conn.channels.add(channel)
        return True

    def unsubscribe(self, ws: WebSocket, channel: str) -> bool:
        """Unsubscribe a connection from a channel. Returns True on success."""
        conn = self._connections.get(ws)
        if conn is None:
            return False
        conn.channels.discard(channel)
        return True

    # -- messaging -----------------------------------------------------------

    async def send_to_connection(
        self, ws: WebSocket, channel: str, msg_type: str, payload: dict[str, Any]
    ) -> bool:
        """Send a message to a specific connection."""
        conn = self._connections.get(ws)
        if conn is None:
            return False
        return await conn.send({"channel": channel, "type": msg_type, "payload": payload})

    async def broadcast_to_user(self, user_id: str, channel: str, message: dict[str, Any]) -> int:
        """Push a message to all connections for a user subscribed to *channel*.

        Returns the number of connections that received the message.
        """
        envelope = {"channel": channel, "type": message.get("type", "event"), "payload": message}
        sent = 0
        # Snapshot the set to avoid holding the lock during I/O.
        async with self._lock:
            ws_set = set(self._user_connections.get(user_id, set()))
        for ws in ws_set:
            conn = self._connections.get(ws)
            if conn is not None and channel in conn.channels and await conn.send(envelope):
                sent += 1
        return sent

    # -- introspection -------------------------------------------------------

    @property
    def active_connections(self) -> int:
        return len(self._connections)

    def user_connections(self, user_id: str) -> int:
        return len(self._user_connections.get(user_id, set()))


# Singleton instance used by the application.
manager = WebSocketManager()


# ---------------------------------------------------------------------------
# Auth helper (shared with other WS endpoints)
# ---------------------------------------------------------------------------


async def get_ws_user(websocket: WebSocket) -> dict[str, Any] | None:
    """Extract user from session cookie on a WebSocket connection."""
    session_id = websocket.cookies.get("exo_session")
    if not session_id:
        return None

    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT u.id, u.email, u.created_at
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.id = ? AND s.expires_at > datetime('now')
            """,
            (session_id,),
        )
        row = await cursor.fetchone()

    if row is None:
        return None
    return dict(row)


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


async def _heartbeat(ws: WebSocket) -> None:
    """Send periodic pings. Raises if the peer doesn't respond within timeout."""
    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL)
        try:
            await asyncio.wait_for(
                ws.send_json({"channel": "system", "type": "ping", "payload": {}}),
                timeout=_HEARTBEAT_TIMEOUT,
            )
        except Exception:
            raise WebSocketDisconnect(code=1001, reason="heartbeat timeout") from None


# ---------------------------------------------------------------------------
# Message dispatcher
# ---------------------------------------------------------------------------


async def _handle_message(ws: WebSocket, raw: str) -> None:
    """Parse and dispatch a single incoming message."""
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        await manager.send_to_connection(ws, "system", "error", {"message": "Invalid JSON"})
        return

    msg_type = msg.get("type")

    if msg_type == "subscribe":
        channel = msg.get("channel", "")
        if channel not in VALID_CHANNELS:
            await manager.send_to_connection(
                ws,
                "system",
                "error",
                {"message": f"Invalid channel: {channel}"},
            )
            return
        manager.subscribe(ws, channel)
        await manager.send_to_connection(ws, "system", "subscribed", {"channel": channel})

    elif msg_type == "unsubscribe":
        channel = msg.get("channel", "")
        manager.unsubscribe(ws, channel)
        await manager.send_to_connection(ws, "system", "unsubscribed", {"channel": channel})

    elif msg_type == "pong":
        # Client responded to our heartbeat ping — nothing to do.
        pass

    else:
        # Forward to any future per-channel handlers.
        # For now, acknowledge receipt.
        await manager.send_to_connection(
            ws,
            "system",
            "error",
            {"message": f"Unknown message type: {msg_type}"},
        )


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@router.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Multiplexed WebSocket endpoint.

    Authentication: the browser sends the ``exo_session`` cookie
    automatically.  We validate it before accepting the connection.
    """
    user = await get_ws_user(websocket)
    if user is None:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()
    conn = await manager.connect(websocket, user["id"])

    # Auto-subscribe to system channel.
    conn.channels.add("system")

    # Send a welcome message with connection info.
    await manager.send_to_connection(
        websocket,
        "system",
        "connected",
        {"user_id": user["id"], "channels": list(VALID_CHANNELS)},
    )

    heartbeat_task: asyncio.Task[None] | None = None
    try:
        heartbeat_task = asyncio.create_task(_heartbeat(websocket))

        while True:
            raw = await websocket.receive_text()
            await _handle_message(websocket, raw)

    except WebSocketDisconnect:
        pass
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
        await manager.disconnect(websocket)
