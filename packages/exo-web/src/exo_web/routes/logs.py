"""Structured log storage with REST filtering and WebSocket streaming."""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user
from exo_web.websocket import get_ws_user

router = APIRouter(prefix="/api/v1/logs", tags=["logs"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LogEntry(BaseModel):
    id: str = Field(description="Unique identifier")
    timestamp: str = Field(description="Timestamp")
    level: str = Field(description="Level")
    source: str = Field(description="Source")
    agent_id: str | None = Field(None, description="Associated agent identifier")
    message: str = Field(description="Message")
    metadata_json: Any | None = Field(None, description="Metadata json")


class CreateLogEntry(BaseModel):
    level: str = Field(..., pattern=r"^(debug|info|warn|error)$", description="Level")
    source: str = Field(..., pattern=r"^(agent|tool|model|system)$", description="Source")
    agent_id: str | None = Field(None, description="Associated agent identifier")
    message: str = Field(..., min_length=1, description="Message")
    metadata_json: Any | None = Field(None, description="Metadata json")


# ---------------------------------------------------------------------------
# Streaming subscribers
# ---------------------------------------------------------------------------

_subscribers: dict[WebSocket, dict[str, Any]] = {}
_sub_lock = asyncio.Lock()


async def _notify_subscribers(entry: dict[str, Any], user_id: str) -> None:
    """Push a new log entry to all subscribed WebSocket clients for this user."""
    async with _sub_lock:
        targets = list(_subscribers.items())
    for ws, info in targets:
        if info.get("user_id") != user_id:
            continue
        try:
            await ws.send_json({"type": "log", "payload": entry})
        except Exception:
            async with _sub_lock:
                _subscribers.pop(ws, None)


# ---------------------------------------------------------------------------
# POST /api/logs — create a log entry
# ---------------------------------------------------------------------------


@router.post("", status_code=201)
async def create_log(
    body: CreateLogEntry,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> LogEntry:
    """Create a structured log entry."""
    log_id = str(uuid.uuid4())
    uid = user["id"]
    meta = json.dumps(body.metadata_json) if body.metadata_json is not None else None

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO logs (id, level, source, agent_id, message, metadata_json, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (log_id, body.level, body.source, body.agent_id, body.message, meta, uid),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM logs WHERE id = ?", (log_id,))
        row = await cursor.fetchone()

    entry = _parse_log_row(row)
    await _notify_subscribers(entry, uid)
    return LogEntry(**entry)


# ---------------------------------------------------------------------------
# GET /api/logs — filtered query
# ---------------------------------------------------------------------------


@router.get("")
async def list_logs(
    level: str | None = Query(default=None),
    source: str | None = Query(default=None),
    agent_id: str | None = Query(default=None),
    start_time: str | None = Query(default=None),
    end_time: str | None = Query(default=None),
    search: str | None = Query(default=None),
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[LogEntry]:
    """Query logs with optional filtering by level, source, agent, time range, and text search."""
    uid = user["id"]
    conditions = ["user_id = ?"]
    params: list[Any] = [uid]

    if level:
        conditions.append("level = ?")
        params.append(level)
    if source:
        conditions.append("source = ?")
        params.append(source)
    if agent_id:
        conditions.append("agent_id = ?")
        params.append(agent_id)
    if start_time:
        conditions.append("timestamp >= ?")
        params.append(start_time)
    if end_time:
        conditions.append("timestamp <= ?")
        params.append(end_time)
    if search:
        conditions.append("message LIKE ?")
        params.append(f"%{search}%")

    where = " AND ".join(conditions)
    params.extend([limit, offset])

    async with get_db() as db:
        cursor = await db.execute(
            f"SELECT * FROM logs WHERE {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            params,
        )
        rows = await cursor.fetchall()

    return [LogEntry(**_parse_log_row(row)) for row in rows]


# ---------------------------------------------------------------------------
# WebSocket /api/logs/stream — live log streaming
# ---------------------------------------------------------------------------


@router.websocket("/stream")
async def logs_stream(websocket: WebSocket) -> None:
    """Stream new log entries in real time.

    Clients may send filter messages to narrow the stream::

        {"level": "error", "source": "agent", "agent_id": "..."}

    Unset fields mean "no filter" (receive all).
    """
    user = await get_ws_user(websocket)
    if user is None:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()
    uid = user["id"]

    async with _sub_lock:
        _subscribers[websocket] = {"user_id": uid, "filters": {}}

    try:
        while True:
            raw = await websocket.receive_text()
            with contextlib.suppress(json.JSONDecodeError):
                filters = json.loads(raw)
                if isinstance(filters, dict):
                    async with _sub_lock:
                        if websocket in _subscribers:
                            _subscribers[websocket]["filters"] = filters
    except WebSocketDisconnect:
        pass
    finally:
        async with _sub_lock:
            _subscribers.pop(websocket, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_log_row(row: Any) -> dict[str, Any]:
    """Convert a DB row to a dict, parsing metadata_json."""
    r = dict(row)
    if r.get("metadata_json"):
        with contextlib.suppress(json.JSONDecodeError):
            r["metadata_json"] = json.loads(r["metadata_json"])
    r.pop("user_id", None)
    return r
