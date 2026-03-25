"""Unified runs REST API and WebSocket streaming."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/v1/runs", tags=["runs"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_user_from_cookie(websocket: WebSocket) -> dict[str, Any] | None:
    """Extract user from session cookie on the WebSocket connection."""
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

    return dict(row) if row else None


def _parse_run_row(row: Any) -> dict[str, Any]:
    """Convert a DB row to a response dict, parsing JSON fields."""
    r = dict(row)
    if r.get("steps_json"):
        r["steps_json"] = json.loads(r["steps_json"])
    return r


# ---------------------------------------------------------------------------
# GET /api/runs — list with filters
# ---------------------------------------------------------------------------


@router.get("")
async def list_runs(
    agent_id: str | None = None,
    status: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_cost: float | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return paginated runs with optional filters."""
    where_clauses = ["r.user_id = ?"]
    params: list[Any] = [user["id"]]

    if agent_id:
        where_clauses.append("r.agent_id = ?")
        params.append(agent_id)
    if status:
        where_clauses.append("r.status = ?")
        params.append(status)
    if start_date:
        where_clauses.append("r.created_at >= ?")
        params.append(start_date)
    if end_date:
        where_clauses.append("r.created_at <= ?")
        params.append(end_date)
    if min_cost is not None:
        where_clauses.append("r.total_cost >= ?")
        params.append(min_cost)

    where_sql = " AND ".join(where_clauses)

    async with get_db() as db:
        cursor = await db.execute(
            f"""SELECT COUNT(*) as cnt FROM runs r WHERE {where_sql}""",
            params,
        )
        total = (await cursor.fetchone())["cnt"]

        cursor = await db.execute(
            f"""SELECT r.*, a.name as agent_name, w.name as workflow_name
            FROM runs r
            LEFT JOIN agents a ON r.agent_id = a.id
            LEFT JOIN workflows w ON r.workflow_id = w.id
            WHERE {where_sql}
            ORDER BY r.created_at DESC LIMIT ? OFFSET ?""",
            [*params, limit, offset],
        )
        rows = await cursor.fetchall()

    return {
        "runs": [_parse_run_row(r) for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


# ---------------------------------------------------------------------------
# GET /api/runs/:id — full run detail with steps
# ---------------------------------------------------------------------------


@router.get("/{run_id}")
async def get_run(
    run_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return full run detail including steps."""
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT r.*, a.name as agent_name, w.name as workflow_name
            FROM runs r
            LEFT JOIN agents a ON r.agent_id = a.id
            LEFT JOIN workflows w ON r.workflow_id = w.id
            WHERE r.id = ? AND r.user_id = ?""",
            (run_id, user["id"]),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")

    return _parse_run_row(row)


# ---------------------------------------------------------------------------
# WebSocket /api/runs/:id/stream — live execution updates
# ---------------------------------------------------------------------------

# Active run subscriptions: run_id -> set of queues.
_run_subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}


def publish_run_event(run_id: str, event: dict[str, Any]) -> None:
    """Publish an event to all WebSocket subscribers for a run.

    Call this from execution engines to push real-time updates.
    """
    for queue in _run_subscribers.get(run_id, set()):
        queue.put_nowait(event)


@router.websocket("/{run_id}/stream")
async def stream_run(
    websocket: WebSocket,
    run_id: str,
) -> None:
    """WebSocket endpoint for live execution updates on a run."""
    user = await _get_user_from_cookie(websocket)
    if user is None:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    # Verify ownership.
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM runs WHERE id = ? AND user_id = ?",
            (run_id, user["id"]),
        )
        run_row = await cursor.fetchone()

    if run_row is None:
        await websocket.close(code=4004, reason="Run not found")
        return

    await websocket.accept()

    # If the run is already done, send final status and close.
    if run_row["status"] in ("completed", "failed", "cancelled"):
        await websocket.send_json(
            {"type": "run_completed", "status": run_row["status"], "run_id": run_id}
        )
        await websocket.close()
        return

    # Subscribe to live events.
    event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    if run_id not in _run_subscribers:
        _run_subscribers[run_id] = set()
    _run_subscribers[run_id].add(event_queue)

    # Poll DB as fallback if no events arrive via publish.
    poll_task = asyncio.create_task(_poll_run(run_id, event_queue))

    try:
        while True:
            event = await event_queue.get()
            await websocket.send_json(event)
            if event.get("type") == "run_completed":
                break
    except WebSocketDisconnect:
        pass
    finally:
        poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poll_task
        _run_subscribers.get(run_id, set()).discard(event_queue)
        if run_id in _run_subscribers and not _run_subscribers[run_id]:
            del _run_subscribers[run_id]
        with contextlib.suppress(Exception):
            await websocket.close()


async def _poll_run(run_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
    """Fallback poller that checks run status periodically."""
    last_status = "pending"

    while True:
        await asyncio.sleep(1.0)

        async with get_db() as db:
            cursor = await db.execute(
                "SELECT status, steps_json, total_tokens, total_cost, finish_reason FROM runs WHERE id = ?",
                (run_id,),
            )
            row = await cursor.fetchone()

        if row is None:
            await queue.put({"type": "run_completed", "status": "failed", "error": "Run not found"})
            return

        current_status = row["status"]

        if current_status != last_status:
            if current_status == "running":
                await queue.put({"type": "run_started", "run_id": run_id})
            elif current_status in ("completed", "failed", "cancelled"):
                await queue.put(
                    {
                        "type": "run_completed",
                        "run_id": run_id,
                        "status": current_status,
                        "total_tokens": row["total_tokens"],
                        "total_cost": row["total_cost"],
                        "finish_reason": row["finish_reason"],
                    }
                )
                return
            last_status = current_status
