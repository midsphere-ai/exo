"""Workflow execution REST + WebSocket endpoints."""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect

from exo_web.database import get_db
from exo_web.engine import (
    cancel_run,
    execute_single_node,
    execute_workflow,
    execute_workflow_debug,
    register_debug_session,
    send_debug_command,
)
from exo_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/v1/workflows", tags=["workflow-runs"])

# Keep references to background tasks so they aren't garbage-collected.
_background_tasks: set[asyncio.Task[Any]] = set()


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


async def _verify_workflow_ownership(db: Any, workflow_id: str, user_id: str) -> dict[str, Any]:
    """Return workflow row or raise 404."""
    cursor = await db.execute(
        "SELECT * FROM workflows WHERE id = ? AND user_id = ?",
        (workflow_id, user_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return dict(row)


# ---------------------------------------------------------------------------
# GET /api/workflows/:id/runs — paginated run history
# ---------------------------------------------------------------------------


@router.get("/{workflow_id}/runs")
async def list_workflow_runs(
    workflow_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: str | None = None,
    trigger_type: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return paginated run history for a workflow with optional filters."""
    async with get_db() as db:
        await _verify_workflow_ownership(db, workflow_id, user["id"])

        where_clauses = ["workflow_id = ?"]
        params: list[Any] = [workflow_id]

        if status:
            where_clauses.append("status = ?")
            params.append(status)
        if trigger_type:
            where_clauses.append("trigger_type = ?")
            params.append(trigger_type)
        if start_date:
            where_clauses.append("created_at >= ?")
            params.append(start_date)
        if end_date:
            where_clauses.append("created_at <= ?")
            params.append(end_date)

        where_sql = " AND ".join(where_clauses)

        # Total count for pagination.
        cursor = await db.execute(
            f"SELECT COUNT(*) as cnt FROM workflow_runs WHERE {where_sql}",
            params,
        )
        total = (await cursor.fetchone())["cnt"]

        # Fetch the page.
        cursor = await db.execute(
            f"SELECT id, workflow_id, status, trigger_type, input_json, started_at, completed_at, step_count, total_tokens, total_cost, error, created_at FROM workflow_runs WHERE {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        rows = await cursor.fetchall()

    runs = []
    for row in rows:
        r = dict(row)
        if r.get("input_json"):
            r["input_json"] = json.loads(r["input_json"])
        runs.append(r)

    return {"runs": runs, "total": total, "limit": limit, "offset": offset}


# ---------------------------------------------------------------------------
# GET /api/workflows/:id/runs/:runId — full run detail with node executions
# ---------------------------------------------------------------------------


@router.get("/{workflow_id}/runs/{run_id}")
async def get_workflow_run(
    workflow_id: str,
    run_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return full run detail including all node executions."""
    async with get_db() as db:
        await _verify_workflow_ownership(db, workflow_id, user["id"])

        cursor = await db.execute(
            "SELECT id, workflow_id, status, trigger_type, input_json, started_at, completed_at, step_count, total_tokens, total_cost, error, created_at FROM workflow_runs WHERE id = ? AND workflow_id = ?",
            (run_id, workflow_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")

        run = dict(row)
        if run.get("input_json"):
            run["input_json"] = json.loads(run["input_json"])

        # Fetch all node executions for this run.
        cursor = await db.execute(
            "SELECT id, run_id, node_id, status, input_json, output_json, logs_text, token_usage_json, started_at, completed_at, error FROM workflow_run_logs WHERE run_id = ? ORDER BY started_at",
            (run_id,),
        )
        node_rows = await cursor.fetchall()

    node_executions = []
    for nr in node_rows:
        n = dict(nr)
        for field in ("input_json", "output_json", "token_usage_json"):
            if n.get(field):
                n[field] = json.loads(n[field])
        node_executions.append(n)

    run["node_executions"] = node_executions
    return run


# ---------------------------------------------------------------------------
# POST /api/workflows/:id/run — start execution
# ---------------------------------------------------------------------------


@router.post("/{workflow_id}/run")
async def start_workflow_run(
    workflow_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Start executing a workflow. Returns the run_id immediately.

    If the concurrent run limit is reached, the run is queued instead.
    """
    from exo_web.services.run_queue import can_start_run, enqueue_run

    async with get_db() as db:
        wf = await _verify_workflow_ownership(db, workflow_id, user["id"])

        nodes = json.loads(wf["nodes_json"] or "[]")
        edges = json.loads(wf["edges_json"] or "[]")

        if not nodes:
            raise HTTPException(status_code=422, detail="Workflow has no nodes")

    # Check concurrency limits before starting.
    if not await can_start_run(workflow_id):
        result = await enqueue_run(workflow_id, user["id"], nodes, edges)
        return {"queue_id": result["queue_id"], "status": "queued", "position": result["position"]}

    async with get_db() as db:
        run_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            "INSERT INTO workflow_runs (id, workflow_id, status, user_id, created_at) VALUES (?, ?, 'pending', ?, ?)",
            (run_id, workflow_id, user["id"], now),
        )
        await db.commit()

    # Fire-and-forget: run the engine in the background.
    task = asyncio.create_task(execute_workflow(run_id, workflow_id, user["id"], nodes, edges))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {"run_id": run_id, "status": "pending"}


# ---------------------------------------------------------------------------
# POST /api/workflows/:id/nodes/:nodeId/run — single-node execution
# ---------------------------------------------------------------------------


@router.post("/{workflow_id}/nodes/{node_id}/run")
async def run_single_node(
    workflow_id: str,
    node_id: str,
    body: dict[str, Any] | None = None,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Execute a single node in isolation with optional mock input."""
    async with get_db() as db:
        wf = await _verify_workflow_ownership(db, workflow_id, user["id"])

        nodes = json.loads(wf["nodes_json"] or "[]")
        node = next((n for n in nodes if n["id"] == node_id), None)
        if node is None:
            raise HTTPException(status_code=404, detail="Node not found in workflow")

        run_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            "INSERT INTO workflow_runs (id, workflow_id, status, user_id, created_at) VALUES (?, ?, 'pending', ?, ?)",
            (run_id, workflow_id, user["id"], now),
        )
        await db.commit()

    mock_input = (body or {}).get("mock_input", {})
    result = await execute_single_node(run_id, workflow_id, user["id"], node, mock_input)
    return result


# ---------------------------------------------------------------------------
# DELETE /api/workflows/:id/runs/:runId — cancel execution
# ---------------------------------------------------------------------------


@router.delete("/{workflow_id}/runs/{run_id}")
async def cancel_workflow_run(
    workflow_id: str,
    run_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, str]:
    """Cancel a running workflow execution."""
    async with get_db() as db:
        await _verify_workflow_ownership(db, workflow_id, user["id"])

        cursor = await db.execute(
            "SELECT status FROM workflow_runs WHERE id = ? AND workflow_id = ?",
            (run_id, workflow_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")

        if row["status"] not in ("pending", "running"):
            raise HTTPException(
                status_code=422,
                detail=f"Run is already {row['status']}",
            )

    cancelled = cancel_run(run_id)
    if not cancelled:
        # Run might have finished between the check and cancel.
        async with get_db() as db:
            await db.execute(
                "UPDATE workflow_runs SET status = 'cancelled', completed_at = datetime('now') WHERE id = ?",
                (run_id,),
            )
            await db.commit()

    return {"status": "cancelled"}


# ---------------------------------------------------------------------------
# GET /api/workflows/:id/runs/:runId/nodes/:nodeId — node execution data
# ---------------------------------------------------------------------------


@router.get("/{workflow_id}/runs/{run_id}/nodes/{node_id}")
async def get_node_execution(
    workflow_id: str,
    run_id: str,
    node_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return execution data for a specific node within a run."""
    async with get_db() as db:
        await _verify_workflow_ownership(db, workflow_id, user["id"])

        # Verify the run belongs to this workflow.
        cursor = await db.execute(
            "SELECT id FROM workflow_runs WHERE id = ? AND workflow_id = ?",
            (run_id, workflow_id),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Run not found")

        cursor = await db.execute(
            "SELECT id, run_id, node_id, status, input_json, output_json, logs_text, started_at, completed_at, error, token_usage_json FROM workflow_run_logs WHERE run_id = ? AND node_id = ?",
            (run_id, node_id),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Node execution not found")

    result = dict(row)
    # Parse JSON fields for the response.
    for field in ("input_json", "output_json", "token_usage_json"):
        if result.get(field):
            result[field] = json.loads(result[field])

    return result


# ---------------------------------------------------------------------------
# WebSocket /api/workflows/:id/runs/:runId/stream — live execution events
# ---------------------------------------------------------------------------


@router.websocket("/{workflow_id}/runs/{run_id}/stream")
async def stream_workflow_run(
    websocket: WebSocket,
    workflow_id: str,
    run_id: str,
) -> None:
    """WebSocket endpoint for streaming workflow execution events."""
    user = await _get_user_from_cookie(websocket)
    if user is None:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    # Verify ownership.
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM workflows WHERE id = ? AND user_id = ?",
            (workflow_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            await websocket.close(code=4004, reason="Workflow not found")
            return

        cursor = await db.execute(
            "SELECT status FROM workflow_runs WHERE id = ? AND workflow_id = ?",
            (run_id, workflow_id),
        )
        run_row = await cursor.fetchone()
        if run_row is None:
            await websocket.close(code=4004, reason="Run not found")
            return

    await websocket.accept()

    # If run is already done, send the final status and close.
    if run_row["status"] in ("completed", "failed", "cancelled"):
        await websocket.send_json({"type": "execution_completed", "status": run_row["status"]})
        await websocket.close()
        return

    # Create an event queue to receive events from the engine.
    event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def event_callback(event: dict[str, Any]) -> None:
        await event_queue.put(event)

    # If the run hasn't started yet, start it with our callback.
    async with get_db() as db:
        cursor = await db.execute("SELECT status FROM workflow_runs WHERE id = ?", (run_id,))
        current = await cursor.fetchone()

    if current and current["status"] == "pending":
        async with get_db() as db:
            cursor = await db.execute(
                "SELECT nodes_json, edges_json FROM workflows WHERE id = ?",
                (workflow_id,),
            )
            wf = await cursor.fetchone()

        if wf:
            nodes = json.loads(wf["nodes_json"] or "[]")
            edges = json.loads(wf["edges_json"] or "[]")
            task = asyncio.create_task(
                execute_workflow(run_id, workflow_id, user["id"], nodes, edges, event_callback)
            )
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
    else:
        # Run is already in-progress (started via POST). We can't retroactively
        # attach a callback, so poll the DB for updates instead.
        task = asyncio.create_task(_poll_run_status(run_id, event_queue))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    try:
        while True:
            event = await event_queue.get()
            await websocket.send_json(event)
            if event.get("type") == "execution_completed":
                break
    except WebSocketDisconnect:
        pass
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


# ---------------------------------------------------------------------------
# POST /api/workflows/:id/debug — start debug execution
# ---------------------------------------------------------------------------


@router.post("/{workflow_id}/debug")
async def start_debug_run(
    workflow_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Start a debug execution of a workflow. Returns run_id immediately.

    Connect to the debug WebSocket to control stepping.
    """
    async with get_db() as db:
        wf = await _verify_workflow_ownership(db, workflow_id, user["id"])

        nodes = json.loads(wf["nodes_json"] or "[]")

        if not nodes:
            raise HTTPException(status_code=422, detail="Workflow has no nodes")

        run_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            "INSERT INTO workflow_runs (id, workflow_id, status, user_id, created_at) VALUES (?, ?, 'pending', ?, ?)",
            (run_id, workflow_id, user["id"], now),
        )
        await db.commit()

    # Register the debug session command queue but don't start execution yet.
    # The debug WebSocket will start execution when it connects.
    register_debug_session(run_id)

    return {"run_id": run_id, "status": "pending", "mode": "debug"}


# ---------------------------------------------------------------------------
# WebSocket /api/workflows/:id/runs/:runId/debug — debug step-through control
# ---------------------------------------------------------------------------


@router.websocket("/{workflow_id}/runs/{run_id}/debug")
async def debug_workflow_run(
    websocket: WebSocket,
    workflow_id: str,
    run_id: str,
) -> None:
    """WebSocket for debug step-through execution.

    Receives commands: {action: 'continue'|'skip'|'stop'|'set_breakpoint'|'set_variable', ...}
    Sends events: debug_paused, node_started, node_completed, node_failed, node_skipped, execution_completed
    """
    user = await _get_user_from_cookie(websocket)
    if user is None:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM workflows WHERE id = ? AND user_id = ?",
            (workflow_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            await websocket.close(code=4004, reason="Workflow not found")
            return

        cursor = await db.execute(
            "SELECT status FROM workflow_runs WHERE id = ? AND workflow_id = ?",
            (run_id, workflow_id),
        )
        run_row = await cursor.fetchone()
        if run_row is None:
            await websocket.close(code=4004, reason="Run not found")
            return

    await websocket.accept()

    if run_row["status"] in ("completed", "failed", "cancelled"):
        await websocket.send_json({"type": "execution_completed", "status": run_row["status"]})
        await websocket.close()
        return

    # Event queue for engine -> client communication.
    event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def event_callback(event: dict[str, Any]) -> None:
        await event_queue.put(event)

    # Ensure the debug session command queue exists (created by POST /debug).
    command_queue = register_debug_session(run_id)

    # Load workflow and start debug execution.
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT nodes_json, edges_json FROM workflows WHERE id = ?",
            (workflow_id,),
        )
        wf = await cursor.fetchone()

    if not wf:
        await websocket.send_json(
            {"type": "execution_completed", "status": "failed", "error": "Workflow not found"}
        )
        await websocket.close()
        return

    nodes = json.loads(wf["nodes_json"] or "[]")
    edges = json.loads(wf["edges_json"] or "[]")

    engine_task = asyncio.create_task(
        execute_workflow_debug(
            run_id, workflow_id, user["id"], nodes, edges, command_queue, event_callback
        )
    )
    _background_tasks.add(engine_task)
    engine_task.add_done_callback(_background_tasks.discard)

    # Run two concurrent loops: send events to client, receive commands from client.
    async def _send_events() -> None:
        while True:
            event = await event_queue.get()
            await websocket.send_json(event)
            if event.get("type") == "execution_completed":
                return

    async def _receive_commands() -> None:
        while True:
            data = await websocket.receive_json()
            send_debug_command(run_id, data)

    try:
        # Run both loops; when either finishes or raises, cancel the other.
        _done, pending = await asyncio.wait(
            [asyncio.create_task(_send_events()), asyncio.create_task(_receive_commands())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
    except WebSocketDisconnect:
        # Client disconnected — stop the debug run.
        send_debug_command(run_id, {"action": "stop"})
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


async def _poll_run_status(run_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
    """Fallback poller for runs started before the WS connected."""
    seen_nodes: set[str] = set()

    while True:
        await asyncio.sleep(0.3)

        async with get_db() as db:
            cursor = await db.execute("SELECT status FROM workflow_runs WHERE id = ?", (run_id,))
            run_row = await cursor.fetchone()
            if run_row is None:
                await queue.put(
                    {"type": "execution_completed", "status": "failed", "error": "Run not found"}
                )
                return

            # Check for new node completions.
            cursor = await db.execute(
                "SELECT node_id, status, output_json, error FROM workflow_run_logs WHERE run_id = ? ORDER BY started_at",
                (run_id,),
            )
            logs = await cursor.fetchall()

        for log in logs:
            nid = log["node_id"]
            if nid not in seen_nodes and log["status"] != "pending":
                if log["status"] == "running":
                    await queue.put({"type": "node_started", "node_id": nid})
                elif log["status"] == "completed":
                    output = json.loads(log["output_json"]) if log["output_json"] else {}
                    await queue.put({"type": "node_completed", "node_id": nid, "output": output})
                    seen_nodes.add(nid)
                elif log["status"] == "failed":
                    await queue.put(
                        {"type": "node_failed", "node_id": nid, "error": log["error"] or ""}
                    )
                    seen_nodes.add(nid)

        if run_row["status"] in ("completed", "failed", "cancelled"):
            await queue.put({"type": "execution_completed", "status": run_row["status"]})
            return
