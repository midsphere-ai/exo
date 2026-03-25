"""Run queue — enforces concurrency limits on workflow execution."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from exo_web.database import get_db

_log = logging.getLogger(__name__)

# Keep references to background tasks so they aren't GC'd.
_background_tasks: set[asyncio.Task[Any]] = set()


async def get_concurrent_run_limit() -> int:
    """Return the workspace-level concurrent run limit from DB settings."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT value FROM workspace_settings WHERE key = 'concurrent_run_limit'"
        )
        row = await cursor.fetchone()
    return int(row["value"]) if row else 5


async def get_active_run_count(workflow_id: str | None = None) -> int:
    """Count currently running workflow executions.

    If *workflow_id* is given, count only runs for that workflow.
    """
    async with get_db() as db:
        if workflow_id:
            cursor = await db.execute(
                "SELECT COUNT(*) AS cnt FROM workflow_runs WHERE status IN ('pending', 'running') AND workflow_id = ?",
                (workflow_id,),
            )
        else:
            cursor = await db.execute(
                "SELECT COUNT(*) AS cnt FROM workflow_runs WHERE status IN ('pending', 'running')"
            )
        row = await cursor.fetchone()
    return row["cnt"] if row else 0


async def get_workflow_max_concurrent(workflow_id: str) -> int | None:
    """Return the per-workflow max_concurrent setting, or None if not set."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT max_concurrent FROM workflows WHERE id = ?",
            (workflow_id,),
        )
        row = await cursor.fetchone()
    if row and row["max_concurrent"] is not None:
        return int(row["max_concurrent"])
    return None


async def can_start_run(workflow_id: str) -> bool:
    """Return True if a new run can start (both global and per-workflow limits)."""
    # Check global limit.
    limit = await get_concurrent_run_limit()
    active = await get_active_run_count()
    if active >= limit:
        return False

    # Check per-workflow limit.
    wf_limit = await get_workflow_max_concurrent(workflow_id)
    if wf_limit is not None:
        wf_active = await get_active_run_count(workflow_id)
        if wf_active >= wf_limit:
            return False

    return True


async def enqueue_run(
    workflow_id: str,
    user_id: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    trigger_type: str = "manual",
) -> dict[str, Any]:
    """Add a run to the queue. Returns the queue entry."""
    queue_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        await db.execute(
            """INSERT INTO run_queue (id, workflow_id, user_id, status, trigger_type, nodes_json, edges_json, queued_at)
               VALUES (?, ?, ?, 'queued', ?, ?, ?, ?)""",
            (
                queue_id,
                workflow_id,
                user_id,
                trigger_type,
                json.dumps(nodes),
                json.dumps(edges),
                now,
            ),
        )
        await db.commit()

    return {
        "queue_id": queue_id,
        "status": "queued",
        "position": await get_queue_position(queue_id),
    }


async def get_queue_position(queue_id: str) -> int:
    """Return 1-based position of a queued entry."""
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT COUNT(*) AS pos FROM run_queue
               WHERE status = 'queued'
                 AND queued_at <= (SELECT queued_at FROM run_queue WHERE id = ?)""",
            (queue_id,),
        )
        row = await cursor.fetchone()
    return row["pos"] if row else 0


async def list_queued_runs(user_id: str) -> list[dict[str, Any]]:
    """Return all queued runs for a user, ordered by position."""
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT id, workflow_id, status, trigger_type, queued_at, started_at, run_id
               FROM run_queue
               WHERE user_id = ? AND status = 'queued'
               ORDER BY queued_at ASC""",
            (user_id,),
        )
        rows = await cursor.fetchall()

    result = []
    for i, row in enumerate(rows, 1):
        entry = dict(row)
        entry["position"] = i
        result.append(entry)
    return result


async def cancel_queued_run(queue_id: str, user_id: str) -> bool:
    """Cancel a queued run. Returns True if cancelled, False if not found or not queued."""
    async with get_db() as db:
        cursor = await db.execute(
            "UPDATE run_queue SET status = 'cancelled' WHERE id = ? AND user_id = ? AND status = 'queued'",
            (queue_id, user_id),
        )
        await db.commit()
    return cursor.rowcount > 0


async def process_queue() -> None:
    """Check for queued runs that can be started and launch them.

    Called whenever a workflow run completes or fails.
    """
    from exo_web.engine import execute_workflow

    async with get_db() as db:
        # Fetch oldest queued entries.
        cursor = await db.execute(
            "SELECT * FROM run_queue WHERE status = 'queued' ORDER BY queued_at ASC"
        )
        queued = await cursor.fetchall()

    for entry in queued:
        entry = dict(entry)
        workflow_id = entry["workflow_id"]

        if not await can_start_run(workflow_id):
            # If global limit hit, no point checking more entries.
            break

        # Start this queued run.
        run_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        nodes = json.loads(entry["nodes_json"])
        edges = json.loads(entry["edges_json"])

        async with get_db() as db:
            await db.execute(
                "INSERT INTO workflow_runs (id, workflow_id, status, trigger_type, user_id, created_at) VALUES (?, ?, 'pending', ?, ?, ?)",
                (run_id, workflow_id, entry["trigger_type"], entry["user_id"], now),
            )
            await db.execute(
                "UPDATE run_queue SET status = 'started', started_at = ?, run_id = ? WHERE id = ?",
                (now, run_id, entry["id"]),
            )
            await db.commit()

        _log.info(
            "Starting queued run %s for workflow %s (queue entry %s)",
            run_id,
            workflow_id,
            entry["id"],
        )

        task = asyncio.create_task(
            execute_workflow(run_id, workflow_id, entry["user_id"], nodes, edges)
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
