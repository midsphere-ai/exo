"""Background scheduler — polls for due cron schedules and triggers workflow runs."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from croniter import croniter

from exo_web.database import get_db
from exo_web.engine import execute_workflow

_log = logging.getLogger(__name__)

# Keep references to workflow-execution tasks so they aren't GC'd.
_background_tasks: set[asyncio.Task[Any]] = set()

# Polling interval in seconds.
_POLL_INTERVAL = 15

# Sentinel for the background polling loop.
_scheduler_task: asyncio.Task[Any] | None = None


def compute_next_run(cron_expression: str, after: datetime | None = None) -> str:
    """Return the next fire time as a UTC ISO string for a given cron expression."""
    base = after or datetime.now(UTC)
    # croniter expects a naive datetime in the target timezone.  We work in UTC.
    naive = base.replace(tzinfo=None)
    cron = croniter(cron_expression, naive)
    next_dt = cron.get_next(datetime)
    return next_dt.strftime("%Y-%m-%d %H:%M:%S")


async def _fire_schedule(schedule: dict[str, Any]) -> None:
    """Trigger a workflow run for a single due schedule."""
    schedule_id = schedule["id"]
    workflow_id = schedule["workflow_id"]
    user_id = schedule["user_id"]
    cron_expr = schedule["cron_expression"]

    _log.info("Firing schedule %s for workflow %s", schedule_id, workflow_id)

    async with get_db() as db:
        # Load the workflow to get nodes/edges.
        cursor = await db.execute(
            "SELECT nodes_json, edges_json FROM workflows WHERE id = ?",
            (workflow_id,),
        )
        wf = await cursor.fetchone()
        if wf is None:
            _log.warning(
                "Workflow %s not found for schedule %s — disabling", workflow_id, schedule_id
            )
            await db.execute("UPDATE schedules SET enabled = 0 WHERE id = ?", (schedule_id,))
            await db.commit()
            return

        nodes = json.loads(wf["nodes_json"] or "[]")
        edges = json.loads(wf["edges_json"] or "[]")

        if not nodes:
            _log.warning(
                "Workflow %s has no nodes — skipping schedule %s", workflow_id, schedule_id
            )
            now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
            next_run = compute_next_run(cron_expr)
            await db.execute(
                "UPDATE schedules SET last_run_at = ?, next_run_at = ?, updated_at = ? WHERE id = ?",
                (now, next_run, now, schedule_id),
            )
            await db.commit()
            return

        # Create the workflow run record.
        run_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            "INSERT INTO workflow_runs (id, workflow_id, status, trigger_type, user_id, created_at) VALUES (?, ?, 'pending', 'schedule', ?, ?)",
            (run_id, workflow_id, user_id, now),
        )

        # Update the schedule: last_run_at = now, compute next_run_at.
        next_run = compute_next_run(cron_expr)
        _log.debug("Next run for schedule %s: %s", schedule_id, next_run)
        await db.execute(
            "UPDATE schedules SET last_run_at = ?, next_run_at = ?, updated_at = ? WHERE id = ?",
            (now, next_run, now, schedule_id),
        )
        await db.commit()

    # Fire-and-forget: execute in the background.
    task = asyncio.create_task(execute_workflow(run_id, workflow_id, user_id, nodes, edges))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _poll_loop() -> None:
    """Continuously poll for due schedules and fire them."""
    while True:
        try:
            now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

            async with get_db() as db:
                cursor = await db.execute(
                    "SELECT * FROM schedules WHERE enabled = 1 AND next_run_at <= ?",
                    (now,),
                )
                due = await cursor.fetchall()

            for row in due:
                schedule = dict(row)
                try:
                    await _fire_schedule(schedule)
                except Exception:
                    _log.exception("Error firing schedule %s", schedule["id"])

        except Exception:
            _log.exception("Error in scheduler poll loop")

        await asyncio.sleep(_POLL_INTERVAL)


async def start_scheduler() -> None:
    """Start the background scheduler polling loop."""
    global _scheduler_task
    if _scheduler_task is not None:
        return
    _log.info("Starting workflow scheduler (poll every %ds)", _POLL_INTERVAL)
    _scheduler_task = asyncio.create_task(_poll_loop())


async def stop_scheduler() -> None:
    """Stop the background scheduler."""
    global _scheduler_task
    if _scheduler_task is None:
        return
    _log.info("Stopping workflow scheduler")
    _scheduler_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await _scheduler_task
    _scheduler_task = None
