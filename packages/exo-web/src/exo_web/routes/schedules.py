"""Schedule/cron trigger CRUD REST API."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from croniter import croniter
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user
from exo_web.services.scheduler import compute_next_run

router = APIRouter(prefix="/api/v1/schedules", tags=["schedules"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ScheduleCreate(BaseModel):
    workflow_id: str = Field(..., min_length=1, description="Associated workflow identifier")
    cron_expression: str = Field(
        ..., min_length=1, max_length=255, description="Cron schedule expression"
    )
    timezone: str = Field(default="UTC", max_length=64, description="Timezone")
    enabled: bool = Field(True, description="Whether this item is active")


class ScheduleUpdate(BaseModel):
    cron_expression: str | None = Field(
        None, min_length=1, max_length=255, description="Cron schedule expression"
    )
    timezone: str | None = Field(None, max_length=64, description="Timezone")
    enabled: bool | None = Field(None, description="Whether this item is active")


class ScheduleResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    workflow_id: str = Field(description="Associated workflow identifier")
    user_id: str = Field(description="Owning user identifier")
    cron_expression: str = Field(description="Cron schedule expression")
    timezone: str = Field(description="Timezone")
    enabled: bool = Field(description="Whether this item is active")
    last_run_at: str | None = Field(description="Last execution timestamp")
    next_run_at: str | None = Field(description="Next scheduled run timestamp")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    d = dict(row)
    # SQLite stores enabled as int; convert to bool for JSON.
    d["enabled"] = bool(d.get("enabled", 0))
    return d


def _validate_cron(expr: str) -> None:
    """Raise 422 if cron expression is invalid."""
    if not croniter.is_valid(expr):
        raise HTTPException(status_code=422, detail=f"Invalid cron expression: {expr}")


async def _verify_ownership(db: Any, schedule_id: str, user_id: str) -> dict[str, Any]:
    """Return schedule row or raise 404."""
    cursor = await db.execute(
        "SELECT * FROM schedules WHERE id = ? AND user_id = ?",
        (schedule_id, user_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("")
async def list_schedules(
    workflow_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """List schedules for the current user, optionally filtered by workflow."""
    async with get_db() as db:
        conditions = ["user_id = ?"]
        params: list[Any] = [user["id"]]

        if workflow_id:
            conditions.append("workflow_id = ?")
            params.append(workflow_id)

        where = " AND ".join(conditions)
        cursor = await db.execute(
            f"SELECT * FROM schedules WHERE {where} ORDER BY created_at DESC",
            params,
        )
        rows = await cursor.fetchall()

    return [_row_to_dict(r) for r in rows]


@router.post("", response_model=ScheduleResponse, status_code=201)
async def create_schedule(
    body: ScheduleCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a schedule for a workflow."""
    _validate_cron(body.cron_expression)

    async with get_db() as db:
        # Verify the workflow exists and belongs to this user.
        cursor = await db.execute(
            "SELECT id FROM workflows WHERE id = ? AND user_id = ?",
            (body.workflow_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Workflow not found")

        schedule_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        next_run = compute_next_run(body.cron_expression) if body.enabled else None

        await db.execute(
            """
            INSERT INTO schedules (id, workflow_id, user_id, cron_expression, timezone, enabled, next_run_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                schedule_id,
                body.workflow_id,
                user["id"],
                body.cron_expression,
                body.timezone,
                int(body.enabled),
                next_run,
                now,
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM schedules WHERE id = ?", (schedule_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.get("/{schedule_id}", response_model=ScheduleResponse)
async def get_schedule(
    schedule_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single schedule by ID."""
    async with get_db() as db:
        return await _verify_ownership(db, schedule_id, user["id"])


@router.put("/{schedule_id}", response_model=ScheduleResponse)
async def update_schedule(
    schedule_id: str,
    body: ScheduleUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a schedule's cron expression, timezone, or enabled status."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    if "cron_expression" in updates:
        _validate_cron(updates["cron_expression"])

    async with get_db() as db:
        existing = await _verify_ownership(db, schedule_id, user["id"])

        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        updates["updated_at"] = now

        # Convert bool -> int for SQLite.
        if "enabled" in updates:
            updates["enabled"] = int(updates["enabled"])

        # Recompute next_run_at if cron or enabled changed.
        cron_expr = updates.get("cron_expression", existing["cron_expression"])
        is_enabled = bool(updates.get("enabled", existing["enabled"]))
        if is_enabled:
            updates["next_run_at"] = compute_next_run(cron_expr)
        else:
            updates["next_run_at"] = None

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), schedule_id]

        await db.execute(
            f"UPDATE schedules SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM schedules WHERE id = ?", (schedule_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.delete("/{schedule_id}", status_code=204)
async def delete_schedule(
    schedule_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a schedule."""
    async with get_db() as db:
        await _verify_ownership(db, schedule_id, user["id"])
        await db.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
        await db.commit()
