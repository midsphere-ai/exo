"""Alert rules and alerts REST API."""

from __future__ import annotations

import contextlib
import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user
from exo_web.sanitize import sanitize_html

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AlertRuleCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="Display name")
    condition_type: str = Field(
        ..., pattern=r"^(error_rate|latency|cost)$", description="Condition type"
    )
    condition_threshold: float = Field(..., gt=0, description="Condition threshold")
    action_type: str = Field(..., pattern=r"^(toast|email|webhook)$", description="Action type")
    action_config_json: dict[str, Any] | None = Field(None, description="Action config json")
    enabled: bool = Field(True, description="Whether this item is active")


class AlertRuleUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200, description="Display name")
    condition_type: str | None = Field(
        default=None, pattern=r"^(error_rate|latency|cost)$", description="Condition type"
    )
    condition_threshold: float | None = Field(default=None, gt=0, description="Condition threshold")
    action_type: str | None = Field(
        default=None, pattern=r"^(toast|email|webhook)$", description="Action type"
    )
    action_config_json: dict[str, Any] | None = Field(None, description="Action config json")
    enabled: bool | None = Field(None, description="Whether this item is active")


class AlertRuleResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    condition_type: str = Field(description="Condition type")
    condition_threshold: float = Field(description="Condition threshold")
    action_type: str = Field(description="Action type")
    action_config_json: dict[str, Any] | None = Field(None, description="Action config json")
    enabled: bool = Field(description="Whether this item is active")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


class AlertResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    rule_id: str = Field(description="Rule id")
    severity: str = Field(description="Severity")
    agent_id: str | None = Field(None, description="Associated agent identifier")
    message: str = Field(description="Message")
    acknowledged: bool = Field(description="Acknowledged")
    created_at: str = Field(description="ISO 8601 creation timestamp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_rule_row(row: Any) -> dict[str, Any]:
    """Convert a DB row to a dict, parsing action_config_json."""
    r = dict(row)
    if r.get("action_config_json"):
        with contextlib.suppress(json.JSONDecodeError):
            r["action_config_json"] = json.loads(r["action_config_json"])
    r["enabled"] = bool(r.get("enabled", 0))
    r.pop("user_id", None)
    return r


def _parse_alert_row(row: Any) -> dict[str, Any]:
    """Convert a DB row to a dict."""
    r = dict(row)
    r["acknowledged"] = bool(r.get("acknowledged", 0))
    r.pop("user_id", None)
    return r


# ---------------------------------------------------------------------------
# POST /api/alerts/rules — create an alert rule
# ---------------------------------------------------------------------------


@router.post("/rules", status_code=201, response_model=AlertRuleResponse)
async def create_alert_rule(
    body: AlertRuleCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new alert rule."""
    rule_id = str(uuid.uuid4())
    uid = user["id"]
    name = sanitize_html(body.name)
    action_config = json.dumps(body.action_config_json) if body.action_config_json else None

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO alert_rules (id, name, condition_type, condition_threshold,
                action_type, action_config_json, enabled, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rule_id,
                name,
                body.condition_type,
                body.condition_threshold,
                body.action_type,
                action_config,
                int(body.enabled),
                uid,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM alert_rules WHERE id = ?", (rule_id,))
        row = await cursor.fetchone()

    return _parse_rule_row(row)


# ---------------------------------------------------------------------------
# GET /api/alerts/rules — list alert rules
# ---------------------------------------------------------------------------


@router.get("/rules", response_model=list[AlertRuleResponse])
async def list_alert_rules(
    enabled: bool | None = None,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """List alert rules for the current user."""
    conditions = ["user_id = ?"]
    params: list[Any] = [user["id"]]

    if enabled is not None:
        conditions.append("enabled = ?")
        params.append(int(enabled))

    where = " AND ".join(conditions)

    async with get_db() as db:
        cursor = await db.execute(
            f"SELECT * FROM alert_rules WHERE {where} ORDER BY created_at DESC",
            params,
        )
        rows = await cursor.fetchall()

    return [_parse_rule_row(row) for row in rows]


# ---------------------------------------------------------------------------
# GET /api/alerts/rules/:id — get a single alert rule
# ---------------------------------------------------------------------------


@router.get("/rules/{rule_id}", response_model=AlertRuleResponse)
async def get_alert_rule(
    rule_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Get a single alert rule by ID."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM alert_rules WHERE id = ? AND user_id = ?",
            (rule_id, user["id"]),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    return _parse_rule_row(row)


# ---------------------------------------------------------------------------
# PUT /api/alerts/rules/:id — update an alert rule
# ---------------------------------------------------------------------------


@router.put("/rules/{rule_id}", response_model=AlertRuleResponse)
async def update_alert_rule(
    rule_id: str,
    body: AlertRuleUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update an existing alert rule."""
    uid = user["id"]

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM alert_rules WHERE id = ? AND user_id = ?",
            (rule_id, uid),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Alert rule not found")

        updates = body.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        if "name" in updates:
            updates["name"] = sanitize_html(updates["name"])
        if "action_config_json" in updates:
            updates["action_config_json"] = json.dumps(updates["action_config_json"])
        if "enabled" in updates:
            updates["enabled"] = int(updates["enabled"])

        set_clauses = [f"{k} = ?" for k in updates]
        set_clauses.append("updated_at = datetime('now')")
        values = [*updates.values(), rule_id, uid]

        await db.execute(
            f"UPDATE alert_rules SET {', '.join(set_clauses)} WHERE id = ? AND user_id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM alert_rules WHERE id = ?", (rule_id,))
        row = await cursor.fetchone()

    return _parse_rule_row(row)


# ---------------------------------------------------------------------------
# DELETE /api/alerts/rules/:id — delete an alert rule
# ---------------------------------------------------------------------------


@router.delete("/rules/{rule_id}", status_code=204)
async def delete_alert_rule(
    rule_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete an alert rule and its associated alerts."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM alert_rules WHERE id = ? AND user_id = ?",
            (rule_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Alert rule not found")

        await db.execute("DELETE FROM alert_rules WHERE id = ?", (rule_id,))
        await db.commit()


# ---------------------------------------------------------------------------
# GET /api/alerts — list alerts
# ---------------------------------------------------------------------------


@router.get("", response_model=list[AlertResponse])
async def list_alerts(
    severity: str | None = Query(default=None),
    agent_id: str | None = Query(default=None),
    acknowledged: bool | None = None,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """List alerts with optional filtering."""
    conditions = ["a.user_id = ?"]
    params: list[Any] = [user["id"]]

    if severity:
        conditions.append("a.severity = ?")
        params.append(severity)
    if agent_id:
        conditions.append("a.agent_id = ?")
        params.append(agent_id)
    if acknowledged is not None:
        conditions.append("a.acknowledged = ?")
        params.append(int(acknowledged))

    where = " AND ".join(conditions)
    params.extend([limit, offset])

    async with get_db() as db:
        cursor = await db.execute(
            f"SELECT a.* FROM alerts a WHERE {where} ORDER BY a.created_at DESC LIMIT ? OFFSET ?",
            params,
        )
        rows = await cursor.fetchall()

    return [_parse_alert_row(row) for row in rows]


# ---------------------------------------------------------------------------
# PUT /api/alerts/:id/acknowledge — acknowledge an alert
# ---------------------------------------------------------------------------


@router.put("/{alert_id}/acknowledge", response_model=AlertResponse)
async def acknowledge_alert(
    alert_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Mark an alert as acknowledged."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM alerts WHERE id = ? AND user_id = ?",
            (alert_id, user["id"]),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Alert not found")

        await db.execute(
            "UPDATE alerts SET acknowledged = 1 WHERE id = ?",
            (alert_id,),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,))
        row = await cursor.fetchone()

    return _parse_alert_row(row)
