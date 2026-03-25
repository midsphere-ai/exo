"""Webhook triggers and notification templates REST API."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import secrets
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user

router = APIRouter(tags=["webhooks"])

_log = logging.getLogger(__name__)

# Keep references to background tasks so they aren't GC'd.
_background_tasks: set[asyncio.Task[Any]] = set()

_MAX_REQUEST_LOG_ENTRIES = 50


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class WebhookCreate(BaseModel):
    workflow_id: str = Field(..., min_length=1, description="Associated workflow identifier")
    hook_id: str = Field(..., min_length=1, description="Hook id")


class WebhookUpdate(BaseModel):
    enabled: bool | None = Field(None, description="Whether this item is active")


class WebhookResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    workflow_id: str = Field(description="Associated workflow identifier")
    hook_id: str = Field(description="Hook id")
    url_token: str = Field(description="Url token")
    webhook_url: str = Field(description="Webhook url")
    enabled: bool = Field(description="Whether this item is active")
    request_log: list[dict[str, Any]] = Field(description="Request log")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


class NotificationTemplateCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    type: str = Field(..., pattern=r"^(slack|discord|email)$", description="Type")
    config_json: dict[str, Any] = Field(
        default_factory=dict, description="JSON configuration object"
    )


class NotificationTemplateUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255, description="Display name")
    type: str | None = Field(None, pattern=r"^(slack|discord|email)$", description="Type")
    config_json: dict[str, Any] | None = Field(None, description="JSON configuration object")


class NotificationTemplateResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    type: str = Field(description="Type")
    config_json: dict[str, Any] = Field(description="JSON configuration object")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_url_token() -> str:
    """Generate a secure random URL token."""
    return secrets.token_urlsafe(32)


def _build_webhook_url(workflow_id: str, hook_id: str) -> str:
    """Build the webhook URL path."""
    return f"/api/v1/webhooks/{workflow_id}/{hook_id}"


def _parse_webhook_row(row: Any) -> dict[str, Any]:
    """Convert a DB row to a response-ready dict."""
    r = dict(row)
    r["enabled"] = bool(r.get("enabled", 0))
    r["request_log"] = json.loads(r.get("request_log_json", "[]"))
    r.pop("request_log_json", None)
    r.pop("user_id", None)
    r["webhook_url"] = _build_webhook_url(r["workflow_id"], r["hook_id"])
    return r


def _parse_template_row(row: Any) -> dict[str, Any]:
    """Convert a DB row to a response-ready dict."""
    r = dict(row)
    r["config_json"] = json.loads(r.get("config_json", "{}"))
    r.pop("user_id", None)
    return r


async def _append_request_log(
    webhook_id: str, payload: dict[str, Any], status: str, response_status: int
) -> None:
    """Append a request entry to the webhook's request log (capped at N entries)."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT request_log_json FROM webhooks WHERE id = ?", (webhook_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return

        log: list[dict[str, Any]] = json.loads(row["request_log_json"] or "[]")
        log.append(
            {
                "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
                "payload_preview": json.dumps(payload)[:500],
                "status": status,
                "response_status": response_status,
            }
        )
        # Keep only the last N entries.
        log = log[-_MAX_REQUEST_LOG_ENTRIES:]

        await db.execute(
            "UPDATE webhooks SET request_log_json = ?, updated_at = datetime('now') WHERE id = ?",
            (json.dumps(log), webhook_id),
        )
        await db.commit()


# ---------------------------------------------------------------------------
# POST /api/v1/webhooks/:workflowId/:hookId — trigger webhook (no auth)
# ---------------------------------------------------------------------------


@router.post("/api/v1/webhooks/{workflow_id}/{hook_id}")
async def trigger_webhook(
    workflow_id: str,
    hook_id: str,
    request: Request,
    url_token: str | None = Query(default=None),
) -> dict[str, Any]:
    """Receive an incoming webhook POST and trigger workflow execution.

    This endpoint is unauthenticated — it validates via the url_token stored
    in the webhooks table matching the workflow_id/hook_id pair.
    """
    from exo_web.engine import execute_workflow
    from exo_web.services.run_queue import can_start_run, enqueue_run

    # Parse request body.
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    # Look up the webhook.
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM webhooks WHERE workflow_id = ? AND hook_id = ?",
            (workflow_id, hook_id),
        )
        webhook = await cursor.fetchone()

    if webhook is None:
        raise HTTPException(status_code=404, detail="Webhook not found")

    webhook = dict(webhook)

    # Validate url_token using constant-time comparison to prevent timing attacks.
    stored_token: str = webhook.get("url_token") or ""
    if not url_token or not hmac.compare_digest(url_token, stored_token):
        await _append_request_log(webhook["id"], payload, "rejected_invalid_token", 403)
        raise HTTPException(status_code=403, detail="Invalid token")

    if not webhook["enabled"]:
        await _append_request_log(webhook["id"], payload, "rejected_disabled", 403)
        raise HTTPException(status_code=403, detail="Webhook is disabled")

    # Load the workflow.
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
        wf_row = await cursor.fetchone()

    if wf_row is None:
        await _append_request_log(webhook["id"], payload, "rejected_no_workflow", 404)
        raise HTTPException(status_code=404, detail="Workflow not found")

    wf = dict(wf_row)
    nodes = json.loads(wf["nodes_json"] or "[]")
    edges = json.loads(wf["edges_json"] or "[]")

    if not nodes:
        await _append_request_log(webhook["id"], payload, "rejected_empty", 422)
        raise HTTPException(status_code=422, detail="Workflow has no nodes")

    user_id = wf["user_id"]

    # Check concurrency limits.
    if not await can_start_run(workflow_id):
        result = await enqueue_run(workflow_id, user_id, nodes, edges, trigger_type="webhook")
        await _append_request_log(webhook["id"], payload, "queued", 202)
        return {
            "status": "queued",
            "queue_id": result["queue_id"],
            "position": result["position"],
        }

    # Create a run and start execution.
    run_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        await db.execute(
            "INSERT INTO workflow_runs (id, workflow_id, status, trigger_type, input_json, user_id, created_at) VALUES (?, ?, 'pending', 'webhook', ?, ?, ?)",
            (run_id, workflow_id, json.dumps(payload), user_id, now),
        )
        await db.commit()

    task = asyncio.create_task(execute_workflow(run_id, workflow_id, user_id, nodes, edges))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    await _append_request_log(webhook["id"], payload, "triggered", 200)

    _log.info(
        "Webhook triggered: workflow=%s hook=%s run=%s",
        workflow_id,
        hook_id,
        run_id,
    )

    return {"status": "triggered", "run_id": run_id}


# ---------------------------------------------------------------------------
# Webhook CRUD (authenticated)
# ---------------------------------------------------------------------------


@router.get("/api/v1/webhook-configs", response_model=list[WebhookResponse])
async def list_webhooks(
    workflow_id: str | None = Query(default=None),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """List webhooks, optionally filtered by workflow_id."""
    async with get_db() as db:
        if workflow_id:
            cursor = await db.execute(
                "SELECT * FROM webhooks WHERE user_id = ? AND workflow_id = ? ORDER BY created_at DESC",
                (user["id"], workflow_id),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM webhooks WHERE user_id = ? ORDER BY created_at DESC",
                (user["id"],),
            )
        rows = await cursor.fetchall()

    return [_parse_webhook_row(row) for row in rows]


@router.post(
    "/api/v1/webhook-configs",
    response_model=WebhookResponse,
    status_code=201,
)
async def create_webhook(
    body: WebhookCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new webhook trigger for a workflow."""
    # Verify workflow ownership.
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM workflows WHERE id = ? AND user_id = ?",
            (body.workflow_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Check for duplicate hook_id on this workflow.
        cursor = await db.execute(
            "SELECT id FROM webhooks WHERE workflow_id = ? AND hook_id = ?",
            (body.workflow_id, body.hook_id),
        )
        if await cursor.fetchone() is not None:
            raise HTTPException(
                status_code=409,
                detail="Webhook with this hook_id already exists for this workflow",
            )

        webhook_id = str(uuid.uuid4())
        url_token = _generate_url_token()
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """INSERT INTO webhooks (id, workflow_id, hook_id, url_token, enabled, request_log_json, user_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, 1, '[]', ?, ?, ?)""",
            (webhook_id, body.workflow_id, body.hook_id, url_token, user["id"], now, now),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM webhooks WHERE id = ?", (webhook_id,))
        row = await cursor.fetchone()

    return _parse_webhook_row(row)


@router.get("/api/v1/webhook-configs/{webhook_id}", response_model=WebhookResponse)
async def get_webhook(
    webhook_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Get a single webhook by ID."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM webhooks WHERE id = ? AND user_id = ?",
            (webhook_id, user["id"]),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Webhook not found")

    return _parse_webhook_row(row)


@router.put("/api/v1/webhook-configs/{webhook_id}", response_model=WebhookResponse)
async def update_webhook(
    webhook_id: str,
    body: WebhookUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a webhook (enable/disable)."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM webhooks WHERE id = ? AND user_id = ?",
            (webhook_id, user["id"]),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Webhook not found")

        updates = body.model_dump(exclude_none=True)
        if not updates:
            return _parse_webhook_row(row)

        set_parts = [f"{k} = ?" for k in updates]
        set_parts.append("updated_at = datetime('now')")
        values = list(updates.values())
        values.append(webhook_id)

        await db.execute(
            f"UPDATE webhooks SET {', '.join(set_parts)} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM webhooks WHERE id = ?", (webhook_id,))
        row = await cursor.fetchone()

    return _parse_webhook_row(row)


@router.delete("/api/v1/webhook-configs/{webhook_id}", status_code=204)
async def delete_webhook(
    webhook_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a webhook."""
    async with get_db() as db:
        cursor = await db.execute(
            "DELETE FROM webhooks WHERE id = ? AND user_id = ?",
            (webhook_id, user["id"]),
        )
        await db.commit()

    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Webhook not found")


@router.post("/api/v1/webhook-configs/{webhook_id}/regenerate", response_model=WebhookResponse)
async def regenerate_webhook_token(
    webhook_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Regenerate the URL token for a webhook."""
    new_token = _generate_url_token()

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM webhooks WHERE id = ? AND user_id = ?",
            (webhook_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Webhook not found")

        await db.execute(
            "UPDATE webhooks SET url_token = ?, updated_at = datetime('now') WHERE id = ?",
            (new_token, webhook_id),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM webhooks WHERE id = ?", (webhook_id,))
        row = await cursor.fetchone()

    return _parse_webhook_row(row)


@router.get(
    "/api/v1/webhook-configs/{webhook_id}/request-log",
)
async def get_webhook_request_log(
    webhook_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Get the request log for a webhook."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT request_log_json FROM webhooks WHERE id = ? AND user_id = ?",
            (webhook_id, user["id"]),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Webhook not found")

    log: list[dict[str, Any]] = json.loads(row["request_log_json"] or "[]")
    # Return most recent first.
    log.reverse()
    return log[:limit]


# ---------------------------------------------------------------------------
# Notification templates CRUD
# ---------------------------------------------------------------------------

# Default templates for each notification type.
NOTIFICATION_TEMPLATE_DEFAULTS: dict[str, dict[str, Any]] = {
    "slack": {
        "webhook_url": "",
        "channel": "",
        "username": "Exo",
        "icon_emoji": ":rocket:",
        "message_template": "Workflow *{{workflow_name}}* completed with status: {{status}}",
    },
    "discord": {
        "webhook_url": "",
        "username": "Exo",
        "avatar_url": "",
        "message_template": "Workflow **{{workflow_name}}** completed with status: {{status}}",
    },
    "email": {
        "smtp_host": "",
        "smtp_port": 587,
        "smtp_user": "",
        "smtp_password": "",
        "use_tls": True,
        "from_address": "",
        "to_addresses": [],
        "subject_template": "Exo: {{workflow_name}} — {{status}}",
        "body_template": "Workflow {{workflow_name}} completed with status: {{status}}",
    },
}


@router.get("/api/v1/notification-templates/defaults")
async def get_notification_defaults() -> dict[str, dict[str, Any]]:
    """Return default config schemas for each notification template type."""
    return NOTIFICATION_TEMPLATE_DEFAULTS


@router.get(
    "/api/v1/notification-templates",
    response_model=list[NotificationTemplateResponse],
)
async def list_notification_templates(
    type: str | None = Query(default=None),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """List notification templates, optionally filtered by type."""
    async with get_db() as db:
        if type:
            cursor = await db.execute(
                "SELECT * FROM notification_templates WHERE user_id = ? AND type = ? ORDER BY created_at DESC",
                (user["id"], type),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM notification_templates WHERE user_id = ? ORDER BY created_at DESC",
                (user["id"],),
            )
        rows = await cursor.fetchall()

    return [_parse_template_row(row) for row in rows]


@router.post(
    "/api/v1/notification-templates",
    response_model=NotificationTemplateResponse,
    status_code=201,
)
async def create_notification_template(
    body: NotificationTemplateCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new notification template."""
    template_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        await db.execute(
            """INSERT INTO notification_templates (id, name, type, config_json, user_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                template_id,
                body.name,
                body.type,
                json.dumps(body.config_json),
                user["id"],
                now,
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute(
            "SELECT * FROM notification_templates WHERE id = ?", (template_id,)
        )
        row = await cursor.fetchone()

    return _parse_template_row(row)


@router.get(
    "/api/v1/notification-templates/{template_id}",
    response_model=NotificationTemplateResponse,
)
async def get_notification_template(
    template_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Get a notification template by ID."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM notification_templates WHERE id = ? AND user_id = ?",
            (template_id, user["id"]),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Notification template not found")

    return _parse_template_row(row)


@router.put(
    "/api/v1/notification-templates/{template_id}",
    response_model=NotificationTemplateResponse,
)
async def update_notification_template(
    template_id: str,
    body: NotificationTemplateUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a notification template."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM notification_templates WHERE id = ? AND user_id = ?",
            (template_id, user["id"]),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Notification template not found")

        updates: dict[str, Any] = {}
        if body.name is not None:
            updates["name"] = body.name
        if body.type is not None:
            updates["type"] = body.type
        if body.config_json is not None:
            updates["config_json"] = json.dumps(body.config_json)

        if not updates:
            return _parse_template_row(row)

        set_parts = [f"{k} = ?" for k in updates]
        set_parts.append("updated_at = datetime('now')")
        values = list(updates.values())
        values.append(template_id)

        await db.execute(
            f"UPDATE notification_templates SET {', '.join(set_parts)} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute(
            "SELECT * FROM notification_templates WHERE id = ?", (template_id,)
        )
        row = await cursor.fetchone()

    return _parse_template_row(row)


@router.delete("/api/v1/notification-templates/{template_id}", status_code=204)
async def delete_notification_template(
    template_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a notification template."""
    async with get_db() as db:
        cursor = await db.execute(
            "DELETE FROM notification_templates WHERE id = ? AND user_id = ?",
            (template_id, user["id"]),
        )
        await db.commit()

    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Notification template not found")
