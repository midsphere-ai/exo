"""Observability integration routes.

Manages external observability platform connections (Langfuse, LangSmith,
Datadog, Opik, Custom Webhook) for trace export.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from exo_web.crypto import decrypt_api_key, encrypt_api_key
from exo_web.database import get_db
from exo_web.routes.auth import get_current_user
from exo_web.sanitize import sanitize_html

router = APIRouter(prefix="/api/v1/observability", tags=["observability"])

_log = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = ("langfuse", "langsmith", "datadog", "opik", "custom_webhook")

# Default endpoint URLs per platform
_DEFAULT_ENDPOINTS: dict[str, str] = {
    "langfuse": "https://cloud.langfuse.com",
    "langsmith": "https://api.smith.langchain.com",
    "datadog": "https://api.datadoghq.com",
    "opik": "https://www.comet.com/opik/api",
    "custom_webhook": "",
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class IntegrationCreate(BaseModel):
    platform: str = Field(..., min_length=1, description="Platform")
    display_name: str = Field(..., min_length=1, max_length=255, description="Display name")
    endpoint_url: str = Field(default="", description="Endpoint url")
    api_key: str = Field(default="", description="API key (stored encrypted)")
    project_name: str = Field(default="", description="Project name")
    extra_config: dict[str, Any] = Field(default_factory=dict, description="Extra config")


class IntegrationUpdate(BaseModel):
    display_name: str | None = Field(None, description="Display name")
    endpoint_url: str | None = Field(None, description="Endpoint url")
    api_key: str | None = Field(None, description="API key (stored encrypted)")
    project_name: str | None = Field(None, description="Project name")
    enabled: bool | None = Field(None, description="Whether this item is active")
    extra_config: dict[str, Any] | None = Field(None, description="Extra config")


class IntegrationResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    platform: str = Field(description="Platform")
    display_name: str = Field(description="Display name")
    enabled: bool = Field(description="Whether this item is active")
    endpoint_url: str = Field(description="Endpoint url")
    api_key_set: bool = Field(description="Whether an API key is configured")
    project_name: str = Field(description="Project name")
    extra_config: dict[str, Any] = Field(description="Extra config")
    last_test_at: str | None = Field(None, description="Last test at")
    last_test_status: str | None = Field(None, description="Last test status")
    last_test_error: str | None = Field(None, description="Last test error")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_response(row: Any) -> dict[str, Any]:
    """Convert a DB row to an IntegrationResponse-compatible dict."""
    d = dict(row)
    d["enabled"] = bool(d.get("enabled", 0))
    d["api_key_set"] = bool(d.get("encrypted_api_key"))
    d.pop("encrypted_api_key", None)
    d.pop("user_id", None)
    extra = d.pop("extra_config_json", "{}")
    try:
        d["extra_config"] = json.loads(extra) if extra else {}
    except (json.JSONDecodeError, TypeError):
        d["extra_config"] = {}
    return d


async def _verify_ownership(db: Any, integration_id: str, user_id: str) -> dict[str, Any]:
    cursor = await db.execute(
        "SELECT * FROM observability_integrations WHERE id = ? AND user_id = ?",
        (integration_id, user_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Observability integration not found")
    return dict(row)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("", response_model=list[IntegrationResponse])
async def list_integrations(
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all observability integrations for the current user."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM observability_integrations WHERE user_id = ? ORDER BY created_at DESC",
            (user["id"],),
        )
        rows = await cursor.fetchall()
        return [_row_to_response(r) for r in rows]


@router.post("", response_model=IntegrationResponse, status_code=201)
async def create_integration(
    body: IntegrationCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new observability integration."""
    if body.platform not in SUPPORTED_PLATFORMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported platform: {body.platform}. Supported: {', '.join(SUPPORTED_PLATFORMS)}",
        )

    integration_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    endpoint_url = body.endpoint_url or _DEFAULT_ENDPOINTS.get(body.platform, "")
    encrypted_key = encrypt_api_key(body.api_key) if body.api_key else ""
    display_name = sanitize_html(body.display_name)
    project_name = sanitize_html(body.project_name) if body.project_name else ""

    async with get_db() as db:
        await db.execute(
            """INSERT INTO observability_integrations
               (id, user_id, platform, display_name, enabled, endpoint_url,
                encrypted_api_key, project_name, extra_config_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)""",
            (
                integration_id,
                user["id"],
                body.platform,
                display_name,
                endpoint_url,
                encrypted_key,
                project_name,
                json.dumps(body.extra_config),
                now,
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute(
            "SELECT * FROM observability_integrations WHERE id = ?",
            (integration_id,),
        )
        row = await cursor.fetchone()
        return _row_to_response(row)


@router.get("/{integration_id}", response_model=IntegrationResponse)
async def get_integration(
    integration_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Get a single observability integration."""
    async with get_db() as db:
        await _verify_ownership(db, integration_id, user["id"])
        cursor = await db.execute(
            "SELECT * FROM observability_integrations WHERE id = ?",
            (integration_id,),
        )
        row = await cursor.fetchone()
        return _row_to_response(row)


@router.put("/{integration_id}", response_model=IntegrationResponse)
async def update_integration(
    integration_id: str,
    body: IntegrationUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update an observability integration."""
    async with get_db() as db:
        await _verify_ownership(db, integration_id, user["id"])

        updates: list[str] = []
        params: list[Any] = []

        if body.display_name is not None:
            updates.append("display_name = ?")
            params.append(sanitize_html(body.display_name))
        if body.endpoint_url is not None:
            updates.append("endpoint_url = ?")
            params.append(body.endpoint_url)
        if body.api_key is not None:
            updates.append("encrypted_api_key = ?")
            params.append(encrypt_api_key(body.api_key) if body.api_key else "")
        if body.project_name is not None:
            updates.append("project_name = ?")
            params.append(sanitize_html(body.project_name))
        if body.enabled is not None:
            updates.append("enabled = ?")
            params.append(1 if body.enabled else 0)
        if body.extra_config is not None:
            updates.append("extra_config_json = ?")
            params.append(json.dumps(body.extra_config))

        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        updates.append("updated_at = ?")
        params.append(now)
        params.append(integration_id)

        set_clause = ", ".join(updates)
        await db.execute(
            f"UPDATE observability_integrations SET {set_clause} WHERE id = ?",
            params,
        )
        await db.commit()

        cursor = await db.execute(
            "SELECT * FROM observability_integrations WHERE id = ?",
            (integration_id,),
        )
        row = await cursor.fetchone()
        return _row_to_response(row)


@router.delete("/{integration_id}", status_code=204)
async def delete_integration(
    integration_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete an observability integration."""
    async with get_db() as db:
        await _verify_ownership(db, integration_id, user["id"])
        await db.execute(
            "DELETE FROM observability_integrations WHERE id = ?",
            (integration_id,),
        )
        await db.commit()


@router.post("/{integration_id}/test")
async def test_integration(
    integration_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Send a sample trace to the observability platform and report success/error."""
    async with get_db() as db:
        row = await _verify_ownership(db, integration_id, user["id"])

    platform = row["platform"]
    endpoint_url = row["endpoint_url"]
    encrypted_key = row["encrypted_api_key"]
    project_name = row["project_name"]
    extra_config = json.loads(row.get("extra_config_json") or "{}")

    api_key = decrypt_api_key(encrypted_key) if encrypted_key else ""

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    test_status = "success"
    test_error: str | None = None

    try:
        await _send_test_trace(platform, endpoint_url, api_key, project_name, extra_config)
    except Exception as exc:
        _log.warning("Observability test failed for %s: %s", integration_id, exc)
        test_status = "error"
        test_error = str(exc)

    # Persist test result
    async with get_db() as db:
        await db.execute(
            """UPDATE observability_integrations
               SET last_test_at = ?, last_test_status = ?, last_test_error = ?, updated_at = ?
               WHERE id = ?""",
            (now, test_status, test_error, now, integration_id),
        )
        await db.commit()

    return {
        "status": test_status,
        "error": test_error,
        "tested_at": now,
    }


# ---------------------------------------------------------------------------
# Test-trace senders per platform
# ---------------------------------------------------------------------------


async def _send_test_trace(
    platform: str,
    endpoint_url: str,
    api_key: str,
    project_name: str,
    extra_config: dict[str, Any],
) -> None:
    """Send a sample trace payload to the given observability platform."""
    import httpx

    sample_trace = {
        "name": "exo-test-trace",
        "input": {"message": "Hello from Exo"},
        "output": {"message": "Test response"},
        "model": "test-model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "latency_ms": 42,
        "status": "success",
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        if platform == "langfuse":
            await _test_langfuse(
                client, endpoint_url, api_key, project_name, extra_config, sample_trace
            )
        elif platform == "langsmith":
            await _test_langsmith(client, endpoint_url, api_key, project_name, sample_trace)
        elif platform == "datadog":
            await _test_datadog(client, endpoint_url, api_key, sample_trace)
        elif platform == "opik":
            await _test_opik(client, endpoint_url, api_key, project_name, sample_trace)
        elif platform == "custom_webhook":
            await _test_custom_webhook(client, endpoint_url, api_key, extra_config, sample_trace)
        else:
            msg = f"Unsupported platform: {platform}"
            raise ValueError(msg)


async def _test_langfuse(
    client: Any,
    endpoint_url: str,
    api_key: str,
    project_name: str,
    extra_config: dict[str, Any],
    trace: dict[str, Any],
) -> None:
    """Test Langfuse connection by posting a sample trace via their ingestion API."""
    public_key = extra_config.get("public_key", "")
    url = f"{endpoint_url.rstrip('/')}/api/public/ingestion"
    headers = {"Content-Type": "application/json"}
    if public_key and api_key:
        import base64

        creds = base64.b64encode(f"{public_key}:{api_key}".encode()).decode()
        headers["Authorization"] = f"Basic {creds}"

    payload = {
        "batch": [
            {
                "id": str(uuid.uuid4()),
                "type": "trace-create",
                "timestamp": datetime.now(UTC).isoformat(),
                "body": {
                    "name": trace["name"],
                    "input": trace["input"],
                    "output": trace["output"],
                    "metadata": {"source": "exo-test", "project": project_name},
                },
            }
        ],
    }

    resp = await client.post(url, json=payload, headers=headers)
    resp.raise_for_status()


async def _test_langsmith(
    client: Any,
    endpoint_url: str,
    api_key: str,
    project_name: str,
    trace: dict[str, Any],
) -> None:
    """Test LangSmith connection by posting a sample run."""
    url = f"{endpoint_url.rstrip('/')}/runs"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }

    payload = {
        "name": trace["name"],
        "run_type": "llm",
        "inputs": trace["input"],
        "outputs": trace["output"],
        "session_name": project_name or "exo-test",
        "start_time": datetime.now(UTC).isoformat(),
        "end_time": datetime.now(UTC).isoformat(),
    }

    resp = await client.post(url, json=payload, headers=headers)
    resp.raise_for_status()


async def _test_datadog(
    client: Any,
    endpoint_url: str,
    api_key: str,
    trace: dict[str, Any],
) -> None:
    """Test Datadog connection by sending a log entry."""
    url = f"{endpoint_url.rstrip('/')}/api/v2/logs"
    headers = {
        "Content-Type": "application/json",
        "DD-API-KEY": api_key,
    }

    payload = [
        {
            "ddsource": "exo",
            "service": "exo-agent",
            "message": json.dumps(trace),
            "ddtags": "env:test,source:exo",
        }
    ]

    resp = await client.post(url, json=payload, headers=headers)
    resp.raise_for_status()


async def _test_opik(
    client: Any,
    endpoint_url: str,
    api_key: str,
    project_name: str,
    trace: dict[str, Any],
) -> None:
    """Test Opik connection by posting a sample trace."""
    url = f"{endpoint_url.rstrip('/')}/v1/private/traces"
    headers = {
        "Content-Type": "application/json",
        "authorization": api_key,
    }
    if project_name:
        headers["Comet-Workspace"] = project_name

    payload = {
        "name": trace["name"],
        "input": trace["input"],
        "output": trace["output"],
        "start_time": datetime.now(UTC).isoformat(),
        "end_time": datetime.now(UTC).isoformat(),
    }

    resp = await client.post(url, json=payload, headers=headers)
    resp.raise_for_status()


async def _test_custom_webhook(
    client: Any,
    endpoint_url: str,
    api_key: str,
    extra_config: dict[str, Any],
    trace: dict[str, Any],
) -> None:
    """Test a custom webhook by POSTing a sample trace payload."""
    if not endpoint_url:
        msg = "Endpoint URL is required for custom webhook"
        raise ValueError(msg)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        header_name = extra_config.get("auth_header", "Authorization")
        header_value = (
            extra_config.get("auth_prefix", "Bearer") + " " + api_key
            if extra_config.get("auth_prefix")
            else api_key
        )
        headers[header_name] = header_value

    resp = await client.post(endpoint_url, json=trace, headers=headers)
    resp.raise_for_status()
