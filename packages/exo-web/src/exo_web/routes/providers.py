"""Provider CRUD REST API."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from exo_web.crypto import decrypt_api_key, encrypt_api_key
from exo_web.database import get_db
from exo_web.routes.auth import require_role

router = APIRouter(prefix="/api/v1/providers", tags=["providers"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_PROVIDER_TYPES = ("openai", "anthropic", "gemini", "vertex", "ollama", "custom")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ProviderCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    provider_type: str = Field(
        ..., min_length=1, description="Provider type (openai, anthropic, etc.)"
    )
    api_key: str | None = Field(None, description="API key (stored encrypted)")
    base_url: str | None = Field(None, description="Provider base URL override")
    max_retries: int = Field(3, description="Maximum retry attempts")
    timeout: int = Field(30, description="Request timeout in seconds")


class ProviderUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255, description="Display name")
    provider_type: str | None = Field(None, description="Provider type (openai, anthropic, etc.)")
    api_key: str | None = Field(None, description="API key (stored encrypted)")
    base_url: str | None = Field(None, description="Provider base URL override")
    max_retries: int | None = Field(None, description="Maximum retry attempts")
    timeout: int | None = Field(None, description="Request timeout in seconds")


class ProviderResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    provider_type: str = Field(description="Provider type (openai, anthropic, etc.)")
    api_key_set: bool = Field(description="Whether an API key is configured")
    base_url: str | None = Field(description="Provider base URL override")
    max_retries: int = Field(description="Maximum retry attempts")
    timeout: int = Field(description="Request timeout in seconds")
    load_balance_strategy: str = Field(
        description="Key selection strategy (round_robin, random, least_used)"
    )
    user_id: str = Field(description="Owning user identifier")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


class TestResult(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    error: str | None = Field(None, description="Error message if failed")
    latency_ms: float = Field(description="Response latency in milliseconds")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mask_key(encrypted_key: str | None) -> bool:
    """Return whether an API key is set (never expose actual key)."""
    return encrypted_key is not None and encrypted_key != ""


def _row_to_response(row: Any) -> dict[str, Any]:
    """Convert a DB row to a ProviderResponse-compatible dict."""
    d = dict(row)
    d["api_key_set"] = _mask_key(d.pop("encrypted_api_key", None))
    return d


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[ProviderResponse])
async def list_providers(
    user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all providers for the current user."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM providers WHERE user_id = ? ORDER BY created_at DESC",
            (user["id"],),
        )
        rows = await cursor.fetchall()
        return [_row_to_response(r) for r in rows]


@router.post("", response_model=ProviderResponse, status_code=201)
async def create_provider(
    body: ProviderCreate,
    user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> dict[str, Any]:
    """Create a new provider."""
    if body.provider_type not in VALID_PROVIDER_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid provider_type. Must be one of: {', '.join(VALID_PROVIDER_TYPES)}",
        )

    provider_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    encrypted_key = encrypt_api_key(body.api_key) if body.api_key else None

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO providers (id, name, provider_type, encrypted_api_key, base_url,
                                   max_retries, timeout, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                provider_id,
                body.name,
                body.provider_type,
                encrypted_key,
                body.base_url,
                body.max_retries,
                body.timeout,
                user["id"],
                now,
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM providers WHERE id = ?", (provider_id,))
        row = await cursor.fetchone()
        return _row_to_response(row)


@router.get("/{provider_id}", response_model=ProviderResponse)
async def get_provider(
    provider_id: str,
    user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> dict[str, Any]:
    """Return a single provider by ID."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM providers WHERE id = ? AND user_id = ?",
            (provider_id, user["id"]),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Provider not found")
        return _row_to_response(row)


@router.put("/{provider_id}", response_model=ProviderResponse)
async def update_provider(
    provider_id: str,
    body: ProviderUpdate,
    user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> dict[str, Any]:
    """Update a provider's editable fields."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    # Validate provider_type if being changed.
    if "provider_type" in updates and updates["provider_type"] not in VALID_PROVIDER_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid provider_type. Must be one of: {', '.join(VALID_PROVIDER_TYPES)}",
        )

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM providers WHERE id = ? AND user_id = ?",
            (provider_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Provider not found")

        # Encrypt api_key if provided.
        if "api_key" in updates:
            raw_key = updates.pop("api_key")
            updates["encrypted_api_key"] = encrypt_api_key(raw_key) if raw_key else None

        updates["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), provider_id]

        await db.execute(
            f"UPDATE providers SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM providers WHERE id = ?", (provider_id,))
        row = await cursor.fetchone()
        return _row_to_response(row)


@router.delete("/{provider_id}", status_code=204)
async def delete_provider(
    provider_id: str,
    user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> None:
    """Delete a provider."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM providers WHERE id = ? AND user_id = ?",
            (provider_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Provider not found")

        await db.execute("DELETE FROM providers WHERE id = ?", (provider_id,))
        await db.commit()


@router.post("/{provider_id}/test", response_model=TestResult)
async def test_provider(
    provider_id: str,
    user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> dict[str, Any]:
    """Test a provider's API key by making a lightweight API call."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM providers WHERE id = ? AND user_id = ?",
            (provider_id, user["id"]),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Provider not found")

    provider = dict(row)
    if not provider["encrypted_api_key"]:
        return {"success": False, "error": "No API key configured", "latency_ms": 0}

    api_key = decrypt_api_key(provider["encrypted_api_key"])
    provider_type = provider["provider_type"]
    base_url = provider["base_url"]

    return await _test_provider_connection(provider_type, api_key, base_url, provider["timeout"])


async def _test_provider_connection(
    provider_type: str,
    api_key: str,
    base_url: str | None,
    timeout: int,
) -> dict[str, Any]:
    """Make a lightweight API call to validate the provider connection."""
    import httpx

    start = time.monotonic()

    try:
        if provider_type == "openai":
            url = f"{base_url}/v1/models" if base_url else "https://api.openai.com/v1/models"
            headers = {"Authorization": f"Bearer {api_key}"}
        elif provider_type == "anthropic":
            url = f"{base_url}/v1/models" if base_url else "https://api.anthropic.com/v1/models"
            headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        elif provider_type == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            headers = {}
        elif provider_type == "vertex":
            # Vertex uses service account auth — just check that the base_url is reachable.
            if not base_url:
                elapsed = (time.monotonic() - start) * 1000
                return {
                    "success": False,
                    "error": "Vertex requires a base_url",
                    "latency_ms": elapsed,
                }
            url = base_url
            headers = {"Authorization": f"Bearer {api_key}"}
        elif provider_type == "ollama":
            url = f"{base_url}/api/tags" if base_url else "http://localhost:11434/api/tags"
            headers = {}
        else:
            # Custom provider — check base_url is reachable.
            if not base_url:
                elapsed = (time.monotonic() - start) * 1000
                return {
                    "success": False,
                    "error": "Custom provider requires a base_url",
                    "latency_ms": elapsed,
                }
            url = base_url
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)

        elapsed = (time.monotonic() - start) * 1000

        if resp.status_code < 400:
            return {"success": True, "error": None, "latency_ms": round(elapsed, 1)}
        return {
            "success": False,
            "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            "latency_ms": round(elapsed, 1),
        }

    except httpx.TimeoutException:
        elapsed = (time.monotonic() - start) * 1000
        return {"success": False, "error": "Connection timed out", "latency_ms": round(elapsed, 1)}
    except Exception as exc:
        elapsed = (time.monotonic() - start) * 1000
        return {"success": False, "error": str(exc)[:200], "latency_ms": round(elapsed, 1)}
