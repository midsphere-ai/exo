"""Provider keys CRUD and load balancing REST API."""

from __future__ import annotations

import random as rand_module
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from exo_web.crypto import decrypt_api_key, encrypt_api_key
from exo_web.database import get_db
from exo_web.routes.auth import require_role
from exo_web.services.audit import audit_log

router = APIRouter(prefix="/api/v1/providers", tags=["provider-keys"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STRATEGIES = ("round_robin", "random", "least_recently_used")
VALID_KEY_STATUSES = ("active", "rate_limited", "invalid")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class KeyCreate(BaseModel):
    api_key: str = Field(..., min_length=1, description="API key (stored encrypted)")
    label: str = Field("", description="Label")


class KeyResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    provider_id: str = Field(description="Associated provider identifier")
    label: str = Field(description="Label")
    strategy_position: int = Field(description="Strategy position")
    status: str = Field(description="Current status")
    total_requests: int = Field(description="Total requests")
    total_tokens: int = Field(description="Total token count")
    error_count: int = Field(description="Error count")
    last_used: str | None = Field(description="Last used")
    cooldown_until: str | None = Field(description="Cooldown until")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


class KeySelectResponse(BaseModel):
    """Response from the key selection / load-balancing endpoint."""

    key_id: str = Field(description="Key id")
    api_key: str = Field(description="API key (stored encrypted)")
    strategy: str = Field(description="Strategy")


class StrategyUpdate(BaseModel):
    strategy: str = Field(..., min_length=1, description="Strategy")


class KeyUsageReport(BaseModel):
    """Report usage stats after an API call."""

    tokens_used: int = Field(0, description="Tokens used")
    error: bool = Field(False, description="Error message if failed")
    rate_limited: bool = Field(False, description="Rate limited")
    cooldown_seconds: int = Field(300, description="Cooldown seconds")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_key_response(row: Any) -> dict[str, Any]:
    """Convert a DB row to a KeyResponse-compatible dict (no encrypted key)."""
    d = dict(row)
    d.pop("encrypted_key", None)
    return d


async def _verify_provider_ownership(provider_id: str, user_id: str) -> None:
    """Raise 404 if the provider doesn't exist or doesn't belong to user."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM providers WHERE id = ? AND user_id = ?",
            (provider_id, user_id),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Provider not found")


# ---------------------------------------------------------------------------
# Key CRUD endpoints
# ---------------------------------------------------------------------------


@router.get("/{provider_id}/keys", response_model=list[KeyResponse])
async def list_keys(
    provider_id: str,
    user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> list[dict[str, Any]]:
    """List all keys for a provider."""
    await _verify_provider_ownership(provider_id, user["id"])

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM provider_keys WHERE provider_id = ? ORDER BY strategy_position",
            (provider_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_key_response(r) for r in rows]


@router.post("/{provider_id}/keys", response_model=KeyResponse, status_code=201)
async def create_key(
    provider_id: str,
    body: KeyCreate,
    user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> dict[str, Any]:
    """Add a new API key to a provider."""
    await _verify_provider_ownership(provider_id, user["id"])

    key_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    encrypted = encrypt_api_key(body.api_key)

    async with get_db() as db:
        # Determine next strategy_position.
        cursor = await db.execute(
            "SELECT COALESCE(MAX(strategy_position), -1) + 1 FROM provider_keys WHERE provider_id = ?",
            (provider_id,),
        )
        row = await cursor.fetchone()
        position = row[0]

        await db.execute(
            """
            INSERT INTO provider_keys (id, provider_id, encrypted_key, label, strategy_position,
                                       created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (key_id, provider_id, encrypted, body.label, position, now, now),
        )
        await db.commit()

        await audit_log(
            user["id"],
            "add_provider_key",
            "provider_key",
            key_id,
            details={"provider_id": provider_id, "label": body.label},
        )

        cursor = await db.execute("SELECT * FROM provider_keys WHERE id = ?", (key_id,))
        row = await cursor.fetchone()
        return _row_to_key_response(row)


@router.delete("/{provider_id}/keys/{key_id}", status_code=204)
async def delete_key(
    provider_id: str,
    key_id: str,
    user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> None:
    """Delete an API key from a provider."""
    await _verify_provider_ownership(provider_id, user["id"])

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM provider_keys WHERE id = ? AND provider_id = ?",
            (key_id, provider_id),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Key not found")

        await db.execute("DELETE FROM provider_keys WHERE id = ?", (key_id,))
        await db.commit()

    await audit_log(
        user["id"],
        "delete_provider_key",
        "provider_key",
        key_id,
        details={"provider_id": provider_id},
    )


# ---------------------------------------------------------------------------
# Load balancing strategy
# ---------------------------------------------------------------------------


@router.get("/{provider_id}/strategy")
async def get_strategy(
    provider_id: str,
    user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> dict[str, str]:
    """Get the current load balancing strategy for a provider."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT load_balance_strategy FROM providers WHERE id = ? AND user_id = ?",
            (provider_id, user["id"]),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Provider not found")
        return {"strategy": row["load_balance_strategy"]}


@router.put("/{provider_id}/strategy")
async def update_strategy(
    provider_id: str,
    body: StrategyUpdate,
    user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> dict[str, str]:
    """Update the load balancing strategy for a provider."""
    if body.strategy not in VALID_STRATEGIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid strategy. Must be one of: {', '.join(VALID_STRATEGIES)}",
        )

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM providers WHERE id = ? AND user_id = ?",
            (provider_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Provider not found")

        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        await db.execute(
            "UPDATE providers SET load_balance_strategy = ?, updated_at = ? WHERE id = ?",
            (body.strategy, now, provider_id),
        )
        await db.commit()

    return {"strategy": body.strategy}


# ---------------------------------------------------------------------------
# Key selection (load balancing)
# ---------------------------------------------------------------------------


@router.post("/{provider_id}/keys/select", response_model=KeySelectResponse)
async def select_key(
    provider_id: str,
    user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> dict[str, Any]:
    """Select the next API key using the provider's load balancing strategy.

    Skips keys that are rate_limited (with active cooldown) or invalid.
    """
    async with get_db() as db:
        # Get provider and its strategy.
        cursor = await db.execute(
            "SELECT id, load_balance_strategy FROM providers WHERE id = ? AND user_id = ?",
            (provider_id, user["id"]),
        )
        provider = await cursor.fetchone()
        if provider is None:
            raise HTTPException(status_code=404, detail="Provider not found")

        strategy = provider["load_balance_strategy"]

        # Get available keys (active, or rate_limited with expired cooldown).
        cursor = await db.execute(
            """
            SELECT * FROM provider_keys
            WHERE provider_id = ?
              AND (
                status = 'active'
                OR (status = 'rate_limited' AND (cooldown_until IS NULL OR cooldown_until < datetime('now')))
              )
            ORDER BY strategy_position
            """,
            (provider_id,),
        )
        available = await cursor.fetchall()

        if not available:
            raise HTTPException(
                status_code=503,
                detail="No available API keys. All keys are rate-limited or invalid.",
            )

        # Select key based on strategy.
        selected: Any
        if strategy == "round_robin":
            selected = _select_round_robin(available)
        elif strategy == "random":
            selected = _select_random(available)
        elif strategy == "least_recently_used":
            selected = _select_lru(available)
        else:
            selected = available[0]

        decrypted = decrypt_api_key(selected["encrypted_key"])

        return {
            "key_id": selected["id"],
            "api_key": decrypted,
            "strategy": strategy,
        }


def _select_round_robin(keys: list[Any]) -> Any:
    """Round robin: pick the key with the lowest total_requests."""
    return min(keys, key=lambda k: k["total_requests"])


def _select_random(keys: list[Any]) -> Any:
    """Random: pick a random key from the available pool."""
    return rand_module.choice(keys)


def _select_lru(keys: list[Any]) -> Any:
    """Least recently used: pick the key with the oldest last_used (or never used)."""
    return min(keys, key=lambda k: k["last_used"] or "")


# ---------------------------------------------------------------------------
# Usage reporting (for auto-failover)
# ---------------------------------------------------------------------------


@router.post("/{provider_id}/keys/{key_id}/report", status_code=204)
async def report_key_usage(
    provider_id: str,
    key_id: str,
    body: KeyUsageReport,
    user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> None:
    """Report the result of an API call for auto-failover and stats tracking.

    Call this after each provider API call to update key health and stats.
    If rate_limited=True, the key is marked rate_limited with a cooldown.
    If error=True (non-rate-limit), the error count increments. After 5
    consecutive errors the key is marked invalid.
    """
    await _verify_provider_ownership(provider_id, user["id"])

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM provider_keys WHERE id = ? AND provider_id = ?",
            (key_id, provider_id),
        )
        key_row = await cursor.fetchone()
        if key_row is None:
            raise HTTPException(status_code=404, detail="Key not found")

        key = dict(key_row)

        if body.rate_limited:
            # Mark as rate_limited with cooldown.
            cooldown_seconds = max(body.cooldown_seconds, 30)
            await db.execute(
                """
                UPDATE provider_keys
                SET status = 'rate_limited',
                    cooldown_until = datetime('now', '+' || ? || ' seconds'),
                    error_count = error_count + 1,
                    total_requests = total_requests + 1,
                    total_tokens = total_tokens + ?,
                    last_used = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (cooldown_seconds, body.tokens_used, now, now, key_id),
            )
        elif body.error:
            new_error_count = key["error_count"] + 1
            new_status = "invalid" if new_error_count >= 5 else key["status"]
            await db.execute(
                """
                UPDATE provider_keys
                SET status = ?,
                    error_count = ?,
                    total_requests = total_requests + 1,
                    total_tokens = total_tokens + ?,
                    last_used = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (new_status, new_error_count, body.tokens_used, now, now, key_id),
            )
        else:
            # Successful call — reset error count, ensure status is active.
            await db.execute(
                """
                UPDATE provider_keys
                SET status = 'active',
                    error_count = 0,
                    cooldown_until = NULL,
                    total_requests = total_requests + 1,
                    total_tokens = total_tokens + ?,
                    last_used = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (body.tokens_used, now, now, key_id),
            )

        await db.commit()
