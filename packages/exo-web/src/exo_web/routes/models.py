"""Model catalog CRUD and discovery REST API."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from exo_web.crypto import decrypt_api_key
from exo_web.database import get_db
from exo_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/v1", tags=["models"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ModelCreate(BaseModel):
    provider_id: str = Field(..., min_length=1, description="Associated provider identifier")
    model_name: str = Field(..., min_length=1, max_length=255, description="Model name")
    context_window: int | None = Field(None, description="Context window")
    capabilities: list[str] = Field([], description="Capabilities")
    pricing_input: float | None = Field(None, description="Pricing input")
    pricing_output: float | None = Field(None, description="Pricing output")
    is_custom: bool = Field(True, description="Is custom")


class ModelResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    provider_id: str = Field(description="Associated provider identifier")
    model_name: str = Field(description="Model name")
    context_window: int | None = Field(description="Context window")
    capabilities: list[str] = Field(description="Capabilities")
    pricing_input: float | None = Field(description="Pricing input")
    pricing_output: float | None = Field(description="Pricing output")
    is_custom: bool = Field(description="Is custom")
    provider_name: str | None = Field(None, description="Provider name")
    provider_type: str | None = Field(None, description="Provider type (openai, anthropic, etc.)")
    user_id: str = Field(description="Owning user identifier")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


class DiscoverResult(BaseModel):
    discovered: int = Field(description="Discovered")
    models: list[ModelResponse] = Field(description="Models")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_response(row: Any) -> dict[str, Any]:
    """Convert a DB row to a ModelResponse-compatible dict."""
    d = dict(row)
    # Parse capabilities JSON string to list.
    caps = d.get("capabilities", "[]")
    d["capabilities"] = json.loads(caps) if isinstance(caps, str) else caps
    d["is_custom"] = bool(d.get("is_custom", 0))
    return d


# ---------------------------------------------------------------------------
# GET /api/models — list all models with optional filters
# ---------------------------------------------------------------------------


@router.get("/models", response_model=list[ModelResponse])
async def list_models(
    provider_id: str | None = Query(None),
    provider_type: str | None = Query(None),
    capability: str | None = Query(None),
    search: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all models across configured providers with optional filters."""
    async with get_db() as db:
        query = """
            SELECT m.*, p.name AS provider_name, p.provider_type AS provider_type
            FROM models m
            JOIN providers p ON m.provider_id = p.id
            WHERE m.user_id = ?
        """
        params: list[Any] = [user["id"]]

        if provider_id:
            query += " AND m.provider_id = ?"
            params.append(provider_id)

        if provider_type:
            query += " AND p.provider_type = ?"
            params.append(provider_type)

        if capability:
            query += " AND m.capabilities LIKE ?"
            params.append(f"%{capability}%")

        if search:
            query += " AND m.model_name LIKE ?"
            params.append(f"%{search}%")

        query += " ORDER BY p.name, m.model_name"

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_response(r) for r in rows]


# ---------------------------------------------------------------------------
# POST /api/models — manually add a model (custom / fine-tuned)
# ---------------------------------------------------------------------------


@router.post("/models", response_model=ModelResponse, status_code=201)
async def create_model(
    body: ModelCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Manually add a model entry (for custom or fine-tuned models)."""
    async with get_db() as db:
        # Verify the provider exists and belongs to the user.
        cursor = await db.execute(
            "SELECT id, name, provider_type FROM providers WHERE id = ? AND user_id = ?",
            (body.provider_id, user["id"]),
        )
        provider = await cursor.fetchone()
        if provider is None:
            raise HTTPException(status_code=404, detail="Provider not found")

        # Check for duplicate model_name under same provider.
        cursor = await db.execute(
            "SELECT id FROM models WHERE provider_id = ? AND model_name = ?",
            (body.provider_id, body.model_name),
        )
        if await cursor.fetchone() is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Model '{body.model_name}' already exists for this provider",
            )

        model_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        capabilities_json = json.dumps(body.capabilities)

        await db.execute(
            """
            INSERT INTO models (id, provider_id, model_name, context_window, capabilities,
                                pricing_input, pricing_output, is_custom, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                body.provider_id,
                body.model_name,
                body.context_window,
                capabilities_json,
                body.pricing_input,
                body.pricing_output,
                1 if body.is_custom else 0,
                user["id"],
                now,
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute(
            """
            SELECT m.*, p.name AS provider_name, p.provider_type AS provider_type
            FROM models m
            JOIN providers p ON m.provider_id = p.id
            WHERE m.id = ?
            """,
            (model_id,),
        )
        row = await cursor.fetchone()
        return _row_to_response(row)


# ---------------------------------------------------------------------------
# POST /api/providers/:id/discover — auto-discover models from provider API
# ---------------------------------------------------------------------------


@router.post("/providers/{provider_id}/discover", response_model=DiscoverResult)
async def discover_models(
    provider_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Query a provider's API for available models and upsert them to DB."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM providers WHERE id = ? AND user_id = ?",
            (provider_id, user["id"]),
        )
        provider = await cursor.fetchone()
        if provider is None:
            raise HTTPException(status_code=404, detail="Provider not found")

    provider_dict = dict(provider)
    provider_type = provider_dict["provider_type"]

    # Get a working API key — try provider-level first, then provider_keys.
    api_key = None
    if provider_dict["encrypted_api_key"]:
        api_key = decrypt_api_key(provider_dict["encrypted_api_key"])
    else:
        async with get_db() as db:
            cursor = await db.execute(
                """
                SELECT encrypted_key FROM provider_keys
                WHERE provider_id = ? AND status = 'active'
                ORDER BY strategy_position LIMIT 1
                """,
                (provider_id,),
            )
            key_row = await cursor.fetchone()
            if key_row:
                api_key = decrypt_api_key(key_row["encrypted_key"])

    if not api_key and provider_type not in ("ollama",):
        raise HTTPException(status_code=400, detail="No API key available for this provider")

    raw_models = await _fetch_models_from_provider(
        provider_type, api_key, provider_dict["base_url"], provider_dict["timeout"]
    )

    # Upsert discovered models.
    upserted: list[dict[str, Any]] = []
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        for model_info in raw_models:
            model_name = model_info["model_name"]
            capabilities_json = json.dumps(model_info.get("capabilities", []))

            # Check if model already exists.
            cursor = await db.execute(
                "SELECT id FROM models WHERE provider_id = ? AND model_name = ?",
                (provider_id, model_name),
            )
            existing = await cursor.fetchone()

            if existing:
                # Update existing model metadata.
                model_id = existing["id"]
                await db.execute(
                    """
                    UPDATE models SET context_window = ?, capabilities = ?,
                                      pricing_input = ?, pricing_output = ?,
                                      updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        model_info.get("context_window"),
                        capabilities_json,
                        model_info.get("pricing_input"),
                        model_info.get("pricing_output"),
                        now,
                        model_id,
                    ),
                )
            else:
                model_id = str(uuid.uuid4())
                await db.execute(
                    """
                    INSERT INTO models (id, provider_id, model_name, context_window, capabilities,
                                        pricing_input, pricing_output, is_custom, user_id,
                                        created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                    """,
                    (
                        model_id,
                        provider_id,
                        model_name,
                        model_info.get("context_window"),
                        capabilities_json,
                        model_info.get("pricing_input"),
                        model_info.get("pricing_output"),
                        user["id"],
                        now,
                        now,
                    ),
                )

        await db.commit()

        # Fetch all models for this provider to return.
        cursor = await db.execute(
            """
            SELECT m.*, p.name AS provider_name, p.provider_type AS provider_type
            FROM models m
            JOIN providers p ON m.provider_id = p.id
            WHERE m.provider_id = ? AND m.user_id = ?
            ORDER BY m.model_name
            """,
            (provider_id, user["id"]),
        )
        rows = await cursor.fetchall()
        upserted = [_row_to_response(r) for r in rows]

    return {"discovered": len(raw_models), "models": upserted}


# ---------------------------------------------------------------------------
# Provider-specific model fetching
# ---------------------------------------------------------------------------


async def _fetch_models_from_provider(
    provider_type: str,
    api_key: str | None,
    base_url: str | None,
    timeout: int,
) -> list[dict[str, Any]]:
    """Fetch the list of available models from a provider's API."""
    import httpx

    models: list[dict[str, Any]] = []

    try:
        if provider_type == "openai":
            url = f"{base_url}/v1/models" if base_url else "https://api.openai.com/v1/models"
            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                for m in data.get("data", []):
                    models.append({"model_name": m["id"]})

        elif provider_type == "anthropic":
            url = f"{base_url}/v1/models" if base_url else "https://api.anthropic.com/v1/models"
            headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                for m in data.get("data", []):
                    models.append({"model_name": m["id"]})

        elif provider_type == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
                for m in data.get("models", []):
                    name = m.get("name", "")
                    # Gemini returns "models/gemini-pro" — strip "models/" prefix.
                    model_name = name.removeprefix("models/")
                    info: dict[str, Any] = {"model_name": model_name}
                    if "inputTokenLimit" in m:
                        info["context_window"] = m["inputTokenLimit"]
                    models.append(info)

        elif provider_type == "ollama":
            url = f"{base_url}/api/tags" if base_url else "http://localhost:11434/api/tags"
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
                for m in data.get("models", []):
                    models.append({"model_name": m.get("name", m.get("model", ""))})

        elif provider_type == "vertex":
            if not base_url:
                return []
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(base_url, headers={"Authorization": f"Bearer {api_key}"})
                resp.raise_for_status()
                data = resp.json()
                for m in data.get("models", data.get("data", [])):
                    name = m.get("name", m.get("id", ""))
                    models.append({"model_name": name})

        else:
            # Custom provider — attempt OpenAI-compatible /v1/models.
            if base_url:
                url = f"{base_url}/v1/models"
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.get(url, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    for m in data.get("data", []):
                        models.append({"model_name": m.get("id", m.get("name", ""))})

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to discover models from provider: {exc!s}"[:300],
        ) from exc

    return models
