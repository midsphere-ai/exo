"""Applications CRUD REST API."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user
from exo_web.sanitize import sanitize_html

router = APIRouter(prefix="/api/v1/applications", tags=["applications"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_APP_TYPES = ("chatbot", "chatflow", "workflow", "agent", "text_generator")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ApplicationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    type: str = Field(..., min_length=1, description="Type")
    project_id: str = Field(..., min_length=1, description="Associated project identifier")
    config_json: str = Field("{}", description="JSON configuration object")


class ApplicationUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255, description="Display name")
    config_json: str | None = Field(None, description="JSON configuration object")
    status: str | None = Field(None, description="Current status")


class ApplicationResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    type: str = Field(description="Type")
    project_id: str = Field(description="Associated project identifier")
    config_json: str = Field(description="JSON configuration object")
    status: str = Field(description="Current status")
    last_run_at: str | None = Field(description="Last execution timestamp")
    user_id: str = Field(description="Owning user identifier")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict."""
    return dict(row)


async def _verify_ownership(db: Any, app_id: str, user_id: str) -> dict[str, Any]:
    """Verify application exists and belongs to user. Returns row dict or raises 404."""
    cursor = await db.execute(
        "SELECT * FROM applications WHERE id = ? AND user_id = ?",
        (app_id, user_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Application not found")
    return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[ApplicationResponse])
async def list_applications(
    project_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all applications for the current user, optionally filtered by project."""
    async with get_db() as db:
        if project_id:
            cursor = await db.execute(
                "SELECT * FROM applications WHERE user_id = ? AND project_id = ? ORDER BY created_at DESC",
                (user["id"], project_id),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM applications WHERE user_id = ? ORDER BY created_at DESC",
                (user["id"],),
            )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


@router.post("", response_model=ApplicationResponse, status_code=201)
async def create_application(
    body: ApplicationCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new application."""
    if body.type not in VALID_APP_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid type. Must be one of: {', '.join(VALID_APP_TYPES)}",
        )

    async with get_db() as db:
        # Verify project exists and belongs to user.
        cursor = await db.execute(
            "SELECT id FROM projects WHERE id = ? AND user_id = ?",
            (body.project_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")

        app_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO applications (id, name, type, project_id, config_json, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                app_id,
                sanitize_html(body.name),
                body.type,
                body.project_id,
                body.config_json,
                user["id"],
                now,
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM applications WHERE id = ?", (app_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.get("/{app_id}", response_model=ApplicationResponse)
async def get_application(
    app_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single application by ID."""
    async with get_db() as db:
        return await _verify_ownership(db, app_id, user["id"])


@router.put("/{app_id}", response_model=ApplicationResponse)
async def update_application(
    app_id: str,
    body: ApplicationUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update an application's editable fields."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    if "name" in updates and isinstance(updates["name"], str):
        updates["name"] = sanitize_html(updates["name"])

    async with get_db() as db:
        await _verify_ownership(db, app_id, user["id"])

        updates["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), app_id]

        await db.execute(
            f"UPDATE applications SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM applications WHERE id = ?", (app_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.delete("/{app_id}", status_code=204)
async def delete_application(
    app_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete an application."""
    async with get_db() as db:
        await _verify_ownership(db, app_id, user["id"])
        await db.execute("DELETE FROM applications WHERE id = ?", (app_id,))
        await db.commit()


@router.post("/{app_id}/duplicate", response_model=ApplicationResponse, status_code=201)
async def duplicate_application(
    app_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Duplicate an application with '(Copy)' suffix."""
    async with get_db() as db:
        original = await _verify_ownership(db, app_id, user["id"])

        new_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO applications (id, name, type, project_id, config_json, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                f"{original['name']} (Copy)",
                original["type"],
                original["project_id"],
                original["config_json"],
                user["id"],
                now,
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM applications WHERE id = ?", (new_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)
