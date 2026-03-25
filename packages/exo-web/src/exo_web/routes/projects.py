"""Projects CRUD REST API."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.pagination import paginate
from exo_web.routes.auth import get_current_user
from exo_web.sanitize import sanitize_html

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    description: str = Field("", description="Human-readable description")


class ProjectUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255, description="Display name")
    description: str | None = Field(None, description="Human-readable description")
    default_model: str | None = Field(None, description="Default model for the project")
    default_provider: str | None = Field(None, description="Default provider for the project")


class ProjectResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    description: str = Field(description="Human-readable description")
    default_model: str | None = Field(description="Default model for the project")
    default_provider: str | None = Field(description="Default provider for the project")
    user_id: str = Field(description="Owning user identifier")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict."""
    return dict(row)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("")
async def list_projects(
    cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return projects for the current user with cursor-based pagination."""
    async with get_db() as db:
        result = await paginate(
            db,
            table="projects",
            conditions=["user_id = ?"],
            params=[user["id"]],
            cursor=cursor,
            limit=limit,
            row_mapper=_row_to_dict,
        )
        return result.model_dump()


@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(
    body: ProjectCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new project."""
    project_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO projects (id, name, description, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                sanitize_html(body.name),
                sanitize_html(body.description),
                user["id"],
                now,
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single project by ID."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM projects WHERE id = ? AND user_id = ?",
            (project_id, user["id"]),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Project not found")
        return _row_to_dict(row)


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    body: ProjectUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a project's editable fields."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    for field in ("name", "description"):
        if field in updates and isinstance(updates[field], str):
            updates[field] = sanitize_html(updates[field])

    async with get_db() as db:
        # Verify the project exists and belongs to the user.
        cursor = await db.execute(
            "SELECT id FROM projects WHERE id = ? AND user_id = ?",
            (project_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")

        # Build dynamic SET clause.
        updates["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), project_id]

        await db.execute(
            f"UPDATE projects SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a project and all cascading data."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM projects WHERE id = ? AND user_id = ?",
            (project_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")

        await db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        await db.commit()
