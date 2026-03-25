"""Artifact storage and retrieval REST API."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from exo_web.config import settings
from exo_web.database import get_db
from exo_web.routes.auth import get_current_user
from exo_web.sanitize import sanitize_html

router = APIRouter(prefix="/api/v1/artifacts", tags=["artifacts"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ArtifactResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    run_id: str | None = Field(None, description="Associated run identifier")
    agent_id: str | None = Field(None, description="Associated agent identifier")
    filename: str = Field(description="Filename")
    file_type: str = Field(description="File type")
    file_size: int = Field(description="File size")
    storage_path: str = Field(description="Storage path")
    created_at: str = Field(description="ISO 8601 creation timestamp")


class ArtifactUpdateContent(BaseModel):
    content: str = Field(min_length=0, description="Text content")


class ArtifactVersionResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    artifact_id: str = Field(description="Artifact id")
    version_number: int = Field(description="Version number")
    file_size: int = Field(description="File size")
    created_at: str = Field(description="ISO 8601 creation timestamp")


class ArtifactVersionDetailResponse(ArtifactVersionResponse):
    content: str


class ArtifactRegenerateRequest(BaseModel):
    instructions: str = Field(min_length=1, description="System instructions for the agent")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _artifact_dir() -> Path:
    """Return the configured artifact storage directory, creating it if needed."""
    p = Path(settings.artifact_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[ArtifactResponse])
async def list_artifacts(
    file_type: str | None = Query(None),
    agent_id: str | None = Query(None),
    run_id: str | None = Query(None),
    date_from: str | None = Query(None, description="ISO date, e.g. 2026-01-01"),
    date_to: str | None = Query(None, description="ISO date, e.g. 2026-12-31"),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """List artifacts with optional filters."""
    query = "SELECT * FROM artifacts WHERE user_id = ?"
    params: list[Any] = [user["id"]]

    if file_type is not None:
        query += " AND file_type = ?"
        params.append(file_type)
    if agent_id is not None:
        query += " AND agent_id = ?"
        params.append(agent_id)
    if run_id is not None:
        query += " AND run_id = ?"
        params.append(run_id)
    if date_from is not None:
        query += " AND created_at >= ?"
        params.append(date_from)
    if date_to is not None:
        query += " AND created_at <= ?"
        params.append(date_to)

    query += " ORDER BY created_at DESC"

    async with get_db() as db:
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


@router.get("/{artifact_id}", response_model=ArtifactResponse)
async def get_artifact(
    artifact_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return metadata for a single artifact."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM artifacts WHERE id = ? AND user_id = ?",
            (artifact_id, user["id"]),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return dict(row)


@router.get("/{artifact_id}/download")
async def download_artifact(
    artifact_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> FileResponse:
    """Download the artifact file content."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM artifacts WHERE id = ? AND user_id = ?",
            (artifact_id, user["id"]),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    artifact = dict(row)
    file_path = Path(artifact["storage_path"])

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found on disk")

    return FileResponse(
        path=str(file_path),
        filename=artifact["filename"],
        media_type="application/octet-stream",
    )


@router.delete("/{artifact_id}", status_code=204)
async def delete_artifact(
    artifact_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete an artifact record and its file from disk."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM artifacts WHERE id = ? AND user_id = ?",
            (artifact_id, user["id"]),
        )
        row = await cursor.fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Artifact not found")

        artifact = dict(row)

        # Remove from database
        await db.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))
        await db.commit()

    # Remove file from disk (best-effort)
    file_path = Path(artifact["storage_path"])
    if file_path.is_file():
        os.remove(file_path)


# ---------------------------------------------------------------------------
# Content read / edit
# ---------------------------------------------------------------------------

_TEXT_EXTS = frozenset(
    [
        "txt",
        "md",
        "log",
        "csv",
        "tsv",
        "xml",
        "yaml",
        "yml",
        "toml",
        "ini",
        "cfg",
        "env",
        "py",
        "js",
        "ts",
        "jsx",
        "tsx",
        "html",
        "css",
        "scss",
        "json",
        "sql",
        "sh",
        "bash",
        "go",
        "rs",
        "java",
        "c",
        "cpp",
        "h",
        "rb",
        "php",
        "svg",
    ]
)


def _is_text_artifact(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in _TEXT_EXTS


@router.get("/{artifact_id}/content")
async def get_artifact_content(
    artifact_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> PlainTextResponse:
    """Return raw text content for editable artifacts."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM artifacts WHERE id = ? AND user_id = ?",
            (artifact_id, user["id"]),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    artifact = dict(row)
    if not _is_text_artifact(artifact["filename"]):
        raise HTTPException(status_code=400, detail="Not a text-based artifact")

    file_path = Path(artifact["storage_path"])
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found on disk")

    content = file_path.read_text(encoding="utf-8", errors="replace")
    return PlainTextResponse(content)


@router.put("/{artifact_id}/content")
async def update_artifact_content(
    artifact_id: str,
    body: ArtifactUpdateContent,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update text content of an artifact, creating a new version."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM artifacts WHERE id = ? AND user_id = ?",
            (artifact_id, user["id"]),
        )
        row = await cursor.fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Artifact not found")

        artifact = dict(row)
        if not _is_text_artifact(artifact["filename"]):
            raise HTTPException(status_code=400, detail="Not a text-based artifact")

        file_path = Path(artifact["storage_path"])

        # Read old content for version snapshot
        old_content = ""
        if file_path.is_file():
            old_content = file_path.read_text(encoding="utf-8", errors="replace")

        # Get next version number
        cursor = await db.execute(
            "SELECT COALESCE(MAX(version_number), 0) AS max_ver FROM artifact_versions WHERE artifact_id = ?",
            (artifact_id,),
        )
        max_row = await cursor.fetchone()
        next_ver = (dict(max_row)["max_ver"] if max_row else 0) + 1

        # If this is first edit, save original as version 1
        if next_ver == 1:
            ver_id = str(uuid.uuid4())
            await db.execute(
                "INSERT INTO artifact_versions (id, artifact_id, version_number, content, file_size) VALUES (?, ?, 1, ?, ?)",
                (ver_id, artifact_id, old_content, len(old_content.encode("utf-8"))),
            )
            next_ver = 2

        # Save new content as next version
        new_content = sanitize_html(body.content)
        new_size = len(new_content.encode("utf-8"))
        ver_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO artifact_versions (id, artifact_id, version_number, content, file_size) VALUES (?, ?, ?, ?, ?)",
            (ver_id, artifact_id, next_ver, new_content, new_size),
        )

        # Write to disk
        file_path.write_text(new_content, encoding="utf-8")

        # Update file_size on artifact record
        await db.execute(
            "UPDATE artifacts SET file_size = ? WHERE id = ?",
            (new_size, artifact_id),
        )
        await db.commit()

    return {"version_number": next_ver, "file_size": new_size}


# ---------------------------------------------------------------------------
# Version history
# ---------------------------------------------------------------------------


@router.get("/{artifact_id}/versions", response_model=list[ArtifactVersionResponse])
async def list_artifact_versions(
    artifact_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """List all saved versions of an artifact."""
    async with get_db() as db:
        # Verify ownership
        cursor = await db.execute(
            "SELECT id FROM artifacts WHERE id = ? AND user_id = ?",
            (artifact_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Artifact not found")

        cursor = await db.execute(
            "SELECT id, artifact_id, version_number, file_size, created_at FROM artifact_versions WHERE artifact_id = ? ORDER BY version_number DESC",
            (artifact_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


@router.get(
    "/{artifact_id}/versions/{version_number}", response_model=ArtifactVersionDetailResponse
)
async def get_artifact_version(
    artifact_id: str,
    version_number: int,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Get a specific version's content."""
    async with get_db() as db:
        # Verify ownership
        cursor = await db.execute(
            "SELECT id FROM artifacts WHERE id = ? AND user_id = ?",
            (artifact_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Artifact not found")

        cursor = await db.execute(
            "SELECT * FROM artifact_versions WHERE artifact_id = ? AND version_number = ?",
            (artifact_id, version_number),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Version not found")
    return dict(row)


# ---------------------------------------------------------------------------
# Regenerate
# ---------------------------------------------------------------------------


@router.post("/{artifact_id}/regenerate")
async def regenerate_artifact(
    artifact_id: str,
    body: ArtifactRegenerateRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Send artifact content back to its agent with modification instructions.

    Returns the regenerated content (or error) and saves as a new version.
    """
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM artifacts WHERE id = ? AND user_id = ?",
            (artifact_id, user["id"]),
        )
        row = await cursor.fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Artifact not found")

        artifact = dict(row)
        if not artifact["agent_id"]:
            raise HTTPException(status_code=400, detail="Artifact has no linked agent")

        if not _is_text_artifact(artifact["filename"]):
            raise HTTPException(status_code=400, detail="Not a text-based artifact")

        # Read current content
        file_path = Path(artifact["storage_path"])
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="Artifact file not found on disk")

        current_content = file_path.read_text(encoding="utf-8", errors="replace")

        # Resolve agent's provider and run the modification prompt
        cursor = await db.execute("SELECT * FROM agents WHERE id = ?", (artifact["agent_id"],))
        agent_row = await cursor.fetchone()
        if agent_row is None:
            raise HTTPException(status_code=404, detail="Linked agent not found")

        agent_data = dict(agent_row)

        # Build modification prompt
        instructions = sanitize_html(body.instructions)
        prompt = (
            f"Here is the current content of the file '{artifact['filename']}':\n\n"
            f"```\n{current_content}\n```\n\n"
            f"Modification instructions: {instructions}\n\n"
            "Return ONLY the modified file content, with no extra explanation or markdown fencing."
        )

        # Try to run through agent's provider
        try:
            from exo_web.services.agent_runtime import _resolve_provider

            provider_type = agent_data.get("provider_type", "openai")
            model_name = agent_data.get("model_name", "gpt-4o")
            provider = await _resolve_provider(provider_type, model_name, user["id"])
            resp = await provider.complete(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
            )
            new_content = resp.content
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"Agent regeneration failed: {exc}"
            ) from exc

    # Save the new content as a version via the update endpoint logic
    # (re-use update_artifact_content by calling it internally isn't clean,
    #  so we inline the version-save logic)
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT COALESCE(MAX(version_number), 0) AS max_ver FROM artifact_versions WHERE artifact_id = ?",
            (artifact_id,),
        )
        max_row = await cursor.fetchone()
        next_ver = (dict(max_row)["max_ver"] if max_row else 0) + 1

        if next_ver == 1:
            ver_id = str(uuid.uuid4())
            await db.execute(
                "INSERT INTO artifact_versions (id, artifact_id, version_number, content, file_size) VALUES (?, ?, 1, ?, ?)",
                (ver_id, artifact_id, current_content, len(current_content.encode("utf-8"))),
            )
            next_ver = 2

        new_size = len(new_content.encode("utf-8"))
        ver_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO artifact_versions (id, artifact_id, version_number, content, file_size) VALUES (?, ?, ?, ?, ?)",
            (ver_id, artifact_id, next_ver, new_content, new_size),
        )

        file_path.write_text(new_content, encoding="utf-8")
        await db.execute(
            "UPDATE artifacts SET file_size = ? WHERE id = ?",
            (new_size, artifact_id),
        )
        await db.commit()

    return {"version_number": next_ver, "content": new_content, "file_size": new_size}
