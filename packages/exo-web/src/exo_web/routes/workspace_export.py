"""Workspace export and import endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse

from exo_web.routes.auth import get_current_user
from exo_web.services.audit import audit_log
from exo_web.services.workspace_export import (
    create_export,
    get_export_path,
    import_workspace,
)

router = APIRouter(prefix="/api/v1/settings", tags=["workspace-export"])


@router.post("/export")
async def trigger_export(
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Trigger a workspace export. Returns export metadata with download id."""
    result = await create_export(user["id"])
    await audit_log(
        user_id=user["id"],
        action="workspace_export",
        entity_type="workspace",
        entity_id=result["export_id"],
    )
    return result


@router.get("/export/{export_id}/download")
async def download_export(
    export_id: str,
    _user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> FileResponse:
    """Download a workspace export ZIP file."""
    path = get_export_path(export_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Export not found")
    return FileResponse(
        path=str(path),
        filename=path.name,
        media_type="application/zip",
    )


@router.post("/import")
async def trigger_import(
    file: UploadFile,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Import workspace data from a ZIP file."""
    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(status_code=422, detail="File must be a .zip archive")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=422, detail="File is empty")

    # Cap at max upload size
    max_bytes = 50 * 1024 * 1024  # 50 MB
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    try:
        result = await import_workspace(user["id"], content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    await audit_log(
        user_id=user["id"],
        action="workspace_import",
        entity_type="workspace",
        entity_id="import",
        details=result,
    )
    return result
