"""Retention policy and storage usage endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from exo_web.config import settings
from exo_web.database import _DB_PATH, get_db
from exo_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/v1/settings", tags=["retention"])

_RETENTION_KEYS = {
    "retention_artifacts_days": 90,
    "retention_runs_days": 30,
    "retention_logs_days": 14,
}


class RetentionConfig(BaseModel):
    retention_artifacts_days: int = Field(default=90, ge=1, description="Retention artifacts days")
    retention_runs_days: int = Field(default=30, ge=1, description="Retention runs days")
    retention_logs_days: int = Field(default=14, ge=1, description="Retention logs days")


@router.get("/retention")
async def get_retention_settings(
    _user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> RetentionConfig:
    """Return current retention policy settings."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT key, value FROM workspace_settings WHERE key IN (?, ?, ?)",
            tuple(_RETENTION_KEYS),
        )
        rows = await cursor.fetchall()

    values = {row["key"]: row["value"] for row in rows}
    return RetentionConfig(
        retention_artifacts_days=int(values.get("retention_artifacts_days", 90)),
        retention_runs_days=int(values.get("retention_runs_days", 30)),
        retention_logs_days=int(values.get("retention_logs_days", 14)),
    )


@router.put("/retention")
async def update_retention_settings(
    body: RetentionConfig,
    _user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> RetentionConfig:
    """Update retention policy settings."""
    async with get_db() as db:
        for key, value in body.model_dump().items():
            await db.execute(
                "INSERT INTO workspace_settings (key, value, updated_at) VALUES (?, ?, datetime('now')) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (key, str(value)),
            )
        await db.commit()
    return body


def _dir_size(path: str) -> int:
    """Calculate total size of all files in a directory (non-recursive is fine for flat dirs)."""
    total = 0
    p = Path(path)
    if not p.is_dir():
        return 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


@router.get("/storage")
async def get_storage_usage(
    _user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, int]:
    """Return storage usage: database size, artifact dir size, upload dir size (in bytes)."""
    # Database file size
    db_path = Path(_DB_PATH)
    total_db_size = 0
    if db_path.is_file():
        total_db_size = db_path.stat().st_size
        # Include WAL and SHM files if they exist
        for suffix in ("-wal", "-shm"):
            wal = db_path.with_suffix(db_path.suffix + suffix)
            if wal.is_file():
                total_db_size += wal.stat().st_size

    artifact_dir_size = _dir_size(settings.artifact_dir)
    upload_dir_size = _dir_size(settings.upload_dir)

    return {
        "total_db_size": total_db_size,
        "artifact_dir_size": artifact_dir_size,
        "upload_dir_size": upload_dir_size,
    }
