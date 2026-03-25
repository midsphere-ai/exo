"""Background cleanup — removes expired sessions, stale tokens, orphaned uploads, and enforces retention policies."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from exo_web.config import settings
from exo_web.database import get_db

_log = logging.getLogger(__name__)

_cleanup_task: asyncio.Task[Any] | None = None

_RETENTION_DEFAULTS = {
    "retention_artifacts_days": 90,
    "retention_runs_days": 30,
    "retention_logs_days": 14,
}


async def _get_retention_days() -> dict[str, int]:
    """Read retention settings from workspace_settings, falling back to defaults."""
    result = dict(_RETENTION_DEFAULTS)
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT key, value FROM workspace_settings WHERE key IN (?, ?, ?)",
            tuple(_RETENTION_DEFAULTS),
        )
        for row in await cursor.fetchall():
            with contextlib.suppress(ValueError):
                result[row["key"]] = int(row["value"])
    return result


async def _run_retention_cleanup() -> None:
    """Delete artifacts, runs, and logs older than their retention period.

    Order: artifacts first (since they reference runs), then runs, then logs.
    Cascade: when deleting a run, first delete its artifacts and related logs.
    Note: audit_log is intentionally exempt — entries are never deleted.
    """
    retention = await _get_retention_days()
    deleted_artifacts = 0
    deleted_artifact_files = 0
    deleted_runs = 0
    deleted_logs = 0

    async with get_db() as db:
        # 1. Delete old artifacts (and their files on disk)
        artifact_cutoff = f"-{retention['retention_artifacts_days']} days"
        cursor = await db.execute(
            "SELECT id, storage_path FROM artifacts WHERE created_at < datetime('now', ?)",
            (artifact_cutoff,),
        )
        old_artifacts = await cursor.fetchall()
        for row in old_artifacts:
            # Remove file from disk
            path = Path(row["storage_path"])
            if path.is_file():
                try:
                    path.unlink()
                    deleted_artifact_files += 1
                except OSError:
                    _log.warning("Failed to remove artifact file: %s", path)
            # Delete artifact_versions (cascade via FK) and the artifact record
            await db.execute("DELETE FROM artifact_versions WHERE artifact_id = ?", (row["id"],))
            await db.execute("DELETE FROM artifacts WHERE id = ?", (row["id"],))
            deleted_artifacts += 1

        # 2. Delete old runs — first cascade-delete their artifacts and logs
        run_cutoff = f"-{retention['retention_runs_days']} days"
        cursor = await db.execute(
            "SELECT id FROM runs WHERE created_at < datetime('now', ?)",
            (run_cutoff,),
        )
        old_runs = await cursor.fetchall()
        for row in old_runs:
            run_id = row["id"]
            # Cascade: delete artifacts belonging to this run
            art_cursor = await db.execute(
                "SELECT id, storage_path FROM artifacts WHERE run_id = ?", (run_id,)
            )
            for art in await art_cursor.fetchall():
                path = Path(art["storage_path"])
                if path.is_file():
                    with contextlib.suppress(OSError):
                        path.unlink()
                await db.execute(
                    "DELETE FROM artifact_versions WHERE artifact_id = ?", (art["id"],)
                )
                await db.execute("DELETE FROM artifacts WHERE id = ?", (art["id"],))
            # Cascade: delete logs referencing this run (via agent_id is not direct,
            # but runs and logs share no FK — logs are deleted by timestamp below)
            # Delete the run itself
            await db.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            deleted_runs += 1

        # 3. Delete old logs
        log_cutoff = f"-{retention['retention_logs_days']} days"
        cursor = await db.execute(
            "DELETE FROM logs WHERE timestamp < datetime('now', ?)",
            (log_cutoff,),
        )
        deleted_logs = cursor.rowcount

        await db.commit()

    if deleted_artifacts or deleted_runs or deleted_logs:
        _log.info(
            "Retention cleanup: %d artifacts (%d files), %d runs, %d logs removed",
            deleted_artifacts,
            deleted_artifact_files,
            deleted_runs,
            deleted_logs,
        )


async def _run_cleanup() -> None:
    """Execute one cleanup pass and log results."""
    expired_sessions = 0
    stale_tokens = 0
    orphaned_files = 0

    async with get_db() as db:
        # 1. Delete expired sessions (expires_at < now)
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        cursor = await db.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
        expired_sessions = cursor.rowcount

        # 2. Delete used or expired password reset tokens
        cursor = await db.execute(
            "DELETE FROM password_resets WHERE used = 1 OR expires_at < ?", (now,)
        )
        stale_tokens = cursor.rowcount

        await db.commit()

    # 3. Delete orphaned upload temp files older than 24 hours
    upload_dir = Path(settings.upload_dir)
    if upload_dir.is_dir():
        cutoff = datetime.now(UTC) - timedelta(hours=24)
        # Check which files are still referenced in documents table
        referenced: set[str] = set()
        async with get_db() as db:
            cursor = await db.execute("SELECT file_path FROM documents")
            for row in await cursor.fetchall():
                referenced.add(row["file_path"])

        for file_path in upload_dir.iterdir():
            if not file_path.is_file():
                continue
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=UTC)
            if mtime < cutoff and str(file_path) not in referenced:
                try:
                    file_path.unlink()
                    orphaned_files += 1
                except OSError:
                    _log.warning("Failed to remove orphaned file: %s", file_path)

    _log.info(
        "Cleanup complete: %d expired sessions, %d stale tokens, %d orphaned files removed",
        expired_sessions,
        stale_tokens,
        orphaned_files,
    )

    # 4. Run retention-based cleanup
    await _run_retention_cleanup()


async def _cleanup_loop() -> None:
    """Run cleanup on startup, then repeat at the configured interval."""
    interval_seconds = settings.cleanup_interval_hours * 3600
    while True:
        try:
            await _run_cleanup()
        except Exception:
            _log.exception("Error during cleanup")
        await asyncio.sleep(interval_seconds)


async def start_cleanup() -> None:
    """Start the background cleanup task."""
    global _cleanup_task
    if _cleanup_task is not None:
        return
    _log.info("Starting cleanup task (every %dh)", settings.cleanup_interval_hours)
    _cleanup_task = asyncio.create_task(_cleanup_loop())


async def stop_cleanup() -> None:
    """Stop the background cleanup task."""
    global _cleanup_task
    if _cleanup_task is None:
        return
    _log.info("Stopping cleanup task")
    _cleanup_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await _cleanup_task
    _cleanup_task = None
