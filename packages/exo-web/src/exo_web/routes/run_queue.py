"""Run queue endpoints — list queued runs and cancel entries."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from exo_web.routes.auth import get_current_user
from exo_web.services.run_queue import cancel_queued_run, list_queued_runs

router = APIRouter(prefix="/api/v1/runs", tags=["run-queue"])


@router.get("/queue")
async def get_queued_runs(
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return queued runs with their queue position."""
    return await list_queued_runs(user["id"])


@router.delete("/queue/{queue_id}")
async def cancel_queue_entry(
    queue_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, str]:
    """Cancel a queued run."""
    cancelled = await cancel_queued_run(queue_id, user["id"])
    if not cancelled:
        raise HTTPException(status_code=404, detail="Queued run not found or already started")
    return {"status": "cancelled"}
