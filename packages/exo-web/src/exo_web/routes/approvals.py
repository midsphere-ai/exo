"""Workflow approval gate REST API."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user
from exo_web.sanitize import sanitize_html

router = APIRouter(prefix="/api/v1/approvals", tags=["approvals"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ApprovalResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    run_id: str = Field(description="Associated run identifier")
    node_id: str = Field(description="Associated node identifier")
    status: str = Field(description="Current status")
    timeout_minutes: int = Field(description="Timeout minutes")
    comment: str | None = Field(None, description="Comment")
    requested_at: str = Field(description="Requested at")
    responded_at: str | None = Field(None, description="Responded at")


class ApprovalRespondBody(BaseModel):
    approved: bool = Field(description="Approved")
    comment: str | None = Field(default=None, max_length=2000, description="Comment")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_approval_row(row: Any) -> dict[str, Any]:
    """Convert a DB row to a dict, stripping internal fields."""
    r = dict(row)
    r.pop("user_id", None)
    return r


# ---------------------------------------------------------------------------
# GET /api/approvals/pending — list pending approvals for current user
# ---------------------------------------------------------------------------


@router.get("/pending", response_model=list[ApprovalResponse])
async def list_pending_approvals(
    run_id: str | None = Query(default=None),
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """List pending approval gates for the current user."""
    conditions = ["wa.user_id = ?", "wa.status = 'pending'"]
    params: list[Any] = [user["id"]]

    if run_id:
        conditions.append("wa.run_id = ?")
        params.append(run_id)

    where = " AND ".join(conditions)
    params.extend([limit, offset])

    async with get_db() as db:
        cursor = await db.execute(
            f"SELECT wa.* FROM workflow_approvals wa WHERE {where} ORDER BY wa.requested_at DESC LIMIT ? OFFSET ?",
            params,
        )
        rows = await cursor.fetchall()

    return [_parse_approval_row(row) for row in rows]


# ---------------------------------------------------------------------------
# POST /api/approvals/:id/respond — approve or reject
# ---------------------------------------------------------------------------


@router.post("/{approval_id}/respond", response_model=ApprovalResponse)
async def respond_to_approval(
    approval_id: str,
    body: ApprovalRespondBody,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Approve or reject a pending workflow approval gate."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM workflow_approvals WHERE id = ? AND user_id = ?",
            (approval_id, user["id"]),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Approval not found")

        approval = dict(row)
        if approval["status"] != "pending":
            raise HTTPException(
                status_code=400,
                detail=f"Approval already resolved: {approval['status']}",
            )

        new_status = "approved" if body.approved else "rejected"
        comment = sanitize_html(body.comment) if body.comment else None

        await db.execute(
            "UPDATE workflow_approvals SET status = ?, comment = ?, responded_at = datetime('now') WHERE id = ?",
            (new_status, comment, approval_id),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM workflow_approvals WHERE id = ?", (approval_id,))
        row = await cursor.fetchone()

    return _parse_approval_row(row)


# ---------------------------------------------------------------------------
# GET /api/approvals/count — count of pending approvals (for badge)
# ---------------------------------------------------------------------------


@router.get("/count")
async def count_pending_approvals(
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, int]:
    """Return the count of pending approvals for the current user."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM workflow_approvals WHERE user_id = ? AND status = 'pending'",
            (user["id"],),
        )
        row = await cursor.fetchone()
    return {"count": row["cnt"] if row else 0}


# ---------------------------------------------------------------------------
# GET /api/approvals/history — approval history for a specific node
# ---------------------------------------------------------------------------


@router.get("/history", response_model=list[ApprovalResponse])
async def approval_history(
    node_id: str = Query(),
    limit: int = Query(default=50, le=200),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return approval history for a specific node across all runs."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM workflow_approvals WHERE node_id = ? AND user_id = ? ORDER BY requested_at DESC LIMIT ?",
            (node_id, user["id"], limit),
        )
        rows = await cursor.fetchall()
    return [_parse_approval_row(row) for row in rows]


# ---------------------------------------------------------------------------
# Engine-facing helpers (called from engine.py, not HTTP routes)
# ---------------------------------------------------------------------------


async def create_approval(
    run_id: str,
    node_id: str,
    user_id: str,
    timeout_minutes: int = 60,
) -> str:
    """Create a pending approval record. Returns the approval ID."""
    approval_id = str(uuid.uuid4())
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO workflow_approvals (id, run_id, node_id, status, timeout_minutes, user_id)
            VALUES (?, ?, ?, 'pending', ?, ?)
            """,
            (approval_id, run_id, node_id, timeout_minutes, user_id),
        )
        await db.commit()
    return approval_id


async def poll_approval(approval_id: str) -> str:
    """Check the status of an approval. Returns 'pending', 'approved', 'rejected', or 'timed_out'."""
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT status, timeout_minutes, requested_at FROM workflow_approvals WHERE id = ?
            """,
            (approval_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return "rejected"

        status = row["status"]

        # Check timeout: if pending and past the deadline, auto-reject.
        if status == "pending":
            cursor2 = await db.execute(
                """
                SELECT 1 FROM workflow_approvals
                WHERE id = ?
                  AND status = 'pending'
                  AND datetime(requested_at, '+' || timeout_minutes || ' minutes') < datetime('now')
                """,
                (approval_id,),
            )
            if await cursor2.fetchone() is not None:
                await db.execute(
                    "UPDATE workflow_approvals SET status = 'timed_out', responded_at = datetime('now'), comment = 'Auto-rejected: timeout exceeded' WHERE id = ?",
                    (approval_id,),
                )
                await db.commit()
                return "timed_out"

        return status
