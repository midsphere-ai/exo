"""Cost estimation, pricing, and budget REST API."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/v1/costs", tags=["costs"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ModelPricingResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    model_name: str = Field(description="Model name")
    input_price_per_1k: float = Field(description="Input price per 1k")
    output_price_per_1k: float = Field(description="Output price per 1k")


class ModelPricingUpdate(BaseModel):
    input_price_per_1k: float = Field(ge=0, description="Input price per 1k")
    output_price_per_1k: float = Field(ge=0, description="Output price per 1k")


class CostSummary(BaseModel):
    total_cost: float = Field(description="Total cost in USD")
    total_runs: int = Field(description="Total runs")
    total_tokens: int = Field(description="Total token count")
    cost_by_model: list[dict[str, Any]] = Field(description="Cost by model")
    cost_by_agent: list[dict[str, Any]] = Field(description="Cost by agent")


class BudgetResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    scope: str = Field(description="Scope")
    scope_id: str = Field(description="Scope id")
    budget_amount: float = Field(description="Budget amount")
    period: str = Field(description="Period")
    alert_threshold: float = Field(description="Alert threshold")


class BudgetCreate(BaseModel):
    scope: str = Field(pattern=r"^(workspace|agent)$", description="Scope")
    scope_id: str = Field("", description="Scope id")
    budget_amount: float = Field(gt=0, description="Budget amount")
    period: str = Field(default="monthly", pattern=r"^(daily|monthly)$", description="Period")
    alert_threshold: float = Field(default=80.0, ge=0, le=100, description="Alert threshold")


class BudgetUpdate(BaseModel):
    budget_amount: float | None = Field(default=None, gt=0, description="Budget amount")
    period: str | None = Field(default=None, pattern=r"^(daily|monthly)$", description="Period")
    alert_threshold: float | None = Field(default=None, ge=0, le=100, description="Alert threshold")


class BudgetCheck(BaseModel):
    budget_id: str = Field(description="Budget id")
    scope: str = Field(description="Scope")
    budget_amount: float = Field(description="Budget amount")
    spent: float = Field(description="Spent")
    percent_used: float = Field(description="Percent used")
    status: str  # "ok", "warning", "exceeded" = Field(description="Current status")


# ---------------------------------------------------------------------------
# GET /api/costs/pricing — return model pricing catalog
# ---------------------------------------------------------------------------


@router.get("/pricing", response_model=list[ModelPricingResponse])
async def list_pricing(
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return pricing information from the model_pricing catalog."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, model_name, input_price_per_1k, output_price_per_1k FROM model_pricing ORDER BY model_name"
        )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# PUT /api/costs/pricing/:id — update pricing for a model
# ---------------------------------------------------------------------------


@router.put("/pricing/{pricing_id}", response_model=ModelPricingResponse)
async def update_pricing(
    pricing_id: str,
    body: ModelPricingUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update pricing for a specific model."""
    async with get_db() as db:
        cursor = await db.execute("SELECT id FROM model_pricing WHERE id = ?", (pricing_id,))
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Pricing entry not found")

        await db.execute(
            """UPDATE model_pricing
            SET input_price_per_1k = ?, output_price_per_1k = ?, updated_at = datetime('now')
            WHERE id = ?""",
            (body.input_price_per_1k, body.output_price_per_1k, pricing_id),
        )
        await db.commit()

        cursor = await db.execute(
            "SELECT id, model_name, input_price_per_1k, output_price_per_1k FROM model_pricing WHERE id = ?",
            (pricing_id,),
        )
        row = await cursor.fetchone()
    return dict(row)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# GET /api/costs/summary — aggregate cost data
# ---------------------------------------------------------------------------


@router.get("/summary", response_model=CostSummary)
async def get_cost_summary(
    range: str = Query(default="30d"),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> CostSummary:
    """Return aggregate cost data for the user."""
    now = datetime.now(UTC)
    offsets = {
        "1h": timedelta(hours=1),
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
    }
    delta = offsets.get(range, timedelta(days=30))
    ts_start = (now - delta).isoformat()
    uid = user["id"]

    async with get_db() as db:
        # Total cost, runs, tokens from runs table
        cursor = await db.execute(
            """SELECT
                COALESCE(SUM(total_cost), 0.0) AS total_cost,
                COUNT(*) AS total_runs,
                COALESCE(SUM(total_tokens), 0) AS total_tokens
            FROM runs
            WHERE user_id = ? AND created_at >= ?""",
            (uid, ts_start),
        )
        totals = dict(await cursor.fetchone())  # type: ignore[arg-type]

        # Also include workflow_runs
        cursor = await db.execute(
            """SELECT
                COALESCE(SUM(total_cost), 0.0) AS total_cost,
                COUNT(*) AS total_runs,
                COALESCE(SUM(total_tokens), 0) AS total_tokens
            FROM workflow_runs
            WHERE user_id = ? AND created_at >= ?""",
            (uid, ts_start),
        )
        wf_totals = dict(await cursor.fetchone())  # type: ignore[arg-type]

        totals["total_cost"] = round(
            (totals["total_cost"] or 0) + (wf_totals["total_cost"] or 0), 6
        )
        totals["total_runs"] = (totals["total_runs"] or 0) + (wf_totals["total_runs"] or 0)
        totals["total_tokens"] = (totals["total_tokens"] or 0) + (wf_totals["total_tokens"] or 0)

        # Cost by model — approximate using steps_json in runs
        # For simplicity, aggregate from runs table which has model info in steps
        cost_by_model: list[dict[str, Any]] = []

        # Cost by agent from runs
        cursor = await db.execute(
            """SELECT
                COALESCE(a.name, 'Unknown') AS agent_name,
                r.agent_id,
                COALESCE(SUM(r.total_cost), 0.0) AS cost,
                COUNT(*) AS run_count
            FROM runs r
            LEFT JOIN agents a ON r.agent_id = a.id
            WHERE r.user_id = ? AND r.created_at >= ?
            GROUP BY r.agent_id, a.name
            ORDER BY cost DESC""",
            (uid, ts_start),
        )
        agent_rows = await cursor.fetchall()

        # Also from workflow runs
        cursor = await db.execute(
            """SELECT
                COALESCE(w.name, 'Unknown') AS agent_name,
                wr.workflow_id AS agent_id,
                COALESCE(SUM(wr.total_cost), 0.0) AS cost,
                COUNT(*) AS run_count
            FROM workflow_runs wr
            LEFT JOIN workflows w ON wr.workflow_id = w.id
            WHERE wr.user_id = ? AND wr.created_at >= ?
            GROUP BY wr.workflow_id, w.name
            ORDER BY cost DESC""",
            (uid, ts_start),
        )
        wf_agent_rows = await cursor.fetchall()

    cost_by_agent = [
        {
            "name": r["agent_name"],
            "id": r["agent_id"],
            "cost": round(r["cost"] or 0, 6),
            "runs": r["run_count"],
        }
        for r in [*agent_rows, *wf_agent_rows]
    ]

    return CostSummary(
        total_cost=totals["total_cost"],
        total_runs=totals["total_runs"],
        total_tokens=totals["total_tokens"],
        cost_by_model=cost_by_model,
        cost_by_agent=cost_by_agent,
    )


# ---------------------------------------------------------------------------
# GET /api/costs/budgets — list budgets
# ---------------------------------------------------------------------------


@router.get("/budgets", response_model=list[BudgetResponse])
async def list_budgets(
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all budgets for the user."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, scope, scope_id, budget_amount, period, alert_threshold FROM cost_budgets WHERE user_id = ? ORDER BY scope, scope_id",
            (user["id"],),
        )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# PUT /api/costs/budgets — create or update a budget
# ---------------------------------------------------------------------------


@router.put("/budgets", response_model=BudgetResponse)
async def upsert_budget(
    body: BudgetCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create or update a budget for the given scope."""
    uid = user["id"]

    async with get_db() as db:
        # Check if budget already exists for this scope
        cursor = await db.execute(
            "SELECT id FROM cost_budgets WHERE user_id = ? AND scope = ? AND scope_id = ?",
            (uid, body.scope, body.scope_id),
        )
        existing = await cursor.fetchone()

        if existing:
            budget_id = existing["id"]
            await db.execute(
                """UPDATE cost_budgets
                SET budget_amount = ?, period = ?, alert_threshold = ?, updated_at = datetime('now')
                WHERE id = ?""",
                (body.budget_amount, body.period, body.alert_threshold, budget_id),
            )
        else:
            budget_id = str(uuid.uuid4())
            await db.execute(
                """INSERT INTO cost_budgets (id, scope, scope_id, budget_amount, period, alert_threshold, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    budget_id,
                    body.scope,
                    body.scope_id,
                    body.budget_amount,
                    body.period,
                    body.alert_threshold,
                    uid,
                ),
            )
        await db.commit()

        cursor = await db.execute(
            "SELECT id, scope, scope_id, budget_amount, period, alert_threshold FROM cost_budgets WHERE id = ?",
            (budget_id,),
        )
        row = await cursor.fetchone()
    return dict(row)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Budget check helper — called before runs
# ---------------------------------------------------------------------------


async def check_budget(user_id: str, agent_id: str | None = None) -> BudgetCheck | None:
    """Check if a budget exists and whether it's been exceeded.

    Returns a BudgetCheck if a budget applies, or None if no budget is set.
    The ``status`` field is:
    - ``"ok"`` — under threshold
    - ``"warning"`` — at or above alert threshold
    - ``"exceeded"`` — at or above 100%
    """
    async with get_db() as db:
        # Check agent-specific budget first, then workspace-level
        budget_row = None
        if agent_id:
            cursor = await db.execute(
                "SELECT * FROM cost_budgets WHERE user_id = ? AND scope = 'agent' AND scope_id = ?",
                (user_id, agent_id),
            )
            budget_row = await cursor.fetchone()

        if budget_row is None:
            cursor = await db.execute(
                "SELECT * FROM cost_budgets WHERE user_id = ? AND scope = 'workspace' AND scope_id = ''",
                (user_id,),
            )
            budget_row = await cursor.fetchone()

        if budget_row is None:
            return None

        budget = dict(budget_row)

        # Calculate the period start
        now = datetime.now(UTC)
        if budget["period"] == "daily":
            period_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        else:  # monthly
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()

        # Sum cost from runs in this period
        cursor = await db.execute(
            """SELECT COALESCE(SUM(total_cost), 0.0) AS spent
            FROM runs WHERE user_id = ? AND created_at >= ?""",
            (user_id, period_start),
        )
        runs_spent = (await cursor.fetchone())["spent"] or 0.0  # type: ignore[index]

        # Also include workflow_runs
        cursor = await db.execute(
            """SELECT COALESCE(SUM(total_cost), 0.0) AS spent
            FROM workflow_runs WHERE user_id = ? AND created_at >= ?""",
            (user_id, period_start),
        )
        wf_spent = (await cursor.fetchone())["spent"] or 0.0  # type: ignore[index]

    spent = round(runs_spent + wf_spent, 6)
    budget_amount = budget["budget_amount"]
    percent_used = round((spent / budget_amount * 100) if budget_amount > 0 else 0.0, 2)

    if percent_used >= 100:
        status = "exceeded"
    elif percent_used >= budget["alert_threshold"]:
        status = "warning"
    else:
        status = "ok"

    return BudgetCheck(
        budget_id=budget["id"],
        scope=budget["scope"],
        budget_amount=budget_amount,
        spent=spent,
        percent_used=percent_used,
        status=status,
    )


# ---------------------------------------------------------------------------
# GET /api/costs/budget-check — check budget status before a run
# ---------------------------------------------------------------------------


@router.delete("/budgets/{budget_id}", status_code=204)
async def delete_budget(
    budget_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a budget by ID."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM cost_budgets WHERE id = ? AND user_id = ?",
            (budget_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Budget not found")

        await db.execute("DELETE FROM cost_budgets WHERE id = ?", (budget_id,))
        await db.commit()


@router.get("/budget-check", response_model=BudgetCheck | None)
async def budget_check_endpoint(
    agent_id: str | None = None,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> BudgetCheck | None:
    """Check budget status. Returns null if no budget is configured."""
    return await check_budget(user["id"], agent_id)
