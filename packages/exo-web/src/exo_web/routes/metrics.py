"""Metrics aggregation REST API."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class DashboardMetrics(BaseModel):
    total_runs: int = Field(description="Total runs")
    success_rate: float = Field(description="Success rate")
    avg_latency_ms: float = Field(description="Average latency in milliseconds")
    total_tokens: int = Field(description="Total token count")
    total_cost: float = Field(description="Total cost in USD")


class AgentMetrics(BaseModel):
    agent_name: str = Field(description="Agent name")
    agent_id: str | None = Field(None, description="Associated agent identifier")
    run_count: int = Field(description="Run count")
    success_rate: float = Field(description="Success rate")
    avg_tokens: float = Field(description="Avg tokens")
    avg_cost: float = Field(description="Avg cost")


class TimeseriesPoint(BaseModel):
    bucket: str = Field(description="Bucket")
    value: float = Field(description="Value")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_RANGES = {"1h", "24h", "7d", "30d"}


def _range_to_start(range_str: str, start: str | None, end: str | None) -> tuple[str, str]:
    """Convert a range string or custom start/end to ISO-8601 timestamps."""
    now = datetime.now(UTC)
    if start and end:
        return start, end
    offsets = {
        "1h": timedelta(hours=1),
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
    }
    delta = offsets.get(range_str, timedelta(hours=24))
    return (now - delta).isoformat(), now.isoformat()


def _bucket_format(range_str: str) -> str:
    """Return strftime format for time bucketing based on range."""
    if range_str == "1h":
        return "%Y-%m-%dT%H:%M"  # per-minute
    if range_str == "24h":
        return "%Y-%m-%dT%H:00"  # per-hour
    return "%Y-%m-%d"  # per-day


# ---------------------------------------------------------------------------
# GET /api/metrics/dashboard — aggregate dashboard metrics
# ---------------------------------------------------------------------------


@router.get("/dashboard", response_model=DashboardMetrics)
async def get_dashboard_metrics(
    range: str = Query(default="24h"),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> DashboardMetrics:
    """Return aggregate metrics: total runs, success rate, avg latency, tokens, cost."""
    ts_start, ts_end = _range_to_start(range, start, end)
    uid = user["id"]

    async with get_db() as db:
        # Workflow runs
        cursor = await db.execute(
            """
            SELECT
                COUNT(*) AS cnt,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS ok,
                AVG(
                    CASE WHEN completed_at IS NOT NULL AND started_at IS NOT NULL
                    THEN (julianday(completed_at) - julianday(started_at)) * 86400000
                    END
                ) AS avg_ms,
                COALESCE(SUM(total_tokens), 0) AS tokens,
                COALESCE(SUM(total_cost), 0.0) AS cost
            FROM workflow_runs
            WHERE user_id = ? AND created_at >= ? AND created_at <= ?
            """,
            (uid, ts_start, ts_end),
        )
        wf = dict(await cursor.fetchone())  # type: ignore[arg-type]

        # Crew runs
        cursor = await db.execute(
            """
            SELECT
                COUNT(*) AS cnt,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS ok,
                AVG(
                    CASE WHEN completed_at IS NOT NULL AND started_at IS NOT NULL
                    THEN (julianday(completed_at) - julianday(started_at)) * 86400000
                    END
                ) AS avg_ms
            FROM crew_runs
            WHERE user_id = ? AND created_at >= ? AND created_at <= ?
            """,
            (uid, ts_start, ts_end),
        )
        cr = dict(await cursor.fetchone())  # type: ignore[arg-type]

    total = (wf["cnt"] or 0) + (cr["cnt"] or 0)
    ok_total = (wf["ok"] or 0) + (cr["ok"] or 0)
    success_rate = (ok_total / total * 100) if total > 0 else 0.0

    # Weighted average latency
    wf_ms = wf["avg_ms"] or 0.0
    cr_ms = cr["avg_ms"] or 0.0
    wf_cnt = wf["cnt"] or 0
    cr_cnt = cr["cnt"] or 0
    avg_latency = (wf_ms * wf_cnt + cr_ms * cr_cnt) / total if total > 0 else 0.0

    return DashboardMetrics(
        total_runs=total,
        success_rate=round(success_rate, 2),
        avg_latency_ms=round(avg_latency, 2),
        total_tokens=wf["tokens"] or 0,
        total_cost=round(wf["cost"] or 0.0, 6),
    )


# ---------------------------------------------------------------------------
# GET /api/metrics/agents — per-agent breakdown
# ---------------------------------------------------------------------------


@router.get("/agents", response_model=list[AgentMetrics])
async def get_agent_metrics(
    range: str = Query(default="24h"),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[AgentMetrics]:
    """Return per-agent run metrics: name, count, success rate, avg tokens, avg cost."""
    ts_start, ts_end = _range_to_start(range, start, end)
    uid = user["id"]

    async with get_db() as db:
        # Workflow runs contain agent nodes — aggregate by workflow name as proxy
        # Also pull direct agent-node data from workflow_run_logs
        cursor = await db.execute(
            """
            SELECT
                COALESCE(w.name, 'Unknown workflow') AS agent_name,
                w.id AS agent_id,
                COUNT(*) AS run_count,
                SUM(CASE WHEN wr.status = 'completed' THEN 1 ELSE 0 END) AS ok,
                AVG(COALESCE(wr.total_tokens, 0)) AS avg_tokens,
                AVG(COALESCE(wr.total_cost, 0.0)) AS avg_cost
            FROM workflow_runs wr
            LEFT JOIN workflows w ON wr.workflow_id = w.id
            WHERE wr.user_id = ? AND wr.created_at >= ? AND wr.created_at <= ?
            GROUP BY w.id, w.name
            ORDER BY run_count DESC
            """,
            (uid, ts_start, ts_end),
        )
        wf_rows = await cursor.fetchall()

        # Crew runs — aggregate by crew name
        cursor = await db.execute(
            """
            SELECT
                COALESCE(c.name, 'Unknown crew') AS agent_name,
                c.id AS agent_id,
                COUNT(*) AS run_count,
                SUM(CASE WHEN cr.status = 'completed' THEN 1 ELSE 0 END) AS ok,
                0 AS avg_tokens,
                0.0 AS avg_cost
            FROM crew_runs cr
            LEFT JOIN crews c ON cr.crew_id = c.id
            WHERE cr.user_id = ? AND cr.created_at >= ? AND cr.created_at <= ?
            GROUP BY c.id, c.name
            ORDER BY run_count DESC
            """,
            (uid, ts_start, ts_end),
        )
        cr_rows = await cursor.fetchall()

    results: list[AgentMetrics] = []
    for row in wf_rows:
        r = dict(row)
        count = r["run_count"]
        ok = r["ok"] or 0
        results.append(
            AgentMetrics(
                agent_name=r["agent_name"],
                agent_id=r["agent_id"],
                run_count=count,
                success_rate=round(ok / count * 100, 2) if count > 0 else 0.0,
                avg_tokens=round(r["avg_tokens"] or 0, 2),
                avg_cost=round(r["avg_cost"] or 0, 6),
            )
        )
    for row in cr_rows:
        r = dict(row)
        count = r["run_count"]
        ok = r["ok"] or 0
        results.append(
            AgentMetrics(
                agent_name=r["agent_name"],
                agent_id=r["agent_id"],
                run_count=count,
                success_rate=round(ok / count * 100, 2) if count > 0 else 0.0,
                avg_tokens=0,
                avg_cost=0,
            )
        )

    return results


# ---------------------------------------------------------------------------
# GET /api/metrics/timeseries — time-bucketed data for charts
# ---------------------------------------------------------------------------

_VALID_METRICS = {"runs", "tokens", "cost", "latency", "success_rate"}


@router.get("/timeseries", response_model=list[TimeseriesPoint])
async def get_timeseries(
    range: str = Query(default="7d"),
    metric: str = Query(default="runs"),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[TimeseriesPoint]:
    """Return time-bucketed data for the requested metric."""
    ts_start, ts_end = _range_to_start(range, start, end)
    uid = user["id"]
    bucket_fmt = _bucket_format(range)

    if metric not in _VALID_METRICS:
        metric = "runs"

    # Build the value expression based on the requested metric
    value_expr = _metric_sql(metric)

    async with get_db() as db:
        cursor = await db.execute(
            f"""
            SELECT
                strftime('{bucket_fmt}', created_at) AS bucket,
                {value_expr}
            FROM workflow_runs
            WHERE user_id = ? AND created_at >= ? AND created_at <= ?
            GROUP BY bucket
            ORDER BY bucket
            """,
            (uid, ts_start, ts_end),
        )
        rows = await cursor.fetchall()

    return [TimeseriesPoint(bucket=r["bucket"], value=round(r["value"] or 0, 4)) for r in rows]


def _metric_sql(metric: str) -> str:
    """Return the SQL expression for the given metric name."""
    if metric == "runs":
        return "COUNT(*) AS value"
    if metric == "tokens":
        return "COALESCE(SUM(total_tokens), 0) AS value"
    if metric == "cost":
        return "COALESCE(SUM(total_cost), 0.0) AS value"
    if metric == "latency":
        return """AVG(
            CASE WHEN completed_at IS NOT NULL AND started_at IS NOT NULL
            THEN (julianday(completed_at) - julianday(started_at)) * 86400000
            END
        ) AS value"""
    if metric == "success_rate":
        return """(CAST(SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS REAL)
            / NULLIF(COUNT(*), 0) * 100) AS value"""
    return "COUNT(*) AS value"
