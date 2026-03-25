"""Agent benchmarking REST API.

Run the same evaluation suite against multiple agents and compare results.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.pagination import paginate
from exo_web.routes.auth import get_current_user
from exo_web.sanitize import sanitize_html
from exo_web.services.evaluators import EVALUATORS, run_evaluator

router = APIRouter(prefix="/api/v1/benchmarks", tags=["benchmarks"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class BenchmarkCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    description: str = Field("", description="Human-readable description")
    evaluation_id: str = Field(..., min_length=1, description="Associated evaluation identifier")


class BenchmarkUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255, description="Display name")
    description: str | None = Field(None, description="Human-readable description")


class BenchmarkRunCreate(BaseModel):
    agent_ids: list[str] = Field(..., min_length=2, description="Agent ids")


class BenchmarkResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    description: str = Field(description="Human-readable description")
    evaluation_id: str = Field(description="Associated evaluation identifier")
    user_id: str = Field(description="Owning user identifier")
    created_at: str = Field(description="ISO 8601 creation timestamp")


class BenchmarkRunResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    benchmark_id: str = Field(description="Associated benchmark identifier")
    agent_ids_json: str = Field(description="Agent ids json")
    status: str = Field(description="Current status")
    started_at: str | None = Field(description="Started at")
    completed_at: str | None = Field(description="Completed at")
    created_at: str = Field(description="ISO 8601 creation timestamp")


class BenchmarkResultResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    run_id: str = Field(description="Associated run identifier")
    agent_id: str = Field(description="Associated agent identifier")
    agent_name: str = Field(description="Agent name")
    results_json: str = Field(description="Results json")
    overall_score: float = Field(description="Overall score")
    pass_rate: float = Field(description="Pass rate")
    total_latency_ms: float = Field(description="Total latency ms")
    avg_latency_ms: float = Field(description="Average latency in milliseconds")
    estimated_cost: float = Field(description="Estimated cost")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row)


async def _send_to_agent(agent_id: str, user_id: str, message: str) -> tuple[str, float]:
    """Send a message to an agent and return (response_text, latency_ms)."""
    from exo_web.services.agent_runtime import _load_agent_row, _resolve_provider

    row = await _load_agent_row(agent_id)
    provider_type = row.get("model_provider", "")
    model_name = row.get("model_name", "")
    if not provider_type or not model_name:
        return "[error: agent has no model configured]", 0.0

    try:
        provider = await _resolve_provider(provider_type, model_name, user_id)
        instructions = row.get("instructions", "")
        messages: list[dict[str, str]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": message})
        t0 = time.monotonic()
        resp = await provider.complete(messages=messages, model=model_name)
        latency_ms = (time.monotonic() - t0) * 1000
        return resp.content, latency_ms
    except Exception as exc:
        return f"[error: {exc}]", 0.0


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


@router.get("")
async def list_benchmarks(
    cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """List benchmark suites for the current user."""
    async with get_db() as db:
        result = await paginate(
            db,
            table="benchmarks",
            conditions=["user_id = ?"],
            params=[user["id"]],
            cursor=cursor,
            limit=limit,
            row_mapper=_row_to_dict,
        )
        return result.model_dump()


@router.post("", response_model=BenchmarkResponse, status_code=201)
async def create_benchmark(
    body: BenchmarkCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new benchmark suite linked to an evaluation."""
    async with get_db() as db:
        # Validate evaluation exists and belongs to user
        cur = await db.execute(
            "SELECT id FROM evaluations WHERE id = ? AND user_id = ?",
            (body.evaluation_id, user["id"]),
        )
        if await cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="Evaluation not found")

        bench_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        await db.execute(
            """
            INSERT INTO benchmarks (id, name, description, evaluation_id, user_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                bench_id,
                sanitize_html(body.name),
                sanitize_html(body.description),
                body.evaluation_id,
                user["id"],
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM benchmarks WHERE id = ?", (bench_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.get("/{benchmark_id}", response_model=BenchmarkResponse)
async def get_benchmark(
    benchmark_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single benchmark suite."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM benchmarks WHERE id = ? AND user_id = ?",
            (benchmark_id, user["id"]),
        )
        row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return _row_to_dict(row)


@router.put("/{benchmark_id}", response_model=BenchmarkResponse)
async def update_benchmark(
    benchmark_id: str,
    body: BenchmarkUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a benchmark suite."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM benchmarks WHERE id = ? AND user_id = ?",
            (benchmark_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Benchmark not found")

        updates: dict[str, Any] = {}
        if body.name is not None:
            updates["name"] = sanitize_html(body.name)
        if body.description is not None:
            updates["description"] = sanitize_html(body.description)

        if not updates:
            raise HTTPException(status_code=422, detail="No fields to update")

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), benchmark_id]
        await db.execute(
            f"UPDATE benchmarks SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM benchmarks WHERE id = ?", (benchmark_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.delete("/{benchmark_id}", status_code=204)
async def delete_benchmark(
    benchmark_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a benchmark and all its runs/results."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM benchmarks WHERE id = ? AND user_id = ?",
            (benchmark_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Benchmark not found")

        await db.execute("DELETE FROM benchmarks WHERE id = ?", (benchmark_id,))
        await db.commit()


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------


@router.post(
    "/{benchmark_id}/run",
    response_model=BenchmarkRunResponse,
)
async def run_benchmark(
    benchmark_id: str,
    body: BenchmarkRunCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Run the evaluation test cases against all specified agents."""
    async with get_db() as db:
        # Verify benchmark ownership
        cur = await db.execute(
            "SELECT * FROM benchmarks WHERE id = ? AND user_id = ?",
            (benchmark_id, user["id"]),
        )
        bench_row = await cur.fetchone()
        if bench_row is None:
            raise HTTPException(status_code=404, detail="Benchmark not found")
        bench = _row_to_dict(bench_row)

        # Load evaluation test cases
        cur = await db.execute(
            "SELECT * FROM evaluations WHERE id = ? AND user_id = ?",
            (bench["evaluation_id"], user["id"]),
        )
        eval_row = await cur.fetchone()
        if eval_row is None:
            raise HTTPException(status_code=404, detail="Linked evaluation not found")
        evaluation = _row_to_dict(eval_row)

        try:
            test_cases: list[dict[str, Any]] = json.loads(evaluation["test_cases_json"])
        except (json.JSONDecodeError, TypeError):
            test_cases = []

        if not test_cases:
            raise HTTPException(status_code=422, detail="No test cases in evaluation")

        # Validate all agents exist and belong to user
        agent_names: dict[str, str] = {}
        for aid in body.agent_ids:
            cur = await db.execute(
                "SELECT id, name FROM agents WHERE id = ? AND user_id = ?",
                (aid, user["id"]),
            )
            agent_row = await cur.fetchone()
            if agent_row is None:
                raise HTTPException(status_code=404, detail=f"Agent {aid} not found")
            agent_names[aid] = dict(agent_row)["name"]

    # Create the benchmark run
    run_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO benchmark_runs (id, benchmark_id, agent_ids_json, status, started_at, created_at)
            VALUES (?, ?, ?, 'running', ?, ?)
            """,
            (run_id, benchmark_id, json.dumps(body.agent_ids), now, now),
        )
        await db.commit()

    # Run each agent against all test cases
    for agent_id in body.agent_ids:
        results: list[dict[str, Any]] = []
        total_score = 0.0
        pass_count = 0
        total_latency_ms = 0.0

        for tc in test_cases:
            input_msg = tc.get("input", "")
            expected = tc.get("expected", "")
            evaluator_type = tc.get("evaluator", "exact_match")
            if evaluator_type not in EVALUATORS:
                evaluator_type = "exact_match"

            actual, latency_ms = await _send_to_agent(agent_id, user["id"], input_msg)
            total_latency_ms += latency_ms

            kwargs: dict[str, Any] = {}
            if evaluator_type == "llm_as_judge":
                from exo_web.services.agent_runtime import _resolve_provider

                async with get_db() as db2:
                    cur2 = await db2.execute(
                        "SELECT model_provider, model_name FROM agents WHERE id = ?",
                        (agent_id,),
                    )
                    a_row = await cur2.fetchone()
                if a_row:
                    a_data = dict(a_row)
                    kwargs["provider_resolver"] = _resolve_provider
                    kwargs["provider_type"] = a_data.get("model_provider", "")
                    kwargs["model_name"] = a_data.get("model_name", "")
                    kwargs["user_id"] = user["id"]

            score = await run_evaluator(evaluator_type, expected, actual, **kwargs)
            passed = score >= 0.5
            if passed:
                pass_count += 1
            total_score += score

            results.append(
                {
                    "input": input_msg,
                    "expected": expected,
                    "actual": actual,
                    "evaluator": evaluator_type,
                    "score": round(score, 4),
                    "passed": passed,
                    "latency_ms": round(latency_ms, 1),
                }
            )

        num_cases = len(test_cases)
        overall_score = round(total_score / num_cases, 4) if num_cases else 0.0
        pass_rate = round(pass_count / num_cases, 4) if num_cases else 0.0
        avg_latency = round(total_latency_ms / num_cases, 1) if num_cases else 0.0

        result_id = str(uuid.uuid4())
        async with get_db() as db:
            await db.execute(
                """
                INSERT INTO benchmark_results
                    (id, run_id, agent_id, agent_name, results_json,
                     overall_score, pass_rate, total_latency_ms, avg_latency_ms, estimated_cost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result_id,
                    run_id,
                    agent_id,
                    agent_names.get(agent_id, ""),
                    json.dumps(results),
                    overall_score,
                    pass_rate,
                    round(total_latency_ms, 1),
                    avg_latency,
                    0.0,
                ),
            )
            await db.commit()

    # Mark run as completed
    completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    async with get_db() as db:
        await db.execute(
            "UPDATE benchmark_runs SET status = 'completed', completed_at = ? WHERE id = ?",
            (completed_at, run_id),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM benchmark_runs WHERE id = ?", (run_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Runs & results
# ---------------------------------------------------------------------------


@router.get("/{benchmark_id}/runs", response_model=list[BenchmarkRunResponse])
async def list_benchmark_runs(
    benchmark_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """List all runs for a benchmark."""
    async with get_db() as db:
        cur = await db.execute(
            "SELECT id FROM benchmarks WHERE id = ? AND user_id = ?",
            (benchmark_id, user["id"]),
        )
        if await cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="Benchmark not found")

        cursor = await db.execute(
            "SELECT * FROM benchmark_runs WHERE benchmark_id = ? ORDER BY created_at DESC",
            (benchmark_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


@router.get(
    "/{benchmark_id}/runs/{run_id}/results",
    response_model=list[BenchmarkResultResponse],
)
async def get_benchmark_results(
    benchmark_id: str,
    run_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return per-agent results for a benchmark run."""
    async with get_db() as db:
        cur = await db.execute(
            "SELECT id FROM benchmarks WHERE id = ? AND user_id = ?",
            (benchmark_id, user["id"]),
        )
        if await cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="Benchmark not found")

        cursor = await db.execute(
            """
            SELECT * FROM benchmark_results
            WHERE run_id = ?
            ORDER BY overall_score DESC
            """,
            (run_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


@router.get("/{benchmark_id}/leaderboard")
async def get_leaderboard(
    benchmark_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Aggregate leaderboard across all runs — best score per agent."""
    async with get_db() as db:
        cur = await db.execute(
            "SELECT id FROM benchmarks WHERE id = ? AND user_id = ?",
            (benchmark_id, user["id"]),
        )
        if await cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="Benchmark not found")

        cursor = await db.execute(
            """
            SELECT
                br.agent_id,
                br.agent_name,
                MAX(br.overall_score) AS best_score,
                AVG(br.overall_score) AS avg_score,
                MAX(br.pass_rate) AS best_pass_rate,
                AVG(br.avg_latency_ms) AS avg_latency_ms,
                SUM(br.estimated_cost) AS total_cost,
                COUNT(*) AS run_count
            FROM benchmark_results br
            JOIN benchmark_runs brun ON brun.id = br.run_id
            WHERE brun.benchmark_id = ? AND brun.status = 'completed'
            GROUP BY br.agent_id
            ORDER BY best_score DESC, avg_latency_ms ASC
            """,
            (benchmark_id,),
        )
        rows = await cursor.fetchall()
        result = []
        for i, r in enumerate(rows):
            d = _row_to_dict(r)
            d["rank"] = i + 1
            # Cost-efficiency metric: score per dollar (higher is better)
            cost = d.get("total_cost", 0.0) or 0.0
            d["score_per_dollar"] = round(d["best_score"] / cost, 4) if cost > 0 else None
            result.append(d)
        return result


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@router.get("/{benchmark_id}/runs/{run_id}/export")
async def export_benchmark_results(
    benchmark_id: str,
    run_id: str,
    fmt: str = Query("json", pattern="^(json|csv)$"),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> StreamingResponse:
    """Export benchmark results as JSON or CSV."""
    async with get_db() as db:
        cur = await db.execute(
            "SELECT id FROM benchmarks WHERE id = ? AND user_id = ?",
            (benchmark_id, user["id"]),
        )
        if await cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="Benchmark not found")

        cursor = await db.execute(
            """
            SELECT * FROM benchmark_results
            WHERE run_id = ?
            ORDER BY overall_score DESC
            """,
            (run_id,),
        )
        rows = await cursor.fetchall()

    results = [_row_to_dict(r) for r in rows]

    if fmt == "csv":
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "Agent",
                "Overall Score",
                "Pass Rate",
                "Avg Latency (ms)",
                "Total Latency (ms)",
                "Est. Cost",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["agent_name"],
                    r["overall_score"],
                    r["pass_rate"],
                    r["avg_latency_ms"],
                    r["total_latency_ms"],
                    r["estimated_cost"],
                ]
            )
        content = output.getvalue()
        return StreamingResponse(
            iter([content]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=benchmark-{run_id}.csv"},
        )

    # JSON export
    export_data = []
    for r in results:
        detail_results = json.loads(r.get("results_json", "[]"))
        export_data.append(
            {
                "agent_id": r["agent_id"],
                "agent_name": r["agent_name"],
                "overall_score": r["overall_score"],
                "pass_rate": r["pass_rate"],
                "avg_latency_ms": r["avg_latency_ms"],
                "total_latency_ms": r["total_latency_ms"],
                "estimated_cost": r["estimated_cost"],
                "test_results": detail_results,
            }
        )

    content = json.dumps(export_data, indent=2)
    return StreamingResponse(
        iter([content]),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=benchmark-{run_id}.json"},
    )
