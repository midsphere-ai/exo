"""Plan/Execute/Verify REST API for autonomous agent execution.

Endpoints for creating plans from goals, executing steps one at a time,
and verifying each step's output before proceeding.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user
from exo_web.services.planner import (
    build_executor_prompt,
    build_planner_prompt,
    build_verifier_prompt,
    complete_step,
    create_plan,
    get_active_plan,
    get_next_step,
    get_plan,
    list_plan_versions,
    mark_step_running,
)

router = APIRouter(prefix="/api/v1/agents", tags=["plans"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PlanStepInput(BaseModel):
    step_number: int = Field(0, description="Step number")
    description: str = Field(..., min_length=1, description="Human-readable description")
    dependencies: list[int] = Field([], description="Dependencies")


class PlanCreateRequest(BaseModel):
    goal: str = Field(..., min_length=1, description="Goal")


class PlanWithStepsRequest(BaseModel):
    goal: str = Field(..., min_length=1, description="Goal")
    steps: list[PlanStepInput] = Field(..., min_length=1, description="Steps")


class PlanStepResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    plan_id: str = Field(description="Plan id")
    step_number: int = Field(description="Step number")
    description: str = Field(description="Human-readable description")
    dependencies_json: str = Field(description="JSON array of dependency identifiers")
    status: str = Field(description="Current status")
    executor_output: str = Field(description="Executor output")
    verifier_result: str = Field(description="Verifier result")
    verifier_passed: bool | None = Field(description="Verifier passed")
    started_at: str | None = Field(description="Started at")
    completed_at: str | None = Field(description="Completed at")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


class PlanResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    agent_id: str = Field(description="Associated agent identifier")
    goal: str = Field(description="Goal")
    version: int = Field(description="Version identifier")
    status: str = Field(description="Current status")
    user_id: str = Field(description="Owning user identifier")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")
    steps: list[PlanStepResponse] = Field([], description="Steps")


class StepExecuteRequest(BaseModel):
    executor_output: str = Field(..., min_length=1, description="Executor output")


class StepVerifyRequest(BaseModel):
    verifier_result: str = Field(..., min_length=1, description="Verifier result")
    verifier_passed: bool = Field(description="Verifier passed")


class StepAddRequest(BaseModel):
    description: str = Field(..., min_length=1, description="Human-readable description")
    dependencies: list[int] = Field([], description="Dependencies")
    after_step_number: int | None = None  # Insert after this step; None = append


class StepUpdateRequest(BaseModel):
    description: str | None = Field(None, description="Human-readable description")
    dependencies: list[int] | None = Field(None, description="Dependencies")


class StepReorderRequest(BaseModel):
    step_ids: list[str] = Field(..., min_length=1, description="Step ids")


class ExecuteStepResponse(BaseModel):
    step: PlanStepResponse = Field(description="Step")
    executor_prompt: str = Field(description="Executor prompt")


class VerifyStepResponse(BaseModel):
    step: PlanStepResponse = Field(description="Step")
    plan_status: str = Field(description="Plan status")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _verify_agent_ownership(agent_id: str, user_id: str) -> dict[str, Any]:
    """Ensure agent exists and belongs to user."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM agents WHERE id = ? AND user_id = ?",
            (agent_id, user_id),
        )
        row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return dict(row)


# ---------------------------------------------------------------------------
# Plan endpoints
# ---------------------------------------------------------------------------


@router.post("/{agent_id}/plans", response_model=PlanResponse, status_code=201)
async def create_agent_plan(
    agent_id: str,
    body: PlanCreateRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new execution plan for an autonomous agent.

    Creates a single-step plan from the goal. Use POST /plans/generate
    to create a plan with explicit steps parsed from LLM output.
    """
    await _verify_agent_ownership(agent_id, user["id"])

    plan = await create_plan(
        agent_id=agent_id,
        goal=body.goal,
        steps_json=[{"step_number": 1, "description": body.goal, "dependencies": []}],
        user_id=user["id"],
    )
    return plan


@router.post("/{agent_id}/plans/generate", response_model=PlanResponse, status_code=201)
async def create_plan_with_steps(
    agent_id: str,
    body: PlanWithStepsRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a plan with explicit steps (parsed from LLM output)."""
    await _verify_agent_ownership(agent_id, user["id"])

    steps = []
    for i, s in enumerate(body.steps):
        steps.append(
            {
                "step_number": s.step_number if s.step_number else i + 1,
                "description": s.description,
                "dependencies": s.dependencies,
            }
        )

    return await create_plan(
        agent_id=agent_id,
        goal=body.goal,
        steps_json=steps,
        user_id=user["id"],
    )


@router.get("/{agent_id}/plans/active", response_model=PlanResponse)
async def get_active_agent_plan(
    agent_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Get the currently active plan for an agent."""
    await _verify_agent_ownership(agent_id, user["id"])
    plan = await get_active_plan(agent_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="No active plan")
    return plan


@router.get("/{agent_id}/plans", response_model=list[PlanResponse])
async def list_agent_plans(
    agent_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """List all plan versions for an agent (newest first)."""
    await _verify_agent_ownership(agent_id, user["id"])
    return await list_plan_versions(agent_id)


@router.get("/{agent_id}/plans/{plan_id}", response_model=PlanResponse)
async def get_agent_plan(
    agent_id: str,
    plan_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Get a specific plan by ID."""
    await _verify_agent_ownership(agent_id, user["id"])
    try:
        return await get_plan(plan_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Step execution endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/{agent_id}/plans/{plan_id}/next-step",
    response_model=ExecuteStepResponse,
)
async def get_next_plan_step(
    agent_id: str,
    plan_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Get the next executable step (one whose dependencies are met).

    Returns the step along with a pre-built executor prompt that can be
    sent to the model. Marks the step as running.
    """
    agent = await _verify_agent_ownership(agent_id, user["id"])
    step = await get_next_step(plan_id)
    if step is None:
        raise HTTPException(status_code=404, detail="No pending step with met dependencies")

    # Gather prior outputs for context
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT step_number, executor_output FROM agent_plan_steps WHERE plan_id = ? AND status = 'completed' ORDER BY step_number",
            (plan_id,),
        )
        prior = [dict(r) for r in await cursor.fetchall()]

    prompt = build_executor_prompt(
        step["description"],
        agent.get("instructions", ""),
        prior,
    )

    # Mark step as running
    await mark_step_running(step["id"])

    return {"step": step, "executor_prompt": prompt}


@router.post(
    "/{agent_id}/plans/{plan_id}/steps/{step_id}/execute",
    response_model=VerifyStepResponse,
)
async def submit_step_result(
    agent_id: str,
    plan_id: str,
    step_id: str,
    body: StepExecuteRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Submit executor output for a step.

    Stores the executor output. The caller should then call the verify
    endpoint with the verifier's assessment.
    """
    await _verify_agent_ownership(agent_id, user["id"])

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM agent_plan_steps WHERE id = ? AND plan_id = ?",
            (step_id, plan_id),
        )
        step_row = await cursor.fetchone()
    if step_row is None:
        raise HTTPException(status_code=404, detail="Step not found")

    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    async with get_db() as db:
        await db.execute(
            "UPDATE agent_plan_steps SET executor_output = ?, status = 'running', updated_at = ? WHERE id = ?",
            (body.executor_output, now_str, step_id),
        )
        await db.commit()

    plan = await get_plan(plan_id)
    updated_step = next((s for s in plan["steps"] if s["id"] == step_id), plan["steps"][0])
    return {"step": updated_step, "plan_status": plan["status"]}


@router.post(
    "/{agent_id}/plans/{plan_id}/steps/{step_id}/verify",
    response_model=VerifyStepResponse,
)
async def verify_plan_step(
    agent_id: str,
    plan_id: str,
    step_id: str,
    body: StepVerifyRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Submit verification result for a step.

    The caller runs the verifier prompt against the model and sends the
    result here to record whether the step passed.
    """
    await _verify_agent_ownership(agent_id, user["id"])

    # Load current executor_output so we don't overwrite it
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT executor_output FROM agent_plan_steps WHERE id = ? AND plan_id = ?",
            (step_id, plan_id),
        )
        row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Step not found")

    await complete_step(
        step_id=step_id,
        executor_output=row["executor_output"],
        verifier_result=body.verifier_result,
        verifier_passed=body.verifier_passed,
    )

    plan = await get_plan(plan_id)
    updated_step = next((s for s in plan["steps"] if s["id"] == step_id), plan["steps"][0])
    return {"step": updated_step, "plan_status": plan["status"]}


# ---------------------------------------------------------------------------
# Step modification endpoints (add / update / delete / reorder)
# ---------------------------------------------------------------------------


@router.post(
    "/{agent_id}/plans/{plan_id}/steps",
    response_model=PlanResponse,
    status_code=201,
)
async def add_plan_step(
    agent_id: str,
    plan_id: str,
    body: StepAddRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Add a new step to an existing plan."""
    await _verify_agent_ownership(agent_id, user["id"])
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        # Determine step_number
        cursor = await db.execute(
            "SELECT MAX(step_number) as mx FROM agent_plan_steps WHERE plan_id = ?",
            (plan_id,),
        )
        row = await cursor.fetchone()
        max_num = row["mx"] if row and row["mx"] else 0

        if body.after_step_number is not None:
            new_num = body.after_step_number + 1
            # Shift all steps after insertion point
            await db.execute(
                "UPDATE agent_plan_steps SET step_number = step_number + 1, updated_at = ? WHERE plan_id = ? AND step_number >= ?",
                (now_str, plan_id, new_num),
            )
        else:
            new_num = max_num + 1

        step_id = str(__import__("uuid").uuid4())
        await db.execute(
            """
            INSERT INTO agent_plan_steps
                (id, plan_id, step_number, description, dependencies_json, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (
                step_id,
                plan_id,
                new_num,
                body.description,
                json.dumps(body.dependencies),
                now_str,
                now_str,
            ),
        )
        await db.execute("UPDATE agent_plans SET updated_at = ? WHERE id = ?", (now_str, plan_id))
        await db.commit()

    return await get_plan(plan_id)


@router.put(
    "/{agent_id}/plans/{plan_id}/steps/{step_id}",
    response_model=PlanResponse,
)
async def update_plan_step(
    agent_id: str,
    plan_id: str,
    step_id: str,
    body: StepUpdateRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a pending step's description or dependencies."""
    await _verify_agent_ownership(agent_id, user["id"])
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM agent_plan_steps WHERE id = ? AND plan_id = ?",
            (step_id, plan_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Step not found")
        if row["status"] not in ("pending",):
            raise HTTPException(status_code=422, detail="Only pending steps can be edited")

        sets: list[str] = ["updated_at = ?"]
        params: list[Any] = [now_str]
        if body.description is not None:
            sets.append("description = ?")
            params.append(body.description)
        if body.dependencies is not None:
            sets.append("dependencies_json = ?")
            params.append(json.dumps(body.dependencies))
        params.append(step_id)

        await db.execute(
            f"UPDATE agent_plan_steps SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        await db.execute("UPDATE agent_plans SET updated_at = ? WHERE id = ?", (now_str, plan_id))
        await db.commit()

    return await get_plan(plan_id)


@router.delete(
    "/{agent_id}/plans/{plan_id}/steps/{step_id}",
    status_code=204,
)
async def delete_plan_step(
    agent_id: str,
    plan_id: str,
    step_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Remove a pending step from a plan."""
    await _verify_agent_ownership(agent_id, user["id"])
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM agent_plan_steps WHERE id = ? AND plan_id = ?",
            (step_id, plan_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Step not found")
        if row["status"] not in ("pending",):
            raise HTTPException(status_code=422, detail="Only pending steps can be removed")

        deleted_num = row["step_number"]
        await db.execute("DELETE FROM agent_plan_steps WHERE id = ?", (step_id,))
        # Renumber subsequent steps
        await db.execute(
            "UPDATE agent_plan_steps SET step_number = step_number - 1, updated_at = ? WHERE plan_id = ? AND step_number > ?",
            (now_str, plan_id, deleted_num),
        )
        await db.execute("UPDATE agent_plans SET updated_at = ? WHERE id = ?", (now_str, plan_id))
        await db.commit()


@router.put(
    "/{agent_id}/plans/{plan_id}/reorder",
    response_model=PlanResponse,
)
async def reorder_plan_steps(
    agent_id: str,
    plan_id: str,
    body: StepReorderRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Reorder plan steps. Only pending steps can be reordered."""
    await _verify_agent_ownership(agent_id, user["id"])
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        for i, sid in enumerate(body.step_ids, start=1):
            await db.execute(
                "UPDATE agent_plan_steps SET step_number = ?, updated_at = ? WHERE id = ? AND plan_id = ?",
                (i, now_str, sid, plan_id),
            )
        await db.execute("UPDATE agent_plans SET updated_at = ? WHERE id = ?", (now_str, plan_id))
        await db.commit()

    return await get_plan(plan_id)


# ---------------------------------------------------------------------------
# Planner prompt endpoint (for client-side LLM calls)
# ---------------------------------------------------------------------------


@router.get("/{agent_id}/planner-prompt")
async def get_planner_prompt(
    agent_id: str,
    goal: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, str]:
    """Return the planner system prompt for a given goal.

    The client sends this prompt to the model and parses the resulting
    JSON steps to create a plan via POST /plans/generate.
    """
    agent = await _verify_agent_ownership(agent_id, user["id"])
    prompt = build_planner_prompt(goal, agent.get("instructions", ""))
    return {"prompt": prompt}


@router.get("/{agent_id}/plans/{plan_id}/steps/{step_id}/verifier-prompt")
async def get_verifier_prompt(
    agent_id: str,
    plan_id: str,
    step_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, str]:
    """Return the verifier prompt for a step that has executor output."""
    await _verify_agent_ownership(agent_id, user["id"])

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM agent_plan_steps WHERE id = ? AND plan_id = ?",
            (step_id, plan_id),
        )
        row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Step not found")

    step = dict(row)
    prompt = build_verifier_prompt(step["description"], step["executor_output"])
    return {"prompt": prompt}
