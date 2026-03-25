"""Planner / Executor / Verifier orchestration service.

Implements autonomous task decomposition with a plan - execute - verify cycle.
When an agent has ``autonomous_mode`` enabled the service:

1. **Plans** - sends the user goal to the agent's model and produces a numbered
   list of steps with descriptions and dependency information.
2. **Executes** - runs one step at a time, awaiting the result before moving to
   the next (respects dependency ordering).
3. **Verifies** - after each step the verifier checks output against the step's
   acceptance criteria and marks it passed or failed.

Plan versioning is built-in - when a plan is regenerated mid-execution the
previous version is marked *superseded* and the new version gets an incremented
``version`` number.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from exo_web.database import get_db

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """\
You are a planning agent. Given a user goal, decompose it into a numbered list
of concrete, actionable steps. Return ONLY valid JSON — an array of objects,
each with keys: "step_number" (int), "description" (string), and
"dependencies" (array of step_number ints that must complete first).

Example:
[
  {"step_number": 1, "description": "Research API options", "dependencies": []},
  {"step_number": 2, "description": "Write API client", "dependencies": [1]},
  {"step_number": 3, "description": "Add tests", "dependencies": [2]}
]
"""

_VERIFIER_SYSTEM_PROMPT = """\
You are a verification agent. Given a step description and its executor output,
determine whether the step was completed successfully. Return ONLY valid JSON
with keys: "passed" (boolean) and "reason" (string explaining your assessment).
"""


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Plan CRUD
# ---------------------------------------------------------------------------


async def create_plan(
    agent_id: str,
    goal: str,
    steps_json: list[dict[str, Any]],
    user_id: str,
) -> dict[str, Any]:
    """Create a new plan (and its steps) for an agent.

    If the agent already has an *active* plan it is marked ``superseded`` and
    the new plan gets an incremented version number.
    """
    plan_id = str(uuid.uuid4())
    now = _now()

    async with get_db() as db:
        # Determine version number & supersede any active plan
        cursor = await db.execute(
            "SELECT id, version FROM agent_plans WHERE agent_id = ? AND status = 'active' ORDER BY version DESC LIMIT 1",
            (agent_id,),
        )
        active = await cursor.fetchone()
        version = 1
        if active:
            version = active["version"] + 1
            await db.execute(
                "UPDATE agent_plans SET status = 'superseded', updated_at = ? WHERE id = ?",
                (now, active["id"]),
            )

        await db.execute(
            """
            INSERT INTO agent_plans (id, agent_id, goal, version, status, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'active', ?, ?, ?)
            """,
            (plan_id, agent_id, goal, version, user_id, now, now),
        )

        for step in steps_json:
            step_id = str(uuid.uuid4())
            await db.execute(
                """
                INSERT INTO agent_plan_steps
                    (id, plan_id, step_number, description, dependencies_json, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    step_id,
                    plan_id,
                    step["step_number"],
                    step["description"],
                    json.dumps(step.get("dependencies", [])),
                    now,
                    now,
                ),
            )

        await db.commit()

        return await _load_plan(db, plan_id)


async def get_plan(plan_id: str) -> dict[str, Any]:
    """Load a single plan with its steps."""
    async with get_db() as db:
        return await _load_plan(db, plan_id)


async def get_active_plan(agent_id: str) -> dict[str, Any] | None:
    """Return the currently active plan for an agent, or None."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM agent_plans WHERE agent_id = ? AND status = 'active' LIMIT 1",
            (agent_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return await _load_plan(db, row["id"])


async def list_plan_versions(agent_id: str) -> list[dict[str, Any]]:
    """Return all plan versions for an agent (newest first)."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM agent_plans WHERE agent_id = ? ORDER BY version DESC",
            (agent_id,),
        )
        rows = await cursor.fetchall()
        plans: list[dict[str, Any]] = []
        for row in rows:
            plans.append(await _load_plan(db, row["id"]))
        return plans


# ---------------------------------------------------------------------------
# Step execution tracking
# ---------------------------------------------------------------------------


async def get_next_step(plan_id: str) -> dict[str, Any] | None:
    """Return the next pending step whose dependencies are all completed."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM agent_plan_steps WHERE plan_id = ? ORDER BY step_number",
            (plan_id,),
        )
        steps = [dict(r) for r in await cursor.fetchall()]

    completed_steps = {s["step_number"] for s in steps if s["status"] == "completed"}

    for step in steps:
        if step["status"] != "pending":
            continue
        deps = json.loads(step["dependencies_json"])
        if all(d in completed_steps for d in deps):
            return step

    return None


async def mark_step_running(step_id: str) -> None:
    """Mark a step as running."""
    now = _now()
    async with get_db() as db:
        await db.execute(
            "UPDATE agent_plan_steps SET status = 'running', started_at = ?, updated_at = ? WHERE id = ?",
            (now, now, step_id),
        )
        await db.commit()


async def complete_step(
    step_id: str,
    executor_output: str,
    verifier_result: str,
    verifier_passed: bool,
) -> None:
    """Record step completion with executor output and verifier assessment."""
    now = _now()
    status = "completed" if verifier_passed else "failed"
    async with get_db() as db:
        await db.execute(
            """
            UPDATE agent_plan_steps
            SET status = ?, executor_output = ?, verifier_result = ?,
                verifier_passed = ?, completed_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, executor_output, verifier_result, int(verifier_passed), now, now, step_id),
        )

        # Check if all steps completed → mark plan completed
        cursor = await db.execute("SELECT plan_id FROM agent_plan_steps WHERE id = ?", (step_id,))
        row = await cursor.fetchone()
        if row:
            cursor2 = await db.execute(
                "SELECT COUNT(*) as total, SUM(CASE WHEN status IN ('completed','failed','skipped') THEN 1 ELSE 0 END) as done FROM agent_plan_steps WHERE plan_id = ?",
                (row["plan_id"],),
            )
            counts = await cursor2.fetchone()
            if counts and counts["total"] == counts["done"]:
                await db.execute(
                    "UPDATE agent_plans SET status = 'completed', updated_at = ? WHERE id = ?",
                    (now, row["plan_id"]),
                )

        await db.commit()


# ---------------------------------------------------------------------------
# Prompt builders (used by routes to send to model)
# ---------------------------------------------------------------------------


def build_planner_prompt(goal: str, agent_instructions: str) -> str:
    """Build the prompt sent to the LLM for plan generation."""
    parts = [_PLANNER_SYSTEM_PROMPT]
    if agent_instructions:
        parts.append(f"\nAgent context:\n{agent_instructions}")
    parts.append(f"\nUser goal:\n{goal}")
    return "\n".join(parts)


def build_executor_prompt(
    step_description: str, agent_instructions: str, prior_outputs: list[dict[str, Any]]
) -> str:
    """Build the prompt for executing a single step."""
    parts = [agent_instructions or "You are an execution agent. Complete the given task."]
    if prior_outputs:
        parts.append("\nPrevious step results:")
        for po in prior_outputs:
            parts.append(f"  Step {po['step_number']}: {po['executor_output'][:500]}")
    parts.append(f"\nYour current task:\n{step_description}")
    parts.append("\nProvide a clear, detailed response completing this task.")
    return "\n".join(parts)


def build_verifier_prompt(step_description: str, executor_output: str) -> str:
    """Build the prompt for verifying a step's output."""
    return f"""{_VERIFIER_SYSTEM_PROMPT}

Step description: {step_description}

Executor output:
{executor_output}

Return JSON: {{"passed": true/false, "reason": "..."}}"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _load_plan(db: Any, plan_id: str) -> dict[str, Any]:
    """Load a plan and its steps from the database."""
    cursor = await db.execute("SELECT * FROM agent_plans WHERE id = ?", (plan_id,))
    row = await cursor.fetchone()
    if row is None:
        msg = f"Plan not found: {plan_id}"
        raise ValueError(msg)
    plan = dict(row)

    cursor = await db.execute(
        "SELECT * FROM agent_plan_steps WHERE plan_id = ? ORDER BY step_number",
        (plan_id,),
    )
    plan["steps"] = [dict(r) for r in await cursor.fetchall()]
    # Convert verifier_passed int to bool
    for s in plan["steps"]:
        if s["verifier_passed"] is not None:
            s["verifier_passed"] = bool(s["verifier_passed"])
    return plan
