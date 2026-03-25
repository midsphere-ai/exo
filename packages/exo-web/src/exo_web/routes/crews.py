"""Crews CRUD REST API and crew execution with WebSocket streaming."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user
from exo_web.sanitize import sanitize_html

router = APIRouter(prefix="/api/v1/crews", tags=["crews"])

# Keep references to background tasks so they aren't garbage-collected.
_background_tasks: set[asyncio.Task[Any]] = set()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CrewTaskCreate(BaseModel):
    agent_id: str = Field(..., min_length=1, description="Associated agent identifier")
    task_description: str = Field("", description="Task description")
    expected_output: str = Field("", description="Expected output format or content")
    task_order: int = Field(0, description="Execution order within the crew")
    dependencies_json: str = Field("[]", description="JSON array of dependency identifiers")


class CrewTaskUpdate(BaseModel):
    agent_id: str | None = Field(None, min_length=1, description="Associated agent identifier")
    task_description: str | None = Field(None, description="Task description")
    expected_output: str | None = Field(None, description="Expected output format or content")
    task_order: int | None = Field(None, description="Execution order within the crew")
    dependencies_json: str | None = Field(None, description="JSON array of dependency identifiers")


class CrewTaskResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    crew_id: str = Field(description="Associated crew identifier")
    agent_id: str = Field(description="Associated agent identifier")
    task_description: str = Field(description="Task description")
    expected_output: str = Field(description="Expected output format or content")
    task_order: int = Field(description="Execution order within the crew")
    dependencies_json: str = Field(description="JSON array of dependency identifiers")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


class CrewCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    project_id: str = Field(..., min_length=1, description="Associated project identifier")
    description: str = Field("", description="Human-readable description")
    process_type: str = Field(
        "sequential",
        pattern=r"^(sequential|parallel)$",
        description="Crew execution mode (sequential, parallel)",
    )
    config_json: str = Field("{}", description="JSON configuration object")


class CrewUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255, description="Display name")
    description: str | None = Field(None, description="Human-readable description")
    process_type: str | None = Field(
        None,
        pattern=r"^(sequential|parallel)$",
        description="Crew execution mode (sequential, parallel)",
    )
    config_json: str | None = Field(None, description="JSON configuration object")


class CrewResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    description: str = Field(description="Human-readable description")
    process_type: str = Field(description="Crew execution mode (sequential, parallel)")
    config_json: str = Field(description="JSON configuration object")
    project_id: str = Field(description="Associated project identifier")
    user_id: str = Field(description="Owning user identifier")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")
    tasks: list[CrewTaskResponse] = Field([], description="Tasks")


class CrewRunRequest(BaseModel):
    input: str = Field("", description="Input text or data")


class CrewRunTaskResult(BaseModel):
    task_id: str = Field(description="Task id")
    agent_id: str = Field(description="Associated agent identifier")
    task_description: str = Field(description="Task description")
    status: str = Field(description="Current status")
    output: str = Field("", description="Output text or data")
    error: str = Field("", description="Error message if failed")


class CrewRunResponse(BaseModel):
    crew_id: str = Field(description="Associated crew identifier")
    status: str = Field(description="Current status")
    process_type: str = Field(description="Crew execution mode (sequential, parallel)")
    results: list[CrewRunTaskResult] = Field(description="Results")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict."""
    return dict(row)


async def _verify_crew_ownership(db: Any, crew_id: str, user_id: str) -> dict[str, Any]:
    """Verify crew exists and belongs to user. Returns row dict or raises 404."""
    cursor = await db.execute(
        "SELECT * FROM crews WHERE id = ? AND user_id = ?",
        (crew_id, user_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Crew not found")
    return _row_to_dict(row)


async def _get_crew_tasks(db: Any, crew_id: str) -> list[dict[str, Any]]:
    """Return tasks for a crew, ordered by task_order."""
    cursor = await db.execute(
        "SELECT * FROM crew_tasks WHERE crew_id = ? ORDER BY task_order ASC",
        (crew_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_dict(r) for r in rows]


async def _crew_with_tasks(db: Any, crew_id: str) -> dict[str, Any]:
    """Return crew dict with nested tasks list."""
    cursor = await db.execute("SELECT * FROM crews WHERE id = ?", (crew_id,))
    row = await cursor.fetchone()
    crew = _row_to_dict(row)
    crew["tasks"] = await _get_crew_tasks(db, crew_id)
    return crew


async def _execute_single_task(
    task: dict[str, Any], crew_input: str, previous_outputs: list[str]
) -> dict[str, Any]:
    """Execute a single crew task by calling the agent's configured model.

    Returns a result dict with status/output/error/duration_ms.
    """
    import httpx

    from exo_web.crypto import decrypt_api_key

    agent_id = task["agent_id"]
    task_description = task["task_description"] or "Complete the assigned task."
    expected_output = task["expected_output"]
    run_task_id = task.get("run_task_id", task["id"])

    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM agents WHERE id = ?", (agent_id,))
        agent_row = await cursor.fetchone()
        if agent_row is None:
            return {
                "task_id": run_task_id,
                "agent_id": agent_id,
                "task_description": task_description,
                "status": "failed",
                "output": "",
                "error": "Agent not found",
                "duration_ms": 0,
            }
        agent = _row_to_dict(agent_row)

        # Resolve provider and key
        provider_id = agent.get("model_provider", "")
        model_name = agent.get("model_name", "")
        if not provider_id or not model_name:
            return {
                "task_id": run_task_id,
                "agent_id": agent_id,
                "task_description": task_description,
                "status": "failed",
                "output": "",
                "error": "Agent has no model configured",
                "duration_ms": 0,
            }

        cursor = await db.execute(
            "SELECT * FROM providers WHERE id = ?",
            (provider_id,),
        )
        provider_row = await cursor.fetchone()
        if provider_row is None:
            return {
                "task_id": run_task_id,
                "agent_id": agent_id,
                "task_description": task_description,
                "status": "failed",
                "output": "",
                "error": "Provider not found",
                "duration_ms": 0,
            }
        provider = _row_to_dict(provider_row)

        api_key = ""
        if provider.get("encrypted_api_key"):
            api_key = decrypt_api_key(provider["encrypted_api_key"])
        else:
            cursor = await db.execute(
                "SELECT encrypted_api_key FROM provider_keys WHERE provider_id = ? AND status = 'active' LIMIT 1",
                (provider_id,),
            )
            key_row = await cursor.fetchone()
            if key_row:
                api_key = decrypt_api_key(key_row["encrypted_api_key"])

        if not api_key:
            return {
                "task_id": run_task_id,
                "agent_id": agent_id,
                "task_description": task_description,
                "status": "failed",
                "output": "",
                "error": "No API key configured for provider",
                "duration_ms": 0,
            }

    # Build the prompt
    context_parts: list[str] = []
    if agent.get("instructions"):
        context_parts.append(f"System instructions: {agent['instructions']}")
    if crew_input:
        context_parts.append(f"Crew input: {crew_input}")
    if previous_outputs:
        context_parts.append("Previous task outputs:\n" + "\n---\n".join(previous_outputs))

    prompt = f"{task_description}"
    if expected_output:
        prompt += f"\n\nExpected output: {expected_output}"
    if context_parts:
        prompt = "\n".join(context_parts) + "\n\n" + prompt

    provider_type = provider["provider_type"]
    base_url = provider.get("base_url") or ""
    start_time = time.monotonic()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            if provider_type in ("openai", "custom"):
                url = (base_url or "https://api.openai.com") + "/v1/chat/completions"
                resp = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 2048,
                    },
                )
            elif provider_type == "anthropic":
                url = "https://api.anthropic.com/v1/messages"
                resp = await client.post(
                    url,
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 2048,
                    },
                )
            elif provider_type == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                resp = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                )
            elif provider_type == "ollama":
                url = (base_url or "http://localhost:11434") + "/api/generate"
                resp = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json={"model": model_name, "prompt": prompt, "stream": False},
                )
            else:
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                return {
                    "task_id": run_task_id,
                    "agent_id": agent_id,
                    "task_description": task_description,
                    "status": "failed",
                    "output": "",
                    "error": f"Unsupported provider type: {provider_type}",
                    "duration_ms": elapsed_ms,
                }

            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            if resp.status_code >= 400:
                error_text = resp.text[:500]
                return {
                    "task_id": run_task_id,
                    "agent_id": agent_id,
                    "task_description": task_description,
                    "status": "failed",
                    "output": "",
                    "error": f"API error ({resp.status_code}): {error_text}",
                    "duration_ms": elapsed_ms,
                }

            data = resp.json()

            output = ""
            if provider_type in ("openai", "custom"):
                output = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            elif provider_type == "anthropic":
                content_blocks = data.get("content", [])
                output = "".join(
                    b.get("text", "") for b in content_blocks if b.get("type") == "text"
                )
            elif provider_type == "gemini":
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    output = "".join(p.get("text", "") for p in parts)
            elif provider_type == "ollama":
                output = data.get("response", "")

            return {
                "task_id": run_task_id,
                "agent_id": agent_id,
                "task_description": task_description,
                "status": "completed",
                "output": output,
                "error": "",
                "duration_ms": elapsed_ms,
            }

    except Exception as exc:
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        return {
            "task_id": run_task_id,
            "agent_id": agent_id,
            "task_description": task_description,
            "status": "failed",
            "output": "",
            "error": f"Connection error: {exc!s}",
            "duration_ms": elapsed_ms,
        }


# ---------------------------------------------------------------------------
# Crew CRUD endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[CrewResponse])
async def list_crews(
    project_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all crews for the current user, optionally filtered by project."""
    async with get_db() as db:
        if project_id:
            cursor = await db.execute(
                "SELECT * FROM crews WHERE user_id = ? AND project_id = ? ORDER BY created_at DESC",
                (user["id"], project_id),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM crews WHERE user_id = ? ORDER BY created_at DESC",
                (user["id"],),
            )
        rows = await cursor.fetchall()
        result = []
        for r in rows:
            crew = _row_to_dict(r)
            crew["tasks"] = await _get_crew_tasks(db, crew["id"])
            result.append(crew)
        return result


@router.post("", response_model=CrewResponse, status_code=201)
async def create_crew(
    body: CrewCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new crew."""
    async with get_db() as db:
        # Verify project exists and belongs to user
        cursor = await db.execute(
            "SELECT id FROM projects WHERE id = ? AND user_id = ?",
            (body.project_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")

        crew_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO crews (id, name, description, process_type, config_json, project_id, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                crew_id,
                sanitize_html(body.name),
                sanitize_html(body.description),
                body.process_type,
                body.config_json,
                body.project_id,
                user["id"],
                now,
                now,
            ),
        )
        await db.commit()

        return await _crew_with_tasks(db, crew_id)


@router.get("/{crew_id}", response_model=CrewResponse)
async def get_crew(
    crew_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single crew by ID with its tasks."""
    async with get_db() as db:
        await _verify_crew_ownership(db, crew_id, user["id"])
        return await _crew_with_tasks(db, crew_id)


@router.put("/{crew_id}", response_model=CrewResponse)
async def update_crew(
    crew_id: str,
    body: CrewUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a crew's fields."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    for field in ("name", "description"):
        if field in updates and isinstance(updates[field], str):
            updates[field] = sanitize_html(updates[field])

    async with get_db() as db:
        await _verify_crew_ownership(db, crew_id, user["id"])

        updates["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), crew_id]

        await db.execute(f"UPDATE crews SET {set_clause} WHERE id = ?", values)
        await db.commit()

        return await _crew_with_tasks(db, crew_id)


@router.delete("/{crew_id}", status_code=204)
async def delete_crew(
    crew_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a crew and its tasks (cascade)."""
    async with get_db() as db:
        await _verify_crew_ownership(db, crew_id, user["id"])
        await db.execute("DELETE FROM crews WHERE id = ?", (crew_id,))
        await db.commit()


# ---------------------------------------------------------------------------
# Crew task endpoints
# ---------------------------------------------------------------------------


@router.get("/{crew_id}/tasks", response_model=list[CrewTaskResponse])
async def list_crew_tasks(
    crew_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """List all tasks in a crew."""
    async with get_db() as db:
        await _verify_crew_ownership(db, crew_id, user["id"])
        return await _get_crew_tasks(db, crew_id)


@router.post("/{crew_id}/tasks", response_model=CrewTaskResponse, status_code=201)
async def add_crew_task(
    crew_id: str,
    body: CrewTaskCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Add a task to a crew."""
    async with get_db() as db:
        await _verify_crew_ownership(db, crew_id, user["id"])

        # Verify agent exists
        cursor = await db.execute(
            "SELECT id FROM agents WHERE id = ? AND user_id = ?",
            (body.agent_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Agent not found")

        task_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO crew_tasks (id, crew_id, agent_id, task_description, expected_output, task_order, dependencies_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                crew_id,
                body.agent_id,
                sanitize_html(body.task_description),
                sanitize_html(body.expected_output),
                body.task_order,
                body.dependencies_json,
                now,
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM crew_tasks WHERE id = ?", (task_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.put("/{crew_id}/tasks/{task_id}", response_model=CrewTaskResponse)
async def update_crew_task(
    crew_id: str,
    task_id: str,
    body: CrewTaskUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a crew task."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    for field in ("task_description", "expected_output"):
        if field in updates and isinstance(updates[field], str):
            updates[field] = sanitize_html(updates[field])

    async with get_db() as db:
        await _verify_crew_ownership(db, crew_id, user["id"])

        # Verify task belongs to crew
        cursor = await db.execute(
            "SELECT id FROM crew_tasks WHERE id = ? AND crew_id = ?",
            (task_id, crew_id),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Crew task not found")

        if "agent_id" in updates:
            cursor = await db.execute(
                "SELECT id FROM agents WHERE id = ? AND user_id = ?",
                (updates["agent_id"], user["id"]),
            )
            if await cursor.fetchone() is None:
                raise HTTPException(status_code=404, detail="Agent not found")

        updates["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), task_id]

        await db.execute(f"UPDATE crew_tasks SET {set_clause} WHERE id = ?", values)
        await db.commit()

        cursor = await db.execute("SELECT * FROM crew_tasks WHERE id = ?", (task_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.delete("/{crew_id}/tasks/{task_id}", status_code=204)
async def delete_crew_task(
    crew_id: str,
    task_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Remove a task from a crew."""
    async with get_db() as db:
        await _verify_crew_ownership(db, crew_id, user["id"])
        cursor = await db.execute(
            "SELECT id FROM crew_tasks WHERE id = ? AND crew_id = ?",
            (task_id, crew_id),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Crew task not found")
        await db.execute("DELETE FROM crew_tasks WHERE id = ?", (task_id,))
        await db.commit()


# ---------------------------------------------------------------------------
# Crew execution (persistent runs + WebSocket streaming)
# ---------------------------------------------------------------------------


async def _get_user_from_cookie(websocket: WebSocket) -> dict[str, Any] | None:
    """Extract user from session cookie on the WebSocket connection."""
    session_id = websocket.cookies.get("exo_session")
    if not session_id:
        return None

    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT u.id, u.email, u.created_at
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.id = ? AND s.expires_at > datetime('now')
            """,
            (session_id,),
        )
        row = await cursor.fetchone()

    return dict(row) if row else None


async def _execute_crew_run(
    run_id: str,
    crew_id: str,
    process_type: str,
    crew_input: str,
    tasks: list[dict[str, Any]],
    event_callback: Any | None = None,
) -> None:
    """Execute a crew run, persisting task results and emitting events."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    # Mark run as running
    async with get_db() as db:
        await db.execute(
            "UPDATE crew_runs SET status = 'running', started_at = ? WHERE id = ?",
            (now, run_id),
        )
        await db.commit()

    if event_callback:
        await event_callback(
            {"type": "execution_started", "run_id": run_id, "process_type": process_type}
        )

    # Build run task entries and resolve agent names
    run_tasks: list[dict[str, Any]] = []
    async with get_db() as db:
        for t in tasks:
            rt_id = str(uuid.uuid4())
            cursor = await db.execute("SELECT name FROM agents WHERE id = ?", (t["agent_id"],))
            agent_row = await cursor.fetchone()
            agent_name = agent_row["name"] if agent_row else "Unknown"

            await db.execute(
                """
                INSERT INTO crew_run_tasks (id, run_id, task_id, agent_id, agent_name, task_description, expected_output, task_order, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    rt_id,
                    run_id,
                    t["id"],
                    t["agent_id"],
                    agent_name,
                    t["task_description"] or "",
                    t["expected_output"] or "",
                    t["task_order"],
                ),
            )
            run_tasks.append(
                {
                    **t,
                    "run_task_id": rt_id,
                    "agent_name": agent_name,
                }
            )
        await db.commit()

    results: list[dict[str, Any]] = []

    if process_type == "parallel":
        # Emit task_started for all tasks
        for rt in run_tasks:
            task_now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
            async with get_db() as db:
                await db.execute(
                    "UPDATE crew_run_tasks SET status = 'running', started_at = ? WHERE id = ?",
                    (task_now, rt["run_task_id"]),
                )
                await db.commit()
            if event_callback:
                await event_callback(
                    {
                        "type": "task_started",
                        "task_id": rt["run_task_id"],
                        "agent_id": rt["agent_id"],
                        "agent_name": rt["agent_name"],
                        "task_description": rt["task_description"] or "",
                        "task_order": rt["task_order"],
                    }
                )

        coros = [_execute_single_task(rt, crew_input, []) for rt in run_tasks]
        raw_results = await asyncio.gather(*coros)

        for i, result in enumerate(raw_results):
            rt = run_tasks[i]
            task_completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
            async with get_db() as db:
                await db.execute(
                    "UPDATE crew_run_tasks SET status = ?, output = ?, error = ?, duration_ms = ?, completed_at = ? WHERE id = ?",
                    (
                        result["status"],
                        result.get("output", ""),
                        result.get("error", ""),
                        result.get("duration_ms", 0),
                        task_completed_at,
                        rt["run_task_id"],
                    ),
                )
                await db.commit()
            if event_callback:
                await event_callback(
                    {
                        "type": "task_completed",
                        "task_id": rt["run_task_id"],
                        "agent_id": rt["agent_id"],
                        "agent_name": rt["agent_name"],
                        "status": result["status"],
                        "output": result.get("output", ""),
                        "error": result.get("error", ""),
                        "duration_ms": result.get("duration_ms", 0),
                    }
                )
            results.append(result)
    else:
        # Sequential: feed previous outputs as context
        previous_outputs: list[str] = []
        for rt in run_tasks:
            task_started_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
            async with get_db() as db:
                await db.execute(
                    "UPDATE crew_run_tasks SET status = 'running', started_at = ? WHERE id = ?",
                    (task_started_at, rt["run_task_id"]),
                )
                await db.commit()
            if event_callback:
                await event_callback(
                    {
                        "type": "task_started",
                        "task_id": rt["run_task_id"],
                        "agent_id": rt["agent_id"],
                        "agent_name": rt["agent_name"],
                        "task_description": rt["task_description"] or "",
                        "task_order": rt["task_order"],
                    }
                )

            result = await _execute_single_task(rt, crew_input, previous_outputs)

            task_completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
            async with get_db() as db:
                await db.execute(
                    "UPDATE crew_run_tasks SET status = ?, output = ?, error = ?, duration_ms = ?, completed_at = ? WHERE id = ?",
                    (
                        result["status"],
                        result.get("output", ""),
                        result.get("error", ""),
                        result.get("duration_ms", 0),
                        task_completed_at,
                        rt["run_task_id"],
                    ),
                )
                await db.commit()

            if event_callback:
                await event_callback(
                    {
                        "type": "task_completed",
                        "task_id": rt["run_task_id"],
                        "agent_id": rt["agent_id"],
                        "agent_name": rt["agent_name"],
                        "status": result["status"],
                        "output": result.get("output", ""),
                        "error": result.get("error", ""),
                        "duration_ms": result.get("duration_ms", 0),
                    }
                )

            results.append(result)
            if result["status"] == "completed" and result.get("output"):
                previous_outputs.append(result["output"])

    # Determine overall status
    all_completed = all(r["status"] == "completed" for r in results)
    overall_status = "completed" if all_completed else "partial"
    completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        await db.execute(
            "UPDATE crew_runs SET status = ?, completed_at = ? WHERE id = ?",
            (overall_status, completed_at, run_id),
        )
        await db.commit()

    if event_callback:
        await event_callback(
            {
                "type": "execution_completed",
                "run_id": run_id,
                "status": overall_status,
            }
        )


@router.post("/{crew_id}/run", response_model=CrewRunResponse)
async def run_crew(
    crew_id: str,
    body: CrewRunRequest | None = None,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Execute the crew — runs all tasks sequentially or in parallel based on process_type.

    Creates a persistent crew_run record and returns the run_id.
    """
    crew_input = body.input if body else ""

    async with get_db() as db:
        crew = await _verify_crew_ownership(db, crew_id, user["id"])
        tasks = await _get_crew_tasks(db, crew_id)

    if not tasks:
        raise HTTPException(status_code=422, detail="Crew has no tasks to run")

    process_type = crew["process_type"]
    run_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        await db.execute(
            "INSERT INTO crew_runs (id, crew_id, status, process_type, input, user_id, created_at) VALUES (?, ?, 'pending', ?, ?, ?, ?)",
            (run_id, crew_id, process_type, crew_input, user["id"], now),
        )
        await db.commit()

    # Fire-and-forget: execute in background
    task = asyncio.create_task(_execute_crew_run(run_id, crew_id, process_type, crew_input, tasks))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {
        "crew_id": crew_id,
        "status": "pending",
        "process_type": process_type,
        "results": [],
        "run_id": run_id,
    }


# ---------------------------------------------------------------------------
# Crew run history
# ---------------------------------------------------------------------------


@router.get("/{crew_id}/runs")
async def list_crew_runs(
    crew_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return paginated run history for a crew."""
    async with get_db() as db:
        await _verify_crew_ownership(db, crew_id, user["id"])

        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM crew_runs WHERE crew_id = ?",
            (crew_id,),
        )
        total = (await cursor.fetchone())["cnt"]

        cursor = await db.execute(
            "SELECT * FROM crew_runs WHERE crew_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (crew_id, limit, offset),
        )
        rows = await cursor.fetchall()

    runs = []
    for row in rows:
        r = _row_to_dict(row)
        runs.append(r)

    return {"runs": runs, "total": total, "limit": limit, "offset": offset}


@router.get("/{crew_id}/runs/{run_id}")
async def get_crew_run(
    crew_id: str,
    run_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single crew run with its task results."""
    async with get_db() as db:
        await _verify_crew_ownership(db, crew_id, user["id"])

        cursor = await db.execute(
            "SELECT * FROM crew_runs WHERE id = ? AND crew_id = ?",
            (run_id, crew_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Crew run not found")

        run = _row_to_dict(row)

        cursor = await db.execute(
            "SELECT * FROM crew_run_tasks WHERE run_id = ? ORDER BY task_order ASC",
            (run_id,),
        )
        task_rows = await cursor.fetchall()
        run["tasks"] = [_row_to_dict(r) for r in task_rows]

    return run


# ---------------------------------------------------------------------------
# WebSocket: live crew execution stream
# ---------------------------------------------------------------------------


@router.websocket("/{crew_id}/runs/{run_id}/stream")
async def stream_crew_run(
    websocket: WebSocket,
    crew_id: str,
    run_id: str,
) -> None:
    """WebSocket endpoint for streaming crew execution events."""
    user = await _get_user_from_cookie(websocket)
    if user is None:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    # Verify ownership
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM crews WHERE id = ? AND user_id = ?",
            (crew_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            await websocket.close(code=4004, reason="Crew not found")
            return

        cursor = await db.execute(
            "SELECT * FROM crew_runs WHERE id = ? AND crew_id = ?",
            (run_id, crew_id),
        )
        run_row = await cursor.fetchone()
        if run_row is None:
            await websocket.close(code=4004, reason="Run not found")
            return

    await websocket.accept()

    run = _row_to_dict(run_row)

    # If run is already done, send final status and close
    if run["status"] in ("completed", "partial", "failed"):
        # Send all task results
        async with get_db() as db:
            cursor = await db.execute(
                "SELECT * FROM crew_run_tasks WHERE run_id = ? ORDER BY task_order ASC",
                (run_id,),
            )
            task_rows = await cursor.fetchall()

        for tr in task_rows:
            t = _row_to_dict(tr)
            await websocket.send_json(
                {
                    "type": "task_completed",
                    "task_id": t["id"],
                    "agent_id": t["agent_id"],
                    "agent_name": t["agent_name"],
                    "status": t["status"],
                    "output": t["output"],
                    "error": t["error"],
                    "duration_ms": t["duration_ms"],
                }
            )

        await websocket.send_json(
            {"type": "execution_completed", "run_id": run_id, "status": run["status"]}
        )
        await websocket.close()
        return

    # Run is pending or running — create event queue
    event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def event_callback(event: dict[str, Any]) -> None:
        await event_queue.put(event)

    # If still pending, start execution with our callback
    if run["status"] == "pending":
        async with get_db() as db:
            tasks = await _get_crew_tasks(db, crew_id)

        if tasks:
            bg_task = asyncio.create_task(
                _execute_crew_run(
                    run_id, crew_id, run["process_type"], run["input"], tasks, event_callback
                )
            )
            _background_tasks.add(bg_task)
            bg_task.add_done_callback(_background_tasks.discard)
    else:
        # Already running — poll DB for updates
        bg_task = asyncio.create_task(_poll_crew_run(run_id, event_queue))
        _background_tasks.add(bg_task)
        bg_task.add_done_callback(_background_tasks.discard)

    try:
        while True:
            event = await event_queue.get()
            await websocket.send_json(event)
            if event.get("type") == "execution_completed":
                break
    except WebSocketDisconnect:
        pass
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


async def _poll_crew_run(run_id: str, event_queue: asyncio.Queue[dict[str, Any]]) -> None:
    """Poll the DB for crew run updates when we can't attach a direct callback."""
    seen_completed: set[str] = set()

    while True:
        await asyncio.sleep(1.0)

        async with get_db() as db:
            cursor = await db.execute("SELECT status FROM crew_runs WHERE id = ?", (run_id,))
            run_row = await cursor.fetchone()
            if run_row is None:
                break

            run_status = run_row["status"]

            cursor = await db.execute(
                "SELECT * FROM crew_run_tasks WHERE run_id = ? ORDER BY task_order ASC",
                (run_id,),
            )
            task_rows = await cursor.fetchall()

        for tr in task_rows:
            t = _row_to_dict(tr)
            if t["status"] in ("completed", "failed") and t["id"] not in seen_completed:
                seen_completed.add(t["id"])
                await event_queue.put(
                    {
                        "type": "task_completed",
                        "task_id": t["id"],
                        "agent_id": t["agent_id"],
                        "agent_name": t["agent_name"],
                        "status": t["status"],
                        "output": t["output"],
                        "error": t["error"],
                        "duration_ms": t["duration_ms"],
                    }
                )

        if run_status in ("completed", "partial", "failed"):
            await event_queue.put(
                {
                    "type": "execution_completed",
                    "run_id": run_id,
                    "status": run_status,
                }
            )
            break
