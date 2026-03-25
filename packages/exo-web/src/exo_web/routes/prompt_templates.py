"""Prompt templates CRUD REST API, version history, test/compare, and optimization endpoints."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user
from exo_web.sanitize import sanitize_html

router = APIRouter(prefix="/api/v1/prompt-templates", tags=["prompt_templates"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TemplateCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    content: str = Field("", description="Text content")
    variables_json: str = Field("{}", description="JSON object of variables")


class TemplateUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255, description="Display name")
    content: str | None = Field(None, description="Text content")
    variables_json: str | None = Field(None, description="JSON object of variables")


class TemplateResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    content: str = Field(description="Text content")
    variables_json: str = Field(description="JSON object of variables")
    user_id: str = Field(description="Owning user identifier")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


class VersionResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    template_id: str = Field(description="Associated template identifier")
    content: str = Field(description="Text content")
    variables_json: str = Field(description="JSON object of variables")
    version_number: int = Field(description="Version number")
    user_id: str = Field(description="Owning user identifier")
    created_at: str = Field(description="ISO 8601 creation timestamp")


class TestPromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt")
    variables: dict[str, str] = {}
    provider_id: str = Field(..., min_length=1, description="Associated provider identifier")
    model_name: str = Field(..., min_length=1, description="Model name")


class TestPromptResponse(BaseModel):
    output: str = Field(description="Output text or data")
    model: str = Field(description="Model identifier")
    tokens_used: int | None = Field(None, description="Tokens used")
    response_time_ms: int | None = Field(None, description="Response time ms")


class CompareModelItem(BaseModel):
    provider_id: str = Field(..., min_length=1, description="Associated provider identifier")
    model_name: str = Field(..., min_length=1, description="Model name")


class CompareRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt")
    variables: dict[str, str] = {}
    models: list[CompareModelItem] = Field(..., min_length=2, max_length=3, description="Models")


class CompareResultItem(BaseModel):
    provider_id: str = Field(description="Associated provider identifier")
    model_name: str = Field(description="Model name")
    output: str = Field(description="Output text or data")
    tokens_used: int | None = Field(None, description="Tokens used")
    response_time_ms: int | None = Field(None, description="Response time ms")
    error: str | None = Field(None, description="Error message if failed")


class CompareResponse(BaseModel):
    results: list[CompareResultItem] = Field(description="Results")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict."""
    return dict(row)


async def _verify_ownership(db: Any, template_id: str, user_id: str) -> dict[str, Any]:
    """Verify template exists and belongs to user. Returns row dict or raises 404."""
    cursor = await db.execute(
        "SELECT * FROM prompt_templates WHERE id = ? AND user_id = ?",
        (template_id, user_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Template not found")
    return _row_to_dict(row)


async def _create_version(
    db: Any, template_id: str, content: str, variables_json: str, user_id: str
) -> None:
    """Create a new version entry for a template."""
    # Get next version number
    cursor = await db.execute(
        "SELECT COALESCE(MAX(version_number), 0) + 1 FROM prompt_versions WHERE template_id = ?",
        (template_id,),
    )
    row = await cursor.fetchone()
    next_version = row[0]

    version_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    await db.execute(
        """
        INSERT INTO prompt_versions (id, template_id, content, variables_json, version_number, user_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (version_id, template_id, content, variables_json, next_version, user_id, now),
    )


async def _send_prompt_to_model(
    provider_id: str, model_name: str, prompt: str, user_id: str
) -> dict[str, Any]:
    """Send a prompt to a specific model and return the result dict."""
    import time

    import httpx

    from exo_web.crypto import decrypt_api_key

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM providers WHERE id = ? AND user_id = ?",
            (provider_id, user_id),
        )
        provider_row = await cursor.fetchone()
        if provider_row is None:
            return {
                "provider_id": provider_id,
                "model_name": model_name,
                "output": "",
                "error": "Provider not found",
            }
        provider = dict(provider_row)

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
                "provider_id": provider_id,
                "model_name": model_name,
                "output": "",
                "error": "No API key configured",
            }

    provider_type = provider["provider_type"]
    base_url = provider.get("base_url") or ""

    start_time = time.monotonic()
    tokens_used = None

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
                        "max_tokens": 1024,
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
                        "max_tokens": 1024,
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
                return {
                    "provider_id": provider_id,
                    "model_name": model_name,
                    "output": "",
                    "error": f"Unsupported provider type: {provider_type}",
                }

            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            if resp.status_code >= 400:
                error_text = resp.text[:500]
                return {
                    "provider_id": provider_id,
                    "model_name": model_name,
                    "output": "",
                    "error": f"API error ({resp.status_code}): {error_text}",
                    "response_time_ms": elapsed_ms,
                }

            data = resp.json()

            output = ""
            if provider_type in ("openai", "custom"):
                output = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = data.get("usage", {})
                tokens_used = usage.get("total_tokens")
            elif provider_type == "anthropic":
                content_blocks = data.get("content", [])
                output = "".join(
                    b.get("text", "") for b in content_blocks if b.get("type") == "text"
                )
                usage = data.get("usage", {})
                tokens_used = (usage.get("input_tokens", 0) or 0) + (
                    usage.get("output_tokens", 0) or 0
                )
            elif provider_type == "gemini":
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    output = "".join(p.get("text", "") for p in parts)
            elif provider_type == "ollama":
                output = data.get("response", "")

            return {
                "provider_id": provider_id,
                "model_name": model_name,
                "output": output,
                "tokens_used": tokens_used,
                "response_time_ms": elapsed_ms,
            }

    except httpx.HTTPError as exc:
        return {
            "provider_id": provider_id,
            "model_name": model_name,
            "output": "",
            "error": f"Connection error: {exc!s}",
        }


# ---------------------------------------------------------------------------
# Template CRUD Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[TemplateResponse])
async def list_templates(
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all prompt templates for the current user."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM prompt_templates WHERE user_id = ? ORDER BY updated_at DESC",
            (user["id"],),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


@router.post("", response_model=TemplateResponse, status_code=201)
async def create_template(
    body: TemplateCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new prompt template and its initial version."""
    async with get_db() as db:
        template_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO prompt_templates (id, name, content, variables_json, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                template_id,
                sanitize_html(body.name),
                sanitize_html(body.content),
                body.variables_json,
                user["id"],
                now,
                now,
            ),
        )

        # Create initial version
        await _create_version(db, template_id, body.content, body.variables_json, user["id"])

        await db.commit()

        cursor = await db.execute("SELECT * FROM prompt_templates WHERE id = ?", (template_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.get("/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single prompt template by ID."""
    async with get_db() as db:
        return await _verify_ownership(db, template_id, user["id"])


@router.put("/{template_id}", response_model=TemplateResponse)
async def update_template(
    template_id: str,
    body: TemplateUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a prompt template. Creates a new version if content changes."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    for field in ("name", "content"):
        if field in updates and isinstance(updates[field], str):
            updates[field] = sanitize_html(updates[field])

    async with get_db() as db:
        existing = await _verify_ownership(db, template_id, user["id"])

        # Create version if content or variables changed
        new_content = updates.get("content", existing["content"])
        new_variables = updates.get("variables_json", existing["variables_json"])
        if new_content != existing["content"] or new_variables != existing["variables_json"]:
            await _create_version(db, template_id, new_content, new_variables, user["id"])

        updates["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), template_id]

        await db.execute(
            f"UPDATE prompt_templates SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM prompt_templates WHERE id = ?", (template_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.delete("/{template_id}", status_code=204)
async def delete_template(
    template_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a prompt template and all its versions."""
    async with get_db() as db:
        await _verify_ownership(db, template_id, user["id"])
        await db.execute("DELETE FROM prompt_versions WHERE template_id = ?", (template_id,))
        await db.execute("DELETE FROM prompt_templates WHERE id = ?", (template_id,))
        await db.commit()


# ---------------------------------------------------------------------------
# Version History Endpoints
# ---------------------------------------------------------------------------


@router.get("/{template_id}/versions", response_model=list[VersionResponse])
async def list_versions(
    template_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all versions of a template, newest first."""
    async with get_db() as db:
        await _verify_ownership(db, template_id, user["id"])
        cursor = await db.execute(
            "SELECT * FROM prompt_versions WHERE template_id = ? ORDER BY version_number DESC",
            (template_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


@router.get("/{template_id}/versions/{version_id}", response_model=VersionResponse)
async def get_version(
    template_id: str,
    version_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single version by ID."""
    async with get_db() as db:
        await _verify_ownership(db, template_id, user["id"])
        cursor = await db.execute(
            "SELECT * FROM prompt_versions WHERE id = ? AND template_id = ?",
            (version_id, template_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Version not found")
        return _row_to_dict(row)


@router.post("/{template_id}/versions/{version_id}/restore", response_model=TemplateResponse)
async def restore_version(
    template_id: str,
    version_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Restore a previous version as the current template content."""
    async with get_db() as db:
        await _verify_ownership(db, template_id, user["id"])
        cursor = await db.execute(
            "SELECT * FROM prompt_versions WHERE id = ? AND template_id = ?",
            (version_id, template_id),
        )
        version_row = await cursor.fetchone()
        if version_row is None:
            raise HTTPException(status_code=404, detail="Version not found")
        version = _row_to_dict(version_row)

        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        # Update template with version content
        await db.execute(
            "UPDATE prompt_templates SET content = ?, variables_json = ?, updated_at = ? WHERE id = ?",
            (version["content"], version["variables_json"], now, template_id),
        )

        # Create a new version to record the restore action
        await _create_version(
            db, template_id, version["content"], version["variables_json"], user["id"]
        )

        await db.commit()

        cursor = await db.execute("SELECT * FROM prompt_templates WHERE id = ?", (template_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Test Prompt Endpoint
# ---------------------------------------------------------------------------


@router.post("/test", response_model=TestPromptResponse)
async def test_prompt(
    body: TestPromptRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Send a prompt to the specified model and return the response."""
    # Fill in template variables
    prompt = body.prompt
    for var_name, var_value in body.variables.items():
        prompt = prompt.replace("{{" + var_name + "}}", var_value)

    result = await _send_prompt_to_model(body.provider_id, body.model_name, prompt, user["id"])

    if result.get("error"):
        raise HTTPException(status_code=502, detail=result["error"])

    return {
        "output": result["output"],
        "model": body.model_name,
        "tokens_used": result.get("tokens_used"),
        "response_time_ms": result.get("response_time_ms"),
    }


# ---------------------------------------------------------------------------
# Model Comparison Endpoint
# ---------------------------------------------------------------------------


@router.post("/compare", response_model=CompareResponse)
async def compare_models(
    body: CompareRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Send the same prompt to 2-3 models in parallel and return results side-by-side."""
    # Fill in template variables
    prompt = body.prompt
    for var_name, var_value in body.variables.items():
        prompt = prompt.replace("{{" + var_name + "}}", var_value)

    # Run all model calls in parallel
    tasks = [
        _send_prompt_to_model(m.provider_id, m.model_name, prompt, user["id"]) for m in body.models
    ]
    results = await asyncio.gather(*tasks)

    return {"results": results}


# ---------------------------------------------------------------------------
# Prompt Optimization Models
# ---------------------------------------------------------------------------


class OptimizeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt")
    strategy: str = Field(
        "clarity", pattern=r"^(clarity|specificity|safety|conciseness)$", description="Strategy"
    )
    provider_id: str = Field(..., min_length=1, description="Associated provider identifier")
    model_name: str = Field(..., min_length=1, description="Model name")
    agent_id: str | None = Field(None, description="Associated agent identifier")
    template_id: str | None = Field(None, description="Associated template identifier")


class PromptChange(BaseModel):
    type: str  # "added", "removed", "modified" = Field(description="Type")
    line: int = Field(description="Line")
    original: str = Field(description="Original")
    optimized: str = Field(description="Optimized")


class OptimizeResponse(BaseModel):
    optimized_prompt: str = Field(description="Optimized prompt")
    changes: list[PromptChange] = Field(description="Changes")
    strategy: str = Field(description="Strategy")
    model_used: str = Field(description="Model used")
    optimization_id: str = Field(description="Optimization id")


class OptimizationHistoryItem(BaseModel):
    id: str = Field(description="Unique identifier")
    agent_id: str | None = Field(description="Associated agent identifier")
    template_id: str | None = Field(description="Associated template identifier")
    original_prompt: str = Field(description="Original prompt")
    optimized_prompt: str = Field(description="Optimized prompt")
    strategy: str = Field(description="Strategy")
    changes_json: str = Field(description="Changes json")
    accepted: bool = Field(description="Accepted")
    eval_score_before: float | None = Field(description="Eval score before")
    eval_score_after: float | None = Field(description="Eval score after")
    model_used: str = Field(description="Model used")
    created_at: str = Field(description="ISO 8601 creation timestamp")


class AcceptOptimizationRequest(BaseModel):
    optimization_id: str = Field(..., min_length=1, description="Optimization id")
    accepted: bool = Field(True, description="Accepted")


# ---------------------------------------------------------------------------
# Optimization Helpers
# ---------------------------------------------------------------------------

_STRATEGY_PROMPTS = {
    "clarity": (
        "Rewrite the following system prompt to be clearer and easier to understand. "
        "Eliminate ambiguity, use direct language, and ensure instructions are unambiguous. "
        "Preserve the original intent and functionality."
    ),
    "specificity": (
        "Rewrite the following system prompt to be more specific and detailed. "
        "Add concrete examples, precise constraints, and explicit expected behaviors. "
        "Preserve the original intent but make instructions more actionable."
    ),
    "safety": (
        "Rewrite the following system prompt to be safer. "
        "Add appropriate guardrails, content boundaries, and ethical guidelines. "
        "Ensure the prompt prevents misuse while preserving the original functionality."
    ),
    "conciseness": (
        "Rewrite the following system prompt to be more concise. "
        "Remove redundancy, tighten language, and eliminate unnecessary words. "
        "Preserve all essential instructions and functionality in fewer words."
    ),
}


def _compute_changes(original: str, optimized: str) -> list[dict[str, Any]]:
    """Compute line-level changes between original and optimized prompts."""
    orig_lines = original.splitlines()
    opt_lines = optimized.splitlines()
    changes: list[dict[str, Any]] = []

    max_lines = max(len(orig_lines), len(opt_lines))
    for i in range(max_lines):
        orig_line = orig_lines[i] if i < len(orig_lines) else ""
        opt_line = opt_lines[i] if i < len(opt_lines) else ""

        if orig_line == opt_line:
            continue

        if i >= len(orig_lines):
            changes.append({"type": "added", "line": i + 1, "original": "", "optimized": opt_line})
        elif i >= len(opt_lines):
            changes.append(
                {"type": "removed", "line": i + 1, "original": orig_line, "optimized": ""}
            )
        else:
            changes.append(
                {"type": "modified", "line": i + 1, "original": orig_line, "optimized": opt_line}
            )

    return changes


# ---------------------------------------------------------------------------
# Optimization Endpoints
# ---------------------------------------------------------------------------


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize_prompt(
    body: OptimizeRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Use AI to optimize a system prompt based on the selected strategy."""
    strategy_instruction = _STRATEGY_PROMPTS.get(body.strategy, _STRATEGY_PROMPTS["clarity"])

    # Build context: include eval results if agent_id is provided
    context_parts: list[str] = []
    if body.agent_id:
        async with get_db() as db:
            # Fetch recent eval results for context
            cursor = await db.execute(
                """
                SELECT er.overall_score, er.pass_rate, er.results_json, er.run_at
                FROM eval_results er
                JOIN evaluations e ON e.id = er.evaluation_id
                WHERE e.agent_id = ? ORDER BY er.run_at DESC LIMIT 3
                """,
                (body.agent_id,),
            )
            eval_rows = await cursor.fetchall()
            if eval_rows:
                context_parts.append("Recent evaluation results for this agent:")
                for row in eval_rows:
                    r = dict(row)
                    context_parts.append(
                        f"  - Score: {r['overall_score']:.1%}, Pass rate: {r['pass_rate']:.1%} (run at {r['run_at']})"
                    )

            # Fetch recent conversation logs for context
            cursor = await db.execute(
                """
                SELECT m.role, m.content FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                WHERE c.agent_id = ?
                ORDER BY m.created_at DESC LIMIT 10
                """,
                (body.agent_id,),
            )
            msg_rows = await cursor.fetchall()
            if msg_rows:
                context_parts.append("\nRecent conversation samples:")
                for row in msg_rows:
                    r = dict(row)
                    text = (r.get("content") or "")[:200]
                    context_parts.append(f"  [{r['role']}]: {text}")

    meta_prompt = strategy_instruction + "\n\n"
    if context_parts:
        meta_prompt += "Context:\n" + "\n".join(context_parts) + "\n\n"
    meta_prompt += (
        "IMPORTANT: Return ONLY the improved prompt text. "
        "Do not include any explanation, preamble, or commentary. "
        "Do not wrap in quotes or code blocks.\n\n"
        "Original prompt:\n" + body.prompt
    )

    result = await _send_prompt_to_model(body.provider_id, body.model_name, meta_prompt, user["id"])

    if result.get("error"):
        raise HTTPException(status_code=502, detail=result["error"])

    optimized = (result.get("output") or "").strip()
    if not optimized:
        raise HTTPException(status_code=502, detail="Model returned empty optimization result")

    changes = _compute_changes(body.prompt, optimized)

    # Store optimization record
    opt_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO prompt_optimizations
                (id, agent_id, template_id, original_prompt, optimized_prompt,
                 strategy, changes_json, model_used, user_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                opt_id,
                body.agent_id,
                body.template_id,
                body.prompt,
                optimized,
                body.strategy,
                json.dumps(changes),
                body.model_name,
                user["id"],
                now,
            ),
        )
        await db.commit()

    return {
        "optimized_prompt": optimized,
        "changes": changes,
        "strategy": body.strategy,
        "model_used": body.model_name,
        "optimization_id": opt_id,
    }


@router.post("/optimize/{optimization_id}/accept")
async def accept_optimization(
    optimization_id: str,
    body: AcceptOptimizationRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, str]:
    """Mark an optimization as accepted or rejected."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM prompt_optimizations WHERE id = ? AND user_id = ?",
            (optimization_id, user["id"]),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Optimization not found")

        await db.execute(
            "UPDATE prompt_optimizations SET accepted = ? WHERE id = ?",
            (1 if body.accepted else 0, optimization_id),
        )
        await db.commit()

    return {"status": "ok"}


@router.get("/optimize/history", response_model=list[OptimizationHistoryItem])
async def optimization_history(
    agent_id: str | None = Query(None),
    template_id: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return optimization history, optionally filtered by agent or template."""
    async with get_db() as db:
        conditions = ["user_id = ?"]
        params: list[Any] = [user["id"]]

        if agent_id:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        if template_id:
            conditions.append("template_id = ?")
            params.append(template_id)

        where_clause = " AND ".join(conditions)
        params.append(limit)

        cursor = await db.execute(
            f"SELECT * FROM prompt_optimizations WHERE {where_clause} ORDER BY created_at DESC LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["accepted"] = bool(d.get("accepted"))
            results.append(d)
        return results
