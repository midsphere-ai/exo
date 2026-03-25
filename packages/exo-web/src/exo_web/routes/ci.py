"""CI/CD integration endpoints and platform API key management.

CI endpoints authenticate via X-API-Key header instead of session cookies.
API key management endpoints use standard session authentication.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user
from exo_web.sanitize import sanitize_html
from exo_web.services.audit import audit_log

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_key(key: str) -> str:
    """SHA-256 hash of an API key for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# API key authentication for CI endpoints
# ---------------------------------------------------------------------------


async def _get_ci_user(
    x_api_key: str = Header(...),
) -> dict[str, Any]:
    """Authenticate a CI request via the X-API-Key header.

    Returns a dict with ``user_id`` and ``permissions`` from the api_keys row.
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    key_hash = _hash_key(x_api_key)

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, user_id, permissions_json FROM api_keys WHERE key_hash = ?",
            (key_hash,),
        )
        row = await cursor.fetchone()

        if row is None:
            raise HTTPException(status_code=401, detail="Invalid API key")

        # Update last_used_at
        await db.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
            (datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"), row["id"]),
        )
        await db.commit()

    permissions: list[str] = json.loads(row["permissions_json"] or "[]")
    return {"user_id": row["user_id"], "permissions": permissions, "api_key_id": row["id"]}


def _require_permission(permission: str):
    """Return a dependency that checks the CI user has a specific permission."""

    async def _check(
        ci_user: dict[str, Any] = Depends(_get_ci_user),  # noqa: B008
    ) -> dict[str, Any]:
        if permission not in ci_user.get("permissions", []):
            raise HTTPException(status_code=403, detail=f"Missing permission: {permission}")
        return ci_user

    return _check


# ---------------------------------------------------------------------------
# CI router — authenticated via X-API-Key
# ---------------------------------------------------------------------------

ci_router = APIRouter(prefix="/api/v1/ci", tags=["ci"])


# -- Pydantic models -------------------------------------------------------


class DeployRequest(BaseModel):
    entity_type: str = Field(..., pattern="^(agent|workflow)$", description="Entity type")
    entity_id: str = Field(..., min_length=1, description="Entity id")
    version_tag: str = Field(..., min_length=1, max_length=100, description="Version tag")


class DeployResponse(BaseModel):
    deployment_id: str = Field(description="Associated deployment identifier")
    entity_type: str = Field(description="Entity type")
    entity_id: str = Field(description="Entity id")
    version_tag: str = Field(description="Version tag")
    status: str = Field(description="Current status")
    deployed_at: str = Field(description="Deployed at")


class EvaluateRequest(BaseModel):
    evaluation_id: str = Field(..., min_length=1, description="Associated evaluation identifier")


class EvaluateResponse(BaseModel):
    result_id: str = Field(description="Result id")
    evaluation_id: str = Field(description="Associated evaluation identifier")
    overall_score: float = Field(description="Overall score")
    pass_rate: float = Field(description="Pass rate")
    run_at: str = Field(description="Run at")


class DeploymentStatusResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    entity_type: str = Field(description="Entity type")
    entity_id: str = Field(description="Entity id")
    status: str = Field(description="Current status")
    usage_count: int = Field(description="Number of times used")
    last_run_status: str | None = Field(None, description="Last run status")
    last_run_at: str | None = Field(None, description="Last execution timestamp")


# -- Endpoints --------------------------------------------------------------


@ci_router.post("/deploy", response_model=DeployResponse, status_code=201)
async def ci_deploy(
    body: DeployRequest,
    ci_user: dict[str, Any] = Depends(_require_permission("ci:deploy")),  # noqa: B008
) -> dict[str, Any]:
    """Deploy a specific version of an agent or workflow from a CI pipeline."""
    user_id = ci_user["user_id"]

    async with get_db() as db:
        # Verify entity exists and belongs to user
        if body.entity_type == "agent":
            cur = await db.execute(
                "SELECT id, name FROM agents WHERE id = ? AND user_id = ?",
                (body.entity_id, user_id),
            )
        else:
            cur = await db.execute(
                "SELECT id, name FROM workflows WHERE id = ? AND user_id = ?",
                (body.entity_id, user_id),
            )
        entity_row = await cur.fetchone()
        if entity_row is None:
            raise HTTPException(status_code=404, detail=f"{body.entity_type.title()} not found")

        # Create or update deployment
        deployment_id = str(uuid.uuid4())
        api_key = f"orb_{secrets.token_urlsafe(32)}"
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO deployments
                (id, name, entity_type, entity_id, api_key_hash,
                 rate_limit, status, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 60, 'active', ?, ?, ?)
            """,
            (
                deployment_id,
                f"{entity_row['name']} ({body.version_tag})",
                body.entity_type,
                body.entity_id,
                api_key_hash,
                user_id,
                now,
                now,
            ),
        )
        await db.commit()

    await audit_log(
        user_id,
        "ci_deploy",
        "deployment",
        deployment_id,
        details={
            "entity_type": body.entity_type,
            "entity_id": body.entity_id,
            "version_tag": body.version_tag,
        },
    )

    return {
        "deployment_id": deployment_id,
        "entity_type": body.entity_type,
        "entity_id": body.entity_id,
        "version_tag": body.version_tag,
        "status": "active",
        "deployed_at": now,
    }


@ci_router.post("/evaluate", response_model=EvaluateResponse)
async def ci_evaluate(
    body: EvaluateRequest,
    ci_user: dict[str, Any] = Depends(_require_permission("ci:evaluate")),  # noqa: B008
) -> dict[str, Any]:
    """Trigger an evaluation run and return results."""
    user_id = ci_user["user_id"]

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM evaluations WHERE id = ? AND user_id = ?",
            (body.evaluation_id, user_id),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    evaluation = dict(row)

    try:
        test_cases: list[dict[str, Any]] = json.loads(evaluation["test_cases_json"])
    except (json.JSONDecodeError, TypeError):
        test_cases = []

    if not test_cases:
        raise HTTPException(status_code=422, detail="No test cases defined")

    # Import evaluator runner
    from exo_web.services.evaluators import EVALUATORS, run_evaluator

    agent_id = evaluation["agent_id"]
    results: list[dict[str, Any]] = []
    total_score = 0.0
    pass_count = 0

    for tc in test_cases:
        input_msg = tc.get("input", "")
        expected = tc.get("expected", "")
        evaluator_type = tc.get("evaluator", "exact_match")

        if evaluator_type not in EVALUATORS:
            evaluator_type = "exact_match"

        # Send to agent
        actual = await _send_to_agent(agent_id, user_id, input_msg)

        score = await run_evaluator(evaluator_type, expected, actual)
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
            }
        )

    num_cases = len(test_cases)
    overall_score = round(total_score / num_cases, 4) if num_cases else 0.0
    pass_rate = round(pass_count / num_cases, 4) if num_cases else 0.0

    # Persist the result
    result_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO eval_results (id, evaluation_id, run_at, results_json, overall_score, pass_rate)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (result_id, body.evaluation_id, now, json.dumps(results), overall_score, pass_rate),
        )
        await db.commit()

    return {
        "result_id": result_id,
        "evaluation_id": body.evaluation_id,
        "overall_score": overall_score,
        "pass_rate": pass_rate,
        "run_at": now,
    }


@ci_router.get("/status/{deployment_id}", response_model=DeploymentStatusResponse)
async def ci_status(
    deployment_id: str,
    ci_user: dict[str, Any] = Depends(_require_permission("ci:status")),  # noqa: B008
) -> dict[str, Any]:
    """Return deployment health and last run status."""
    user_id = ci_user["user_id"]

    async with get_db() as db:
        cur = await db.execute(
            "SELECT * FROM deployments WHERE id = ? AND user_id = ?",
            (deployment_id, user_id),
        )
        row = await cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Deployment not found")

        dep = dict(row)

        # Fetch last run status based on entity type
        last_run_status: str | None = None
        last_run_at: str | None = None

        if dep["entity_type"] == "agent":
            run_cur = await db.execute(
                """
                SELECT status, created_at FROM runs
                WHERE agent_id = ? ORDER BY created_at DESC LIMIT 1
                """,
                (dep["entity_id"],),
            )
        else:
            run_cur = await db.execute(
                """
                SELECT status, created_at FROM workflow_runs
                WHERE workflow_id = ? ORDER BY created_at DESC LIMIT 1
                """,
                (dep["entity_id"],),
            )

        run_row = await run_cur.fetchone()
        if run_row:
            last_run_status = run_row["status"]
            last_run_at = run_row["created_at"]

    return {
        "id": dep["id"],
        "name": dep["name"],
        "entity_type": dep["entity_type"],
        "entity_id": dep["entity_id"],
        "status": dep["status"],
        "usage_count": dep["usage_count"],
        "last_run_status": last_run_status,
        "last_run_at": last_run_at,
    }


# ---------------------------------------------------------------------------
# Helper — send prompt to agent (shared with evaluations.py)
# ---------------------------------------------------------------------------


async def _send_to_agent(agent_id: str, user_id: str, message: str) -> str:
    """Send a message to an agent and return the text response."""
    from exo_web.services.agent_runtime import _load_agent_row, _resolve_provider

    row = await _load_agent_row(agent_id)
    provider_type = row.get("model_provider", "")
    model_name = row.get("model_name", "")
    if not provider_type or not model_name:
        return "[error: agent has no model configured]"

    try:
        provider = await _resolve_provider(provider_type, model_name, user_id)
        instructions = row.get("instructions", "")
        messages: list[dict[str, str]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": message})
        resp = await provider.complete(messages=messages, model=model_name)
        return resp.content
    except Exception as exc:
        return f"[error: {exc}]"


# ---------------------------------------------------------------------------
# API key management router — authenticated via session cookie
# ---------------------------------------------------------------------------

api_keys_router = APIRouter(prefix="/api/v1/settings/api-keys", tags=["api-keys"])


class ApiKeyCreateRequest(BaseModel):
    label: str = Field("", max_length=255, description="Label")
    permissions: list[str] = Field(
        default_factory=lambda: ["ci:deploy", "ci:evaluate", "ci:status"], description="Permissions"
    )


class ApiKeyCreateResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    label: str = Field(description="Label")
    permissions: list[str] = Field(description="Permissions")
    api_key: str = Field(description="API key (stored encrypted)")
    created_at: str = Field(description="ISO 8601 creation timestamp")


class ApiKeyResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    label: str = Field(description="Label")
    permissions: list[str] = Field(description="Permissions")
    key_prefix: str = Field(description="Key prefix")
    last_used_at: str | None = Field(description="Last used at")
    created_at: str = Field(description="ISO 8601 creation timestamp")


@api_keys_router.post("", response_model=ApiKeyCreateResponse, status_code=201)
async def create_api_key(
    body: ApiKeyCreateRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Generate a platform API key for CI usage.

    The plaintext key is returned only once — it cannot be retrieved later.
    """
    key_id = str(uuid.uuid4())
    raw_key = f"orb_ci_{secrets.token_urlsafe(32)}"
    key_hash = _hash_key(raw_key)
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    # Validate permissions
    valid_perms = {"ci:deploy", "ci:evaluate", "ci:status"}
    perms = [p for p in body.permissions if p in valid_perms]
    if not perms:
        perms = list(valid_perms)

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO api_keys (id, user_id, key_hash, label, permissions_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (key_id, user["id"], key_hash, sanitize_html(body.label), json.dumps(perms), now),
        )
        await db.commit()

    ip = request.client.host if request.client else None
    await audit_log(
        user["id"],
        "create_api_key",
        "api_key",
        key_id,
        details={"label": body.label},
        ip_address=ip,
    )

    return {
        "id": key_id,
        "label": body.label,
        "permissions": perms,
        "api_key": raw_key,
        "created_at": now,
    }


@api_keys_router.get("")
async def list_api_keys(
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """List all API keys for the current user (without exposing the key)."""
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT id, label, permissions_json, key_hash, last_used_at, created_at
            FROM api_keys WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user["id"],),
        )
        rows = await cursor.fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        perms: list[str] = json.loads(r["permissions_json"] or "[]")
        result.append(
            {
                "id": r["id"],
                "label": r["label"],
                "permissions": perms,
                "key_prefix": r["key_hash"][:8] + "...",
                "last_used_at": r["last_used_at"],
                "created_at": r["created_at"],
            }
        )

    return result


@api_keys_router.delete("/{key_id}", status_code=204)
async def delete_api_key(
    key_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Revoke (delete) an API key."""
    async with get_db() as db:
        cur = await db.execute(
            "SELECT id FROM api_keys WHERE id = ? AND user_id = ?",
            (key_id, user["id"]),
        )
        if await cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="API key not found")

        await db.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
        await db.commit()

    ip = request.client.host if request.client else None
    await audit_log(user["id"], "delete_api_key", "api_key", key_id, ip_address=ip)
