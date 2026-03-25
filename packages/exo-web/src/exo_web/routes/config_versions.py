"""Config version tracking for agents and workflows."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user

router = APIRouter(tags=["config-versions"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ConfigVersionResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    entity_type: str = Field(description="Entity type")
    entity_id: str = Field(description="Entity id")
    version_num: int = Field(description="Version num")
    config_json: str = Field(description="JSON configuration object")
    author: str = Field(description="Author")
    summary: str = Field(description="Summary")
    tag: str = Field(description="Version tag label")
    created_at: str = Field(description="ISO 8601 creation timestamp")


class TagUpdate(BaseModel):
    tag: str = Field(..., min_length=1, max_length=100, description="Version tag label")


# ---------------------------------------------------------------------------
# Shared helper — called from agents.py and workflows.py on save
# ---------------------------------------------------------------------------


async def create_config_version(
    db: Any,
    entity_type: str,
    entity_id: str,
    config: dict[str, Any],
    author: str = "",
    summary: str = "",
) -> int:
    """Create a new config version. Returns the new version number."""
    cursor = await db.execute(
        "SELECT COALESCE(MAX(version_num), 0) + 1 FROM config_versions WHERE entity_type = ? AND entity_id = ?",
        (entity_type, entity_id),
    )
    row = await cursor.fetchone()
    next_version: int = row[0]

    version_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    await db.execute(
        """
        INSERT INTO config_versions (id, entity_type, entity_id, version_num, config_json, author, summary, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            version_id,
            entity_type,
            entity_id,
            next_version,
            json.dumps(config),
            author,
            summary,
            now,
        ),
    )
    return next_version


def _snapshot_agent(row: dict[str, Any]) -> dict[str, Any]:
    """Extract agent config fields worth versioning."""
    return {
        "name": row.get("name", ""),
        "description": row.get("description", ""),
        "instructions": row.get("instructions", ""),
        "model_provider": row.get("model_provider", ""),
        "model_name": row.get("model_name", ""),
        "temperature": row.get("temperature"),
        "max_tokens": row.get("max_tokens"),
        "max_steps": row.get("max_steps"),
        "output_type_json": row.get("output_type_json", "{}"),
        "tools_json": row.get("tools_json", "[]"),
        "handoffs_json": row.get("handoffs_json", "[]"),
        "hooks_json": row.get("hooks_json", "{}"),
    }


def _snapshot_workflow(row: dict[str, Any]) -> dict[str, Any]:
    """Extract workflow config fields worth versioning."""
    return {
        "name": row.get("name", ""),
        "description": row.get("description", ""),
        "nodes_json": row.get("nodes_json", "[]"),
        "edges_json": row.get("edges_json", "[]"),
        "viewport_json": row.get("viewport_json", '{"x":0,"y":0,"zoom":1}'),
        "status": row.get("status", "draft"),
    }


# ---------------------------------------------------------------------------
# Agent version endpoints
# ---------------------------------------------------------------------------


@router.get("/api/v1/agents/{agent_id}/versions", response_model=list[ConfigVersionResponse])
async def list_agent_versions(
    agent_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """List version history for an agent."""
    async with get_db() as db:
        # Verify ownership
        cursor = await db.execute(
            "SELECT id FROM agents WHERE id = ? AND user_id = ?", (agent_id, user["id"])
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Agent not found")

        cursor = await db.execute(
            "SELECT * FROM config_versions WHERE entity_type = 'agent' AND entity_id = ? ORDER BY version_num DESC",
            (agent_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


@router.post(
    "/api/v1/agents/{agent_id}/versions/{version_id}/rollback", response_model=ConfigVersionResponse
)
async def rollback_agent_version(
    agent_id: str,
    version_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Rollback an agent to a previous version config."""
    async with get_db() as db:
        # Verify ownership
        cursor = await db.execute(
            "SELECT id FROM agents WHERE id = ? AND user_id = ?", (agent_id, user["id"])
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Agent not found")

        # Fetch the version to restore
        cursor = await db.execute(
            "SELECT * FROM config_versions WHERE id = ? AND entity_type = 'agent' AND entity_id = ?",
            (version_id, agent_id),
        )
        version_row = await cursor.fetchone()
        if not version_row:
            raise HTTPException(status_code=404, detail="Version not found")
        version = dict(version_row)

        config = json.loads(version["config_json"])

        # Apply the version's config back to the agent
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        fields = [
            "name",
            "description",
            "instructions",
            "model_provider",
            "model_name",
            "temperature",
            "max_tokens",
            "max_steps",
            "output_type_json",
            "tools_json",
            "handoffs_json",
            "hooks_json",
        ]
        set_parts = [f"{f} = ?" for f in fields]
        set_parts.append("updated_at = ?")
        values = [config.get(f) for f in fields]
        values.append(now)
        values.append(agent_id)

        await db.execute(
            f"UPDATE agents SET {', '.join(set_parts)} WHERE id = ?",
            values,
        )

        # Read back the agent to snapshot the restored state
        cursor = await db.execute("SELECT * FROM agents WHERE id = ?", (agent_id,))
        agent_row = await cursor.fetchone()
        snapshot = _snapshot_agent(dict(agent_row))

        # Create a new version recording the rollback
        new_version_num = await create_config_version(
            db,
            "agent",
            agent_id,
            snapshot,
            author=user["id"],
            summary=f"Rollback to version {version['version_num']}",
        )

        await db.commit()

        # Return the newly created rollback version
        cursor = await db.execute(
            "SELECT * FROM config_versions WHERE entity_type = 'agent' AND entity_id = ? AND version_num = ?",
            (agent_id, new_version_num),
        )
        new_row = await cursor.fetchone()
        return dict(new_row)


@router.put("/api/v1/agents/{agent_id}/versions/{version_id}/tag")
async def tag_agent_version(
    agent_id: str,
    version_id: str,
    body: TagUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Set a tag label on an agent version."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM agents WHERE id = ? AND user_id = ?", (agent_id, user["id"])
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Agent not found")

        cursor = await db.execute(
            "SELECT id FROM config_versions WHERE id = ? AND entity_type = 'agent' AND entity_id = ?",
            (version_id, agent_id),
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Version not found")

        await db.execute("UPDATE config_versions SET tag = ? WHERE id = ?", (body.tag, version_id))
        await db.commit()

        cursor = await db.execute("SELECT * FROM config_versions WHERE id = ?", (version_id,))
        row = await cursor.fetchone()
        return dict(row)


# ---------------------------------------------------------------------------
# Workflow version endpoints
# ---------------------------------------------------------------------------


@router.get("/api/v1/workflows/{workflow_id}/versions", response_model=list[ConfigVersionResponse])
async def list_workflow_versions(
    workflow_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """List version history for a workflow."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM workflows WHERE id = ? AND user_id = ?", (workflow_id, user["id"])
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Workflow not found")

        cursor = await db.execute(
            "SELECT * FROM config_versions WHERE entity_type = 'workflow' AND entity_id = ? ORDER BY version_num DESC",
            (workflow_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


@router.post(
    "/api/v1/workflows/{workflow_id}/versions/{version_id}/rollback",
    response_model=ConfigVersionResponse,
)
async def rollback_workflow_version(
    workflow_id: str,
    version_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Rollback a workflow to a previous version config."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM workflows WHERE id = ? AND user_id = ?", (workflow_id, user["id"])
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Workflow not found")

        cursor = await db.execute(
            "SELECT * FROM config_versions WHERE id = ? AND entity_type = 'workflow' AND entity_id = ?",
            (version_id, workflow_id),
        )
        version_row = await cursor.fetchone()
        if not version_row:
            raise HTTPException(status_code=404, detail="Version not found")
        version = dict(version_row)

        config = json.loads(version["config_json"])

        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        fields = ["name", "description", "nodes_json", "edges_json", "viewport_json", "status"]
        set_parts = [f"{f} = ?" for f in fields]
        set_parts.append("updated_at = ?")
        values = [config.get(f) for f in fields]
        values.append(now)
        values.append(workflow_id)

        await db.execute(
            f"UPDATE workflows SET {', '.join(set_parts)} WHERE id = ?",
            values,
        )

        cursor = await db.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
        wf_row = await cursor.fetchone()
        snapshot = _snapshot_workflow(dict(wf_row))

        new_version_num = await create_config_version(
            db,
            "workflow",
            workflow_id,
            snapshot,
            author=user["id"],
            summary=f"Rollback to version {version['version_num']}",
        )

        await db.commit()

        cursor = await db.execute(
            "SELECT * FROM config_versions WHERE entity_type = 'workflow' AND entity_id = ? AND version_num = ?",
            (workflow_id, new_version_num),
        )
        new_row = await cursor.fetchone()
        return dict(new_row)


@router.put("/api/v1/workflows/{workflow_id}/versions/{version_id}/tag")
async def tag_workflow_version(
    workflow_id: str,
    version_id: str,
    body: TagUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Set a tag label on a workflow version."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM workflows WHERE id = ? AND user_id = ?", (workflow_id, user["id"])
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Workflow not found")

        cursor = await db.execute(
            "SELECT id FROM config_versions WHERE id = ? AND entity_type = 'workflow' AND entity_id = ?",
            (version_id, workflow_id),
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Version not found")

        await db.execute("UPDATE config_versions SET tag = ? WHERE id = ?", (body.tag, version_id))
        await db.commit()

        cursor = await db.execute("SELECT * FROM config_versions WHERE id = ?", (version_id,))
        row = await cursor.fetchone()
        return dict(row)
