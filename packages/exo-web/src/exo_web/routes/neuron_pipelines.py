"""Neuron pipeline CRUD API for agent context engine configuration."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from exo_web.database import get_db_dep
from exo_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/v1/agents", tags=["neuron-pipelines"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class NeuronConfig(BaseModel):
    """Single neuron in the pipeline."""

    type: str = Field(..., min_length=1, description="Type")
    label: str = Field("", description="Label")
    template: str = Field("", description="Template")
    max_tokens: int | None = Field(None, description="Maximum tokens to generate")
    priority: int = Field(0, description="Priority level")
    enabled: bool = Field(True, description="Whether this item is active")


class PipelineCreate(BaseModel):
    name: str = Field("Default Pipeline", min_length=1, max_length=255, description="Display name")
    neurons: list[NeuronConfig] = Field([], description="Neurons")


class PipelineUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255, description="Display name")
    neurons: list[NeuronConfig] | None = Field(None, description="Neurons")


class PipelineResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    agent_id: str = Field(description="Associated agent identifier")
    name: str = Field(description="Display name")
    neurons: list[dict[str, Any]] = Field(description="Neurons")
    user_id: str = Field(description="Owning user identifier")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NEURON_TYPES = [
    "system_instruction",
    "context_state",
    "tool_results",
    "conversation_history",
    "knowledge_retrieval",
    "custom_template",
]


def _row_to_response(row: Any) -> dict[str, Any]:
    d = dict(row)
    d["neurons"] = json.loads(d.pop("neurons_json", "[]"))
    return d


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/{agent_id}/pipelines")
async def list_pipelines(
    agent_id: str,
    user: dict = Depends(get_current_user),  # noqa: B008
    db: Any = Depends(get_db_dep),  # noqa: B008
) -> list[PipelineResponse]:
    """List pipelines."""
    cursor = await db.execute(
        "SELECT * FROM neuron_pipelines WHERE agent_id = ? AND user_id = ? ORDER BY created_at ASC",
        (agent_id, user["id"]),
    )
    rows = await cursor.fetchall()
    return [_row_to_response(r) for r in rows]


@router.post("/{agent_id}/pipelines", status_code=201)
async def create_pipeline(
    agent_id: str,
    body: PipelineCreate,
    user: dict = Depends(get_current_user),  # noqa: B008
    db: Any = Depends(get_db_dep),  # noqa: B008
) -> PipelineResponse:
    # Verify agent ownership
    """Create pipeline."""
    cursor = await db.execute(
        "SELECT id FROM agents WHERE id = ? AND user_id = ?",
        (agent_id, user["id"]),
    )
    if not await cursor.fetchone():
        raise HTTPException(status_code=404, detail="Agent not found")

    pipeline_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    neurons_json = json.dumps([n.model_dump() for n in body.neurons])

    await db.execute(
        """INSERT INTO neuron_pipelines (id, agent_id, name, neurons_json, user_id, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (pipeline_id, agent_id, body.name, neurons_json, user["id"], now, now),
    )
    await db.commit()

    cursor = await db.execute("SELECT * FROM neuron_pipelines WHERE id = ?", (pipeline_id,))
    row = await cursor.fetchone()
    return _row_to_response(row)


@router.get("/{agent_id}/pipelines/{pipeline_id}")
async def get_pipeline(
    agent_id: str,
    pipeline_id: str,
    user: dict = Depends(get_current_user),  # noqa: B008
    db: Any = Depends(get_db_dep),  # noqa: B008
) -> PipelineResponse:
    """Get pipeline."""
    cursor = await db.execute(
        "SELECT * FROM neuron_pipelines WHERE id = ? AND agent_id = ? AND user_id = ?",
        (pipeline_id, agent_id, user["id"]),
    )
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return _row_to_response(row)


@router.put("/{agent_id}/pipelines/{pipeline_id}")
async def update_pipeline(
    agent_id: str,
    pipeline_id: str,
    body: PipelineUpdate,
    user: dict = Depends(get_current_user),  # noqa: B008
    db: Any = Depends(get_db_dep),  # noqa: B008
) -> PipelineResponse:
    """Update pipeline."""
    cursor = await db.execute(
        "SELECT * FROM neuron_pipelines WHERE id = ? AND agent_id = ? AND user_id = ?",
        (pipeline_id, agent_id, user["id"]),
    )
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    updates = {"updated_at": now}
    if body.name is not None:
        updates["name"] = body.name
    if body.neurons is not None:
        updates["neurons_json"] = json.dumps([n.model_dump() for n in body.neurons])

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = [*updates.values(), pipeline_id]
    await db.execute(f"UPDATE neuron_pipelines SET {set_clause} WHERE id = ?", values)
    await db.commit()

    cursor = await db.execute("SELECT * FROM neuron_pipelines WHERE id = ?", (pipeline_id,))
    row = await cursor.fetchone()
    return _row_to_response(row)


@router.delete("/{agent_id}/pipelines/{pipeline_id}", status_code=204)
async def delete_pipeline(
    agent_id: str,
    pipeline_id: str,
    user: dict = Depends(get_current_user),  # noqa: B008
    db: Any = Depends(get_db_dep),  # noqa: B008
) -> None:
    """Delete pipeline."""
    cursor = await db.execute(
        "SELECT id FROM neuron_pipelines WHERE id = ? AND agent_id = ? AND user_id = ?",
        (pipeline_id, agent_id, user["id"]),
    )
    if not await cursor.fetchone():
        raise HTTPException(status_code=404, detail="Pipeline not found")

    await db.execute("DELETE FROM neuron_pipelines WHERE id = ?", (pipeline_id,))
    await db.commit()


@router.get("/{agent_id}/pipelines/{pipeline_id}/preview")
async def preview_pipeline(
    agent_id: str,
    pipeline_id: str,
    user: dict = Depends(get_current_user),  # noqa: B008
    db: Any = Depends(get_db_dep),  # noqa: B008
) -> dict[str, Any]:
    """Generate a preview of the assembled prompt from the neuron pipeline."""
    cursor = await db.execute(
        "SELECT * FROM neuron_pipelines WHERE id = ? AND agent_id = ? AND user_id = ?",
        (pipeline_id, agent_id, user["id"]),
    )
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    neurons = json.loads(row["neurons_json"])

    # Also fetch agent instructions for system_instruction neuron
    agent_cursor = await db.execute(
        "SELECT instructions FROM agents WHERE id = ? AND user_id = ?",
        (agent_id, user["id"]),
    )
    agent_row = await agent_cursor.fetchone()
    agent_instructions = agent_row["instructions"] if agent_row else ""

    preview_sections: list[dict[str, Any]] = []
    total_tokens = 0

    for neuron in neurons:
        if not neuron.get("enabled", True):
            continue

        neuron_type = neuron.get("type", "")
        template = neuron.get("template", "")
        max_tok = neuron.get("max_tokens")
        label = neuron.get("label", "") or neuron_type.replace("_", " ").title()

        # Estimate token count (~4 chars per token)
        if neuron_type == "system_instruction":
            text = template or agent_instructions or "(Agent instructions will be injected here)"
        elif neuron_type == "context_state":
            text = template or "(Context state key-value pairs will be injected here)"
        elif neuron_type == "tool_results":
            text = template or "(Tool call results will be injected here)"
        elif neuron_type == "conversation_history":
            text = template or "(Conversation message history will be injected here)"
        elif neuron_type == "knowledge_retrieval":
            text = template or "(Retrieved knowledge documents will be injected here)"
        elif neuron_type == "custom_template":
            text = template or "(Custom template content)"
        else:
            text = template or f"({neuron_type})"

        estimated_tokens = max(len(text) // 4, 1)
        if max_tok and estimated_tokens > max_tok:
            estimated_tokens = max_tok

        total_tokens += estimated_tokens
        preview_sections.append(
            {
                "type": neuron_type,
                "label": label,
                "text": text,
                "estimated_tokens": estimated_tokens,
                "max_tokens": max_tok,
            }
        )

    return {
        "sections": preview_sections,
        "total_estimated_tokens": total_tokens,
    }
