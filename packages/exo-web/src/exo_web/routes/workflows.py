"""Workflows CRUD REST API with canvas state persistence."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.pagination import paginate
from exo_web.routes.auth import get_current_user
from exo_web.routes.config_versions import _snapshot_workflow, create_config_version
from exo_web.sanitize import sanitize_html
from exo_web.services.audit import audit_log

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class WorkflowCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    project_id: str = Field(..., min_length=1, description="Associated project identifier")
    description: str = Field("", description="Human-readable description")
    nodes_json: str = Field("[]", description="JSON array of canvas nodes")
    edges_json: str = Field("[]", description="JSON array of canvas edges")
    viewport_json: str = Field('{"x":0,"y":0,"zoom":1}', description="JSON canvas viewport state")


class WorkflowUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255, description="Display name")
    description: str | None = Field(None, description="Human-readable description")
    nodes_json: str | None = Field(None, description="JSON array of canvas nodes")
    edges_json: str | None = Field(None, description="JSON array of canvas edges")
    viewport_json: str | None = Field(None, description="JSON canvas viewport state")
    status: str | None = Field(None, description="Current status")


class WorkflowImport(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    project_id: str = Field(..., min_length=1, description="Associated project identifier")
    description: str = Field("", description="Human-readable description")
    nodes_json: str = Field("[]", description="JSON array of canvas nodes")
    edges_json: str = Field("[]", description="JSON array of canvas edges")
    viewport_json: str = Field('{"x":0,"y":0,"zoom":1}', description="JSON canvas viewport state")


class WorkflowResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    description: str = Field(description="Human-readable description")
    project_id: str = Field(description="Associated project identifier")
    nodes_json: str = Field(description="JSON array of canvas nodes")
    edges_json: str = Field(description="JSON array of canvas edges")
    viewport_json: str = Field(description="JSON canvas viewport state")
    version: int = Field(description="Version identifier")
    status: str = Field(description="Current status")
    last_run_at: str | None = Field(description="Last execution timestamp")
    user_id: str = Field(description="Owning user identifier")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


class WorkflowAIGenerateRequest(BaseModel):
    description: str = Field(..., min_length=1, description="Human-readable description")
    model: str | None = None  # "provider_id:model_name" format, or uses default


class WorkflowAIRefineRequest(BaseModel):
    instruction: str = Field(..., min_length=1, description="Instruction")
    model: str | None = Field(None, description="Model identifier")


class WorkflowAIResponse(BaseModel):
    nodes: list[dict[str, Any]] = Field(description="Nodes")
    edges: list[dict[str, Any]] = Field(description="Edges")


# ---------------------------------------------------------------------------
# Node category → color mapping (mirrors NodeSidebar.tsx)
# ---------------------------------------------------------------------------

_NODE_CATEGORY_COLORS: dict[str, str] = {
    # triggers
    "chat_input": "#F76F53",
    "webhook": "#F76F53",
    "schedule": "#F76F53",
    "manual": "#F76F53",
    # llm
    "llm_call": "#6287f5",
    "prompt_template": "#6287f5",
    "model_selector": "#6287f5",
    # agent
    "agent_node": "#a78bfa",
    "sub_agent": "#a78bfa",
    # tools
    "function_tool": "#63f78b",
    "http_request": "#63f78b",
    "code_python": "#63f78b",
    "code_javascript": "#63f78b",
    # logic
    "conditional": "#f59e0b",
    "switch": "#f59e0b",
    "loop_iterator": "#f59e0b",
    "aggregator": "#f59e0b",
    "approval_gate": "#f59e0b",
    # data
    "variable_assigner": "#14b8a6",
    "template_jinja": "#14b8a6",
    "json_transform": "#14b8a6",
    "text_splitter": "#14b8a6",
    # knowledge
    "knowledge_retrieval": "#8b5cf6",
    "document_loader": "#8b5cf6",
    "embedding_node": "#8b5cf6",
    # output
    "chat_response": "#ec4899",
    "api_response": "#ec4899",
    "file_output": "#ec4899",
    "notification": "#ec4899",
}

_VALID_NODE_TYPES = set(_NODE_CATEGORY_COLORS.keys())

# ---------------------------------------------------------------------------
# AI workflow generation system prompt
# ---------------------------------------------------------------------------

_WORKFLOW_SYSTEM_PROMPT = """\
You are a workflow design assistant. Given a natural language description, \
generate a workflow as a JSON object with two arrays: "nodes" and "edges".

## Node format
Each node is an object:
{
  "id": "<unique_string>",
  "type": "workflow",
  "data": {
    "nodeType": "<one of the valid types>",
    "label": "<human-readable label>"
  },
  "position": {"x": <number>, "y": <number>}
}

## Valid node types by category
- Triggers (start a flow, output-only): chat_input, webhook, schedule, manual
- LLM: llm_call, prompt_template, model_selector
- Agent: agent_node, sub_agent
- Tools: function_tool, http_request, code_python, code_javascript
- Logic: conditional, switch, loop_iterator, aggregator, approval_gate
- Data: variable_assigner, template_jinja, json_transform, text_splitter
- Knowledge: knowledge_retrieval, document_loader, embedding_node
- Output (end a flow, input-only): chat_response, api_response, file_output, notification

## Edge format
Each edge connects two nodes:
{
  "id": "<unique_string>",
  "source": "<source_node_id>",
  "target": "<target_node_id>",
  "sourceHandle": "output",
  "targetHandle": "input"
}
For conditional nodes, use sourceHandle "output-true" or "output-false" for branches.

## Layout rules
- Place nodes left-to-right. Triggers on the left (x≈100), outputs on the right.
- Space nodes ~250px apart horizontally, ~150px apart vertically for parallel paths.
- Every workflow should start with a trigger node and end with an output node.

## Response rules
- Return ONLY valid JSON — no markdown fences, no explanation, no extra text.
- Use short, descriptive labels for each node.
- Keep the workflow minimal but complete for the described task.
"""

_REFINE_SYSTEM_PROMPT = """\
You are a workflow refinement assistant. You will receive an existing workflow \
(nodes and edges as JSON) and a user instruction describing how to modify it.

Apply the requested changes and return the COMPLETE updated workflow as a JSON \
object with "nodes" and "edges" arrays. Follow the same format rules:

## Node format
{"id": "<string>", "type": "workflow", "data": {"nodeType": "<type>", "label": "<label>"}, "position": {"x": <num>, "y": <num>}}

## Valid node types
Triggers: chat_input, webhook, schedule, manual
LLM: llm_call, prompt_template, model_selector
Agent: agent_node, sub_agent
Tools: function_tool, http_request, code_python, code_javascript
Logic: conditional, switch, loop_iterator, aggregator, approval_gate
Data: variable_assigner, template_jinja, json_transform, text_splitter
Knowledge: knowledge_retrieval, document_loader, embedding_node
Output: chat_response, api_response, file_output, notification

## Edge format
{"id": "<string>", "source": "<node_id>", "target": "<node_id>", "sourceHandle": "output", "targetHandle": "input"}

## Rules
- Preserve existing node IDs when not removing nodes.
- Re-layout positions if new nodes are added so nothing overlaps.
- Return ONLY valid JSON — no markdown fences, no explanation.
"""


# ---------------------------------------------------------------------------
# AI helpers
# ---------------------------------------------------------------------------


async def _resolve_provider(user_id: str, model_spec: str | None) -> tuple[str, str]:
    """Resolve provider_id and model_name from an optional model spec string.

    Returns (provider_id, model_name).
    """
    provider_id: str | None = None
    model_name: str | None = None

    if model_spec:
        if ":" in model_spec:
            provider_id, model_name = model_spec.split(":", 1)
        else:
            model_name = model_spec

    if not provider_id:
        async with get_db() as db:
            cursor = await db.execute(
                """
                SELECT p.id, p.provider_type FROM providers p
                WHERE p.user_id = ?
                AND (
                    p.encrypted_api_key IS NOT NULL AND p.encrypted_api_key != ''
                    OR EXISTS (
                        SELECT 1 FROM provider_keys pk
                        WHERE pk.provider_id = p.id AND pk.status = 'active'
                    )
                )
                ORDER BY p.created_at ASC LIMIT 1
                """,
                (user_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                raise HTTPException(
                    status_code=400,
                    detail="No provider with API keys configured. Add a provider in Settings > Models first.",
                )
            provider_id = row["id"]
            if not model_name:
                ptype = row["provider_type"]
                model_name = {
                    "openai": "gpt-4o",
                    "anthropic": "claude-sonnet-4-5-20250514",
                    "gemini": "gemini-2.0-flash",
                    "ollama": "llama3.2",
                    "custom": "gpt-4o",
                }.get(ptype, "gpt-4o")

    if not model_name:
        model_name = "gpt-4o"

    return provider_id, model_name  # type: ignore[return-value]


async def _call_model_for_workflow(
    provider_id: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    user_id: str,
) -> str:
    """Call a model and return the raw text output."""
    import httpx

    from exo_web.crypto import decrypt_api_key

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM providers WHERE id = ? AND user_id = ?",
            (provider_id, user_id),
        )
        provider_row = await cursor.fetchone()
        if provider_row is None:
            raise HTTPException(status_code=404, detail="Provider not found")
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
            raise HTTPException(status_code=400, detail="No API key configured for provider")

    provider_type = provider["provider_type"]
    base_url = provider.get("base_url") or ""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

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
                        "messages": messages,
                        "max_tokens": 4096,
                        "temperature": 0.7,
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
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_prompt}],
                        "max_tokens": 4096,
                        "temperature": 0.7,
                    },
                )
            elif provider_type == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                resp = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": system_prompt + "\n\n" + user_prompt}],
                            },
                        ],
                    },
                )
            elif provider_type == "ollama":
                url = (base_url or "http://localhost:11434") + "/api/generate"
                resp = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": model_name,
                        "prompt": system_prompt + "\n\n" + user_prompt,
                        "stream": False,
                    },
                )
            else:
                raise HTTPException(
                    status_code=400, detail=f"Unsupported provider type: {provider_type}"
                )

            if resp.status_code >= 400:
                error_text = resp.text[:500]
                raise HTTPException(
                    status_code=502, detail=f"Model API error ({resp.status_code}): {error_text}"
                )

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

            return output

    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Connection error: {exc!s}") from exc


def _parse_workflow_json(raw_output: str) -> dict[str, Any]:
    """Parse LLM output into validated {nodes, edges} dict."""
    text = raw_output.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        first_newline = text.index("\n")
        text = text[first_newline + 1 :]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Model returned invalid JSON: {exc!s}",
        ) from exc

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=502, detail="Expected JSON object with nodes and edges")

    nodes = parsed.get("nodes", [])
    edges = parsed.get("edges", [])

    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise HTTPException(status_code=502, detail="nodes and edges must be arrays")

    # Validate and enrich nodes
    valid_nodes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id", "")
        if not node_id or node_id in seen_ids:
            continue
        seen_ids.add(node_id)

        data = node.get("data", {})
        if not isinstance(data, dict):
            continue
        node_type = data.get("nodeType", "")
        if node_type not in _VALID_NODE_TYPES:
            continue

        # Ensure type is "workflow" for canvas rendering
        node["type"] = "workflow"
        # Inject category color
        data["categoryColor"] = _NODE_CATEGORY_COLORS.get(node_type, "#999")
        node["data"] = data

        # Ensure position exists
        pos = node.get("position", {})
        if not isinstance(pos, dict) or "x" not in pos or "y" not in pos:
            node["position"] = {"x": 100, "y": 100}

        valid_nodes.append(node)

    # Validate edges — only keep edges referencing valid nodes
    valid_edges: list[dict[str, Any]] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if src not in seen_ids or tgt not in seen_ids:
            continue
        # Ensure required handle fields
        if "sourceHandle" not in edge:
            edge["sourceHandle"] = "output"
        if "targetHandle" not in edge:
            edge["targetHandle"] = "input"
        if "id" not in edge:
            edge["id"] = f"edge_{src}_{tgt}"
        valid_edges.append(edge)

    if not valid_nodes:
        raise HTTPException(status_code=502, detail="Model generated no valid workflow nodes")

    return {"nodes": valid_nodes, "edges": valid_edges}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict."""
    return dict(row)


async def _verify_ownership(db: Any, workflow_id: str, user_id: str) -> dict[str, Any]:
    """Verify workflow exists and belongs to user. Returns row dict or raises 404."""
    cursor = await db.execute(
        "SELECT * FROM workflows WHERE id = ? AND user_id = ?",
        (workflow_id, user_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Endpoints — AI generation (before /{workflow_id} param routes)
# ---------------------------------------------------------------------------


@router.post("/ai-generate", response_model=WorkflowAIResponse)
async def ai_generate_workflow(
    body: WorkflowAIGenerateRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Generate a workflow from a natural language description."""
    provider_id, model_name = await _resolve_provider(user["id"], body.model)

    prompt = f"Generate a workflow for the following task:\n\n{sanitize_html(body.description)}"
    raw_output = await _call_model_for_workflow(
        provider_id, model_name, _WORKFLOW_SYSTEM_PROMPT, prompt, user["id"]
    )
    return _parse_workflow_json(raw_output)


@router.post("/{workflow_id}/ai-refine", response_model=WorkflowAIResponse)
async def ai_refine_workflow(
    workflow_id: str,
    body: WorkflowAIRefineRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Refine an existing workflow based on a natural language instruction."""
    async with get_db() as db:
        wf = await _verify_ownership(db, workflow_id, user["id"])

    provider_id, model_name = await _resolve_provider(user["id"], body.model)

    prompt = (
        f"Here is the current workflow:\n\n"
        f"Nodes:\n{wf['nodes_json']}\n\n"
        f"Edges:\n{wf['edges_json']}\n\n"
        f"Apply this modification:\n{sanitize_html(body.instruction)}"
    )
    raw_output = await _call_model_for_workflow(
        provider_id, model_name, _REFINE_SYSTEM_PROMPT, prompt, user["id"]
    )
    return _parse_workflow_json(raw_output)


# ---------------------------------------------------------------------------
# Endpoints — CRUD
# ---------------------------------------------------------------------------


@router.get("")
async def list_workflows(
    project_id: str | None = Query(None),
    cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return workflows for the current user with cursor-based pagination."""
    conditions = ["user_id = ?"]
    params: list[Any] = [user["id"]]
    if project_id:
        conditions.append("project_id = ?")
        params.append(project_id)

    async with get_db() as db:
        result = await paginate(
            db,
            table="workflows",
            conditions=conditions,
            params=params,
            cursor=cursor,
            limit=limit,
            row_mapper=_row_to_dict,
        )
        return result.model_dump()


@router.post("", response_model=WorkflowResponse, status_code=201)
async def create_workflow(
    body: WorkflowCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new workflow."""
    async with get_db() as db:
        # Verify project exists and belongs to user.
        cursor = await db.execute(
            "SELECT id FROM projects WHERE id = ? AND user_id = ?",
            (body.project_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")

        workflow_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO workflows (id, name, description, project_id, nodes_json, edges_json, viewport_json, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workflow_id,
                sanitize_html(body.name),
                sanitize_html(body.description),
                body.project_id,
                body.nodes_json,
                body.edges_json,
                body.viewport_json,
                user["id"],
                now,
                now,
            ),
        )

        # Auto-create initial config version
        cursor = await db.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
        row = await cursor.fetchone()
        await create_config_version(
            db,
            "workflow",
            workflow_id,
            _snapshot_workflow(dict(row)),
            author=user["id"],
            summary="Initial version",
        )

        await db.commit()

        await audit_log(
            user["id"],
            "create_workflow",
            "workflow",
            workflow_id,
            details={"name": sanitize_html(body.name)},
        )

        return _row_to_dict(row)


@router.post("/import", response_model=WorkflowResponse, status_code=201)
async def import_workflow(
    body: WorkflowImport,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Import and create a workflow from JSON data."""
    # Validate that JSON fields are valid JSON.
    for field_name in ("nodes_json", "edges_json", "viewport_json"):
        value = getattr(body, field_name)
        try:
            json.loads(value)
        except (json.JSONDecodeError, TypeError):
            raise HTTPException(  # noqa: B904
                status_code=422,
                detail=f"Invalid JSON in {field_name}",
            )

    async with get_db() as db:
        # Verify project exists and belongs to user.
        cursor = await db.execute(
            "SELECT id FROM projects WHERE id = ? AND user_id = ?",
            (body.project_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")

        workflow_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO workflows (id, name, description, project_id, nodes_json, edges_json, viewport_json, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workflow_id,
                sanitize_html(body.name),
                sanitize_html(body.description),
                body.project_id,
                body.nodes_json,
                body.edges_json,
                body.viewport_json,
                user["id"],
                now,
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single workflow by ID (including canvas state)."""
    async with get_db() as db:
        return await _verify_ownership(db, workflow_id, user["id"])


@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: str,
    body: WorkflowUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a workflow's fields (including canvas state)."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    for field in ("name", "description"):
        if field in updates and isinstance(updates[field], str):
            updates[field] = sanitize_html(updates[field])

    async with get_db() as db:
        await _verify_ownership(db, workflow_id, user["id"])

        updates["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), workflow_id]

        await db.execute(
            f"UPDATE workflows SET {set_clause} WHERE id = ?",
            values,
        )

        # Auto-create config version on every save
        cursor = await db.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
        row = await cursor.fetchone()
        await create_config_version(
            db, "workflow", workflow_id, _snapshot_workflow(dict(row)), author=user["id"]
        )

        await db.commit()

        return _row_to_dict(row)


@router.delete("/{workflow_id}", status_code=204)
async def delete_workflow(
    workflow_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a workflow."""
    async with get_db() as db:
        wf = await _verify_ownership(db, workflow_id, user["id"])
        await db.execute("DELETE FROM workflows WHERE id = ?", (workflow_id,))
        await db.commit()
    await audit_log(
        user["id"], "delete_workflow", "workflow", workflow_id, details={"name": wf["name"]}
    )


@router.post("/{workflow_id}/export")
async def export_workflow(
    workflow_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> JSONResponse:
    """Export a workflow as downloadable JSON."""
    async with get_db() as db:
        data = await _verify_ownership(db, workflow_id, user["id"])

    export_data = {
        "name": data["name"],
        "description": data["description"],
        "version": data["version"],
        "nodes_json": data["nodes_json"],
        "edges_json": data["edges_json"],
        "viewport_json": data["viewport_json"],
    }

    safe_name = data["name"].replace(" ", "_").lower()
    return JSONResponse(
        content=export_data,
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.json"',
        },
    )


@router.post("/{workflow_id}/duplicate", response_model=WorkflowResponse, status_code=201)
async def duplicate_workflow(
    workflow_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Duplicate a workflow with '(Copy)' suffix."""
    async with get_db() as db:
        data = await _verify_ownership(db, workflow_id, user["id"])

        new_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO workflows (id, name, description, project_id, nodes_json, edges_json, viewport_json, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                data["name"] + " (Copy)",
                data["description"],
                data["project_id"],
                data["nodes_json"],
                data["edges_json"],
                data["viewport_json"],
                user["id"],
                now,
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM workflows WHERE id = ?", (new_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)
