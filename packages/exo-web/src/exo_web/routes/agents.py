"""Agents CRUD REST API and AI-assisted generation."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.pagination import paginate
from exo_web.routes.auth import get_current_user
from exo_web.routes.config_versions import _snapshot_agent, create_config_version
from exo_web.sanitize import sanitize_html
from exo_web.services.audit import audit_log

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# AI-generate models
# ---------------------------------------------------------------------------


class AIGenerateRequest(BaseModel):
    description: str = Field(..., min_length=1, description="Human-readable description")
    model: str | None = None  # "provider_id:model_name" format, or uses default


class GeneratedAgentConfig(BaseModel):
    name: str = Field("", description="Display name")
    description: str = Field("", description="Human-readable description")
    instructions: str = Field("", description="System instructions for the agent")
    persona_role: str = Field("", description="Agent persona role")
    persona_goal: str = Field("", description="Agent persona goal")
    persona_backstory: str = Field("", description="Agent persona backstory")
    suggested_tools: list[str] = Field([], description="Recommended tools for this agent")
    suggested_model: str = Field("", description="Recommended model for this agent")
    task: str = Field("", description="Task description")


class AIGenerateResponse(BaseModel):
    agents: list[GeneratedAgentConfig] = Field(description="Agents")


# ---------------------------------------------------------------------------
# CRUD models
# ---------------------------------------------------------------------------


class AgentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    project_id: str = Field(..., min_length=1, description="Associated project identifier")
    description: str = Field("", description="Human-readable description")
    instructions: str = Field("", description="System instructions for the agent")
    model_provider: str = Field("", description="Model provider")
    model_name: str = Field("", description="Model name")
    temperature: float | None = Field(None, description="Sampling temperature (0.0-2.0)")
    max_tokens: int | None = Field(None, description="Maximum tokens to generate")
    max_steps: int | None = Field(None, description="Maximum agent steps")
    output_type_json: str = Field("{}", description="JSON schema for structured output")
    tools_json: str = Field("[]", description="JSON array of tool configurations")
    handoffs_json: str = Field("[]", description="JSON array of handoff configurations")
    hooks_json: str = Field("{}", description="JSON object of hook configurations")
    knowledge_base_ids: str = Field("[]", description="Knowledge base ids")
    persona_role: str = Field("", description="Agent persona role")
    persona_goal: str = Field("", description="Agent persona goal")
    persona_backstory: str = Field("", description="Agent persona backstory")
    autonomous_mode: bool = Field(False, description="Autonomous mode")
    context_automation_level: str = Field("copilot", description="Context automation level")
    context_max_tokens_per_step: int | None = Field(None, description="Context max tokens per step")
    context_max_total_tokens: int | None = Field(None, description="Context max total tokens")
    context_memory_type: str = Field("conversation", description="Context memory type")
    context_workspace_enabled: bool = Field(False, description="Context workspace enabled")


class AgentUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255, description="Display name")
    description: str | None = Field(None, description="Human-readable description")
    instructions: str | None = Field(None, description="System instructions for the agent")
    model_provider: str | None = Field(None, description="Model provider")
    model_name: str | None = Field(None, description="Model name")
    temperature: float | None = Field(None, description="Sampling temperature (0.0-2.0)")
    max_tokens: int | None = Field(None, description="Maximum tokens to generate")
    max_steps: int | None = Field(None, description="Maximum agent steps")
    output_type_json: str | None = Field(None, description="JSON schema for structured output")
    tools_json: str | None = Field(None, description="JSON array of tool configurations")
    handoffs_json: str | None = Field(None, description="JSON array of handoff configurations")
    hooks_json: str | None = Field(None, description="JSON object of hook configurations")
    knowledge_base_ids: str | None = Field(None, description="Knowledge base ids")
    persona_role: str | None = Field(None, description="Agent persona role")
    persona_goal: str | None = Field(None, description="Agent persona goal")
    persona_backstory: str | None = Field(None, description="Agent persona backstory")
    autonomous_mode: bool | None = Field(None, description="Autonomous mode")
    context_automation_level: str | None = Field(None, description="Context automation level")
    context_max_tokens_per_step: int | None = Field(None, description="Context max tokens per step")
    context_max_total_tokens: int | None = Field(None, description="Context max total tokens")
    context_memory_type: str | None = Field(None, description="Context memory type")
    context_workspace_enabled: bool | None = Field(None, description="Context workspace enabled")


class AgentResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    description: str = Field(description="Human-readable description")
    instructions: str = Field(description="System instructions for the agent")
    model_provider: str = Field(description="Model provider")
    model_name: str = Field(description="Model name")
    temperature: float | None = Field(description="Sampling temperature (0.0-2.0)")
    max_tokens: int | None = Field(description="Maximum tokens to generate")
    max_steps: int | None = Field(description="Maximum agent steps")
    output_type_json: str = Field(description="JSON schema for structured output")
    tools_json: str = Field(description="JSON array of tool configurations")
    handoffs_json: str = Field(description="JSON array of handoff configurations")
    hooks_json: str = Field(description="JSON object of hook configurations")
    knowledge_base_ids: str = Field(description="Knowledge base ids")
    persona_role: str = Field(description="Agent persona role")
    persona_goal: str = Field(description="Agent persona goal")
    persona_backstory: str = Field(description="Agent persona backstory")
    autonomous_mode: bool = Field(description="Autonomous mode")
    context_automation_level: str = Field(description="Context automation level")
    context_max_tokens_per_step: int | None = Field(description="Context max tokens per step")
    context_max_total_tokens: int | None = Field(description="Context max total tokens")
    context_memory_type: str = Field(description="Context memory type")
    context_workspace_enabled: bool = Field(description="Context workspace enabled")
    project_id: str = Field(description="Associated project identifier")
    user_id: str = Field(description="Owning user identifier")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


# ---------------------------------------------------------------------------
# Sub-agent / supervisor models
# ---------------------------------------------------------------------------


class SubAgentAdd(BaseModel):
    sub_agent_id: str = Field(..., min_length=1, description="Sub agent id")
    relationship_type: str = Field("delegation", description="Relationship type")
    routing_rule_json: str = Field("{}", description="Routing rule json")


class SubAgentRelationshipResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    supervisor_id: str = Field(description="Supervisor id")
    sub_agent_id: str = Field(description="Sub agent id")
    sub_agent_name: str = Field(description="Sub agent name")
    sub_agent_description: str = Field(description="Sub agent description")
    relationship_type: str = Field(description="Relationship type")
    routing_rule_json: str = Field(description="Routing rule json")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict."""
    d = dict(row)
    d["autonomous_mode"] = bool(d.get("autonomous_mode", 0))
    d["context_workspace_enabled"] = bool(d.get("context_workspace_enabled", 0))
    return d


async def _verify_ownership(db: Any, agent_id: str, user_id: str) -> dict[str, Any]:
    """Verify agent exists and belongs to user. Returns row dict or raises 404."""
    cursor = await db.execute(
        "SELECT * FROM agents WHERE id = ? AND user_id = ?",
        (agent_id, user_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _row_to_dict(row)


async def _get_sub_agents(db: Any, supervisor_id: str) -> list[dict[str, Any]]:
    """Return sub-agent rows (joined with agent info) for a supervisor."""
    cursor = await db.execute(
        """
        SELECT ar.id, ar.supervisor_id, ar.sub_agent_id,
               a.name AS sub_agent_name, a.description AS sub_agent_description,
               ar.relationship_type, ar.routing_rule_json,
               ar.created_at, ar.updated_at
        FROM agent_relationships ar
        JOIN agents a ON a.id = ar.sub_agent_id
        WHERE ar.supervisor_id = ?
        ORDER BY ar.created_at ASC
        """,
        (supervisor_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_dict(r) for r in rows]


def _build_sub_agent_tools(sub_agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build tool definitions that wrap each sub-agent as a callable tool."""
    tools = []
    for sa in sub_agents:
        tool_id = f"delegate_{sa['sub_agent_id']}"
        routing = sa.get("routing_rule_json", "{}")
        tools.append(
            {
                "id": tool_id,
                "name": f"delegate_to_{sa['sub_agent_name'].lower().replace(' ', '_')}",
                "description": (
                    f"Delegate a task to sub-agent '{sa['sub_agent_name']}': "
                    f"{sa['sub_agent_description']}"
                ),
                "type": "sub_agent",
                "sub_agent_id": sa["sub_agent_id"],
                "routing_rule": json.loads(routing) if routing else {},
            }
        )
    return tools


def _build_sub_agent_prompt_section(sub_agents: list[dict[str, Any]]) -> str:
    """Build system prompt section describing available sub-agents."""
    if not sub_agents:
        return ""
    lines = ["\n\n## Available Sub-Agents\nYou can delegate tasks to these sub-agents:\n"]
    for sa in sub_agents:
        lines.append(f"- **{sa['sub_agent_name']}**: {sa['sub_agent_description']}")
        routing = sa.get("routing_rule_json", "{}")
        rules = json.loads(routing) if routing and routing != "{}" else {}
        if rules:
            lines.append(f"  Routing rules: {json.dumps(rules)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# AI-generate helpers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an AI agent configuration generator for the Exo platform.
Given a natural language description of an agent (or multiple agents), generate \
a JSON configuration for each agent.

Available tools (use these IDs in suggested_tools):
web_search, file_read, file_write, code_interpreter, http_request, \
database_query, knowledge_search, calculator, json_parser, text_splitter

Output ONLY valid JSON matching this schema (no markdown fences, no explanation):
{
  "agents": [
    {
      "name": "Agent Name",
      "description": "One-line description of what the agent does",
      "instructions": "Detailed system prompt / instructions for the agent",
      "persona_role": "The role this agent plays (e.g. Senior Data Analyst)",
      "persona_goal": "The primary goal of this agent",
      "persona_backstory": "Brief backstory giving context to the agent's expertise",
      "suggested_tools": ["tool_id_1", "tool_id_2"],
      "suggested_model": "provider:model (e.g. openai:gpt-4o)",
      "task": "If multi-agent, the specific task assigned to this agent"
    }
  ]
}

Rules:
- For a single agent description, return exactly one agent in the array.
- For multi-agent descriptions (e.g. "a team of agents that..."), return \
multiple agents with complementary roles and task assignments.
- Always provide meaningful instructions (at least 2-3 sentences).
- Pick tools that are relevant to the agent's described purpose.
- suggested_model should be a reasonable default (e.g. openai:gpt-4o for \
general tasks, anthropic:claude-sonnet-4-5-20250514 for coding/analysis).
"""


async def _call_model_for_generation(
    provider_id: str, model_name: str, prompt: str, user_id: str
) -> dict[str, Any]:
    """Call a model to generate agent config JSON. Returns parsed result or raises."""
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
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

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
                        "messages": messages,
                        "max_tokens": 2048,
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
                        "system": _SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 2048,
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
                            {"role": "user", "parts": [{"text": _SYSTEM_PROMPT + "\n\n" + prompt}]},
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
                        "prompt": _SYSTEM_PROMPT + "\n\n" + prompt,
                        "stream": False,
                    },
                )
            else:
                raise HTTPException(
                    status_code=400, detail=f"Unsupported provider type: {provider_type}"
                )

            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            if resp.status_code >= 400:
                error_text = resp.text[:500]
                raise HTTPException(
                    status_code=502, detail=f"Model API error ({resp.status_code}): {error_text}"
                )

            data = resp.json()

            # Extract text output
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

            return {"output": output, "elapsed_ms": elapsed_ms}

    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Connection error: {exc!s}") from exc


def _parse_generated_config(raw_output: str) -> list[dict[str, Any]]:
    """Parse the LLM output into a list of agent config dicts."""
    # Strip markdown code fences if present
    text = raw_output.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
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

    # Accept both {"agents": [...]} and direct [...]
    if isinstance(parsed, list):
        agents = parsed
    elif isinstance(parsed, dict) and "agents" in parsed:
        agents = parsed["agents"]
    elif isinstance(parsed, dict):
        agents = [parsed]
    else:
        raise HTTPException(status_code=502, detail="Unexpected response format from model")

    # Validate and normalize each agent config
    valid_tool_ids = {
        "web_search",
        "file_read",
        "file_write",
        "code_interpreter",
        "http_request",
        "database_query",
        "knowledge_search",
        "calculator",
        "json_parser",
        "text_splitter",
    }

    result = []
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        # Filter suggested_tools to only valid IDs
        tools = agent.get("suggested_tools", [])
        if isinstance(tools, list):
            tools = [t for t in tools if isinstance(t, str) and t in valid_tool_ids]
        else:
            tools = []

        result.append(
            {
                "name": str(agent.get("name", "")),
                "description": str(agent.get("description", "")),
                "instructions": str(agent.get("instructions", "")),
                "persona_role": str(agent.get("persona_role", "")),
                "persona_goal": str(agent.get("persona_goal", "")),
                "persona_backstory": str(agent.get("persona_backstory", "")),
                "suggested_tools": tools,
                "suggested_model": str(agent.get("suggested_model", "")),
                "task": str(agent.get("task", "")),
            }
        )

    if not result:
        raise HTTPException(status_code=502, detail="Model generated no valid agent configurations")

    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/ai-generate", response_model=AIGenerateResponse)
async def ai_generate_agent(
    body: AIGenerateRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Generate agent configuration(s) from a natural language description."""
    # Resolve which provider/model to use
    provider_id: str | None = None
    model_name: str | None = None

    if body.model:
        # Try "provider_id:model_name" format
        if ":" in body.model:
            provider_id, model_name = body.model.split(":", 1)
        else:
            model_name = body.model

    # If no provider_id, find the first available provider with an API key
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
                (user["id"],),
            )
            row = await cursor.fetchone()
            if row is None:
                raise HTTPException(
                    status_code=400,
                    detail="No provider with API keys configured. Add a provider in Settings > Models first.",
                )
            provider_id = row["id"]
            # Default model per provider type if not specified
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

    result = await _call_model_for_generation(provider_id, model_name, body.description, user["id"])
    agents = _parse_generated_config(result["output"])

    return {"agents": agents}


@router.get("")
async def list_agents(
    project_id: str | None = Query(None),
    cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return agents for the current user with cursor-based pagination."""
    conditions = ["user_id = ?"]
    params: list[Any] = [user["id"]]
    if project_id:
        conditions.append("project_id = ?")
        params.append(project_id)

    async with get_db() as db:
        result = await paginate(
            db,
            table="agents",
            conditions=conditions,
            params=params,
            cursor=cursor,
            limit=limit,
            row_mapper=_row_to_dict,
        )
        return result.model_dump()


@router.post("", response_model=AgentResponse, status_code=201)
async def create_agent(
    body: AgentCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new agent."""
    async with get_db() as db:
        # Verify project exists and belongs to user.
        cursor = await db.execute(
            "SELECT id FROM projects WHERE id = ? AND user_id = ?",
            (body.project_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")

        agent_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        # Sanitize user-provided text fields.
        name = sanitize_html(body.name)
        description = sanitize_html(body.description)
        instructions = sanitize_html(body.instructions)
        persona_role = sanitize_html(body.persona_role)
        persona_goal = sanitize_html(body.persona_goal)
        persona_backstory = sanitize_html(body.persona_backstory)

        await db.execute(
            """
            INSERT INTO agents (
                id, name, description, instructions,
                model_provider, model_name, temperature, max_tokens, max_steps,
                output_type_json, tools_json, handoffs_json, hooks_json,
                knowledge_base_ids,
                persona_role, persona_goal, persona_backstory,
                autonomous_mode,
                context_automation_level, context_max_tokens_per_step,
                context_max_total_tokens, context_memory_type,
                context_workspace_enabled,
                project_id, user_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                name,
                description,
                instructions,
                body.model_provider,
                body.model_name,
                body.temperature,
                body.max_tokens,
                body.max_steps,
                body.output_type_json,
                body.tools_json,
                body.handoffs_json,
                body.hooks_json,
                body.knowledge_base_ids,
                persona_role,
                persona_goal,
                persona_backstory,
                int(body.autonomous_mode),
                body.context_automation_level,
                body.context_max_tokens_per_step,
                body.context_max_total_tokens,
                body.context_memory_type,
                int(body.context_workspace_enabled),
                body.project_id,
                user["id"],
                now,
                now,
            ),
        )
        # Auto-create initial config version
        cursor = await db.execute("SELECT * FROM agents WHERE id = ?", (agent_id,))
        row = await cursor.fetchone()
        await create_config_version(
            db,
            "agent",
            agent_id,
            _snapshot_agent(dict(row)),
            author=user["id"],
            summary="Initial version",
        )

        await db.commit()

        await audit_log(user["id"], "create_agent", "agent", agent_id, details={"name": name})

        return _row_to_dict(row)


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single agent by ID."""
    async with get_db() as db:
        return await _verify_ownership(db, agent_id, user["id"])


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    body: AgentUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update an agent's editable fields."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    # Sanitize user-provided text fields.
    for field in (
        "name",
        "description",
        "instructions",
        "persona_role",
        "persona_goal",
        "persona_backstory",
    ):
        if field in updates and isinstance(updates[field], str):
            updates[field] = sanitize_html(updates[field])

    # SQLite stores bools as integers
    if "autonomous_mode" in updates:
        updates["autonomous_mode"] = int(updates["autonomous_mode"])
    if "context_workspace_enabled" in updates:
        updates["context_workspace_enabled"] = int(updates["context_workspace_enabled"])

    async with get_db() as db:
        await _verify_ownership(db, agent_id, user["id"])

        updates["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), agent_id]

        await db.execute(
            f"UPDATE agents SET {set_clause} WHERE id = ?",
            values,
        )

        # Auto-create config version on every save
        cursor = await db.execute("SELECT * FROM agents WHERE id = ?", (agent_id,))
        row = await cursor.fetchone()
        await create_config_version(
            db, "agent", agent_id, _snapshot_agent(dict(row)), author=user["id"]
        )

        await db.commit()

        return _row_to_dict(row)


@router.delete("/{agent_id}", status_code=204)
async def delete_agent(
    agent_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete an agent."""
    async with get_db() as db:
        agent = await _verify_ownership(db, agent_id, user["id"])
        await db.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
        await db.commit()
    await audit_log(user["id"], "delete_agent", "agent", agent_id, details={"name": agent["name"]})


# ---------------------------------------------------------------------------
# Sub-agent / supervisor endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/{agent_id}/sub-agents",
    response_model=list[SubAgentRelationshipResponse],
)
async def list_sub_agents(
    agent_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """List all sub-agents for a supervisor agent."""
    async with get_db() as db:
        await _verify_ownership(db, agent_id, user["id"])
        return await _get_sub_agents(db, agent_id)


@router.post(
    "/{agent_id}/sub-agents",
    response_model=SubAgentRelationshipResponse,
    status_code=201,
)
async def add_sub_agent(
    agent_id: str,
    body: SubAgentAdd,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Add a sub-agent to a supervisor agent."""
    if agent_id == body.sub_agent_id:
        raise HTTPException(status_code=422, detail="Agent cannot be its own sub-agent")

    async with get_db() as db:
        # Verify both agents exist and belong to user
        await _verify_ownership(db, agent_id, user["id"])
        await _verify_ownership(db, body.sub_agent_id, user["id"])

        # Check for duplicate
        cursor = await db.execute(
            "SELECT id FROM agent_relationships WHERE supervisor_id = ? AND sub_agent_id = ?",
            (agent_id, body.sub_agent_id),
        )
        if await cursor.fetchone() is not None:
            raise HTTPException(status_code=409, detail="Sub-agent relationship already exists")

        rel_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO agent_relationships (
                id, supervisor_id, sub_agent_id, relationship_type,
                routing_rule_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rel_id,
                agent_id,
                body.sub_agent_id,
                body.relationship_type,
                body.routing_rule_json,
                now,
                now,
            ),
        )
        await db.commit()

        # Return the joined result
        cursor = await db.execute(
            """
            SELECT ar.id, ar.supervisor_id, ar.sub_agent_id,
                   a.name AS sub_agent_name, a.description AS sub_agent_description,
                   ar.relationship_type, ar.routing_rule_json,
                   ar.created_at, ar.updated_at
            FROM agent_relationships ar
            JOIN agents a ON a.id = ar.sub_agent_id
            WHERE ar.id = ?
            """,
            (rel_id,),
        )
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.delete("/{agent_id}/sub-agents/{sub_agent_id}", status_code=204)
async def remove_sub_agent(
    agent_id: str,
    sub_agent_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Remove a sub-agent from a supervisor agent."""
    async with get_db() as db:
        await _verify_ownership(db, agent_id, user["id"])
        cursor = await db.execute(
            "SELECT id FROM agent_relationships WHERE supervisor_id = ? AND sub_agent_id = ?",
            (agent_id, sub_agent_id),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Sub-agent relationship not found")
        await db.execute(
            "DELETE FROM agent_relationships WHERE supervisor_id = ? AND sub_agent_id = ?",
            (agent_id, sub_agent_id),
        )
        await db.commit()


@router.get(
    "/{agent_id}/effective-config",
)
async def get_effective_config(
    agent_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return agent config with sub-agents injected as tools and prompt additions.

    This merges the agent's own tools_json with auto-generated sub-agent tool
    entries and appends sub-agent descriptions to the instructions.
    """
    async with get_db() as db:
        agent = await _verify_ownership(db, agent_id, user["id"])
        sub_agents = await _get_sub_agents(db, agent_id)

        # Merge tools
        existing_tools = json.loads(agent.get("tools_json", "[]"))
        sub_agent_tools = _build_sub_agent_tools(sub_agents)
        effective_tools = [*existing_tools, *sub_agent_tools]

        # Augment instructions
        prompt_section = _build_sub_agent_prompt_section(sub_agents)
        effective_instructions = agent.get("instructions", "") + prompt_section

        return {
            **agent,
            "tools_json": json.dumps(effective_tools),
            "instructions": effective_instructions,
            "sub_agents": sub_agents,
        }
