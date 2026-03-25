"""MCP server integration routes.

Manages MCP server connections and tool discovery.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/v1/integrations", tags=["integrations"])

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class MCPServerCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    url: str = Field(..., min_length=1, description="Url")
    project_id: str = Field(..., min_length=1, description="Associated project identifier")
    auth_config: dict[str, Any] = Field(default_factory=dict, description="Auth config")


class MCPServerUpdate(BaseModel):
    name: str | None = Field(None, description="Display name")
    url: str | None = Field(None, description="Url")
    auth_config: dict[str, Any] | None = Field(None, description="Auth config")


class MCPServerResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    url: str = Field(description="Url")
    status: str = Field(description="Current status")
    last_check_at: str | None = Field(None, description="Last check at")
    error_message: str | None = Field(None, description="Error message")
    project_id: str = Field(description="Associated project identifier")
    user_id: str = Field(description="Owning user identifier")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    updated_at: str = Field(description="ISO 8601 last-update timestamp")


class DiscoveredToolResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    description: str = Field(description="Human-readable description")
    schema_json: str = Field(description="JSON schema definition")
    source_server_id: str = Field(description="Source server id")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict."""
    return dict(row)


async def _verify_server_ownership(db: Any, server_id: str, user_id: str) -> dict[str, Any]:
    """Verify MCP server exists and belongs to user. Returns row dict or raises 404."""
    cursor = await db.execute(
        "SELECT * FROM mcp_servers WHERE id = ? AND user_id = ?",
        (server_id, user_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return _row_to_dict(row)


async def _query_mcp_server(url: str, auth_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Query an MCP server for available tools via its /tools/list endpoint.

    Returns a list of tool descriptors with name, description, and inputSchema.
    """
    import httpx

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if auth_config.get("type") == "bearer":
        headers["Authorization"] = f"Bearer {auth_config['token']}"
    elif auth_config.get("type") == "api_key":
        header_name = auth_config.get("header", "X-API-Key")
        headers[header_name] = auth_config["key"]

    # MCP uses JSON-RPC 2.0 for tool listing
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    # JSON-RPC response: result.tools is the list
    result = data.get("result", {})
    tools = result.get("tools", [])
    return tools  # type: ignore[no-any-return]


async def _ping_mcp_server(url: str, auth_config: dict[str, Any]) -> bool:
    """Lightweight health check — sends a tools/list request and checks for a valid response."""
    try:
        await _query_mcp_server(url, auth_config)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/mcp", response_model=list[MCPServerResponse])
async def list_mcp_servers(
    project_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all MCP servers for the current user."""
    async with get_db() as db:
        conditions = ["user_id = ?"]
        params: list[str] = [user["id"]]
        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)

        where = " AND ".join(conditions)
        cursor = await db.execute(
            f"SELECT * FROM mcp_servers WHERE {where} ORDER BY created_at DESC",
            params,
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


@router.post("/mcp", response_model=MCPServerResponse, status_code=201)
async def create_mcp_server(
    body: MCPServerCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Register a new MCP server connection."""
    server_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    auth_json = json.dumps(body.auth_config)

    async with get_db() as db:
        await db.execute(
            """INSERT INTO mcp_servers
               (id, name, url, auth_config_json, status, project_id, user_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'disconnected', ?, ?, ?, ?)""",
            (server_id, body.name, body.url, auth_json, body.project_id, user["id"], now, now),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM mcp_servers WHERE id = ?", (server_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.get("/mcp/{server_id}", response_model=MCPServerResponse)
async def get_mcp_server(
    server_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Get a single MCP server by ID."""
    async with get_db() as db:
        return await _verify_server_ownership(db, server_id, user["id"])


@router.put("/mcp/{server_id}", response_model=MCPServerResponse)
async def update_mcp_server(
    server_id: str,
    body: MCPServerUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update an MCP server's configuration."""
    async with get_db() as db:
        await _verify_server_ownership(db, server_id, user["id"])

        updates: list[str] = []
        params: list[Any] = []

        if body.name is not None:
            updates.append("name = ?")
            params.append(body.name)
        if body.url is not None:
            updates.append("url = ?")
            params.append(body.url)
        if body.auth_config is not None:
            updates.append("auth_config_json = ?")
            params.append(json.dumps(body.auth_config))

        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        updates.append("updated_at = ?")
        params.append(now)
        params.append(server_id)

        set_clause = ", ".join(updates)
        await db.execute(
            f"UPDATE mcp_servers SET {set_clause} WHERE id = ?",
            params,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM mcp_servers WHERE id = ?", (server_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.delete("/mcp/{server_id}", status_code=204)
async def delete_mcp_server(
    server_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete an MCP server and its discovered tools."""
    async with get_db() as db:
        await _verify_server_ownership(db, server_id, user["id"])

        # Remove tools discovered from this server
        await db.execute(
            "DELETE FROM tools WHERE tool_type = 'mcp' AND code = ? AND user_id = ?",
            (server_id, user["id"]),
        )
        await db.execute("DELETE FROM mcp_servers WHERE id = ?", (server_id,))
        await db.commit()


@router.post("/mcp/{server_id}/discover", response_model=list[DiscoveredToolResponse])
async def discover_tools(
    server_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Query an MCP server for available tools and store them in the tools table."""
    async with get_db() as db:
        server = await _verify_server_ownership(db, server_id, user["id"])

    url = server["url"]
    auth_config = json.loads(server.get("auth_config_json") or "{}")
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    # Query the MCP server
    try:
        mcp_tools = await _query_mcp_server(url, auth_config)
    except Exception as exc:
        _log.exception("Failed to discover tools from MCP server %s", server_id)
        # Update server status to error
        async with get_db() as db:
            await db.execute(
                "UPDATE mcp_servers SET status = 'error', error_message = ?, last_check_at = ?, updated_at = ? WHERE id = ?",
                (str(exc), now, now, server_id),
            )
            await db.commit()
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to MCP server: {exc}",
        ) from exc

    # Update server status to connected
    async with get_db() as db:
        await db.execute(
            "UPDATE mcp_servers SET status = 'connected', error_message = NULL, last_check_at = ?, updated_at = ? WHERE id = ?",
            (now, now, server_id),
        )

        # Remove previously discovered tools from this server, then re-insert
        await db.execute(
            "DELETE FROM tools WHERE tool_type = 'mcp' AND code = ? AND user_id = ?",
            (server_id, user["id"]),
        )

        results: list[dict[str, Any]] = []
        for t in mcp_tools:
            tool_id = str(uuid.uuid4())
            tool_name = t.get("name", "unknown")
            tool_desc = t.get("description", "")
            input_schema = t.get("inputSchema", {})

            await db.execute(
                """INSERT INTO tools
                   (id, name, description, category, schema_json, code, tool_type, project_id, user_id, created_at)
                   VALUES (?, ?, ?, 'custom', ?, ?, 'mcp', ?, ?, ?)""",
                (
                    tool_id,
                    tool_name,
                    tool_desc,
                    json.dumps(input_schema),
                    server_id,  # store server_id in code column as source reference
                    server["project_id"],
                    user["id"],
                    now,
                ),
            )
            results.append(
                {
                    "id": tool_id,
                    "name": tool_name,
                    "description": tool_desc,
                    "schema_json": json.dumps(input_schema),
                    "source_server_id": server_id,
                }
            )

        await db.commit()

    _log.info(
        "Discovered %d tools from MCP server %s (%s)",
        len(results),
        server["name"],
        server_id,
    )
    return results


@router.get("/mcp/{server_id}/status")
async def get_mcp_server_status(
    server_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Get current connection status for an MCP server."""
    async with get_db() as db:
        server = await _verify_server_ownership(db, server_id, user["id"])

    return {
        "id": server["id"],
        "status": server["status"],
        "last_check_at": server["last_check_at"],
        "error_message": server["error_message"],
    }
