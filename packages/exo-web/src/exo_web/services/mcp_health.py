"""Background health monitor for MCP server connections.

Periodically pings registered MCP servers and updates their status.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

from exo_web.database import get_db

_log = logging.getLogger(__name__)

# Poll every 60 seconds.
_POLL_INTERVAL = 60

# Sentinel for the background polling loop.
_health_task: asyncio.Task[Any] | None = None


async def _check_server(server: dict[str, Any]) -> None:
    """Ping a single MCP server and update its status."""
    import httpx

    server_id = server["id"]
    url = server["url"]
    auth_config = json.loads(server.get("auth_config_json") or "{}")
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if auth_config.get("type") == "bearer":
        headers["Authorization"] = f"Bearer {auth_config['token']}"
    elif auth_config.get("type") == "api_key":
        header_name = auth_config.get("header", "X-API-Key")
        headers[header_name] = auth_config["key"]

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()

        async with get_db() as db:
            await db.execute(
                "UPDATE mcp_servers SET status = 'connected', error_message = NULL, last_check_at = ?, updated_at = ? WHERE id = ?",
                (now, now, server_id),
            )
            await db.commit()
    except Exception as exc:
        error_msg = str(exc)[:500]
        async with get_db() as db:
            await db.execute(
                "UPDATE mcp_servers SET status = 'error', error_message = ?, last_check_at = ?, updated_at = ? WHERE id = ?",
                (error_msg, now, now, server_id),
            )
            await db.commit()
        _log.warning("MCP server %s health check failed: %s", server_id, error_msg)


async def _poll_loop() -> None:
    """Continuously check MCP server health."""
    while True:
        try:
            async with get_db() as db:
                cursor = await db.execute("SELECT * FROM mcp_servers")
                servers = await cursor.fetchall()

            for row in servers:
                server = dict(row)
                try:
                    await _check_server(server)
                except Exception:
                    _log.exception("Error checking MCP server %s", server["id"])

        except Exception:
            _log.exception("Error in MCP health poll loop")

        await asyncio.sleep(_POLL_INTERVAL)


async def start_mcp_health() -> None:
    """Start the background MCP health polling loop."""
    global _health_task
    if _health_task is not None:
        return
    _log.info("Starting MCP health monitor (poll every %ds)", _POLL_INTERVAL)
    _health_task = asyncio.create_task(_poll_loop())


async def stop_mcp_health() -> None:
    """Stop the background MCP health monitor."""
    global _health_task
    if _health_task is None:
        return
    _log.info("Stopping MCP health monitor")
    _health_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await _health_task
    _health_task = None
