"""SSE MCP servers on distributed workers — Orbiter demo.

Demonstrates how an agent with SSE-based MCP tools is serialized,
submitted to a distributed worker queue, and automatically reconnects
to the remote MCP servers on the worker side via lazy reconnection.

Key concepts:
    - MCPServerConfig with transport="sse" for remote MCP servers
    - MCPToolWrapper serialization stores the server_config so workers
      can reconnect without the original MCP client context
    - The distributed() API handles everything: serialize → queue → stream

Prerequisites:
    # Terminal 1 — start Redis
    docker run -p 6379:6379 redis:7

    # Terminal 2 — start one or more SSE MCP servers.
    #   For example, using the official MCP 'everything' demo server:
    #   npx @modelcontextprotocol/server-everything --transport sse --port 3001
    #
    #   Or any SSE-compatible MCP server listening on a URL.

    # Terminal 3 — start a worker
    export ORBITER_REDIS_URL=redis://localhost:6379
    export OPENAI_API_KEY=sk-...
    orbiter start worker

    # Terminal 4 — run this script
    export ORBITER_REDIS_URL=redis://localhost:6379
    export OPENAI_API_KEY=sk-...
    uv run python examples/distributed/mcp_sse_workers.py

How it works:
    1. Configure SSE MCP servers with MCPServerConfig(transport="sse", url=...)
    2. Connect locally to discover available tools
    3. Create an Agent with the discovered MCP tools (+ optional local tools)
    4. Submit the agent to the distributed queue via distributed()
    5. The worker receives the serialized agent, reconstructs MCPToolWrapper
       instances with stored server_config, and lazily connects to the SSE
       servers on first tool execution
    6. Stream events back to the client in real time
"""

import asyncio
import json
import os

from orbiter import Agent, tool
from orbiter.distributed import distributed  # pyright: ignore[reportMissingImports]
from orbiter.mcp.client import MCPClient, MCPServerConfig  # pyright: ignore[reportMissingImports]
from orbiter.mcp.tools import load_tools_from_client  # pyright: ignore[reportMissingImports]
from orbiter.types import (
    ErrorEvent,
    StatusEvent,
    StepEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)

# ---------------------------------------------------------------------------
# Local tools (always available, serialized by import path)
# ---------------------------------------------------------------------------

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    return str(eval(expression))


# ---------------------------------------------------------------------------
# SSE MCP server configs
# ---------------------------------------------------------------------------

# Configure your SSE MCP servers here.  These are network-accessible servers
# that both the submitting client AND the distributed worker can reach.
#
# IMPORTANT: Because workers lazily reconnect, the SSE URL must be reachable
# from the worker's network — use a hostname/IP the worker can resolve.

MCP_SERVERS = [
    MCPServerConfig(
        name="everything",
        transport="sse",
        url=os.environ.get("MCP_EVERYTHING_URL", "http://localhost:3001/sse"),
        timeout=30.0,
        sse_read_timeout=300.0,
    ),
    # Add more SSE servers as needed:
    #
    # MCPServerConfig(
    #     name="my-tools",
    #     transport="sse",
    #     url="http://tools-service.internal:8080/sse",
    #     headers={"Authorization": "Bearer <token>"},
    # ),
]


# ---------------------------------------------------------------------------
# Discover tools & build agent
# ---------------------------------------------------------------------------

async def build_agent() -> Agent:
    """Connect to SSE MCP servers, discover tools, and build an Agent.

    The MCPToolWrappers returned by load_tools_from_client() already carry
    a reference to the server_config.  When the agent is serialized via
    to_dict(), each MCPToolWrapper stores its config so the worker can
    reconnect.
    """
    client = MCPClient()
    for cfg in MCP_SERVERS:
        client.add_server(cfg)

    async with client:
        mcp_tools = await load_tools_from_client(client)

        print(f"Discovered {len(mcp_tools)} MCP tool(s) from SSE servers:")
        for t in mcp_tools:
            print(f"  - {t.name}: {t.description[:80]}")

        # Verify serialization round-trip before submitting
        _verify_serialization(mcp_tools)

        all_tools = [calculate, *mcp_tools]

        return Agent(
            name="sse-mcp-agent",
            model="openai:gpt-4o-mini",
            instructions=(
                "You are a helpful assistant with access to remote MCP tools "
                "via SSE transport, plus a local calculator.\n"
                "Use the available tools to answer the user's questions."
            ),
            tools=all_tools,
        )


def _verify_serialization(mcp_tools: list) -> None:
    """Sanity-check that MCP tools survive serialization round-trip."""
    from orbiter.agent import _deserialize_tool, _serialize_tool

    for t in mcp_tools:
        serialized = _serialize_tool(t)
        assert isinstance(serialized, dict), f"Expected dict, got {type(serialized)}"
        assert serialized.get("__mcp_tool__") is True
        assert "server_config" in serialized, (
            f"Tool '{t.name}' missing server_config — worker won't be able to reconnect"
        )

        # Verify JSON round-trip
        json_str = json.dumps(serialized)
        restored = _deserialize_tool(json.loads(json_str))
        assert restored.name == t.name
        assert restored._server_config is not None
        assert restored._server_config.transport == "sse"

    print(f"\n  Serialization check passed for {len(mcp_tools)} MCP tool(s).\n")


# ---------------------------------------------------------------------------
# Submit to distributed queue & stream events
# ---------------------------------------------------------------------------

async def main() -> None:
    agent = await build_agent()

    # Show the serialized agent config for debugging
    config = agent.to_dict()
    mcp_tool_count = sum(1 for t in config.get("tools", []) if isinstance(t, dict))
    local_tool_count = sum(1 for t in config.get("tools", []) if isinstance(t, str))
    print(f"Agent config: {local_tool_count} local tool(s), {mcp_tool_count} MCP tool(s)")
    print("Submitting to distributed queue...\n")

    handle = await distributed(
        agent,
        "Use the available tools to demonstrate your capabilities. "
        "Then calculate 2**16 - 1.",
        detailed=True,
    )

    print(f"Task submitted: {handle.task_id}\n")

    # Stream events from the worker
    async for event in handle.stream():
        match event:
            case TextEvent():
                print(event.text, end="", flush=True)

            case ToolCallEvent():
                print(f"\n  [tool call] {event.tool_name}({event.arguments})")

            case ToolResultEvent():
                status = "ok" if event.success else "FAILED"
                print(f"  [tool result] {event.tool_name} -> {status} ({event.duration_ms:.0f}ms)")

            case StepEvent() if event.status == "started":
                print(f"\n--- step {event.step_number} ---")

            case StatusEvent():
                print(f"\n[status] {event.status}: {event.message}")

            case UsageEvent():
                u = event.usage
                print(f"[usage] {u.input_tokens} in / {u.output_tokens} out tokens")

            case ErrorEvent():
                print(f"\n[error] {event.error_type}: {event.error}")


if __name__ == "__main__":
    asyncio.run(main())
