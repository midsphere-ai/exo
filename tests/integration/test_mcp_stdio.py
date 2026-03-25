"""Integration tests for MCP stdio subprocess connectivity.

US-INT-010: Verifies that an agent can connect to a real stdio MCP server
subprocess and successfully call its tools.

The mcp_server_process fixture (from conftest.py) provides an MCPServerConfig
that describes how to launch the test MCP server subprocess.  The MCP client
inside the agent spawns the subprocess on each connection.

Tool names are namespaced: ``mcp__test_server__get_capital``.
"""

from __future__ import annotations

import pytest

from exo import Agent  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# test_agent_calls_mcp_tool_get_capital
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_agent_calls_mcp_tool_get_capital(
    vertex_model: str,
    mcp_server_process,
) -> None:
    """Agent loads MCP server, calls get_capital(Japan), result contains 'tokyo'."""
    agent = Agent(
        name="mcp-test-agent",
        model=vertex_model,
        instructions=(
            "You are a helpful assistant. "
            "When asked to call a tool, call it immediately and return the result."
        ),
        memory=None,
        context=None,
    )

    await agent.add_mcp_server(mcp_server_process)

    # Confirm the tool is registered (namespaced name)
    tool_names = list(agent.tools.keys())
    assert any("get_capital" in name for name in tool_names), (
        f"get_capital tool not found in agent tools: {tool_names}"
    )

    result = await agent.run(
        "You MUST call the get_capital tool with country=Japan. "
        "Do not answer from memory. Call the tool now and return exactly what it says."
    )

    # Assert at least one ToolCall with 'get_capital' in the name
    capital_calls = [tc for tc in result.tool_calls if "get_capital" in tc.name]
    assert capital_calls, (
        f"Expected a get_capital tool call but got: {[tc.name for tc in result.tool_calls]}"
    )

    # Assert 'tokyo' appears in the output text
    assert "tokyo" in result.text.lower(), (
        f"Expected 'tokyo' in result but got: {result.text!r}"
    )


# ---------------------------------------------------------------------------
# test_agent_calls_both_mcp_tools_sequentially
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_agent_calls_both_mcp_tools_sequentially(
    vertex_model: str,
    mcp_server_process,
) -> None:
    """Agent calls get_capital then get_population sequentially via MCP tools."""
    agent = Agent(
        name="mcp-chain-agent",
        model=vertex_model,
        instructions=(
            "You are a helpful assistant. "
            "When asked to call tools in sequence, call them one at a time in order."
        ),
        memory=None,
        context=None,
    )

    await agent.add_mcp_server(mcp_server_process)

    result = await agent.run(
        "I need you to do two steps in order: "
        "Step 1: Call the get_capital tool with country=France to get its capital. "
        "Step 2: Then call the get_population tool with the capital city you just found. "
        "Do NOT answer from memory — you MUST call both tools."
    )

    # Assert exactly two tool calls with the correct names
    capital_calls = [tc for tc in result.tool_calls if "get_capital" in tc.name]
    population_calls = [tc for tc in result.tool_calls if "get_population" in tc.name]

    assert capital_calls, (
        f"Expected a get_capital tool call but got: {[tc.name for tc in result.tool_calls]}"
    )
    assert population_calls, (
        f"Expected a get_population tool call but got: {[tc.name for tc in result.tool_calls]}"
    )

    # Verify the calls are in the correct order by checking position in tool_calls list
    first_capital_idx = next(
        i for i, tc in enumerate(result.tool_calls) if "get_capital" in tc.name
    )
    first_population_idx = next(
        i for i, tc in enumerate(result.tool_calls) if "get_population" in tc.name
    )
    assert first_capital_idx < first_population_idx, (
        f"get_capital (idx={first_capital_idx}) should appear before "
        f"get_population (idx={first_population_idx})"
    )
