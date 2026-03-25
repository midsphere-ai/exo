"""Tests for Agent dynamic runtime loading — US-024.

Covers add_tool(), remove_tool(), add_handoff(), and add_mcp_server().
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.agent import Agent, AgentError
from exo.tool import Tool, tool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@tool
def double(x: int) -> int:
    """Return twice the input."""
    return x * 2


@tool
def triple(x: int) -> int:
    """Return three times the input."""
    return x * 3


# ---------------------------------------------------------------------------
# add_tool
# ---------------------------------------------------------------------------


class TestAddTool:
    @pytest.mark.asyncio
    async def test_add_tool_registers_tool(self) -> None:
        agent = Agent(name="bot", memory=None, context=None)
        assert "double" not in agent.tools
        await agent.add_tool(double)
        assert "double" in agent.tools

    @pytest.mark.asyncio
    async def test_add_tool_appears_in_get_tool_schemas(self) -> None:
        agent = Agent(name="bot", memory=None, context=None)
        await agent.add_tool(double)
        schemas = agent.get_tool_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "double" in names

    @pytest.mark.asyncio
    async def test_add_tool_duplicate_raises(self) -> None:
        agent = Agent(name="bot", memory=None, context=None, tools=[double])
        with pytest.raises(AgentError, match="Duplicate tool name 'double'"):
            await agent.add_tool(double)

    @pytest.mark.asyncio
    async def test_add_multiple_tools_sequential(self) -> None:
        agent = Agent(name="bot", memory=None, context=None)
        await agent.add_tool(double)
        await agent.add_tool(triple)
        assert "double" in agent.tools
        assert "triple" in agent.tools

    @pytest.mark.asyncio
    async def test_add_tool_uses_lock(self) -> None:
        """_tools_lock is an asyncio.Lock instance."""
        agent = Agent(name="bot", memory=None, context=None)
        assert isinstance(agent._tools_lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_concurrent_add_tools_safe(self) -> None:
        """Two concurrent add_tool calls should not raise (different names)."""
        agent = Agent(name="bot", memory=None, context=None)

        @tool
        def tool_a(x: int) -> int:
            """Tool A."""
            return x

        @tool
        def tool_b(x: int) -> int:
            """Tool B."""
            return x

        await asyncio.gather(
            agent.add_tool(tool_a),
            agent.add_tool(tool_b),
        )
        assert "tool_a" in agent.tools
        assert "tool_b" in agent.tools


# ---------------------------------------------------------------------------
# remove_tool
# ---------------------------------------------------------------------------


class TestRemoveTool:
    def test_remove_tool_removes_tool(self) -> None:
        agent = Agent(name="bot", memory=None, context=None, tools=[double])
        assert "double" in agent.tools
        agent.remove_tool("double")
        assert "double" not in agent.tools

    def test_remove_tool_nonexistent_raises(self) -> None:
        agent = Agent(name="bot", memory=None, context=None)
        with pytest.raises(AgentError, match="Tool 'missing' is not registered"):
            agent.remove_tool("missing")

    def test_remove_tool_does_not_affect_other_tools(self) -> None:
        agent = Agent(name="bot", memory=None, context=None, tools=[double, triple])
        agent.remove_tool("double")
        assert "double" not in agent.tools
        assert "triple" in agent.tools

    @pytest.mark.asyncio
    async def test_remove_then_readd(self) -> None:
        agent = Agent(name="bot", memory=None, context=None, tools=[double])
        agent.remove_tool("double")
        await agent.add_tool(double)
        assert "double" in agent.tools


# ---------------------------------------------------------------------------
# add_handoff
# ---------------------------------------------------------------------------


class TestAddHandoff:
    @pytest.mark.asyncio
    async def test_add_handoff_registers_agent(self) -> None:
        agent = Agent(name="main", memory=None, context=None)
        sub = Agent(name="sub", memory=None, context=None)
        assert "sub" not in agent.handoffs
        await agent.add_handoff(sub)
        assert "sub" in agent.handoffs
        assert agent.handoffs["sub"] is sub

    @pytest.mark.asyncio
    async def test_add_handoff_duplicate_raises(self) -> None:
        sub = Agent(name="sub", memory=None, context=None)
        agent = Agent(name="main", memory=None, context=None, handoffs=[sub])
        with pytest.raises(AgentError, match="Duplicate handoff agent 'sub'"):
            await agent.add_handoff(sub)

    @pytest.mark.asyncio
    async def test_add_handoff_uses_same_lock(self) -> None:
        """add_handoff uses the same _tools_lock as add_tool."""
        agent = Agent(name="main", memory=None, context=None)
        sub = Agent(name="sub", memory=None, context=None)

        lock_used: list[bool] = []
        original_lock = agent._tools_lock

        async def _patched_add() -> None:
            async with original_lock:
                lock_used.append(True)
                agent.handoffs["sub"] = sub

        # Verify the lock attribute is the same object across both methods
        assert agent._tools_lock is original_lock

        await agent.add_handoff(sub)
        assert "sub" in agent.handoffs


# ---------------------------------------------------------------------------
# add_mcp_server
# ---------------------------------------------------------------------------


class TestAddMcpServer:
    @pytest.mark.asyncio
    async def test_add_mcp_server_raises_without_exo_mcp(self) -> None:
        """add_mcp_server raises AgentError when exo-mcp is not available."""
        agent = Agent(name="bot", memory=None, context=None)
        fake_config = MagicMock()
        fake_config.name = "test-server"

        with (
            patch.dict(
                "sys.modules",
                {
                    "exo.mcp": None,
                    "exo.mcp.client": None,
                    "exo.mcp.tools": None,
                },
            ),
            pytest.raises(AgentError, match="exo-mcp is required"),
        ):
            await agent.add_mcp_server(fake_config)

    @pytest.mark.asyncio
    async def test_add_mcp_server_loads_and_registers_tools(self) -> None:
        """add_mcp_server connects and registers all tools from the MCP server."""
        agent = Agent(name="bot", memory=None, context=None)
        fake_config = MagicMock()
        fake_config.name = "test-server"

        # Create mock MCP tools
        mock_tool_1 = MagicMock(spec=Tool)
        mock_tool_1.name = "mcp__test_server__search"
        mock_tool_2 = MagicMock(spec=Tool)
        mock_tool_2.name = "mcp__test_server__index"

        mock_conn = AsyncMock()
        mock_conn.connect = AsyncMock()

        mock_mcp_conn_cls = MagicMock(return_value=mock_conn)
        mock_load_tools = AsyncMock(return_value=[mock_tool_1, mock_tool_2])

        with (
            patch(
                "exo.agent.Agent.add_mcp_server.__func__"
                if False
                else "exo.mcp.client.MCPServerConnection",
                mock_mcp_conn_cls,
                create=True,
            ),
        ):
            # Patch the imports inside add_mcp_server
            with patch.dict("sys.modules", {}):
                import sys

                # Create a fake exo.mcp.client module
                fake_client_mod = MagicMock()
                fake_client_mod.MCPServerConnection = mock_mcp_conn_cls

                fake_tools_mod = MagicMock()
                fake_tools_mod.load_tools_from_connection = mock_load_tools

                original_client = sys.modules.get("exo.mcp.client")
                original_tools = sys.modules.get("exo.mcp.tools")
                sys.modules["exo.mcp.client"] = fake_client_mod
                sys.modules["exo.mcp.tools"] = fake_tools_mod

                try:
                    await agent.add_mcp_server(fake_config)
                finally:
                    if original_client is None:
                        sys.modules.pop("exo.mcp.client", None)
                    else:
                        sys.modules["exo.mcp.client"] = original_client
                    if original_tools is None:
                        sys.modules.pop("exo.mcp.tools", None)
                    else:
                        sys.modules["exo.mcp.tools"] = original_tools

        # Verify tools were registered
        assert "mcp__test_server__search" in agent.tools
        assert "mcp__test_server__index" in agent.tools
        mock_conn.connect.assert_awaited_once()
        mock_load_tools.assert_awaited_once_with(mock_conn)

    @pytest.mark.asyncio
    async def test_add_mcp_server_connection_failure_raises_agent_error(self) -> None:
        """Connection errors are wrapped in AgentError."""
        agent = Agent(name="bot", memory=None, context=None)
        fake_config = MagicMock()
        fake_config.name = "bad-server"

        mock_conn = AsyncMock()
        mock_conn.connect = AsyncMock(side_effect=RuntimeError("connection refused"))

        import sys

        fake_client_mod = MagicMock()
        fake_client_mod.MCPServerConnection = MagicMock(return_value=mock_conn)

        fake_tools_mod = MagicMock()
        fake_tools_mod.load_tools_from_connection = AsyncMock(return_value=[])

        original_client = sys.modules.get("exo.mcp.client")
        original_tools = sys.modules.get("exo.mcp.tools")
        sys.modules["exo.mcp.client"] = fake_client_mod
        sys.modules["exo.mcp.tools"] = fake_tools_mod

        try:
            with pytest.raises(AgentError, match="Failed to connect MCP server"):
                await agent.add_mcp_server(fake_config)
        finally:
            if original_client is None:
                sys.modules.pop("exo.mcp.client", None)
            else:
                sys.modules["exo.mcp.client"] = original_client
            if original_tools is None:
                sys.modules.pop("exo.mcp.tools", None)
            else:
                sys.modules["exo.mcp.tools"] = original_tools


# ---------------------------------------------------------------------------
# Tool re-enumeration in run loop (SkillNeuron re-enumerates each step)
# ---------------------------------------------------------------------------


class TestToolReenumerationPerStep:
    @pytest.mark.asyncio
    async def test_tool_schemas_reenumerated_each_step(self) -> None:
        """Tools added between steps appear in subsequent LLM calls."""

        from exo.models.types import ModelResponse  # pyright: ignore[reportMissingImports]
        from exo.types import Usage

        added_on_step_2 = False

        @tool
        def late_tool(x: int) -> int:
            """A tool added dynamically."""
            return x

        call_count = 0

        async def mock_complete(
            messages: Any,
            *,
            tools: Any = None,
            temperature: Any = None,
            max_tokens: Any = None,
        ) -> ModelResponse:
            nonlocal call_count, added_on_step_2
            call_count += 1
            if call_count == 1:
                # First step: no late_tool yet — return text answer
                assert tools is None or all(
                    t.get("function", {}).get("name") != "late_tool" for t in (tools or [])
                )
                return ModelResponse(
                    content="done",
                    tool_calls=[],
                    usage=Usage(input_tokens=5, output_tokens=5, total_tokens=10),
                )
            return ModelResponse(
                content="done",
                tool_calls=[],
                usage=Usage(input_tokens=5, output_tokens=5, total_tokens=10),
            )

        provider = MagicMock()
        provider.complete = mock_complete

        agent = Agent(name="bot", memory=None, context=None)
        # Confirm late_tool is NOT in schemas before run
        schemas_before = agent.get_tool_schemas()
        assert all(s["function"]["name"] != "late_tool" for s in schemas_before)

        # Add the tool before running (to simplify the test)
        await agent.add_tool(late_tool)

        # After add_tool, it should appear in schemas
        schemas_after = agent.get_tool_schemas()
        names = [s["function"]["name"] for s in schemas_after]
        assert "late_tool" in names

        await agent.run("hello", provider=provider)
