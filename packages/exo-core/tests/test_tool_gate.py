"""Tests for Agent.tool_gate — conditional tool injection that preserves KV cache."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from exo.agent import Agent, AgentError
from exo.models.types import ModelResponse  # pyright: ignore[reportMissingImports]
from exo.tool import tool
from exo.types import ToolCall, Usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_USAGE = Usage(input_tokens=10, output_tokens=5, total_tokens=15)


@tool
def search(query: str) -> str:
    """Search the database."""
    return f"results for {query}"


@tool
def write_record(data: str) -> str:
    """Write a record to the database."""
    return f"wrote {data}"


@tool
def delete_record(record_id: str) -> str:
    """Delete a record from the database."""
    return f"deleted {record_id}"


@tool
def admin_tool(action: str) -> str:
    """Perform admin action."""
    return f"admin: {action}"


def _multi_step_provider(*responses: ModelResponse) -> AsyncMock:
    provider = AsyncMock()
    provider.complete = AsyncMock(side_effect=list(responses))
    return provider


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------


class TestToolGateInit:
    def test_tool_gate_stored(self) -> None:
        """tool_gate dict is stored on the agent."""
        agent = Agent(
            name="bot",
            tools=[search],
            tool_gate={"search": [write_record, delete_record]},
        )
        assert "search" in agent._tool_gate
        assert len(agent._tool_gate["search"]) == 2

    def test_tool_gate_none_by_default(self) -> None:
        """No tool_gate means empty dict."""
        agent = Agent(name="bot", tools=[search])
        assert agent._tool_gate == {}

    def test_tool_gate_rejects_unknown_trigger(self) -> None:
        """Trigger tool must be registered."""
        with pytest.raises(AgentError, match="tool_gate trigger 'nonexistent'"):
            Agent(
                name="bot",
                tools=[search],
                tool_gate={"nonexistent": [write_record]},
            )

    def test_gated_tools_not_initially_registered(self) -> None:
        """Gated tools are NOT in agent.tools until the trigger fires."""
        agent = Agent(
            name="bot",
            tools=[search],
            tool_gate={"search": [write_record, delete_record]},
        )
        assert "write_record" not in agent.tools
        assert "delete_record" not in agent.tools
        assert "search" in agent.tools


# ---------------------------------------------------------------------------
# Runtime unlock via tool execution
# ---------------------------------------------------------------------------


class TestToolGateUnlock:
    async def test_basic_gate_unlock(self) -> None:
        """Calling the trigger tool injects gated tools on the next step."""
        # Step 1: LLM calls search → triggers gate → gated tools injected
        # Step 2: LLM returns text (gated tools should now be in schemas)
        tc = ToolCall(id="tc-1", name="search", arguments='{"query": "test"}')
        provider = _multi_step_provider(
            ModelResponse(content="", tool_calls=[tc], usage=_USAGE),
            ModelResponse(content="Search complete.", usage=_USAGE),
        )
        agent = Agent(
            name="bot",
            tools=[search],
            tool_gate={"search": [write_record, delete_record]},
        )

        # Before run, gated tools are not present
        assert "write_record" not in agent.tools

        output = await agent.run("find something", provider=provider)

        # After run, gated tools were injected
        assert "write_record" in agent.tools
        assert "delete_record" in agent.tools
        assert output.text == "Search complete."

    async def test_gated_tools_appear_in_schemas_after_unlock(self) -> None:
        """After unlock, get_tool_schemas() includes the gated tools."""
        tc = ToolCall(id="tc-1", name="search", arguments='{"query": "test"}')
        captured_tools: list[list[dict[str, Any]]] = []

        async def capture_complete(messages: Any, **kwargs: Any) -> ModelResponse:
            tools = kwargs.get("tools") or []
            captured_tools.append(tools)
            if len(captured_tools) == 1:
                return ModelResponse(content="", tool_calls=[tc], usage=_USAGE)
            return ModelResponse(content="done", usage=_USAGE)

        provider = AsyncMock()
        provider.complete = capture_complete

        agent = Agent(
            name="bot",
            tools=[search],
            tool_gate={"search": [write_record]},
        )
        await agent.run("go", provider=provider)

        # First LLM call: only search visible
        first_tool_names = {t["function"]["name"] for t in captured_tools[0]}
        assert "search" in first_tool_names
        assert "write_record" not in first_tool_names

        # Second LLM call: write_record now visible (appended)
        second_tool_names = {t["function"]["name"] for t in captured_tools[1]}
        assert "search" in second_tool_names
        assert "write_record" in second_tool_names

    async def test_appended_tools_preserve_order(self) -> None:
        """Gated tools are appended after existing tools (KV-cache safe)."""
        tc = ToolCall(id="tc-1", name="search", arguments='{"query": "x"}')
        captured_tools: list[list[dict[str, Any]]] = []

        async def capture_complete(messages: Any, **kwargs: Any) -> ModelResponse:
            tools = kwargs.get("tools") or []
            captured_tools.append(tools)
            if len(captured_tools) == 1:
                return ModelResponse(content="", tool_calls=[tc], usage=_USAGE)
            return ModelResponse(content="done", usage=_USAGE)

        provider = AsyncMock()
        provider.complete = capture_complete

        agent = Agent(
            name="bot",
            tools=[search],
            tool_gate={"search": [write_record, delete_record]},
        )
        await agent.run("go", provider=provider)

        # First call tools
        first_names = [t["function"]["name"] for t in captured_tools[0]]
        # Second call: original tools still in same position, new ones appended
        second_names = [t["function"]["name"] for t in captured_tools[1]]

        # The original tool(s) must be a prefix of the new list
        for i, name in enumerate(first_names):
            assert second_names[i] == name, (
                f"Tool order changed at index {i}: was {name}, now {second_names[i]}"
            )
        # New tools appended at the end
        appended = second_names[len(first_names):]
        assert "write_record" in appended
        assert "delete_record" in appended

    async def test_idempotent_unlock(self) -> None:
        """Calling trigger tool twice does not duplicate gated tools."""
        tc = ToolCall(id="tc-1", name="search", arguments='{"query": "first"}')
        tc2 = ToolCall(id="tc-2", name="search", arguments='{"query": "second"}')
        provider = _multi_step_provider(
            ModelResponse(content="", tool_calls=[tc], usage=_USAGE),
            ModelResponse(content="", tool_calls=[tc2], usage=_USAGE),
            ModelResponse(content="done", usage=_USAGE),
        )
        agent = Agent(
            name="bot",
            tools=[search],
            tool_gate={"search": [write_record]},
        )

        output = await agent.run("go", provider=provider)

        assert output.text == "done"
        # write_record registered exactly once
        assert "write_record" in agent.tools
        # No duplicate — tool count should be reasonable
        write_count = sum(1 for name in agent.tools if name == "write_record")
        assert write_count == 1

    async def test_gated_tool_executable_after_unlock(self) -> None:
        """After unlock, the gated tool can be called and executes normally."""
        tc_search = ToolCall(id="tc-1", name="search", arguments='{"query": "test"}')
        tc_write = ToolCall(id="tc-2", name="write_record", arguments='{"data": "hello"}')
        provider = _multi_step_provider(
            # Step 1: call search → unlocks write_record
            ModelResponse(content="", tool_calls=[tc_search], usage=_USAGE),
            # Step 2: call write_record (now available)
            ModelResponse(content="", tool_calls=[tc_write], usage=_USAGE),
            # Step 3: final text
            ModelResponse(content="All done!", usage=_USAGE),
        )
        agent = Agent(
            name="bot",
            tools=[search],
            tool_gate={"search": [write_record]},
        )

        output = await agent.run("search and write", provider=provider)
        assert output.text == "All done!"

        # Verify write_record's result was sent back to the LLM
        third_call_msgs = provider.complete.call_args_list[2][0][0]
        tool_result_msgs = [m for m in third_call_msgs if m.role == "tool"]
        assert any("wrote hello" in str(m.content) for m in tool_result_msgs)

    async def test_multiple_independent_gates(self) -> None:
        """Different trigger tools unlock different sets of gated tools."""
        @tool
        def trigger_a(x: str) -> str:
            """Trigger A."""
            return "a"

        @tool
        def trigger_b(x: str) -> str:
            """Trigger B."""
            return "b"

        @tool
        def gated_a(x: str) -> str:
            """Gated behind A."""
            return "ga"

        @tool
        def gated_b(x: str) -> str:
            """Gated behind B."""
            return "gb"

        tc_a = ToolCall(id="tc-1", name="trigger_a", arguments='{"x": "go"}')
        provider = _multi_step_provider(
            ModelResponse(content="", tool_calls=[tc_a], usage=_USAGE),
            ModelResponse(content="done", usage=_USAGE),
        )
        agent = Agent(
            name="bot",
            tools=[trigger_a, trigger_b],
            tool_gate={
                "trigger_a": [gated_a],
                "trigger_b": [gated_b],
            },
        )

        await agent.run("go", provider=provider)

        # Only gate A was triggered
        assert "gated_a" in agent.tools
        assert "gated_b" not in agent.tools

    async def test_unknown_tool_call_before_unlock(self) -> None:
        """If LLM calls a gated tool before unlock, it gets an error result."""
        tc_write = ToolCall(id="tc-1", name="write_record", arguments='{"data": "x"}')
        provider = _multi_step_provider(
            # LLM tries to call write_record before it's unlocked
            ModelResponse(content="", tool_calls=[tc_write], usage=_USAGE),
            ModelResponse(content="ok", usage=_USAGE),
        )
        agent = Agent(
            name="bot",
            tools=[search],
            tool_gate={"search": [write_record]},
        )

        output = await agent.run("go", provider=provider)
        assert output.text == "ok"

        # The tool result should be an error (unknown tool)
        second_call_msgs = provider.complete.call_args_list[1][0][0]
        tool_result_msgs = [m for m in second_call_msgs if m.role == "tool"]
        assert len(tool_result_msgs) == 1
        assert "error" in str(tool_result_msgs[0].content).lower() or tool_result_msgs[0].error
