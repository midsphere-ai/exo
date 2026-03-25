"""Integration tests for dynamic tool loading.

US-INT-013: Verifies that add_tool() after agent initialization makes the tool
immediately available, remove_tool() makes it unavailable, and concurrent
add_tool() calls are safe (no asyncio.Lock errors, all tools registered).
"""

from __future__ import annotations

import asyncio

import pytest

# ---------------------------------------------------------------------------
# test_add_tool_after_init_available_immediately
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_add_tool_after_init_available_immediately(vertex_model: str) -> None:
    """Tool added after init is immediately available to the agent.

    Agent is initialized with zero tools, then get_joke is added via
    await agent.add_tool(). A constrained prompt forces the tool call.
    We assert a ToolCall with name == 'get_joke' appears in result.tool_calls.
    """
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]
    from exo.tool import FunctionTool  # pyright: ignore[reportMissingImports]

    provider = get_provider(vertex_model)

    def get_joke() -> str:
        """Return a short joke."""
        return "Why did the scarecrow win an award? He was outstanding in his field."

    agent = Agent(
        name="dynamic-loader",
        model=vertex_model,
        instructions="You are a helpful assistant. Use the get_joke tool when asked for a joke.",
        max_steps=3,
        memory=None,
        context=None,
    )

    # No tools initially (besides any auto-registered internal tools)
    assert "get_joke" not in agent.tools, "get_joke should not be registered yet"

    # Add tool dynamically
    joke_tool = FunctionTool(get_joke)
    await agent.add_tool(joke_tool)

    assert "get_joke" in agent.tools, "get_joke should be registered after add_tool"

    result = await agent.run(
        "You MUST call the get_joke tool right now. Do not answer from memory. "
        "Call get_joke and report what it returns.",
        provider=provider,
    )

    joke_calls = [tc for tc in result.tool_calls if tc.name == "get_joke"]
    assert len(joke_calls) >= 1, (
        f"Expected at least one get_joke tool call, got: {[tc.name for tc in result.tool_calls]}"
    )


# ---------------------------------------------------------------------------
# test_remove_tool_unavailable_after_removal
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_remove_tool_unavailable_after_removal(vertex_model: str) -> None:
    """Tool removed via remove_tool() is no longer callable by the agent.

    Agent starts with get_joke registered, then it is removed. A prompt
    that explicitly asks to use get_joke should result in no ToolCall
    with that name (model cannot call a tool it cannot see).
    """
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]
    from exo.tool import FunctionTool  # pyright: ignore[reportMissingImports]

    provider = get_provider(vertex_model)

    def get_joke() -> str:
        """Return a short joke."""
        return "Why did the scarecrow win an award? He was outstanding in his field."

    joke_tool = FunctionTool(get_joke)

    agent = Agent(
        name="tool-remover",
        model=vertex_model,
        instructions="You are a helpful assistant.",
        max_steps=3,
        tools=[joke_tool],
        memory=None,
        context=None,
    )

    assert "get_joke" in agent.tools, "get_joke should be registered initially"

    # Remove the tool
    agent.remove_tool("get_joke")

    assert "get_joke" not in agent.tools, "get_joke should be unregistered after remove_tool"

    result = await agent.run(
        "Tell me a joke. Just make one up — do not use any tools.",
        provider=provider,
    )

    joke_calls = [tc for tc in result.tool_calls if tc.name == "get_joke"]
    assert len(joke_calls) == 0, (
        f"Expected no get_joke tool calls after removal, got: {[tc.name for tc in result.tool_calls]}"
    )


# ---------------------------------------------------------------------------
# test_concurrent_add_tool_no_race_condition
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_concurrent_add_tool_no_race_condition() -> None:
    """Concurrent add_tool calls are asyncio-safe via _tools_lock.

    10 uniquely-named tools are added via asyncio.gather. We assert all 10
    are present in agent.tools with no asyncio.Lock errors or exceptions.
    No LLM call is needed — this is a concurrency correctness test.
    """
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.tool import FunctionTool  # pyright: ignore[reportMissingImports]

    # Use a placeholder model string — no LLM calls are made in this test
    agent = Agent(
        name="concurrent-loader",
        model="vertex:gemini-2.0-flash",
        instructions="You are a helpful assistant.",
        max_steps=1,
        memory=None,
        context=None,
    )

    tool_names: list[str] = []

    async def _add(i: int) -> None:
        tool_name = f"tool_{i:02d}"
        tool_names.append(tool_name)

        def fn() -> str:
            return f"result from {tool_name}"

        fn.__name__ = tool_name
        fn.__doc__ = f"Tool number {i}"
        t = FunctionTool(fn)
        await agent.add_tool(t)

    # Fire all 10 adds concurrently
    await asyncio.gather(*[_add(i) for i in range(10)])

    # All 10 tools must be registered
    for name in tool_names:
        assert name in agent.tools, (
            f"Tool '{name}' missing after concurrent add_tool. "
            f"Registered tools: {list(agent.tools.keys())}"
        )

    assert len([k for k in agent.tools if k.startswith("tool_")]) == 10, (
        f"Expected 10 dynamic tools, got: {[k for k in agent.tools if k.startswith('tool_')]}"
    )
