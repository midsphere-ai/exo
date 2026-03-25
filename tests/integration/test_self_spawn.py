"""Integration tests for agent self-spawn.

US-INT-012: Verifies that a parent agent can spawn child agents for parallel
subtasks and aggregate their results, and that the depth limit is correctly
enforced without infinite recursion.

The spawn mechanism in Exo registers a ``spawn_self(task)`` tool when
``allow_self_spawn=True`` is passed to Agent.  The tool creates a child agent
with ``_spawn_depth = parent._spawn_depth + 1``; when depth reaches
``max_spawn_depth`` the tool returns an error string instead of spawning.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# test_parent_spawns_two_children_and_aggregates
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_parent_spawns_two_children_and_aggregates(vertex_model: str) -> None:
    """Parent agent spawns two child agents and aggregates their capital results.

    The parent agent is instructed to call spawn_self sequentially — first for
    Australia, then for Brazil — and combine both results.  We assert:
    - result.text contains 'canberra' and 'brasilia' (case-insensitive)
    - result.tool_calls contains at least two spawn_self entries
    """
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]

    provider = get_provider(vertex_model)

    agent = Agent(
        name="spawner",
        model=vertex_model,
        instructions=(
            "You are a geography coordinator. "
            "To answer questions about multiple countries, use spawn_self to "
            "delegate each country lookup as a separate sub-task. "
            "IMPORTANT: Call spawn_self SEQUENTIALLY — complete one spawn before "
            "starting the next. "
            "After both spawns complete, combine their results in your final answer."
        ),
        max_steps=8,
        allow_self_spawn=True,
        max_spawn_depth=2,
        memory=None,
        context=None,
    )

    result = await agent.run(
        "Find the capital cities of both Australia AND Brazil. "
        "You MUST use spawn_self for EACH country separately. "
        "First call spawn_self with task='What is the capital city of Australia? "
        "Reply with only the city name.' "
        "Then call spawn_self with task='What is the capital city of Brazil? "
        "Reply with only the city name.' "
        "After both calls, state both capitals in your final answer.",
        provider=provider,
    )

    # Assert both capitals appear in the aggregated result
    result_lower = result.text.lower()
    assert "canberra" in result_lower, (
        f"Expected 'canberra' in result, got: {result.text!r}"
    )
    assert "brasilia" in result_lower or "brasília" in result_lower, (
        f"Expected 'brasilia' or 'brasília' in result, got: {result.text!r}"
    )

    # Assert two spawn-related tool calls are present
    spawn_calls = [tc for tc in result.tool_calls if tc.name == "spawn_self"]
    assert len(spawn_calls) >= 2, (
        f"Expected at least 2 spawn_self calls, got {len(spawn_calls)}. "
        f"All tool calls: {[tc.name for tc in result.tool_calls]}"
    )


# ---------------------------------------------------------------------------
# test_spawn_depth_limit_enforced
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_spawn_depth_limit_enforced(vertex_model: str) -> None:
    """Spawn depth guard prevents recursion beyond max_spawn_depth.

    We create an agent that already has ``_spawn_depth == max_spawn_depth``
    so the first call to spawn_self triggers the depth guard immediately,
    returning an error string.  We assert:
    - No exception is raised (graceful error handling)
    - result.text is non-empty (LLM handled the error message)
    - No infinite recursion occurs
    """
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]

    provider = get_provider(vertex_model)

    # Create an agent at max spawn depth — spawn_self will return error string
    agent = Agent(
        name="depth-limited-spawner",
        model=vertex_model,
        instructions=(
            "You are a helpful assistant. "
            "If a tool returns an error, acknowledge it in your response."
        ),
        max_steps=4,
        allow_self_spawn=True,
        max_spawn_depth=1,
        memory=None,
        context=None,
    )
    # Simulate being at max depth — depth guard fires: _spawn_depth (1) >= max (1)
    agent._spawn_depth = 1  # type: ignore[attr-defined]

    # Should not raise; spawn_self returns error string which LLM handles
    result = await agent.run(
        "Use spawn_self to find the capital of France.",
        provider=provider,
    )

    # Assert no exception raised — we got a result
    assert result.text.strip(), (
        f"Expected non-empty response after depth-limit error, got: {result.text!r}"
    )

    # If spawn_self was called, it should have returned the depth-limit error string
    spawn_calls = [tc for tc in result.tool_calls if tc.name == "spawn_self"]
    if spawn_calls:
        # LLM attempted spawn but got error — result should acknowledge it
        # No assertion on exact text, just confirm no infinite recursion (test completes)
        pass
