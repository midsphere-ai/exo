"""Integration tests for MCP progress events in the stream.

US-INT-016: Verifies that MCPProgressEvent items appear in the stream during
MCP tool execution and precede the corresponding ToolResultEvent.

The long_running_task tool added to mcp_test_server.py emits 'steps' progress
notifications before returning its result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HELPERS_DIR = Path(__file__).parent / "helpers"
_MCP_SERVER_SCRIPT = str(_HELPERS_DIR / "mcp_test_server.py")


# ---------------------------------------------------------------------------
# test_mcp_progress_events_fire_before_tool_result
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_mcp_progress_events_fire_before_tool_result(vertex_model: str) -> None:
    """MCPProgressEvent items appear in the stream before the corresponding ToolResultEvent.

    Flow:
    1. Agent loads the stdio MCP test server with the long_running_task tool.
    2. Constrained prompt forces long_running_task(steps=3).
    3. During tool execution, 3 progress notifications are emitted via FastMCP Context.
    4. run.stream() drains the progress queue and yields MCPProgressEvent items
       before yielding the ToolResultEvent.
    5. Assertions:
       - At least 1 MCPProgressEvent present in the stream.
       - All MCPProgressEvent items appear before the first ToolResultEvent.
    """
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.mcp import MCPServerConfig  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]
    from exo.runner import run  # pyright: ignore[reportMissingImports]
    from exo.types import (  # pyright: ignore[reportMissingImports]
        MCPProgressEvent,
        ToolResultEvent,
    )

    mcp_config = MCPServerConfig(
        name="test-server",
        transport="stdio",
        command=sys.executable,
        args=[_MCP_SERVER_SCRIPT],
    )

    provider = get_provider(vertex_model)
    agent = Agent(
        name="progress-stream-agent",
        model=vertex_model,
        instructions=(
            "You are a task runner assistant. "
            "When asked to run a long_running_task, call it immediately with the "
            "specified number of steps. Do not answer without calling the tool."
        ),
        memory=None,
        context=None,
        max_steps=3,
    )

    await agent.add_mcp_server(mcp_config)

    # Confirm the long_running_task tool is registered (namespaced)
    tool_names = list(agent.tools.keys())
    assert any("long_running_task" in name for name in tool_names), (
        f"long_running_task tool not found in agent tools: {tool_names}"
    )

    events = []
    async for event in run.stream(  # type: ignore[attr-defined]
        agent,
        "You MUST call the long_running_task tool with steps=3. "
        "Call the tool immediately — do not answer without calling it first.",
        provider=provider,
        detailed=True,
    ):
        events.append(event)

    progress_events = [e for e in events if isinstance(e, MCPProgressEvent)]
    tool_result_events = [e for e in events if isinstance(e, ToolResultEvent)]

    event_type_names = [type(e).__name__ for e in events]

    # Assert at least 1 MCPProgressEvent is present
    assert len(progress_events) >= 1, (
        f"Expected at least 1 MCPProgressEvent in stream, got {len(progress_events)}. "
        f"All event types: {event_type_names}"
    )

    # Assert at least 1 ToolResultEvent is present (requires detailed=True)
    assert len(tool_result_events) >= 1, (
        f"Expected at least 1 ToolResultEvent in stream, got {len(tool_result_events)}. "
        f"All event types: {event_type_names}"
    )

    # Assert all MCPProgressEvent items appear before the first ToolResultEvent
    first_tool_result_idx = next(
        i for i, e in enumerate(events) if isinstance(e, ToolResultEvent)
    )
    progress_indices = [i for i, e in enumerate(events) if isinstance(e, MCPProgressEvent)]

    for idx in progress_indices:
        assert idx < first_tool_result_idx, (
            f"MCPProgressEvent at stream index {idx} appeared at or after "
            f"first ToolResultEvent at index {first_tool_result_idx}. "
            f"All event types: {event_type_names}"
        )
