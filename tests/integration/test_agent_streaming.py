"""Integration tests for agent streaming event ordering.

Tests that:
- agent.stream() emits events in the correct order with no missing
  or duplicate event types.
- ToolCallEvent appears before ToolResultEvent in the event stream.
- UsageEvent is the last event when filtered to core event types.
- StepEvent is emitted for each step when detailed=True.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_stream_event_ordering_single_step(vertex_model: str) -> None:
    """Agent with no tools emits TextEvent(s) followed by UsageEvent, with no ErrorEvent.

    Uses event_types={"text", "usage"} so that StepEvent/StatusEvent are
    filtered out, making UsageEvent the final event in the sequence.
    """
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]
    from exo.runner import run  # pyright: ignore[reportMissingImports]
    from exo.types import (  # pyright: ignore[reportMissingImports]
        ErrorEvent,
        TextEvent,
        UsageEvent,
    )

    provider = get_provider(vertex_model)
    agent = Agent(
        name="stream-single-step",
        model=vertex_model,
        instructions="You are a helpful assistant. Be very concise.",
        max_steps=1,
    )

    events = []
    async for event in run.stream(  # type: ignore[attr-defined]
        agent,
        "What is 2+2? Reply with just the number.",
        provider=provider,
        detailed=True,
        event_types={"text", "usage"},
    ):
        events.append(event)

    text_events = [e for e in events if isinstance(e, TextEvent)]
    error_events = [e for e in events if isinstance(e, ErrorEvent)]

    assert text_events, "Expected at least one TextEvent"
    assert not error_events, f"Unexpected ErrorEvent(s): {error_events}"
    assert events, "No events received from stream"
    assert isinstance(events[-1], UsageEvent), (
        f"Expected last event to be UsageEvent, got {type(events[-1]).__name__}"
    )


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_stream_event_ordering_with_tool_call(vertex_model: str) -> None:
    """ToolCallEvent precedes ToolResultEvent; TextEvent follows; UsageEvent is last.

    Uses event_types={"text", "tool_call", "tool_result", "usage"} to filter
    out StepEvent/StatusEvent, leaving UsageEvent as the final event.
    """
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]
    from exo.runner import run  # pyright: ignore[reportMissingImports]
    from exo.tool import tool  # pyright: ignore[reportMissingImports]
    from exo.types import (  # pyright: ignore[reportMissingImports]
        TextEvent,
        ToolCallEvent,
        ToolResultEvent,
        UsageEvent,
    )

    @tool
    def get_greeting(name: str) -> str:
        """Return a greeting for a person.

        Args:
            name: The person's name.
        """
        return f"Hello, {name}!"

    provider = get_provider(vertex_model)
    agent = Agent(
        name="stream-tool-call",
        model=vertex_model,
        instructions="You are a helpful assistant. Always use your tools.",
        tools=[get_greeting],
        max_steps=3,
    )

    events = []
    async for event in run.stream(  # type: ignore[attr-defined]
        agent,
        "You MUST call the get_greeting tool with name='World'. Do not answer without calling the tool first.",
        provider=provider,
        detailed=True,
        event_types={"text", "tool_call", "tool_result", "usage"},
    ):
        events.append(event)

    # Find indices of key event types
    tool_call_idx = next(
        (i for i, e in enumerate(events) if isinstance(e, ToolCallEvent)), None
    )
    tool_result_idx = next(
        (i for i, e in enumerate(events) if isinstance(e, ToolResultEvent)), None
    )
    last_tool_result_idx = max(
        (i for i, e in enumerate(events) if isinstance(e, ToolResultEvent)),
        default=None,
    )
    text_after_result_idx = next(
        (
            i
            for i, e in enumerate(events)
            if isinstance(e, TextEvent)
            and (last_tool_result_idx is None or i > last_tool_result_idx)
        ),
        None,
    )

    assert tool_call_idx is not None, "No ToolCallEvent found in stream"
    assert tool_result_idx is not None, "No ToolResultEvent found in stream"
    assert tool_call_idx < tool_result_idx, (
        f"ToolCallEvent (idx={tool_call_idx}) did not precede "
        f"ToolResultEvent (idx={tool_result_idx})"
    )
    assert text_after_result_idx is not None, "No TextEvent found after ToolResultEvent"
    assert isinstance(events[-1], UsageEvent), (
        f"Expected last event to be UsageEvent, got {type(events[-1]).__name__}"
    )


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_stream_step_events_wrap_each_step(vertex_model: str) -> None:
    """Agent making a tool call emits at least 2 StepEvents (tool step + final step).

    Uses detailed=True so StepEvent is emitted for each LLM round-trip.
    Uses two chained tools where the output of the first is needed to call
    the second, forcing sequential steps:
      Step 1: LLM calls get_country_code("France") → "FR"
      Step 2: LLM calls get_capital_for_code("FR") → "Paris"
      Step 3: LLM produces final text response
    This yields at least 3 steps → >= 6 StepEvents, well above the >= 2 target.
    """
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]
    from exo.runner import run  # pyright: ignore[reportMissingImports]
    from exo.tool import tool  # pyright: ignore[reportMissingImports]
    from exo.types import StepEvent  # pyright: ignore[reportMissingImports]

    @tool
    def get_country_code(country: str) -> str:
        """Return the ISO 2-letter country code for a country name.

        Args:
            country: The full country name.
        """
        codes = {"France": "FR", "Germany": "DE", "Japan": "JP"}
        return codes.get(country, "UNKNOWN")

    @tool
    def get_capital_for_code(code: str) -> str:
        """Return the capital city for a 2-letter ISO country code.

        Args:
            code: The ISO 2-letter country code (e.g. 'FR').
        """
        capitals = {"FR": "Paris", "DE": "Berlin", "JP": "Tokyo"}
        return capitals.get(code, f"Unknown capital for code {code}")

    provider = get_provider(vertex_model)
    agent = Agent(
        name="stream-two-steps",
        model=vertex_model,
        instructions=(
            "You are a geography assistant. "
            "To find a capital: FIRST call get_country_code to get the code, "
            "THEN call get_capital_for_code with that code."
        ),
        tools=[get_country_code, get_capital_for_code],
        max_steps=5,
    )

    events = []
    async for event in run.stream(  # type: ignore[attr-defined]
        agent,
        (
            "What is the capital of France? "
            "You MUST use the tools: first get_country_code, then get_capital_for_code."
        ),
        provider=provider,
        detailed=True,
    ):
        events.append(event)

    step_events = [e for e in events if isinstance(e, StepEvent)]
    assert len(step_events) >= 2, (
        f"Expected >= 2 StepEvents, got {len(step_events)}: {step_events}"
    )
