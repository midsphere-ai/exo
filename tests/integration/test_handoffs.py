"""Integration tests for agent handoffs.

US-INT-011: Verifies that Agent A correctly delegates to Agent B via a
handoff and the final output reflects Agent B's work.

The handoff mechanism in Exo's Swarm is output-based: when Agent A's
output exactly matches a registered handoff target's name, the Swarm
transfers control to that agent with the full conversation history.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel


class HandoffResponse(BaseModel):
    handled_by: str
    content: str


# ---------------------------------------------------------------------------
# test_handoff_from_agent_a_to_agent_b
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_handoff_from_agent_a_to_agent_b(vertex_model: str) -> None:
    """Agent A delegates a poetry request to Agent B (poet) via handoff.

    Agent B is a poet that writes haikus.  Agent A is a router that
    responds with exactly 'poet' to trigger the handoff.  We assert:
    - result.output is non-empty and roughly 3 lines (haiku format)
    - An AssistantMessage with content 'poet' appears in messages,
      indicating delegation occurred.
    """
    from exo import Swarm  # pyright: ignore[reportMissingImports]
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]
    from exo.types import AssistantMessage  # pyright: ignore[reportMissingImports]

    provider = get_provider(vertex_model)

    poet_agent = Agent(
        name="poet",
        model=vertex_model,
        instructions=(
            "You are a poet. When asked anything, respond with a haiku "
            "(three lines: 5-7-5 syllables). No other text, just the haiku."
        ),
        max_steps=2,
        memory=None,
        context=None,
    )

    router_agent = Agent(
        name="router",
        model=vertex_model,
        instructions=(
            "You are a routing agent. For any poetry request, your entire "
            "response must be exactly the single word: poet\n"
            "Do not add punctuation, explanation, or extra text. "
            "Respond with ONLY: poet"
        ),
        handoffs=[poet_agent],
        max_steps=2,
        memory=None,
        context=None,
    )

    swarm = Swarm(agents=[router_agent, poet_agent], mode="handoff")

    result = await swarm.run(
        "Write me a haiku about mountains.",
        provider=provider,
    )

    # Assert result is non-empty and has roughly 3 lines (haiku format)
    assert result.output.strip(), (
        f"Expected non-empty output from poet agent, got: {result.output!r}"
    )
    lines = [line for line in result.output.strip().split("\n") if line.strip()]
    assert len(lines) >= 2, (
        f"Expected roughly 3 lines (haiku), got {len(lines)} line(s): {result.output!r}"
    )

    # Assert handoff occurred: an AssistantMessage with content 'poet' in messages
    # indicates the router delegated to the poet agent
    handoff_signals = [
        m
        for m in result.messages
        if isinstance(m, AssistantMessage) and m.content.strip() == "poet"
    ]
    assert handoff_signals, (
        "Expected a handoff signal (AssistantMessage content='poet') in messages "
        "indicating delegation to the poet agent. "
        f"Messages: {[(type(m).__name__, getattr(m, 'content', '?')[:60]) for m in result.messages]}"
    )


# ---------------------------------------------------------------------------
# test_handoff_result_is_from_target_agent
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_handoff_result_is_from_target_agent(vertex_model: str) -> None:
    """Handoff result's structured output identifies the target agent.

    Agent B (specialist) returns a JSON HandoffResponse with handled_by
    set to its own name.  Agent A (coordinator) routes all requests to it
    by outputting 'specialist'.  We parse the final output as
    HandoffResponse and assert handled_by contains the target agent's name.
    """
    from exo import Swarm  # pyright: ignore[reportMissingImports]
    from exo._internal.output_parser import (  # pyright: ignore[reportMissingImports]
        parse_structured_output,
    )
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]

    provider = get_provider(vertex_model)

    specialist_agent = Agent(
        name="specialist",
        model=vertex_model,
        instructions=(
            "You are a specialist agent. When handling any request, reply ONLY with "
            "a valid JSON object matching this schema exactly:\n"
            '{"handled_by": "specialist", "content": "<your answer here>"}\n'
            "No other text outside the JSON object. "
            "Look at the conversation history to understand the original question."
        ),
        max_steps=2,
        memory=None,
        context=None,
    )

    coordinator_agent = Agent(
        name="coordinator",
        model=vertex_model,
        instructions=(
            "You are a coordinator. Route all requests to the specialist agent. "
            "Your entire response must be exactly the single word: specialist\n"
            "Do not add punctuation, explanation, or extra text. "
            "Respond with ONLY: specialist"
        ),
        handoffs=[specialist_agent],
        max_steps=2,
        memory=None,
        context=None,
    )

    swarm = Swarm(agents=[coordinator_agent, specialist_agent], mode="handoff")

    result = await swarm.run(
        "Answer this question: What is the capital of France?",
        provider=provider,
    )

    # Parse the structured output from the specialist agent
    response = parse_structured_output(result.output, HandoffResponse)
    assert isinstance(response, HandoffResponse), (
        f"Expected HandoffResponse, got {type(response)}: {result.output!r}"
    )
    assert "specialist" in response.handled_by, (
        f"Expected handled_by to contain 'specialist', got: {response.handled_by!r}"
    )
