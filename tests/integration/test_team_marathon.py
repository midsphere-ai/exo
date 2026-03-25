"""Integration marathon test: multi-agent team with streaming and shared memory.

US-INT-029: Verifies a 3-agent team (coordinator + worker_a + worker_b) streams
events, delegates tasks to the correct specialists via delegation tool calls, and
produces a structured TeamReport with all 4 fields populated.

All 3 agents share a single SQLiteMemoryStore. Streaming with detailed=True
emits TextEvent, ToolCallEvent, and UsageEvent. At least 2 delegation tool
calls (delegate_to_worker_a, delegate_to_worker_b) appear in the event stream.

Memory item pattern in team mode:
  - Each agent persists HumanMemory (input) + AIMemory (output) to shared store
  - Items tagged with agent_id=agent.name for isolation querying
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel


class TeamReport(BaseModel):
    coordinator_summary: str
    worker_a_contribution: str
    worker_b_contribution: str
    total_steps: int


@pytest.mark.integration
@pytest.mark.marathon
@pytest.mark.timeout(180)
async def test_team_produces_aggregated_structured_output(
    vertex_model: str, tmp_sqlite_db: str
) -> None:
    """Marathon: 3-agent team streams events and produces structured TeamReport.

    Coordinator routes:
    - Geography task ("capital of Japan") → worker_a (geography specialist)
    - Math task ("17 x 23") → worker_b (mathematics specialist)

    After both workers reply, coordinator aggregates results into TeamReport JSON.

    We assert:
    - At least 2 delegation tool calls (delegate_to_worker_a, delegate_to_worker_b)
      appear in the event stream as ToolCallEvent.
    - worker_a_contribution contains 'tokyo' (capital of Japan = Tokyo).
    - worker_b_contribution contains '391' (17 x 23 = 391).
    - All 3 agents persisted messages to the shared SQLiteMemoryStore,
      identifiable by agent_id metadata.
    - TextEvent, ToolCallEvent, and UsageEvent all appear in streamed events.
    """
    from exo import Swarm  # pyright: ignore[reportMissingImports]
    from exo._internal.output_parser import (  # pyright: ignore[reportMissingImports]
        parse_structured_output,
    )
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.memory.backends.sqlite import (  # pyright: ignore[reportMissingImports]
        SQLiteMemoryStore,
    )
    from exo.memory.base import MemoryMetadata  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]
    from exo.types import (  # pyright: ignore[reportMissingImports]
        TextEvent,
        ToolCallEvent,
    )

    provider = get_provider(vertex_model)
    shared_store = SQLiteMemoryStore(db_path=tmp_sqlite_db)
    await shared_store.init()

    try:
        # Worker A: geography specialist
        worker_a = Agent(
            name="worker_a",
            model=vertex_model,
            instructions=(
                "You are a geography specialist. When given a geography task, "
                "answer with ONLY the requested information (e.g., just the city name). "
                "Do not add explanation, punctuation, or extra text."
            ),
            max_steps=2,
            memory=shared_store,
            context=None,
        )

        # Worker B: mathematics specialist
        worker_b = Agent(
            name="worker_b",
            model=vertex_model,
            instructions=(
                "You are a mathematics specialist. When given a calculation task, "
                "compute the exact answer and respond with ONLY the numeric result. "
                "Do not add explanation or extra text."
            ),
            max_steps=2,
            memory=shared_store,
            context=None,
        )

        # Coordinator: lead agent that delegates and aggregates
        coordinator = Agent(
            name="coordinator",
            model=vertex_model,
            instructions=(
                "You are a coordinator agent. Route tasks to specialist workers "
                "and aggregate their results into a structured report.\n\n"
                "You MUST call delegation tools ONE AT A TIME (never in parallel):\n"
                "Step 1: Call delegate_to_worker_a with task="
                "'What is the capital of Japan? Reply with ONLY the city name.'\n"
                "Step 2: Call delegate_to_worker_b with task="
                "'Calculate 17 x 23 and reply with ONLY the numeric result.'\n"
                "Step 3: After receiving BOTH worker results, reply ONLY with "
                "this JSON object (no markdown, no extra text):\n"
                '{"coordinator_summary": "Routed geography to worker_a and math to worker_b.", '
                '"worker_a_contribution": "<exact answer from worker_a>", '
                '"worker_b_contribution": "<exact answer from worker_b>", '
                '"total_steps": 2}'
            ),
            max_steps=6,
            memory=shared_store,
            context=None,
        )

        swarm = Swarm(agents=[coordinator, worker_a, worker_b], mode="team")

        # Stream all events from the swarm (detailed=True enables UsageEvent)
        events = []
        async for event in swarm.stream(
            "Find the capital of Japan AND calculate 17 x 23. "
            "Follow your instructions exactly: call delegate_to_worker_a first "
            "(geography task), then delegate_to_worker_b (math task), "
            "then output the TeamReport JSON.",
            provider=provider,
            detailed=True,
        ):
            events.append(event)

    finally:
        await shared_store.close()

    # -------------------------------------------------------------------------
    # Assert at least 2 delegation (handoff-related) tool calls in stream
    # -------------------------------------------------------------------------
    delegation_calls = [
        e
        for e in events
        if isinstance(e, ToolCallEvent) and e.tool_name.startswith("delegate_to_")
    ]
    assert len(delegation_calls) >= 2, (
        f"Expected >= 2 delegation tool calls (delegate_to_worker_a, delegate_to_worker_b), "
        f"got {len(delegation_calls)}: "
        f"{[e.tool_name for e in delegation_calls]}"
    )

    # -------------------------------------------------------------------------
    # Parse structured output from coordinator's accumulated text events
    # -------------------------------------------------------------------------
    coordinator_text = "".join(
        event.text
        for event in events
        if isinstance(event, TextEvent) and event.agent_name == "coordinator"
    )
    assert coordinator_text, (
        "No text events from coordinator agent. TextEvent agent names found: "
        f"{[e.agent_name for e in events if isinstance(e, TextEvent)]}"
    )

    report = parse_structured_output(coordinator_text, TeamReport)
    assert isinstance(report, TeamReport), (
        f"Expected TeamReport instance, got {type(report)}: {coordinator_text!r}"
    )

    # -------------------------------------------------------------------------
    # Assert worker contributions contain expected values
    # -------------------------------------------------------------------------
    assert "tokyo" in report.worker_a_contribution.lower(), (
        f"Expected 'tokyo' in worker_a_contribution (capital of Japan), "
        f"got: {report.worker_a_contribution!r}"
    )
    assert "391" in report.worker_b_contribution, (
        f"Expected '391' in worker_b_contribution (17 x 23 = 391), "
        f"got: {report.worker_b_contribution!r}"
    )

    # -------------------------------------------------------------------------
    # Assert all 3 agents persisted messages to the shared SQLite memory store
    # -------------------------------------------------------------------------
    verify_store = SQLiteMemoryStore(db_path=tmp_sqlite_db)
    await verify_store.init()
    try:
        items_coordinator = await verify_store.search(
            metadata=MemoryMetadata(agent_id="coordinator"),
            limit=50,
        )
        items_worker_a = await verify_store.search(
            metadata=MemoryMetadata(agent_id="worker_a"),
            limit=50,
        )
        items_worker_b = await verify_store.search(
            metadata=MemoryMetadata(agent_id="worker_b"),
            limit=50,
        )
    finally:
        await verify_store.close()

    assert items_coordinator, (
        "Expected messages from coordinator in shared memory store, found none. "
        "coordinator agent should have persisted at least HumanMemory + AIMemory."
    )
    assert items_worker_a, (
        "Expected messages from worker_a in shared memory store, found none. "
        "worker_a agent should have persisted at least HumanMemory + AIMemory "
        "when handling the delegation task."
    )
    assert items_worker_b, (
        "Expected messages from worker_b in shared memory store, found none. "
        "worker_b agent should have persisted at least HumanMemory + AIMemory "
        "when handling the delegation task."
    )

    # -------------------------------------------------------------------------
    # Assert required event types appear in streamed events
    # -------------------------------------------------------------------------
    event_types_found = {e.type for e in events}
    assert "text" in event_types_found, (
        f"No TextEvent in streamed events. Event types found: {event_types_found}"
    )
    assert "tool_call" in event_types_found, (
        f"No ToolCallEvent in streamed events. Event types found: {event_types_found}"
    )
    assert "usage" in event_types_found, (
        f"No UsageEvent in streamed events. Event types found: {event_types_found}"
    )
