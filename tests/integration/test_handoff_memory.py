"""Integration tests for handoff chain with memory persistence.

US-INT-017: Verifies that a three-agent handoff chain (A→B→C) persists
messages from all three agents to a shared SQLiteMemoryStore, identifiable
by agent_id metadata.

The handoff mechanism is output-based: each routing agent outputs exactly
the name of its target agent.  The terminal agent (C) answers the question
by looking at the accumulated conversation history.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# test_three_agent_chain_all_messages_persisted
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_three_agent_chain_all_messages_persisted(
    vertex_model: str,
    tmp_sqlite_db: str,
) -> None:
    """Three-agent handoff chain A→B→C persists messages from all 3 agents.

    Agent C answers factual questions; Agent B routes all requests to C
    by outputting 'agent_c'; Agent A routes all requests to B by outputting
    'agent_b'.  All three share a single SQLiteMemoryStore.

    After the full A→B→C handoff chain runs, the shared store should
    contain items from all three agents, identifiable by agent_id metadata.
    """
    from exo import Swarm  # pyright: ignore[reportMissingImports]
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.memory.backends.sqlite import (  # pyright: ignore[reportMissingImports]
        SQLiteMemoryStore,
    )
    from exo.memory.base import MemoryMetadata  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]

    provider = get_provider(vertex_model)

    # Shared SQLiteMemoryStore for all three agents
    shared_store = SQLiteMemoryStore(db_path=tmp_sqlite_db)
    await shared_store.init()

    try:
        # Agent C: the terminal answerer — looks at history for the question
        agent_c = Agent(
            name="agent_c",
            model=vertex_model,
            instructions=(
                "You are a factual assistant. Look at the full conversation "
                "history to find the original user question and answer it "
                "directly and briefly. Give only the answer, nothing else."
            ),
            max_steps=2,
            memory=shared_store,
            context=None,
        )

        # Agent B: routes to agent_c by outputting its exact name
        agent_b = Agent(
            name="agent_b",
            model=vertex_model,
            instructions=(
                "You are a routing agent. Your entire response must be exactly "
                "the single word: agent_c\n"
                "Do not add punctuation, explanation, or extra text. "
                "Respond with ONLY: agent_c"
            ),
            handoffs=[agent_c],
            max_steps=2,
            memory=shared_store,
            context=None,
        )

        # Agent A: routes to agent_b by outputting its exact name
        agent_a = Agent(
            name="agent_a",
            model=vertex_model,
            instructions=(
                "You are a routing agent. Your entire response must be exactly "
                "the single word: agent_b\n"
                "Do not add punctuation, explanation, or extra text. "
                "Respond with ONLY: agent_b"
            ),
            handoffs=[agent_b],
            max_steps=2,
            memory=shared_store,
            context=None,
        )

        swarm = Swarm(agents=[agent_a, agent_b, agent_c], mode="handoff")

        await swarm.run(
            "What is the capital of France?",
            provider=provider,
        )

        # Query the shared store for messages from each agent by agent_id
        items_a = await shared_store.search(
            metadata=MemoryMetadata(agent_id="agent_a"),
            limit=50,
        )
        items_b = await shared_store.search(
            metadata=MemoryMetadata(agent_id="agent_b"),
            limit=50,
        )
        items_c = await shared_store.search(
            metadata=MemoryMetadata(agent_id="agent_c"),
            limit=50,
        )

        assert items_a, (
            "Expected messages from agent_a in shared memory store, found none. "
            "agent_a should have persisted at least its HumanMemory (input) "
            "and AIMemory (routing output 'agent_b')."
        )
        assert items_b, (
            "Expected messages from agent_b in shared memory store, found none. "
            "agent_b should have persisted at least its HumanMemory (input) "
            "and AIMemory (routing output 'agent_c')."
        )
        assert items_c, (
            "Expected messages from agent_c in shared memory store, found none. "
            "agent_c should have persisted at least its HumanMemory (input) "
            "and AIMemory (final answer)."
        )

    finally:
        await shared_store.close()
