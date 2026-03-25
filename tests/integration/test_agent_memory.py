"""Integration tests for agent + memory persistence seam.

Tests that:
- An agent's conversation history persists to SQLite and is correctly
  loaded by a new agent instance on the next run.
- Memory items stored with different user_id metadata are isolated from
  each other when queried.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_memory_persists_across_agent_instances(vertex_model: str, tmp_sqlite_db: str) -> None:
    """Conversation history stored by one Agent instance is recalled by a second."""
    from pydantic import BaseModel

    from exo._internal.output_parser import (  # pyright: ignore[reportMissingImports]
        parse_structured_output,
    )
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.memory.backends.sqlite import (  # pyright: ignore[reportMissingImports]
        SQLiteMemoryStore,
    )
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]

    class CityResponse(BaseModel):
        city: str

    provider = get_provider(vertex_model)

    # --- First agent: seed the conversation with the Antananarivo fact ---
    store1 = SQLiteMemoryStore(db_path=tmp_sqlite_db)
    await store1.init()
    try:
        agent1 = Agent(
            name="memory-agent",
            model=vertex_model,
            instructions="You are a helpful geography assistant.",
            memory=store1,
        )
        await agent1.run(
            "Remember this fact: the capital of Madagascar is Antananarivo.",
            provider=provider,
            conversation_id="test-session-001",
        )
    finally:
        await store1.close()

    # --- Second agent: new instance pointing at the same SQLite file ---
    store2 = SQLiteMemoryStore(db_path=tmp_sqlite_db)
    await store2.init()
    try:
        agent2 = Agent(
            name="memory-agent",  # same name → history loaded from DB
            model=vertex_model,
            instructions=(
                "You are a helpful geography assistant. "
                'When asked for a city, reply ONLY with valid JSON: {"city": "<city_name>"}.'
                " No other text, only the JSON object."
            ),
            memory=store2,
        )
        result = await agent2.run(
            "From our previous conversation, what capital city did I ask you to remember? "
            'Reply with ONLY the JSON object: {"city": "<city_name>"}',
            provider=provider,
            conversation_id="test-session-001",
        )
    finally:
        await store2.close()

    city_response = parse_structured_output(result.text, CityResponse)
    assert city_response.city.lower() == "antananarivo"


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_memory_metadata_scoping(tmp_sqlite_db: str) -> None:
    """Memory items stored with different user_id are invisible to each other's queries."""
    from exo.memory.backends.sqlite import (  # pyright: ignore[reportMissingImports]
        SQLiteMemoryStore,
    )
    from exo.memory.base import (  # pyright: ignore[reportMissingImports]
        HumanMemory,
        MemoryMetadata,
    )

    store = SQLiteMemoryStore(db_path=tmp_sqlite_db)
    await store.init()
    try:
        # Write 3 items for user-A (session "shared-session")
        for i in range(3):
            await store.add(
                HumanMemory(
                    content=f"User A message {i}",
                    metadata=MemoryMetadata(user_id="user-A", session_id="shared-session"),
                )
            )

        # Write 2 items for user-B (same session_id, different user_id)
        for i in range(2):
            await store.add(
                HumanMemory(
                    content=f"User B message {i}",
                    metadata=MemoryMetadata(user_id="user-B", session_id="shared-session"),
                )
            )

        # user-A query should return only user-A items
        user_a_items = await store.search(
            metadata=MemoryMetadata(user_id="user-A"), limit=10
        )
        assert len(user_a_items) == 3
        assert all(item.metadata.user_id == "user-A" for item in user_a_items)

        # user-B query should return only user-B items
        user_b_items = await store.search(
            metadata=MemoryMetadata(user_id="user-B"), limit=10
        )
        assert len(user_b_items) == 2
        assert all(item.metadata.user_id == "user-B" for item in user_b_items)
    finally:
        await store.close()
