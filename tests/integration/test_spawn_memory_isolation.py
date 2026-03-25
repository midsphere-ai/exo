"""Integration tests for spawn memory isolation.

US-INT-020: Verifies that memory items written by child agent A are not
visible to child agent B when queried by their respective conversation_id.

Each agent stores items tagged with task_id=conversation_id.  Querying the
shared store with child B's session must not return child A's unique fact,
while querying with child A's session must return it.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_child_agents_have_isolated_memory(
    vertex_model: str,
    tmp_sqlite_db: str,
) -> None:
    """Memory items from child A's session are not visible to child B's session.

    Both child agents share the same SQLiteMemoryStore.  Child A runs with
    conversation_id='child-a-session'; its user prompt contains the unique
    token 'CHILD_A_UNIQUE_FACT_XYZ', which is persisted as HumanMemory with
    task_id='child-a-session'.

    Querying with task_id='child-b-session' must yield no items containing
    the token.  Querying with task_id='child-a-session' must yield the fact.
    """
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.memory.backends.sqlite import (  # pyright: ignore[reportMissingImports]
        SQLiteMemoryStore,
    )
    from exo.memory.base import MemoryMetadata  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]

    provider = get_provider(vertex_model)
    store = SQLiteMemoryStore(db_path=tmp_sqlite_db)
    await store.init()

    try:
        # --- Child agent A: stores unique fact via its user prompt ---
        child_a = Agent(
            name="child-a",
            model=vertex_model,
            instructions="You are a helpful assistant. Reply in one sentence.",
            max_steps=1,
            memory=store,
            context=None,
        )
        await child_a.run(
            "Remember this unique fact: CHILD_A_UNIQUE_FACT_XYZ",
            provider=provider,
            conversation_id="child-a-session",
        )

        # --- Child agent B: runs its own session (no CHILD_A fact) ---
        child_b = Agent(
            name="child-b",
            model=vertex_model,
            instructions="You are a helpful assistant. Reply in one sentence.",
            max_steps=1,
            memory=store,
            context=None,
        )
        await child_b.run(
            "Say hello.",
            provider=provider,
            conversation_id="child-b-session",
        )

        # Query memory scoped to child B's session
        items_b = await store.search(
            metadata=MemoryMetadata(task_id="child-b-session"),
            limit=100,
        )
        content_b = " ".join(item.content for item in items_b)
        assert "CHILD_A_UNIQUE_FACT_XYZ" not in content_b, (
            "Child A's unique fact must not appear in child B's session. "
            f"Child B items: {content_b[:300]}"
        )

        # Query memory scoped to child A's session
        items_a = await store.search(
            metadata=MemoryMetadata(task_id="child-a-session"),
            limit=100,
        )
        assert items_a, "Expected at least one item in child A's session."
        content_a = " ".join(item.content for item in items_a)
        assert "CHILD_A_UNIQUE_FACT_XYZ" in content_a, (
            "Child A's unique fact must appear in child A's session. "
            f"Child A items: {content_a[:300]}"
        )

    finally:
        await store.close()
