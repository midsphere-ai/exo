"""Integration tests for conversation branching isolation.

US-INT-018: Verifies agent.branch() creates isolated conversation histories
where changes in one branch do not affect the other.

Branch A receives VOLCANO_TOPIC prompts; Branch B receives GLACIER_TOPIC
prompts.  Memory is queried by task_id to confirm each branch stores only
its own messages with no cross-contamination.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.timeout(90)
async def test_branch_histories_are_independent(
    vertex_model: str,
    tmp_sqlite_db: str,
) -> None:
    """Branch histories are independent: topic X in branch A, topic Y in branch B.

    Runs 3 initial turns to build shared history, then creates two branches
    from the same message.  Runs 2 turns in branch A about VOLCANO_TOPIC and
    2 turns in branch B about GLACIER_TOPIC.  Queries memory by task_id and
    asserts each branch contains only its own topic messages with no
    cross-contamination.
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
        agent = Agent(
            name="branch_tester",
            model=vertex_model,
            instructions="Reply in one word only.",
            max_steps=1,
            memory=store,
            context=None,
        )

        # Run 3 initial turns to build shared history
        await agent.run("Say the word alpha.", provider=provider)
        await agent.run("Say the word beta.", provider=provider)
        await agent.run("Say the word gamma.", provider=provider)

        # Get the last stored message ID in the current conversation
        current_items = await store.search(
            metadata=MemoryMetadata(
                agent_id="branch_tester",
                task_id=agent.conversation_id,
            ),
            limit=100,
        )
        assert current_items, "Expected items in store after 3 turns"
        current_items_sorted = sorted(current_items, key=lambda x: x.created_at)
        last_message_id = current_items_sorted[-1].id

        # Create two branches from the same point in history
        branch_id_a = await agent.branch(last_message_id)
        branch_id_b = await agent.branch(last_message_id)

        assert branch_id_a != branch_id_b, "Branch IDs must be distinct"

        # Branch A: 2 turns about VOLCANO_TOPIC
        await agent.run(
            "Say exactly: VOLCANO_TOPIC",
            provider=provider,
            conversation_id=branch_id_a,
        )
        await agent.run(
            "Repeat: VOLCANO_TOPIC",
            provider=provider,
            conversation_id=branch_id_a,
        )

        # Branch B: 2 turns about GLACIER_TOPIC
        await agent.run(
            "Say exactly: GLACIER_TOPIC",
            provider=provider,
            conversation_id=branch_id_b,
        )
        await agent.run(
            "Repeat: GLACIER_TOPIC",
            provider=provider,
            conversation_id=branch_id_b,
        )

        # Query memory scoped to each branch
        items_a = await store.search(
            metadata=MemoryMetadata(
                agent_id="branch_tester",
                task_id=branch_id_a,
            ),
            limit=100,
        )
        items_b = await store.search(
            metadata=MemoryMetadata(
                agent_id="branch_tester",
                task_id=branch_id_b,
            ),
            limit=100,
        )

        content_a = " ".join(item.content for item in items_a).lower()
        content_b = " ".join(item.content for item in items_b).lower()

        # Branch A must contain its own topic (human message stored as HumanMemory)
        assert "volcano_topic" in content_a, (
            "Expected VOLCANO_TOPIC in branch A items but not found. "
            f"Branch A content (truncated): {content_a[:300]}"
        )

        # Branch B must contain its own topic
        assert "glacier_topic" in content_b, (
            "Expected GLACIER_TOPIC in branch B items but not found. "
            f"Branch B content (truncated): {content_b[:300]}"
        )

        # No cross-contamination: GLACIER_TOPIC must NOT appear in branch A
        assert "glacier_topic" not in content_a, (
            "GLACIER_TOPIC found in branch A — branch isolation is broken. "
            f"Branch A content (truncated): {content_a[:300]}"
        )

        # No cross-contamination: VOLCANO_TOPIC must NOT appear in branch B
        assert "volcano_topic" not in content_b, (
            "VOLCANO_TOPIC found in branch B — branch isolation is broken. "
            f"Branch B content (truncated): {content_b[:300]}"
        )

    finally:
        await store.close()
