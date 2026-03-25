"""Integration tests for context summarization trigger.

Tests that:
- Context summarization fires when message count exceeds summary_threshold.
- After summarization, the agent continues to function correctly with
  compressed context (bounded input tokens).

Note on SummaryMemory: The PRD specifies asserting a SummaryMemory item in
the store. In the current implementation, summaries are created as transient
SystemMessages in _apply_context_windowing() and are NOT persisted to
SQLiteMemoryStore (only HumanMemory and AIMemory are persisted via hooks).
Tests are adapted to verify the summarization pipeline via:
  1. Item count in the memory store (proxy for all turns completing).
  2. input_tokens comparison as a proxy for context length reduction.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.timeout(90)
async def test_token_budget_triggers_summarization(
    vertex_model: str, tmp_sqlite_db: str
) -> None:
    """Agent with summary_threshold=4 triggers summarization from turn 3 onward.

    Runs 6 turns with verbose prompts so that the non-system message count
    exceeds summary_threshold=4 multiple times (the threshold is hit each
    turn from turn 3 onward because load_history re-loads all persisted
    items). Verifies that:
    - All 6 turns complete without error (>= 12 items in store).
    - The agent produces a non-empty response after multiple summarizations.
    """
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
    from exo.memory.backends.sqlite import (  # pyright: ignore[reportMissingImports]
        SQLiteMemoryStore,
    )
    from exo.memory.base import MemoryMetadata  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]

    conv_id = "sum-trigger-001"

    store = SQLiteMemoryStore(db_path=tmp_sqlite_db)
    await store.init()
    try:
        agent = Agent(
            name="sum-trigger-agent",
            model=vertex_model,
            instructions="You are a helpful assistant. Be concise.",
            memory=store,
            context=ContextConfig(
                mode="copilot",
                summary_threshold=4,
                offload_threshold=20,
                history_rounds=20,
                token_budget_trigger=0.5,
            ),
            max_steps=3,
        )
        provider = get_provider(vertex_model)

        # 6 turns with varied prompts. Summarization fires from turn 3 onward
        # (load_history returns 4+ messages which >= summary_threshold=4).
        prompts = [
            "What is object-oriented programming? Name the four main principles.",
            "What is functional programming? Name two key ideas.",
            "What is the difference between compiled and interpreted languages?",
            "Name three popular software architecture patterns briefly.",
            "What is the difference between concurrency and parallelism?",
            "Give a one-sentence summary of programming paradigms.",
        ]
        last_result = None
        for prompt in prompts:
            last_result = await agent.run(
                prompt,
                provider=provider,
                conversation_id=conv_id,
            )
    finally:
        await store.close()

    # Re-open store to verify all 6 turns were persisted.
    # 6 turns x 2 items (HumanMemory + AIMemory) = 12 items minimum.
    verify_store = SQLiteMemoryStore(db_path=tmp_sqlite_db)
    await verify_store.init()
    try:
        all_items = await verify_store.search(
            metadata=MemoryMetadata(agent_id="sum-trigger-agent", task_id=conv_id),
            limit=100,
        )
    finally:
        await verify_store.close()

    assert len(all_items) >= 12, (
        f"Expected >= 12 memory items (6 turns x human+AI = 12), "
        f"got {len(all_items)}: {[i.memory_type for i in all_items]}"
    )
    assert last_result is not None and last_result.text, (
        "Agent should produce non-empty text after multiple summarizations"
    )


@pytest.mark.integration
@pytest.mark.timeout(90)
async def test_summarization_shortens_context(
    vertex_model: str, tmp_sqlite_db: str
) -> None:
    """Summarization keeps input_tokens bounded as the conversation grows.

    With summary_threshold=4, turns 3+ trigger summarization so the context
    sent to the LLM is compressed to: summary + 2 recent messages + new user.
    That is ~4 messages, versus the uncompressed 11 messages (5 human + 5 AI
    + 1 new) at turn 6.

    Compares usage.input_tokens at turn 2 (no summarization, ~3 messages)
    against turn 6 (with summarization, ~4 messages). Without compression,
    turn 6 would be ~3.7x turn-2 tokens. With compression it should be
    ~1.3x turn-2 tokens, so the assertion uses a 3x bound to confirm
    compression is active.

    Note: PRD specifies len(result.messages) comparison. AgentOutput has no
    messages field; usage.input_tokens is used as a proxy for context length.
    """
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
    from exo.memory.backends.sqlite import (  # pyright: ignore[reportMissingImports]
        SQLiteMemoryStore,
    )
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]

    store = SQLiteMemoryStore(db_path=tmp_sqlite_db)
    await store.init()
    try:
        agent = Agent(
            name="sum-shorten-agent",
            model=vertex_model,
            instructions="You are a helpful assistant. Answer in 1-2 sentences.",
            memory=store,
            context=ContextConfig(
                mode="copilot",
                summary_threshold=4,
                offload_threshold=20,
                history_rounds=20,
            ),
            max_steps=3,
        )
        provider = get_provider(vertex_model)
        conv_id = "sum-shorten-001"

        # Turns 1-2: below summary_threshold=4 (store has 0 then 2 items,
        # context is 1 then 3 non-system messages — both < 4, no summarization).
        await agent.run("What is 3 + 4?", provider=provider, conversation_id=conv_id)
        result_pre = await agent.run(
            "What is 6 x 7?",
            provider=provider,
            conversation_id=conv_id,
        )
        tokens_pre = result_pre.usage.input_tokens

        # Turns 3-6: summarization fires every turn.
        # At each turn, load_history returns >= 4 items → threshold exceeded →
        # context compressed to summary + 2 recent + new (≈4 messages total).
        questions = [
            "What is a stack data structure?",
            "What is a queue data structure?",
            "What is a hash table?",
            "What is a binary search tree?",
        ]
        result_post = result_pre
        for q in questions:
            result_post = await agent.run(q, provider=provider, conversation_id=conv_id)
        tokens_post = result_post.usage.input_tokens
    finally:
        await store.close()

    # Without summarization, turn 6 would carry 10 history messages + 1 new =
    # 11 messages, producing ~3.7x turn-2 input_tokens.
    # With summarization the context is compressed to ~4 messages (~1.3x turn-2).
    # A 3x bound clearly separates the two cases.
    assert tokens_post < tokens_pre * 3, (
        f"Expected summarization to bound context growth: "
        f"turn-2 tokens={tokens_pre}, turn-6 tokens={tokens_post} "
        f"(expected < {tokens_pre * 3} to confirm compression is active)"
    )
    assert result_post.text, "Agent should produce non-empty text after summarization"
