"""Integration tests for agent + context windowing seam.

Tests that:
- Context windowing (history_rounds) correctly limits message history sent to LLM.
- pilot mode retains full history without trimming across turns.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_history_rounds_bounds_messages(
    vertex_model: str, tmp_sqlite_db: str
) -> None:
    """Agent with history_rounds=2 should not see messages from turn 1 after 6 turns.

    Plants a unique secret code in turn 1, then runs 5 padding turns to push turn 1
    beyond the context window.  On turn 7, the agent should be unable to recall the
    secret because only the last 2 non-system messages are visible after windowing.
    """
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
    from exo.memory.backends.sqlite import (  # pyright: ignore[reportMissingImports]
        SQLiteMemoryStore,
    )
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]

    # A unique token the LLM cannot plausibly guess
    secret_code = "CTXTEST-ALPHA-7B4F"

    store = SQLiteMemoryStore(db_path=tmp_sqlite_db)
    await store.init()
    try:
        agent = Agent(
            name="ctx-window-agent",
            model=vertex_model,
            instructions="You are a helpful assistant. Answer all questions concisely.",
            memory=store,
            context=ContextConfig(
                mode="copilot",
                history_rounds=2,  # Only last 2 non-system messages visible to LLM
                summary_threshold=50,  # High threshold — prevent summarization firing
                offload_threshold=100,
            ),
            max_steps=10,  # High enough that memory loads all prior rounds
        )
        provider = get_provider(vertex_model)
        conv_id = "ctx-window-test-001"

        # Turn 1: plant the unique secret code
        await agent.run(
            f"Please remember this secret code: {secret_code}. Just say 'Noted.'",
            provider=provider,
            conversation_id=conv_id,
        )

        # Turns 2-6: unrelated padding turns to push turn 1 past the context window
        padding_questions = [
            "What colour is the sky?",
            "Name the largest ocean.",
            "What is the boiling point of water in Celsius?",
            "How many sides does a hexagon have?",
            "What is the capital of France?",
        ]
        for question in padding_questions:
            await agent.run(question, provider=provider, conversation_id=conv_id)

        # Turn 7: ask about the secret — history_rounds=2 means turn 1 is gone
        result = await agent.run(
            "What secret code did I ask you to remember at the very start of "
            "our conversation? If you do not know, just say 'I don't know'.",
            provider=provider,
            conversation_id=conv_id,
        )
    finally:
        await store.close()

    # With history_rounds=2 the LLM only sees the last 2 non-system messages
    # (the previous AI answer and the current question). Turn 1 is outside the window.
    assert secret_code not in result.text, (
        f"Expected agent to have lost '{secret_code}' from context "
        f"(history_rounds=2), but got: {result.text!r}"
    )


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_context_mode_pilot_disables_windowing(
    vertex_model: str, tmp_sqlite_db: str
) -> None:
    """Agent with mode='pilot' retains full conversation history across turns.

    Plants a unique secret code in turn 1 then runs 2 padding turns.  On turn 4 the
    agent should still recall the secret because pilot mode sets history_rounds=100,
    so all 4 turns (8 messages) remain within the window.
    """
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.context.config import make_config  # pyright: ignore[reportMissingImports]
    from exo.memory.backends.sqlite import (  # pyright: ignore[reportMissingImports]
        SQLiteMemoryStore,
    )
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]

    # A unique token the LLM cannot plausibly guess
    secret_code = "PILOTTEST-BETA-3C8E"

    store = SQLiteMemoryStore(db_path=tmp_sqlite_db)
    await store.init()
    try:
        agent = Agent(
            name="pilot-mode-agent",
            model=vertex_model,
            instructions="You are a helpful assistant. Answer all questions concisely.",
            memory=store,
            context=make_config("pilot"),  # history_rounds=100 — no trimming for 4 turns
            max_steps=10,
        )
        provider = get_provider(vertex_model)
        conv_id = "pilot-mode-test-001"

        # Turn 1: plant the unique secret code
        await agent.run(
            f"Please remember this secret code: {secret_code}. Just say 'Noted.'",
            provider=provider,
            conversation_id=conv_id,
        )

        # Turns 2-3: unrelated padding
        await agent.run(
            "What is 7 times 8?",
            provider=provider,
            conversation_id=conv_id,
        )
        await agent.run(
            "Name a planet in the solar system.",
            provider=provider,
            conversation_id=conv_id,
        )

        # Turn 4: ask about the secret — pilot mode keeps all 4 turns in context
        result = await agent.run(
            "What secret code did I ask you to remember at the very start of "
            "our conversation? Reply with ONLY the code.",
            provider=provider,
            conversation_id=conv_id,
        )
    finally:
        await store.close()

    # With pilot mode (history_rounds=100) all 4 turns are within the window.
    # The agent can still see turn 1 where the secret was planted.
    assert secret_code in result.text, (
        f"Expected pilot-mode agent to recall '{secret_code}' but got: {result.text!r}"
    )
