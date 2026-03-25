"""Tests for Agent.branch() — conversation branching (US-023)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from exo.agent import Agent, AgentError
from exo.models.types import ModelResponse  # pyright: ignore[reportMissingImports]
from exo.types import Usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_provider(content: str = "Reply") -> AsyncMock:
    """Return a mock provider that produces a single text response."""
    provider = AsyncMock()
    provider.complete = AsyncMock(
        return_value=ModelResponse(
            content=content,
            tool_calls=[],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
    )
    return provider


def _make_agent_with_memory() -> tuple[Any, Any, Agent]:
    """Create an agent with in-memory ShortTermMemory stores.

    Returns (memory, short_term_store, agent).
    Skips the test if exo-memory is not installed.
    """
    try:
        from exo.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
        from exo.memory.short_term import ShortTermMemory  # pyright: ignore[reportMissingImports]
    except ImportError:
        pytest.skip("exo-memory not installed")

    short_term = ShortTermMemory()
    memory = AgentMemory(short_term=short_term, long_term=ShortTermMemory())
    agent = Agent(name="branch-bot", memory=memory)
    return memory, short_term, agent


# ---------------------------------------------------------------------------
# Agent.branch() — happy path
# ---------------------------------------------------------------------------


class TestAgentBranch:
    async def test_branch_returns_new_uuid(self) -> None:
        """branch() returns a non-empty string different from parent conversation_id."""
        _, _, agent = _make_agent_with_memory()
        provider = _mock_provider()

        await agent.run("Hello", provider=provider)
        parent_conv = agent.conversation_id
        assert parent_conv is not None

        # Get the ID of the HumanMemory item just stored
        from exo.memory.base import MemoryMetadata  # pyright: ignore[reportMissingImports]

        store = agent._memory_persistence.store
        items = await store.search(
            metadata=MemoryMetadata(agent_id="branch-bot", task_id=parent_conv),
            limit=100,
        )
        assert len(items) >= 1
        first_item_id = items[0].id

        branch_id = await agent.branch(from_message_id=first_item_id)

        assert isinstance(branch_id, str)
        assert len(branch_id) > 0
        assert branch_id != parent_conv

    async def test_branch_copies_messages_up_to_cutoff(self) -> None:
        """Branched conversation contains exactly messages up to from_message_id."""
        _, _, agent = _make_agent_with_memory()

        # First run: generates HumanMemory + AIMemory
        await agent.run("First message", provider=_mock_provider("First reply"))
        # Second run: generates another pair
        await agent.run("Second message", provider=_mock_provider("Second reply"))

        parent_conv = agent.conversation_id
        assert parent_conv is not None
        store = agent._memory_persistence.store

        from exo.memory.base import MemoryMetadata  # pyright: ignore[reportMissingImports]

        # Get all items in parent conversation (raw, ordered chronologically)
        all_items = [
            item
            for item in store._items
            if item.metadata.agent_id == "branch-bot" and item.metadata.task_id == parent_conv
        ]
        assert len(all_items) >= 2

        # Branch at the first item
        first_item = all_items[0]
        branch_id = await agent.branch(from_message_id=first_item.id)

        # Verify branch has exactly 1 item (only up to and including first item)
        branch_items = await store.search(
            metadata=MemoryMetadata(agent_id="branch-bot", task_id=branch_id),
            limit=100,
        )
        assert len(branch_items) == 1
        assert branch_items[0].content == first_item.content

    async def test_branch_full_history_included(self) -> None:
        """Branch at last message includes ALL messages from parent."""
        _, _, agent = _make_agent_with_memory()

        await agent.run("Only message", provider=_mock_provider("Only reply"))
        parent_conv = agent.conversation_id
        assert parent_conv is not None
        store = agent._memory_persistence.store

        from exo.memory.base import MemoryMetadata  # pyright: ignore[reportMissingImports]

        all_items = [item for item in store._items if item.metadata.task_id == parent_conv]
        parent_count = len(all_items)
        assert parent_count >= 1

        last_item = all_items[-1]
        branch_id = await agent.branch(from_message_id=last_item.id)

        branch_items = await store.search(
            metadata=MemoryMetadata(agent_id="branch-bot", task_id=branch_id),
            limit=100,
        )
        assert len(branch_items) == parent_count

    async def test_original_conversation_unaffected(self) -> None:
        """Branching does not modify the original conversation's messages."""
        _, _, agent = _make_agent_with_memory()

        await agent.run("Parent message", provider=_mock_provider("Parent reply"))
        parent_conv = agent.conversation_id
        assert parent_conv is not None
        store = agent._memory_persistence.store

        from exo.memory.base import MemoryMetadata  # pyright: ignore[reportMissingImports]

        all_items = [item for item in store._items if item.metadata.task_id == parent_conv]
        parent_count_before = len(all_items)

        # Create branch
        last_item = all_items[-1]
        await agent.branch(from_message_id=last_item.id)

        # Parent conversation item count is unchanged
        parent_items_after = await store.search(
            metadata=MemoryMetadata(agent_id="branch-bot", task_id=parent_conv),
            limit=100,
        )
        assert len(parent_items_after) == parent_count_before

    async def test_branch_items_are_copies_not_references(self) -> None:
        """Branch messages are deep copies — different IDs, same content."""
        _, _, agent = _make_agent_with_memory()

        await agent.run("Hello", provider=_mock_provider("Hi"))
        parent_conv = agent.conversation_id
        assert parent_conv is not None
        store = agent._memory_persistence.store

        from exo.memory.base import MemoryMetadata  # pyright: ignore[reportMissingImports]

        all_items = [item for item in store._items if item.metadata.task_id == parent_conv]

        last_item = all_items[-1]
        branch_id = await agent.branch(from_message_id=last_item.id)

        parent_ids = {item.id for item in all_items}
        branch_items = await store.search(
            metadata=MemoryMetadata(agent_id="branch-bot", task_id=branch_id),
            limit=100,
        )

        branch_ids = {item.id for item in branch_items}

        # Branch items have new IDs (not the same objects)
        assert not parent_ids.intersection(branch_ids)

        # But same content
        parent_contents = {item.content for item in all_items}
        branch_contents = {item.content for item in branch_items}
        assert parent_contents == branch_contents

    async def test_branch_activity_does_not_affect_parent(self) -> None:
        """Running agent on branch conversation_id does not pollute parent."""
        _, _, agent = _make_agent_with_memory()

        await agent.run("Parent message", provider=_mock_provider("Parent reply"))
        parent_conv = agent.conversation_id
        assert parent_conv is not None
        store = agent._memory_persistence.store

        from exo.memory.base import MemoryMetadata  # pyright: ignore[reportMissingImports]

        all_items = [item for item in store._items if item.metadata.task_id == parent_conv]
        last_item = all_items[-1]
        branch_id = await agent.branch(from_message_id=last_item.id)

        parent_count_before = len(all_items)

        # Continue on branch
        await agent.run(
            "Branch message",
            provider=_mock_provider("Branch reply"),
            conversation_id=branch_id,
        )

        # Parent count still the same
        parent_items = await store.search(
            metadata=MemoryMetadata(agent_id="branch-bot", task_id=parent_conv),
            limit=100,
        )
        assert len(parent_items) == parent_count_before

    async def test_context_fork_called_when_context_set(self) -> None:
        """branch() invokes Context.fork() internally when context is set."""
        try:
            from exo.context.context import Context  # pyright: ignore[reportMissingImports]
        except ImportError:
            pytest.skip("exo-context not installed")

        _, _, agent = _make_agent_with_memory()
        assert agent.context is not None  # auto-created ContextConfig

        await agent.run("Hello", provider=_mock_provider())
        parent_conv = agent.conversation_id
        assert parent_conv is not None
        store = agent._memory_persistence.store

        all_items = [item for item in store._items if item.metadata.task_id == parent_conv]
        last_item = all_items[-1]

        # Should not raise — Context.fork() is called internally
        branch_id = await agent.branch(from_message_id=last_item.id)
        assert isinstance(branch_id, str)


# ---------------------------------------------------------------------------
# Agent.branch() — error cases
# ---------------------------------------------------------------------------


class TestAgentBranchErrors:
    async def test_branch_raises_without_memory(self) -> None:
        """branch() raises AgentError when memory=None."""
        agent = Agent(name="nomem-bot", memory=None)
        agent.conversation_id = "some-conv"  # type: ignore[assignment]

        with pytest.raises(AgentError, match="requires memory"):
            await agent.branch("any-id")

    async def test_branch_raises_without_active_conversation(self) -> None:
        """branch() raises AgentError when no active conversation exists."""
        _, _, agent = _make_agent_with_memory()
        # conversation_id is None until first run()

        with pytest.raises(AgentError, match="no active conversation"):
            await agent.branch("any-id")

    async def test_branch_raises_when_message_id_not_found(self) -> None:
        """branch() raises AgentError when from_message_id is not in the conversation."""
        _, _, agent = _make_agent_with_memory()
        await agent.run("Hello", provider=_mock_provider())

        with pytest.raises(AgentError, match="not found"):
            await agent.branch("nonexistent-message-id")
