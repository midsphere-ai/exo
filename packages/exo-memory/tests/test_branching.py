"""Tests for conversation branching support in ShortTermMemory.

Verifies that ShortTermMemory correctly isolates items by conversation_id
(stored as metadata.task_id), which is the foundation for Agent.branch().
"""

from __future__ import annotations

from exo.memory.base import (
    AIMemory,
    HumanMemory,
    MemoryMetadata,
    SystemMemory,
    ToolMemory,
)
from exo.memory.short_term import ShortTermMemory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_meta(agent_id: str = "bot", task_id: str = "conv-1") -> MemoryMetadata:
    return MemoryMetadata(agent_id=agent_id, task_id=task_id)


# ---------------------------------------------------------------------------
# ShortTermMemory — conversation_id == task_id scope
# ---------------------------------------------------------------------------


class TestShortTermMemoryConversationScope:
    """ShortTermMemory filters items by task_id (= conversation_id) in task scope."""

    async def test_task_scope_isolates_conversations(self) -> None:
        """Items from different conversations do not leak into each other."""
        store = ShortTermMemory(scope="task")
        meta1 = _make_meta(task_id="conv-1")
        meta2 = _make_meta(task_id="conv-2")

        item1 = HumanMemory(content="hello conv-1", metadata=meta1)
        item2 = HumanMemory(content="hello conv-2", metadata=meta2)

        await store.add(item1)
        await store.add(item2)

        results1 = await store.search(metadata=meta1, limit=10)
        results2 = await store.search(metadata=meta2, limit=10)

        assert len(results1) == 1
        assert results1[0].content == "hello conv-1"
        assert len(results2) == 1
        assert results2[0].content == "hello conv-2"

    async def test_conversation_id_maps_to_task_id(self) -> None:
        """conversation_id = metadata.task_id is the canonical scoping convention."""
        store = ShortTermMemory(scope="task")
        conv_id = "my-conversation-id"
        meta = _make_meta(task_id=conv_id)

        item = HumanMemory(content="hello", metadata=meta)
        await store.add(item)

        found = await store.search(metadata=MemoryMetadata(task_id=conv_id), limit=10)
        assert len(found) == 1
        assert found[0].id == item.id

    async def test_search_without_task_id_returns_all_items(self) -> None:
        """search() without task_id filter returns items from all conversations."""
        store = ShortTermMemory(scope="task")

        for i in range(3):
            await store.add(
                HumanMemory(content=f"msg-{i}", metadata=_make_meta(task_id=f"conv-{i}"))
            )

        all_results = await store.search(limit=10)
        assert len(all_results) == 3

    async def test_branched_items_stored_separately(self) -> None:
        """Items copied to branch conv_id are independent of original conversation."""
        store = ShortTermMemory(scope="task")

        parent_meta = _make_meta(task_id="parent-conv")
        branch_meta = _make_meta(task_id="branch-conv")

        parent_item = HumanMemory(content="parent message", metadata=parent_meta)
        await store.add(parent_item)

        # Simulate branch: copy parent item into branch scope
        copied = parent_item.model_copy(update={"id": "branch-copy-id", "metadata": branch_meta})
        await store.add(copied)

        # Verify isolation
        parent_results = await store.search(metadata=parent_meta, limit=10)
        branch_results = await store.search(metadata=branch_meta, limit=10)

        assert len(parent_results) == 1
        assert parent_results[0].id == parent_item.id

        assert len(branch_results) == 1
        assert branch_results[0].id == "branch-copy-id"

    async def test_writes_to_branch_do_not_affect_parent(self) -> None:
        """New messages added to branch conversation do not appear in parent."""
        store = ShortTermMemory(scope="task")

        parent_meta = _make_meta(task_id="parent-conv")
        branch_meta = _make_meta(task_id="branch-conv")

        # Populate parent
        await store.add(HumanMemory(content="parent msg", metadata=parent_meta))

        # Copy to branch (simulate branch())
        branch_copy = HumanMemory(content="parent msg", metadata=branch_meta)
        await store.add(branch_copy)

        # Add new messages to branch only
        await store.add(AIMemory(content="branch reply", metadata=branch_meta))

        parent_items = await store.search(metadata=parent_meta, limit=100)
        branch_items = await store.search(metadata=branch_meta, limit=100)

        assert len(parent_items) == 1
        assert parent_items[0].content == "parent msg"

        assert len(branch_items) == 2
        contents = {i.content for i in branch_items}
        assert "branch reply" in contents

    async def test_branch_inherits_message_types(self) -> None:
        """Branch copies all message types: human, ai, tool, system."""
        store = ShortTermMemory(scope="task")
        parent_meta = _make_meta(task_id="parent")
        branch_meta = _make_meta(task_id="branch")

        # AIMemory must include a matching tool_call so _filter_incomplete_pairs
        # does not drop the subsequent ToolMemory.
        items = [
            SystemMemory(content="sys prompt", metadata=parent_meta),
            HumanMemory(content="user msg", metadata=parent_meta),
            AIMemory(
                content="ai reply",
                metadata=parent_meta,
                tool_calls=[{"id": "tc-1", "name": "search", "arguments": "{}"}],
            ),
            ToolMemory(
                content="tool result",
                metadata=parent_meta,
                tool_call_id="tc-1",
                tool_name="search",
            ),
        ]
        for item in items:
            await store.add(item)

        # Copy all to branch
        for item in items:
            copied = item.model_copy(update={"id": f"copy-{item.id}", "metadata": branch_meta})
            await store.add(copied)

        branch_items = await store.search(metadata=branch_meta, limit=20)
        # All 4 types present in branch
        branch_types = {i.memory_type for i in branch_items}
        assert branch_types == {"system", "human", "ai", "tool"}
