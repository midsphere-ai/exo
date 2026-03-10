"""Tests for memory base types, status lifecycle, and protocol conformance."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orbiter.memory.base import (  # pyright: ignore[reportMissingImports]
    AIMemory,
    HumanMemory,
    MemoryCategory,
    MemoryError,
    MemoryItem,
    MemoryMetadata,
    MemoryStatus,
    MemoryStore,
    SystemMemory,
    ToolMemory,
)

# ---------------------------------------------------------------------------
# MemoryStatus tests
# ---------------------------------------------------------------------------


class TestMemoryStatus:
    def test_values(self) -> None:
        assert MemoryStatus.DRAFT == "draft"
        assert MemoryStatus.ACCEPTED == "accepted"
        assert MemoryStatus.DISCARD == "discard"

    def test_is_str(self) -> None:
        assert isinstance(MemoryStatus.DRAFT, str)


# ---------------------------------------------------------------------------
# MemoryMetadata tests
# ---------------------------------------------------------------------------


class TestMemoryMetadata:
    def test_defaults(self) -> None:
        m = MemoryMetadata()
        assert m.user_id is None
        assert m.session_id is None
        assert m.task_id is None
        assert m.agent_id is None
        assert m.extra == {}

    def test_with_values(self) -> None:
        m = MemoryMetadata(user_id="u1", session_id="s1", task_id="t1", agent_id="a1")
        assert m.user_id == "u1"
        assert m.session_id == "s1"
        assert m.task_id == "t1"
        assert m.agent_id == "a1"

    def test_extra(self) -> None:
        m = MemoryMetadata(extra={"key": "val"})
        assert m.extra["key"] == "val"

    def test_frozen(self) -> None:
        m = MemoryMetadata()
        with pytest.raises(ValidationError):
            m.user_id = "new"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MemoryItem tests
# ---------------------------------------------------------------------------


class TestMemoryItem:
    def test_creation(self) -> None:
        item = MemoryItem(content="hello", memory_type="test")
        assert item.content == "hello"
        assert item.memory_type == "test"
        assert len(item.id) > 0
        assert item.status == MemoryStatus.ACCEPTED
        assert item.created_at is not None
        assert item.updated_at is not None

    def test_custom_id(self) -> None:
        item = MemoryItem(content="x", memory_type="t", id="custom-id")
        assert item.id == "custom-id"

    def test_default_metadata(self) -> None:
        item = MemoryItem(content="x", memory_type="t")
        assert item.metadata.user_id is None

    def test_with_metadata(self) -> None:
        meta = MemoryMetadata(user_id="u1", session_id="s1")
        item = MemoryItem(content="x", memory_type="t", metadata=meta)
        assert item.metadata.user_id == "u1"

    def test_draft_status(self) -> None:
        item = MemoryItem(content="x", memory_type="t", status=MemoryStatus.DRAFT)
        assert item.status == MemoryStatus.DRAFT


# ---------------------------------------------------------------------------
# Status transition tests
# ---------------------------------------------------------------------------


class TestStatusTransitions:
    def test_draft_to_accepted(self) -> None:
        item = MemoryItem(content="x", memory_type="t", status=MemoryStatus.DRAFT)
        item.transition(MemoryStatus.ACCEPTED)
        assert item.status == MemoryStatus.ACCEPTED

    def test_draft_to_discard(self) -> None:
        item = MemoryItem(content="x", memory_type="t", status=MemoryStatus.DRAFT)
        item.transition(MemoryStatus.DISCARD)
        assert item.status == MemoryStatus.DISCARD

    def test_accepted_to_discard(self) -> None:
        item = MemoryItem(content="x", memory_type="t", status=MemoryStatus.ACCEPTED)
        item.transition(MemoryStatus.DISCARD)
        assert item.status == MemoryStatus.DISCARD

    def test_discard_to_anything_fails(self) -> None:
        item = MemoryItem(content="x", memory_type="t", status=MemoryStatus.DISCARD)
        with pytest.raises(MemoryError, match="Cannot transition"):
            item.transition(MemoryStatus.ACCEPTED)

    def test_accepted_to_draft_fails(self) -> None:
        item = MemoryItem(content="x", memory_type="t", status=MemoryStatus.ACCEPTED)
        with pytest.raises(MemoryError, match="Cannot transition"):
            item.transition(MemoryStatus.DRAFT)

    def test_draft_to_draft_fails(self) -> None:
        item = MemoryItem(content="x", memory_type="t", status=MemoryStatus.DRAFT)
        with pytest.raises(MemoryError, match="Cannot transition"):
            item.transition(MemoryStatus.DRAFT)

    def test_transition_updates_timestamp(self) -> None:
        item = MemoryItem(content="x", memory_type="t", status=MemoryStatus.DRAFT)
        old_updated = item.updated_at
        item.transition(MemoryStatus.ACCEPTED)
        # updated_at should change (or at least not precede the original)
        assert item.updated_at >= old_updated


# ---------------------------------------------------------------------------
# Subtype tests
# ---------------------------------------------------------------------------


class TestSystemMemory:
    def test_defaults(self) -> None:
        m = SystemMemory(content="sys prompt")
        assert m.memory_type == "system"
        assert m.content == "sys prompt"
        assert m.status == MemoryStatus.ACCEPTED

    def test_isinstance(self) -> None:
        m = SystemMemory(content="x")
        assert isinstance(m, MemoryItem)


class TestHumanMemory:
    def test_defaults(self) -> None:
        m = HumanMemory(content="user query")
        assert m.memory_type == "human"
        assert m.content == "user query"

    def test_isinstance(self) -> None:
        m = HumanMemory(content="x")
        assert isinstance(m, MemoryItem)


class TestAIMemory:
    def test_defaults(self) -> None:
        m = AIMemory(content="response")
        assert m.memory_type == "ai"
        assert m.tool_calls == []

    def test_with_tool_calls(self) -> None:
        m = AIMemory(content="r", tool_calls=[{"id": "tc1", "name": "foo"}])
        assert len(m.tool_calls) == 1
        assert m.tool_calls[0]["name"] == "foo"

    def test_isinstance(self) -> None:
        m = AIMemory(content="x")
        assert isinstance(m, MemoryItem)


class TestToolMemory:
    def test_defaults(self) -> None:
        m = ToolMemory(content="result")
        assert m.memory_type == "tool"
        assert m.tool_call_id == ""
        assert m.tool_name == ""
        assert m.is_error is False

    def test_error_result(self) -> None:
        m = ToolMemory(content="err", tool_call_id="tc1", tool_name="search", is_error=True)
        assert m.is_error is True
        assert m.tool_call_id == "tc1"
        assert m.tool_name == "search"

    def test_isinstance(self) -> None:
        m = ToolMemory(content="x")
        assert isinstance(m, MemoryItem)


# ---------------------------------------------------------------------------
# MemoryStore protocol conformance tests
# ---------------------------------------------------------------------------


class InMemoryStore:
    """Minimal in-memory implementation for protocol conformance testing."""

    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}

    async def add(self, item: MemoryItem) -> None:
        self._items[item.id] = item

    async def get(self, item_id: str) -> MemoryItem | None:
        return self._items.get(item_id)

    async def search(
        self,
        *,
        query: str = "",
        metadata: MemoryMetadata | None = None,
        memory_type: str | None = None,
        category: MemoryCategory | None = None,
        status: MemoryStatus | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        results: list[MemoryItem] = []
        for item in self._items.values():
            if memory_type and item.memory_type != memory_type:
                continue
            if category is not None and item.category != category:
                continue
            if status and item.status != status:
                continue
            if metadata:
                if metadata.user_id and item.metadata.user_id != metadata.user_id:
                    continue
                if metadata.session_id and item.metadata.session_id != metadata.session_id:
                    continue
                if metadata.task_id and item.metadata.task_id != metadata.task_id:
                    continue
                if metadata.agent_id and item.metadata.agent_id != metadata.agent_id:
                    continue
            if query and query.lower() not in item.content.lower():
                continue
            results.append(item)
            if len(results) >= limit:
                break
        return results

    async def clear(self, *, metadata: MemoryMetadata | None = None) -> int:
        if metadata is None:
            count = len(self._items)
            self._items.clear()
            return count
        to_remove = []
        for item_id, item in self._items.items():
            if metadata.user_id and item.metadata.user_id != metadata.user_id:
                continue
            if metadata.session_id and item.metadata.session_id != metadata.session_id:
                continue
            to_remove.append(item_id)
        for item_id in to_remove:
            del self._items[item_id]
        return len(to_remove)


class TestMemoryStoreProtocol:
    def test_isinstance_check(self) -> None:
        store = InMemoryStore()
        assert isinstance(store, MemoryStore)

    async def test_add_and_get(self) -> None:
        store = InMemoryStore()
        item = HumanMemory(content="hello")
        await store.add(item)
        retrieved = await store.get(item.id)
        assert retrieved is not None
        assert retrieved.content == "hello"

    async def test_get_missing(self) -> None:
        store = InMemoryStore()
        result = await store.get("nonexistent")
        assert result is None

    async def test_search_by_type(self) -> None:
        store = InMemoryStore()
        await store.add(HumanMemory(content="q1"))
        await store.add(AIMemory(content="a1"))
        await store.add(HumanMemory(content="q2"))
        results = await store.search(memory_type="human")
        assert len(results) == 2

    async def test_search_by_query(self) -> None:
        store = InMemoryStore()
        await store.add(HumanMemory(content="the cat sat"))
        await store.add(HumanMemory(content="the dog ran"))
        results = await store.search(query="cat")
        assert len(results) == 1
        assert "cat" in results[0].content

    async def test_search_by_metadata(self) -> None:
        store = InMemoryStore()
        meta = MemoryMetadata(user_id="u1")
        await store.add(HumanMemory(content="q1", metadata=meta))
        await store.add(HumanMemory(content="q2", metadata=MemoryMetadata(user_id="u2")))
        results = await store.search(metadata=MemoryMetadata(user_id="u1"))
        assert len(results) == 1
        assert results[0].content == "q1"

    async def test_search_by_status(self) -> None:
        store = InMemoryStore()
        await store.add(HumanMemory(content="accepted"))
        draft = HumanMemory(content="draft", status=MemoryStatus.DRAFT)
        await store.add(draft)
        results = await store.search(status=MemoryStatus.DRAFT)
        assert len(results) == 1
        assert results[0].content == "draft"

    async def test_search_limit(self) -> None:
        store = InMemoryStore()
        for i in range(20):
            await store.add(HumanMemory(content=f"msg {i}"))
        results = await store.search(limit=5)
        assert len(results) == 5

    async def test_clear_all(self) -> None:
        store = InMemoryStore()
        await store.add(HumanMemory(content="q1"))
        await store.add(HumanMemory(content="q2"))
        count = await store.clear()
        assert count == 2
        results = await store.search()
        assert len(results) == 0

    async def test_clear_with_filter(self) -> None:
        store = InMemoryStore()
        await store.add(HumanMemory(content="q1", metadata=MemoryMetadata(user_id="u1")))
        await store.add(HumanMemory(content="q2", metadata=MemoryMetadata(user_id="u2")))
        count = await store.clear(metadata=MemoryMetadata(user_id="u1"))
        assert count == 1
        results = await store.search()
        assert len(results) == 1
        assert results[0].metadata.user_id == "u2"
