"""Tests for unified SearchManager."""

from __future__ import annotations

import pytest

from orbiter.memory.base import (  # pyright: ignore[reportMissingImports]
    MemoryCategory,
    MemoryItem,
    MemoryMetadata,
    MemoryStatus,
)
from orbiter.memory.search import SearchManager  # pyright: ignore[reportMissingImports]


# ---------------------------------------------------------------------------
# Mock MemoryStore
# ---------------------------------------------------------------------------


class MockStore:
    """In-memory store implementing the MemoryStore protocol for testing."""

    def __init__(self, items: list[MemoryItem] | None = None) -> None:
        self._items: list[MemoryItem] = list(items or [])
        self.search_call_count = 0

    async def add(self, item: MemoryItem) -> None:
        self._items.append(item)

    async def get(self, item_id: str) -> MemoryItem | None:
        for item in self._items:
            if item.id == item_id:
                return item
        return None

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
        self.search_call_count += 1
        results = list(self._items)
        if query:
            results = [i for i in results if query.lower() in i.content.lower()]
        if category is not None:
            results = [i for i in results if i.category == category]
        return results[:limit]

    async def clear(self, *, metadata: MemoryMetadata | None = None) -> int:
        count = len(self._items)
        self._items.clear()
        return count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(
    content: str,
    *,
    item_id: str | None = None,
    category: MemoryCategory | None = None,
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> MemoryItem:
    kwargs: dict = {
        "content": content,
        "memory_type": "human",
        "created_at": created_at,
    }
    if item_id is not None:
        kwargs["id"] = item_id
    if category is not None:
        kwargs["category"] = category
    return MemoryItem(**kwargs)


# ===========================================================================
# SearchManager — constructor
# ===========================================================================


class TestSearchManagerInit:
    def test_repr(self) -> None:
        mgr = SearchManager(stores=[MockStore()])
        assert "SearchManager(stores=1)" in repr(mgr)

    def test_stores_property_returns_copy(self) -> None:
        store = MockStore()
        mgr = SearchManager(stores=[store])
        stores = mgr.stores
        stores.append(MockStore())
        assert len(mgr.stores) == 1

    def test_empty_stores(self) -> None:
        mgr = SearchManager(stores=[])
        assert mgr.stores == []


# ===========================================================================
# SearchManager — search
# ===========================================================================


class TestSearchManagerSearch:
    async def test_empty_stores_returns_empty(self) -> None:
        mgr = SearchManager(stores=[])
        results = await mgr.search("hello")
        assert results == []

    async def test_single_store(self) -> None:
        items = [_make_item("hello world", item_id="a1")]
        store = MockStore(items)
        mgr = SearchManager(stores=[store])
        results = await mgr.search("hello")
        assert len(results) == 1
        assert results[0].id == "a1"

    async def test_multiple_stores_merged(self) -> None:
        store1 = MockStore([_make_item("fact A", item_id="s1a")])
        store2 = MockStore([_make_item("fact B", item_id="s2b")])
        mgr = SearchManager(stores=[store1, store2])
        results = await mgr.search("fact")
        assert len(results) == 2
        ids = {r.id for r in results}
        assert ids == {"s1a", "s2b"}

    async def test_deduplicates_by_id(self) -> None:
        shared_item = _make_item("shared fact", item_id="dup1")
        store1 = MockStore([shared_item])
        store2 = MockStore([shared_item])
        mgr = SearchManager(stores=[store1, store2])
        results = await mgr.search("shared")
        assert len(results) == 1
        assert results[0].id == "dup1"

    async def test_overlapping_results_deduplicated(self) -> None:
        item_a = _make_item("fact alpha", item_id="a1")
        item_b = _make_item("fact beta", item_id="b1")
        item_shared = _make_item("fact shared", item_id="shared")
        store1 = MockStore([item_a, item_shared])
        store2 = MockStore([item_b, item_shared])
        mgr = SearchManager(stores=[store1, store2])
        results = await mgr.search("fact")
        assert len(results) == 3
        ids = {r.id for r in results}
        assert ids == {"a1", "b1", "shared"}

    async def test_sorted_by_created_at_descending(self) -> None:
        old = _make_item("old", item_id="old", created_at="2025-01-01T00:00:00+00:00")
        mid = _make_item("mid", item_id="mid", created_at="2025-06-01T00:00:00+00:00")
        new = _make_item("new", item_id="new", created_at="2026-01-01T00:00:00+00:00")
        store1 = MockStore([old, new])
        store2 = MockStore([mid])
        mgr = SearchManager(stores=[store1, store2])
        results = await mgr.search("")
        assert [r.id for r in results] == ["new", "mid", "old"]

    async def test_respects_global_limit(self) -> None:
        items = [_make_item(f"item {i}", item_id=f"i{i}") for i in range(20)]
        store = MockStore(items)
        mgr = SearchManager(stores=[store])
        results = await mgr.search("item", limit=5)
        assert len(results) == 5

    async def test_per_store_limit_prevents_domination(self) -> None:
        """Each store is queried with the same limit, preventing one from dominating."""
        big_store = MockStore(
            [
                _make_item(f"big {i}", item_id=f"b{i}", created_at=f"2025-01-{i + 1:02d}T00:00:00+00:00")
                for i in range(50)
            ]
        )
        small_store = MockStore(
            [
                _make_item(f"small {i}", item_id=f"s{i}", created_at=f"2026-06-{i + 1:02d}T00:00:00+00:00")
                for i in range(3)
            ]
        )
        mgr = SearchManager(stores=[big_store, small_store])
        results = await mgr.search("", limit=10)
        # big_store returns at most 10, small_store returns 3 → merged 13, trimmed to 10
        assert len(results) == 10
        # small_store items should be present (not crowded out by big_store)
        small_ids = {r.id for r in results if r.id.startswith("s")}
        assert len(small_ids) == 3

    async def test_category_filter(self) -> None:
        semantic = _make_item("sem", item_id="s1", category=MemoryCategory.SEMANTIC)
        episodic = _make_item("ep", item_id="e1", category=MemoryCategory.EPISODIC)
        store = MockStore([semantic, episodic])
        mgr = SearchManager(stores=[store])
        results = await mgr.search("", category=MemoryCategory.SEMANTIC)
        assert len(results) == 1
        assert results[0].id == "s1"

    async def test_all_stores_queried_in_parallel(self) -> None:
        """All stores receive a search call."""
        stores = [MockStore() for _ in range(5)]
        mgr = SearchManager(stores=stores)
        await mgr.search("query")
        for store in stores:
            assert store.search_call_count == 1
