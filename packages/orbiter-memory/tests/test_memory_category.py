"""Tests for MemoryCategory taxonomy and category-based filtering."""

from __future__ import annotations

import pytest

from orbiter.memory.base import (  # pyright: ignore[reportMissingImports]
    HumanMemory,
    MemoryCategory,
    MemoryItem,
    MemoryMetadata,
)
from orbiter.memory.long_term import LongTermMemory  # pyright: ignore[reportMissingImports]
from orbiter.memory.short_term import ShortTermMemory  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# MemoryCategory enum tests
# ---------------------------------------------------------------------------


class TestMemoryCategory:
    def test_values(self) -> None:
        assert MemoryCategory.USER_PROFILE == "user_profile"
        assert MemoryCategory.SEMANTIC == "semantic"
        assert MemoryCategory.EPISODIC == "episodic"
        assert MemoryCategory.VARIABLE == "variable"
        assert MemoryCategory.SUMMARY == "summary"
        assert MemoryCategory.CONVERSATION == "conversation"

    def test_is_str(self) -> None:
        assert isinstance(MemoryCategory.USER_PROFILE, str)

    def test_all_members(self) -> None:
        assert len(MemoryCategory) == 6


# ---------------------------------------------------------------------------
# MemoryItem category field tests
# ---------------------------------------------------------------------------


class TestMemoryItemCategory:
    def test_default_is_none(self) -> None:
        item = MemoryItem(content="hello", memory_type="test")
        assert item.category is None

    def test_set_category(self) -> None:
        item = MemoryItem(
            content="user prefers dark mode",
            memory_type="human",
            category=MemoryCategory.USER_PROFILE,
        )
        assert item.category == MemoryCategory.USER_PROFILE

    def test_category_on_subclass(self) -> None:
        item = HumanMemory(content="hello", category=MemoryCategory.CONVERSATION)
        assert item.category == MemoryCategory.CONVERSATION
        assert item.memory_type == "human"

    def test_category_orthogonal_to_memory_type(self) -> None:
        """Category and memory_type are independent dimensions."""
        item = HumanMemory(content="remember this", category=MemoryCategory.EPISODIC)
        assert item.memory_type == "human"
        assert item.category == MemoryCategory.EPISODIC

    def test_backward_compatible_no_category(self) -> None:
        """Existing code creating MemoryItems without category still works."""
        item = HumanMemory(content="hello")
        assert item.category is None
        # Serialization still works
        data = item.model_dump()
        assert "category" in data
        assert data["category"] is None


# ---------------------------------------------------------------------------
# ShortTermMemory category filtering tests
# ---------------------------------------------------------------------------


class TestShortTermMemoryCategoryFilter:
    async def test_search_by_category(self) -> None:
        stm = ShortTermMemory()
        await stm.add(HumanMemory(content="profile info", category=MemoryCategory.USER_PROFILE))
        await stm.add(HumanMemory(content="a fact", category=MemoryCategory.SEMANTIC))
        await stm.add(HumanMemory(content="an episode", category=MemoryCategory.EPISODIC))

        results = await stm.search(category=MemoryCategory.USER_PROFILE)
        assert len(results) == 1
        assert results[0].content == "profile info"

    async def test_search_no_category_filter_returns_all(self) -> None:
        stm = ShortTermMemory()
        await stm.add(HumanMemory(content="a", category=MemoryCategory.USER_PROFILE))
        await stm.add(HumanMemory(content="b", category=MemoryCategory.SEMANTIC))
        await stm.add(HumanMemory(content="c"))  # no category

        results = await stm.search()
        assert len(results) == 3

    async def test_search_category_with_none_items(self) -> None:
        """Items with no category should not match a specific category filter."""
        stm = ShortTermMemory()
        await stm.add(HumanMemory(content="no category"))
        await stm.add(HumanMemory(content="has category", category=MemoryCategory.VARIABLE))

        results = await stm.search(category=MemoryCategory.VARIABLE)
        assert len(results) == 1
        assert results[0].content == "has category"

    async def test_search_category_combined_with_memory_type(self) -> None:
        stm = ShortTermMemory()
        await stm.add(HumanMemory(content="human profile", category=MemoryCategory.USER_PROFILE))
        await stm.add(
            MemoryItem(
                content="system profile",
                memory_type="system",
                category=MemoryCategory.USER_PROFILE,
            )
        )

        results = await stm.search(
            memory_type="human",
            category=MemoryCategory.USER_PROFILE,
        )
        assert len(results) == 1
        assert results[0].content == "human profile"

    async def test_search_category_combined_with_query(self) -> None:
        stm = ShortTermMemory()
        await stm.add(HumanMemory(content="dark mode preference", category=MemoryCategory.USER_PROFILE))
        await stm.add(HumanMemory(content="light mode preference", category=MemoryCategory.USER_PROFILE))
        await stm.add(HumanMemory(content="dark theme fact", category=MemoryCategory.SEMANTIC))

        results = await stm.search(query="dark", category=MemoryCategory.USER_PROFILE)
        assert len(results) == 1
        assert results[0].content == "dark mode preference"

    async def test_search_empty_category_result(self) -> None:
        stm = ShortTermMemory()
        await stm.add(HumanMemory(content="hello", category=MemoryCategory.CONVERSATION))

        results = await stm.search(category=MemoryCategory.SUMMARY)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# LongTermMemory category filtering tests
# ---------------------------------------------------------------------------


class TestLongTermMemoryCategoryFilter:
    async def test_search_by_category(self) -> None:
        ltm = LongTermMemory()
        await ltm.add(MemoryItem(content="user profile data", memory_type="human", category=MemoryCategory.USER_PROFILE))
        await ltm.add(MemoryItem(content="semantic fact", memory_type="human", category=MemoryCategory.SEMANTIC))
        await ltm.add(MemoryItem(content="an episode", memory_type="human", category=MemoryCategory.EPISODIC))

        results = await ltm.search(category=MemoryCategory.SEMANTIC)
        assert len(results) == 1
        assert results[0].content == "semantic fact"

    async def test_search_no_category_filter_returns_all(self) -> None:
        ltm = LongTermMemory()
        await ltm.add(MemoryItem(content="a", memory_type="human", category=MemoryCategory.USER_PROFILE))
        await ltm.add(MemoryItem(content="b", memory_type="human", category=MemoryCategory.SEMANTIC))
        await ltm.add(MemoryItem(content="c", memory_type="human"))

        results = await ltm.search()
        assert len(results) == 3

    async def test_search_category_with_none_items(self) -> None:
        ltm = LongTermMemory()
        await ltm.add(MemoryItem(content="no cat", memory_type="human"))
        await ltm.add(MemoryItem(content="has cat", memory_type="human", category=MemoryCategory.SUMMARY))

        results = await ltm.search(category=MemoryCategory.SUMMARY)
        assert len(results) == 1
        assert results[0].content == "has cat"

    async def test_search_category_combined_with_query(self) -> None:
        ltm = LongTermMemory()
        await ltm.add(MemoryItem(content="python tip", memory_type="human", category=MemoryCategory.SEMANTIC))
        await ltm.add(MemoryItem(content="python episode", memory_type="human", category=MemoryCategory.EPISODIC))
        await ltm.add(MemoryItem(content="rust fact", memory_type="human", category=MemoryCategory.SEMANTIC))

        results = await ltm.search(query="python", category=MemoryCategory.SEMANTIC)
        assert len(results) == 1
        assert results[0].content == "python tip"

    async def test_search_category_combined_with_metadata(self) -> None:
        ltm = LongTermMemory()
        meta_u1 = MemoryMetadata(user_id="u1")
        meta_u2 = MemoryMetadata(user_id="u2")
        await ltm.add(MemoryItem(content="u1 profile", memory_type="human", category=MemoryCategory.USER_PROFILE, metadata=meta_u1))
        await ltm.add(MemoryItem(content="u2 profile", memory_type="human", category=MemoryCategory.USER_PROFILE, metadata=meta_u2))

        results = await ltm.search(
            category=MemoryCategory.USER_PROFILE,
            metadata=MemoryMetadata(user_id="u1"),
        )
        assert len(results) == 1
        assert results[0].content == "u1 profile"

    async def test_search_empty_category_result(self) -> None:
        ltm = LongTermMemory()
        await ltm.add(MemoryItem(content="hello", memory_type="human", category=MemoryCategory.CONVERSATION))

        results = await ltm.search(category=MemoryCategory.VARIABLE)
        assert len(results) == 0
