"""Tests for memory evolution strategy ABC and pipeline composition."""

from __future__ import annotations

from typing import Any

import pytest

from exo.memory.base import MemoryItem  # pyright: ignore[reportMissingImports]
from exo.memory.evolution import (  # pyright: ignore[reportMissingImports]
    MemoryEvolutionPipeline,
    MemoryEvolutionStrategy,
)

# ---------------------------------------------------------------------------
# Test strategies
# ---------------------------------------------------------------------------


class UpperCaseStrategy(MemoryEvolutionStrategy):
    """Uppercases all memory content."""

    name = "uppercase"

    async def evolve(
        self,
        items: list[MemoryItem],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        return [item.model_copy(update={"content": item.content.upper()}) for item in items]


class SuffixStrategy(MemoryEvolutionStrategy):
    """Appends a suffix to all memory content."""

    name = "suffix"

    def __init__(self, suffix: str = "!") -> None:
        self._suffix = suffix

    async def evolve(
        self,
        items: list[MemoryItem],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        return [item.model_copy(update={"content": item.content + self._suffix}) for item in items]


class FilterStrategy(MemoryEvolutionStrategy):
    """Filters out items with content shorter than min_length."""

    name = "filter"

    def __init__(self, min_length: int = 5) -> None:
        self._min_length = min_length

    async def evolve(
        self,
        items: list[MemoryItem],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        return [item for item in items if len(item.content) >= self._min_length]


class DuplicateStrategy(MemoryEvolutionStrategy):
    """Returns items with modified content but same IDs (for merge testing)."""

    name = "duplicate"

    def __init__(self, prefix: str = "") -> None:
        self._prefix = prefix

    async def evolve(
        self,
        items: list[MemoryItem],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        return [item.model_copy(update={"content": self._prefix + item.content}) for item in items]


class ContextAwareStrategy(MemoryEvolutionStrategy):
    """Uses context dict to modify behavior."""

    name = "context_aware"

    async def evolve(
        self,
        items: list[MemoryItem],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        tag = (context or {}).get("tag", "default")
        return [item.model_copy(update={"content": f"[{tag}] {item.content}"}) for item in items]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_items() -> list[MemoryItem]:
    return [
        MemoryItem(content="hello world", memory_type="test", id="a"),
        MemoryItem(content="foo", memory_type="test", id="b"),
        MemoryItem(content="bar baz", memory_type="test", id="c"),
    ]


# ---------------------------------------------------------------------------
# ABC tests
# ---------------------------------------------------------------------------


class TestMemoryEvolutionStrategy:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            MemoryEvolutionStrategy()  # type: ignore[abstract]

    async def test_concrete_subclass(self, sample_items: list[MemoryItem]) -> None:
        strategy = UpperCaseStrategy()
        assert strategy.name == "uppercase"
        result = await strategy.evolve(sample_items)
        assert len(result) == 3
        assert result[0].content == "HELLO WORLD"

    async def test_context_passed(self, sample_items: list[MemoryItem]) -> None:
        strategy = ContextAwareStrategy()
        result = await strategy.evolve(sample_items, context={"tag": "v1"})
        assert result[0].content == "[v1] hello world"

    async def test_context_default(self, sample_items: list[MemoryItem]) -> None:
        strategy = ContextAwareStrategy()
        result = await strategy.evolve(sample_items)
        assert result[0].content == "[default] hello world"


# ---------------------------------------------------------------------------
# Sequential composition tests
# ---------------------------------------------------------------------------


class TestSequentialComposition:
    async def test_rshift_creates_pipeline(self) -> None:
        a = UpperCaseStrategy()
        b = SuffixStrategy("!")
        pipeline = a >> b
        assert isinstance(pipeline, MemoryEvolutionPipeline)
        assert pipeline.mode == "sequential"
        assert len(pipeline.strategies) == 2

    async def test_sequential_execution(self, sample_items: list[MemoryItem]) -> None:
        pipeline = UpperCaseStrategy() >> SuffixStrategy("!")
        result = await pipeline.evolve(sample_items)
        assert result[0].content == "HELLO WORLD!"
        assert result[1].content == "FOO!"
        assert result[2].content == "BAR BAZ!"

    async def test_sequential_chaining(self, sample_items: list[MemoryItem]) -> None:
        pipeline = UpperCaseStrategy() >> SuffixStrategy("!") >> SuffixStrategy("?")
        result = await pipeline.evolve(sample_items)
        assert result[0].content == "HELLO WORLD!?"

    async def test_sequential_flattens(self) -> None:
        a = UpperCaseStrategy()
        b = SuffixStrategy("!")
        c = SuffixStrategy("?")
        pipeline = a >> b >> c
        assert isinstance(pipeline, MemoryEvolutionPipeline)
        # Should be flat: [a, b, c], not nested
        assert len(pipeline.strategies) == 3

    async def test_sequential_filter_then_transform(self, sample_items: list[MemoryItem]) -> None:
        pipeline = FilterStrategy(min_length=5) >> UpperCaseStrategy()
        result = await pipeline.evolve(sample_items)
        # "foo" (3 chars) filtered out
        assert len(result) == 2
        assert result[0].content == "HELLO WORLD"
        assert result[1].content == "BAR BAZ"


# ---------------------------------------------------------------------------
# Parallel composition tests
# ---------------------------------------------------------------------------


class TestParallelComposition:
    async def test_or_creates_pipeline(self) -> None:
        a = UpperCaseStrategy()
        b = SuffixStrategy("!")
        pipeline = a | b
        assert isinstance(pipeline, MemoryEvolutionPipeline)
        assert pipeline.mode == "parallel"
        assert len(pipeline.strategies) == 2

    async def test_parallel_merge_last_write_wins(self, sample_items: list[MemoryItem]) -> None:
        # Both strategies produce items with same IDs
        # Last strategy's result wins for each ID
        pipeline = DuplicateStrategy("A:") | DuplicateStrategy("B:")
        result = await pipeline.evolve(sample_items)
        assert len(result) == 3
        # Last-write-wins: B: should overwrite A: for each ID
        contents = {item.id: item.content for item in result}
        assert contents["a"] == "B:hello world"
        assert contents["b"] == "B:foo"
        assert contents["c"] == "B:bar baz"

    async def test_parallel_chaining_flattens(self) -> None:
        a = UpperCaseStrategy()
        b = SuffixStrategy("!")
        c = SuffixStrategy("?")
        pipeline = a | b | c
        assert isinstance(pipeline, MemoryEvolutionPipeline)
        assert len(pipeline.strategies) == 3

    async def test_parallel_with_filter_merges(self, sample_items: list[MemoryItem]) -> None:
        # Filter keeps only long items; Uppercase keeps all
        # Union should include all items (uppercase wins for shared IDs)
        pipeline = FilterStrategy(min_length=5) | UpperCaseStrategy()
        result = await pipeline.evolve(sample_items)
        assert len(result) == 3

    async def test_parallel_context_forwarded(self, sample_items: list[MemoryItem]) -> None:
        pipeline = ContextAwareStrategy() | UpperCaseStrategy()
        result = await pipeline.evolve(sample_items, context={"tag": "test"})
        contents = {item.id: item.content for item in result}
        # UpperCaseStrategy runs last, its result overwrites context_aware for same IDs
        assert contents["a"] == "HELLO WORLD"


# ---------------------------------------------------------------------------
# Mixed composition tests
# ---------------------------------------------------------------------------


class TestMixedComposition:
    async def test_parallel_then_sequential(self, sample_items: list[MemoryItem]) -> None:
        parallel = DuplicateStrategy("X:") | DuplicateStrategy("Y:")
        pipeline = parallel >> UpperCaseStrategy()
        result = await pipeline.evolve(sample_items)
        contents = {item.id: item.content for item in result}
        # Parallel: Y: wins, then uppercase
        assert contents["a"] == "Y:HELLO WORLD"

    async def test_sequential_then_parallel(self, sample_items: list[MemoryItem]) -> None:
        sequential = UpperCaseStrategy() >> SuffixStrategy("!")
        pipeline = sequential | DuplicateStrategy("Z:")
        result = await pipeline.evolve(sample_items)
        assert len(result) == 3

    async def test_pipeline_is_strategy(self) -> None:
        pipeline = UpperCaseStrategy() >> SuffixStrategy("!")
        assert isinstance(pipeline, MemoryEvolutionStrategy)

    async def test_pipeline_name(self) -> None:
        pipeline = UpperCaseStrategy() >> SuffixStrategy("!")
        assert pipeline.name == "pipeline"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    async def test_empty_input(self) -> None:
        pipeline = UpperCaseStrategy() >> SuffixStrategy("!")
        result = await pipeline.evolve([])
        assert result == []

    async def test_single_strategy_pipeline(self, sample_items: list[MemoryItem]) -> None:
        pipeline = MemoryEvolutionPipeline([UpperCaseStrategy()], mode="sequential")
        result = await pipeline.evolve(sample_items)
        assert result[0].content == "HELLO WORLD"

    async def test_parallel_empty_input(self) -> None:
        pipeline = UpperCaseStrategy() | SuffixStrategy("!")
        result = await pipeline.evolve([])
        assert result == []

    async def test_item_ids_preserved_sequential(self, sample_items: list[MemoryItem]) -> None:
        pipeline = UpperCaseStrategy() >> SuffixStrategy("!")
        result = await pipeline.evolve(sample_items)
        assert [item.id for item in result] == ["a", "b", "c"]

    async def test_original_items_not_mutated(self, sample_items: list[MemoryItem]) -> None:
        pipeline = UpperCaseStrategy() >> SuffixStrategy("!")
        await pipeline.evolve(sample_items)
        assert sample_items[0].content == "hello world"
