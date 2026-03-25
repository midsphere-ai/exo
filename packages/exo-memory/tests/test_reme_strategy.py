"""Tests for the ReMe (Relevant Memory) evolution strategy."""

from __future__ import annotations

import json

import pytest

from exo.memory.base import MemoryItem  # pyright: ignore[reportMissingImports]
from exo.memory.evolution.reme import (  # pyright: ignore[reportMissingImports]
    ReMeStrategy,
    _keyword_similarity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockProvider:
    """Mock LLM that returns a fixed JSON response or per-content mapping."""

    def __init__(
        self,
        patterns: list[dict[str, str]] | None = None,
        raw: str | None = None,
    ) -> None:
        self._patterns = patterns
        self._raw = raw

    async def __call__(self, prompt: str) -> str:
        if self._raw is not None:
            return self._raw
        if self._patterns is not None:
            return json.dumps(self._patterns)
        return "[]"


@pytest.fixture()
def sample_items() -> list[MemoryItem]:
    return [
        MemoryItem(
            content="Using retry logic with exponential backoff fixed flaky API calls",
            memory_type="test",
            id="m1",
        ),
        MemoryItem(
            content="Caching database queries reduced latency by 40%",
            memory_type="test",
            id="m2",
        ),
        MemoryItem(
            content="Skipping input validation caused data corruption in production",
            memory_type="test",
            id="m3",
        ),
    ]


# ---------------------------------------------------------------------------
# Keyword similarity
# ---------------------------------------------------------------------------


class TestKeywordSimilarity:
    def test_identical(self) -> None:
        assert _keyword_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self) -> None:
        assert _keyword_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self) -> None:
        sim = _keyword_similarity("hello world foo", "hello world bar")
        # intersection={hello, world}, union={hello, world, foo, bar}
        assert sim == pytest.approx(2 / 4)

    def test_empty_string(self) -> None:
        assert _keyword_similarity("", "hello") == 0.0
        assert _keyword_similarity("hello", "") == 0.0

    def test_case_insensitive(self) -> None:
        assert _keyword_similarity("Hello World", "hello world") == 1.0


# ---------------------------------------------------------------------------
# ReMeStrategy basics
# ---------------------------------------------------------------------------


class TestReMeStrategy:
    def test_name(self) -> None:
        strategy = ReMeStrategy()
        assert strategy.name == "reme"

    def test_is_evolution_strategy(self) -> None:
        from exo.memory.evolution import (
            MemoryEvolutionStrategy,  # pyright: ignore[reportMissingImports]
        )

        strategy = ReMeStrategy()
        assert isinstance(strategy, MemoryEvolutionStrategy)

    def test_default_threshold(self) -> None:
        strategy = ReMeStrategy()
        assert strategy._similarity_threshold == 0.85

    def test_custom_threshold(self) -> None:
        strategy = ReMeStrategy(similarity_threshold=0.5)
        assert strategy._similarity_threshold == 0.5


# ---------------------------------------------------------------------------
# evolve() without model
# ---------------------------------------------------------------------------


class TestEvolveWithoutModel:
    async def test_returns_items_without_model(self, sample_items: list[MemoryItem]) -> None:
        strategy = ReMeStrategy()
        result = await strategy.evolve(sample_items)
        assert len(result) == 3

    async def test_empty_input(self) -> None:
        strategy = ReMeStrategy()
        result = await strategy.evolve([])
        assert result == []

    async def test_context_without_model_key(self, sample_items: list[MemoryItem]) -> None:
        strategy = ReMeStrategy()
        result = await strategy.evolve(sample_items, context={"other": "val"})
        assert len(result) == 3

    async def test_deduplicates_similar_items(self) -> None:
        strategy = ReMeStrategy(similarity_threshold=0.5)
        items = [
            MemoryItem(content="retry logic with backoff", memory_type="test", id="a"),
            MemoryItem(content="retry logic with backoff works", memory_type="test", id="b"),
        ]
        result = await strategy.evolve(items)
        # b has more content, so a is dropped
        assert len(result) == 1
        assert result[0].id == "b"


# ---------------------------------------------------------------------------
# evolve() with model
# ---------------------------------------------------------------------------


class TestEvolveWithModel:
    async def test_extracts_patterns_via_model(self, sample_items: list[MemoryItem]) -> None:
        model = MockProvider(
            patterns=[
                {
                    "content": "Use retry with exponential backoff for flaky APIs",
                    "pattern_type": "success",
                    "when_to_use": "When calling external APIs that may fail intermittently",
                },
                {
                    "content": "Always validate input before processing",
                    "pattern_type": "failure",
                    "when_to_use": "When accepting user or external data",
                },
            ]
        )
        strategy = ReMeStrategy()
        result = await strategy.evolve(sample_items, context={"model": model})
        assert len(result) == 2
        # Check when_to_use metadata
        assert result[0].metadata.extra["when_to_use"] != ""
        assert result[0].metadata.extra["pattern_type"] in ("success", "failure")
        assert result[0].memory_type == "pattern"

    async def test_model_returns_invalid_json(self, sample_items: list[MemoryItem]) -> None:
        model = MockProvider(raw="not valid json at all")
        strategy = ReMeStrategy()
        result = await strategy.evolve(sample_items, context={"model": model})
        # Falls back to original items
        assert len(result) == 3

    async def test_model_returns_non_array(self, sample_items: list[MemoryItem]) -> None:
        model = MockProvider(raw='{"not": "an array"}')
        strategy = ReMeStrategy()
        result = await strategy.evolve(sample_items, context={"model": model})
        # Falls back to original items
        assert len(result) == 3

    async def test_model_returns_empty_array(self, sample_items: list[MemoryItem]) -> None:
        model = MockProvider(patterns=[])
        strategy = ReMeStrategy()
        result = await strategy.evolve(sample_items, context={"model": model})
        # Empty patterns fall back to original items
        assert len(result) == 3

    async def test_skips_entries_without_content(self) -> None:
        model = MockProvider(
            patterns=[
                {"content": "valid pattern", "pattern_type": "success", "when_to_use": "always"},
                {"content": "", "pattern_type": "success", "when_to_use": "never"},
                {"pattern_type": "failure", "when_to_use": "sometimes"},
            ]
        )
        items = [MemoryItem(content="source", memory_type="test", id="s1")]
        strategy = ReMeStrategy()
        result = await strategy.evolve(items, context={"model": model})
        assert len(result) == 1
        assert result[0].content == "valid pattern"

    async def test_deduplicates_extracted_patterns(self) -> None:
        model = MockProvider(
            patterns=[
                {
                    "content": "use retry logic with exponential backoff",
                    "pattern_type": "success",
                    "when_to_use": "flaky APIs",
                },
                {
                    "content": "use retry logic with exponential backoff always",
                    "pattern_type": "success",
                    "when_to_use": "unreliable services",
                },
            ]
        )
        strategy = ReMeStrategy(similarity_threshold=0.7)
        items = [MemoryItem(content="source", memory_type="test", id="s1")]
        result = await strategy.evolve(items, context={"model": model})
        # Near-duplicates should be deduplicated
        assert len(result) == 1


# ---------------------------------------------------------------------------
# extract_patterns()
# ---------------------------------------------------------------------------


class TestExtractPatterns:
    async def test_returns_memory_items(self) -> None:
        model = MockProvider(
            patterns=[
                {
                    "content": "Cache heavy queries",
                    "pattern_type": "success",
                    "when_to_use": "When queries take over 100ms",
                },
            ]
        )
        items = [MemoryItem(content="Caching helped", memory_type="test", id="x")]
        strategy = ReMeStrategy()
        result = await strategy.extract_patterns(items, model)
        assert len(result) == 1
        assert result[0].content == "Cache heavy queries"
        assert result[0].metadata.extra["when_to_use"] == "When queries take over 100ms"
        assert result[0].metadata.extra["pattern_type"] == "success"

    async def test_preserves_when_to_use_metadata(self) -> None:
        model = MockProvider(
            patterns=[
                {
                    "content": "Validate inputs at boundaries",
                    "pattern_type": "failure",
                    "when_to_use": "When accepting external data",
                },
                {
                    "content": "Use connection pooling",
                    "pattern_type": "success",
                    "when_to_use": "When making many DB connections",
                },
            ]
        )
        items = [MemoryItem(content="source", memory_type="test", id="s1")]
        strategy = ReMeStrategy()
        result = await strategy.extract_patterns(items, model)
        assert len(result) == 2
        assert result[0].metadata.extra["when_to_use"] == "When accepting external data"
        assert result[1].metadata.extra["when_to_use"] == "When making many DB connections"

    async def test_handles_non_dict_entries(self) -> None:
        model = MockProvider(
            raw='[{"content": "valid", "pattern_type": "success", "when_to_use": "always"}, "not a dict", 42]'
        )
        items = [MemoryItem(content="source", memory_type="test", id="s1")]
        strategy = ReMeStrategy()
        result = await strategy.extract_patterns(items, model)
        assert len(result) == 1
        assert result[0].content == "valid"

    async def test_fallback_on_invalid_json(self) -> None:
        model = MockProvider(raw="this is not json")
        items = [MemoryItem(content="original", memory_type="test", id="o1")]
        strategy = ReMeStrategy()
        result = await strategy.extract_patterns(items, model)
        # Falls back to original items
        assert len(result) == 1
        assert result[0].id == "o1"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_no_duplicates(self) -> None:
        strategy = ReMeStrategy()
        items = [
            MemoryItem(content="alpha bravo", memory_type="test", id="a"),
            MemoryItem(content="charlie delta", memory_type="test", id="b"),
        ]
        result = strategy._deduplicate(items)
        assert len(result) == 2

    def test_exact_duplicates(self) -> None:
        strategy = ReMeStrategy(similarity_threshold=0.85)
        items = [
            MemoryItem(content="exact same content", memory_type="test", id="a"),
            MemoryItem(content="exact same content", memory_type="test", id="b"),
        ]
        result = strategy._deduplicate(items)
        assert len(result) == 1

    def test_keeps_longer_content(self) -> None:
        strategy = ReMeStrategy(similarity_threshold=0.5)
        items = [
            MemoryItem(content="retry logic backoff api", memory_type="test", id="short"),
            MemoryItem(content="retry logic backoff api calls", memory_type="test", id="long"),
        ]
        # Jaccard: {retry,logic,backoff,api} ∩ {retry,logic,backoff,api,calls} = 4/5 = 0.8
        result = strategy._deduplicate(items)
        assert len(result) == 1
        assert result[0].id == "long"

    def test_single_item(self) -> None:
        strategy = ReMeStrategy()
        items = [MemoryItem(content="only one", memory_type="test", id="solo")]
        result = strategy._deduplicate(items)
        assert len(result) == 1

    def test_empty_list(self) -> None:
        strategy = ReMeStrategy()
        result = strategy._deduplicate([])
        assert result == []


# ---------------------------------------------------------------------------
# Composition with pipeline
# ---------------------------------------------------------------------------


class TestComposition:
    async def test_compose_sequential(self, sample_items: list[MemoryItem]) -> None:
        from exo.memory.evolution import (
            MemoryEvolutionPipeline,  # pyright: ignore[reportMissingImports]
        )

        reme = ReMeStrategy()
        pipeline = reme >> reme
        assert isinstance(pipeline, MemoryEvolutionPipeline)
        result = await pipeline.evolve(sample_items)
        assert len(result) == 3

    async def test_compose_parallel(self, sample_items: list[MemoryItem]) -> None:
        from exo.memory.evolution import (
            MemoryEvolutionPipeline,  # pyright: ignore[reportMissingImports]
        )

        reme = ReMeStrategy()
        pipeline = reme | reme
        assert isinstance(pipeline, MemoryEvolutionPipeline)
        result = await pipeline.evolve(sample_items)
        # Parallel merges by ID, so same 3 items
        assert len(result) == 3
