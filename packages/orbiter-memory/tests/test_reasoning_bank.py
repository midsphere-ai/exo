"""Tests for the ReasoningBank memory evolution strategy."""

from __future__ import annotations

import json
from typing import Any

import pytest

from orbiter.memory.base import MemoryItem  # pyright: ignore[reportMissingImports]
from orbiter.memory.evolution.reasoning_bank import (  # pyright: ignore[reportMissingImports]
    ReasoningBankStrategy,
    ReasoningEntry,
    _cosine_similarity,
    _entry_text,
    _keyword_similarity,
    _parse_entry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockEmbeddings:
    """Mock embeddings provider that returns pre-configured vectors."""

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self._vectors = vectors or {}

    async def embed(self, text: str) -> list[float]:
        if text in self._vectors:
            return self._vectors[text]
        # Deterministic hash-based fallback
        h = hash(text)
        return [
            (h % 100) / 100.0,
            ((h >> 8) % 100) / 100.0,
            ((h >> 16) % 100) / 100.0,
        ]


def _make_item(content: str, item_id: str = "") -> MemoryItem:
    kwargs: dict[str, Any] = {"content": content, "memory_type": "test"}
    if item_id:
        kwargs["id"] = item_id
    return MemoryItem(**kwargs)


def _make_structured_item(
    title: str, description: str, content: str, item_id: str = ""
) -> MemoryItem:
    data = json.dumps({"title": title, "description": description, "content": content})
    kwargs: dict[str, Any] = {"content": data, "memory_type": "test"}
    if item_id:
        kwargs["id"] = item_id
    return MemoryItem(**kwargs)


@pytest.fixture()
def sample_items() -> list[MemoryItem]:
    return [
        _make_structured_item(
            "Python basics", "Introduction to Python",
            "Python is a programming language", "m1",
        ),
        _make_structured_item(
            "JavaScript basics", "Introduction to JavaScript",
            "JavaScript runs in the browser", "m2",
        ),
        _make_structured_item(
            "Rust basics", "Introduction to Rust",
            "Rust is a systems programming language", "m3",
        ),
    ]


# ---------------------------------------------------------------------------
# ReasoningEntry
# ---------------------------------------------------------------------------


class TestReasoningEntry:
    def test_creation(self) -> None:
        entry = ReasoningEntry(title="t", description="d", content="c", item_id="x")
        assert entry.title == "t"
        assert entry.description == "d"
        assert entry.content == "c"
        assert entry.item_id == "x"

    def test_frozen(self) -> None:
        entry = ReasoningEntry(title="t", description="d", content="c", item_id="x")
        with pytest.raises(AttributeError):
            entry.title = "new"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestCosine:
    def test_identical(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_opposite(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


class TestKeyword:
    def test_identical(self) -> None:
        assert _keyword_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_disjoint(self) -> None:
        assert _keyword_similarity("hello world", "foo bar") == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        sim = _keyword_similarity("hello world foo", "hello world bar")
        # intersection={hello, world}, union={hello, world, foo, bar}
        assert sim == pytest.approx(2 / 4)

    def test_empty_string(self) -> None:
        assert _keyword_similarity("", "hello") == 0.0

    def test_both_empty(self) -> None:
        assert _keyword_similarity("", "") == 0.0


class TestParseEntry:
    def test_json_format(self) -> None:
        item = _make_structured_item("Title", "Desc", "Body", "x1")
        entry = _parse_entry(item)
        assert entry.title == "Title"
        assert entry.description == "Desc"
        assert entry.content == "Body"
        assert entry.item_id == "x1"

    def test_plain_text_fallback(self) -> None:
        item = _make_item("just plain text", "x2")
        entry = _parse_entry(item)
        assert entry.title == ""
        assert entry.description == ""
        assert entry.content == "just plain text"
        assert entry.item_id == "x2"

    def test_invalid_json(self) -> None:
        item = _make_item("{not valid json!", "x3")
        entry = _parse_entry(item)
        assert entry.content == "{not valid json!"

    def test_json_array_falls_back(self) -> None:
        item = _make_item('[1, 2, 3]', "x4")
        entry = _parse_entry(item)
        assert entry.content == '[1, 2, 3]'
        assert entry.title == ""


class TestEntryText:
    def test_all_fields(self) -> None:
        entry = ReasoningEntry(title="T", description="D", content="C", item_id="x")
        assert _entry_text(entry) == "T D C"

    def test_content_only(self) -> None:
        entry = ReasoningEntry(title="", description="", content="C", item_id="x")
        assert _entry_text(entry) == "C"

    def test_title_and_content(self) -> None:
        entry = ReasoningEntry(title="T", description="", content="C", item_id="x")
        assert _entry_text(entry) == "T C"


# ---------------------------------------------------------------------------
# ReasoningBankStrategy basics
# ---------------------------------------------------------------------------


class TestReasoningBankStrategy:
    def test_name(self) -> None:
        strategy = ReasoningBankStrategy()
        assert strategy.name == "reasoning_bank"

    def test_is_evolution_strategy(self) -> None:
        from orbiter.memory.evolution import MemoryEvolutionStrategy  # pyright: ignore[reportMissingImports]

        strategy = ReasoningBankStrategy()
        assert isinstance(strategy, MemoryEvolutionStrategy)


# ---------------------------------------------------------------------------
# evolve()
# ---------------------------------------------------------------------------


class TestEvolve:
    async def test_empty_input(self) -> None:
        strategy = ReasoningBankStrategy()
        result = await strategy.evolve([])
        assert result == []

    async def test_no_duplicates_keeps_all(
        self, sample_items: list[MemoryItem]
    ) -> None:
        strategy = ReasoningBankStrategy()
        result = await strategy.evolve(sample_items)
        assert len(result) == 3

    async def test_dedup_with_embeddings(self) -> None:
        """Two items with identical embeddings are deduplicated."""
        same_vec = [1.0, 0.0, 0.0]
        embeddings = MockEmbeddings(vectors={
            "Python intro Learn Python basics Python is a programming language": same_vec,
            "Python introduction Learn Python fundamentals Python is a programming language": same_vec,
        })
        items = [
            _make_structured_item(
                "Python intro", "Learn Python basics",
                "Python is a programming language", "s1",
            ),
            _make_structured_item(
                "Python introduction", "Learn Python fundamentals",
                "Python is a programming language", "s2",
            ),
        ]
        strategy = ReasoningBankStrategy(
            embeddings=embeddings, similarity_threshold=0.85,
        )
        result = await strategy.evolve(items)
        assert len(result) == 1

    async def test_dedup_keeps_longer(self) -> None:
        """When deduplicating, the entry with more content is kept."""
        same_vec = [1.0, 0.0, 0.0]
        embeddings = MockEmbeddings(vectors={
            "Short Short": same_vec,
            "Long entry A longer description Much more detailed content here": same_vec,
        })
        items = [
            _make_structured_item("Short", "", "Short", "s1"),
            _make_structured_item(
                "Long entry", "A longer description",
                "Much more detailed content here", "s2",
            ),
        ]
        strategy = ReasoningBankStrategy(
            embeddings=embeddings, similarity_threshold=0.85,
        )
        result = await strategy.evolve(items)
        assert len(result) == 1
        assert result[0].id == "s2"

    async def test_dedup_keyword_fallback(self) -> None:
        """Without embeddings, identical content is deduplicated via keywords."""
        items = [
            _make_item("python programming language basics introduction", "k1"),
            _make_item("python programming language basics introduction", "k2"),
        ]
        strategy = ReasoningBankStrategy(similarity_threshold=0.85)
        result = await strategy.evolve(items)
        assert len(result) == 1

    async def test_dissimilar_items_kept(self) -> None:
        """Items below similarity threshold are all kept."""
        embeddings = MockEmbeddings(vectors={
            "cats": [1.0, 0.0, 0.0],
            "dogs": [0.0, 1.0, 0.0],
        })
        items = [_make_item("cats", "c1"), _make_item("dogs", "c2")]
        strategy = ReasoningBankStrategy(
            embeddings=embeddings, similarity_threshold=0.85,
        )
        result = await strategy.evolve(items)
        assert len(result) == 2

    async def test_context_accepted(
        self, sample_items: list[MemoryItem]
    ) -> None:
        strategy = ReasoningBankStrategy()
        result = await strategy.evolve(sample_items, context={"key": "val"})
        assert len(result) == 3

    async def test_single_item(self) -> None:
        items = [_make_item("solo", "s1")]
        strategy = ReasoningBankStrategy()
        result = await strategy.evolve(items)
        assert len(result) == 1
        assert result[0].id == "s1"

    async def test_three_similar_dedup_to_one(self) -> None:
        """Three items with identical embeddings collapse to one."""
        same_vec = [1.0, 0.0, 0.0]
        embeddings = MockEmbeddings(vectors={
            "a": same_vec,
            "ab": same_vec,
            "abc": same_vec,
        })
        items = [
            _make_item("a", "i1"),
            _make_item("ab", "i2"),
            _make_item("abc", "i3"),
        ]
        strategy = ReasoningBankStrategy(
            embeddings=embeddings, similarity_threshold=0.85,
        )
        result = await strategy.evolve(items)
        assert len(result) == 1
        # Longest content kept
        assert result[0].id == "i3"


# ---------------------------------------------------------------------------
# recall()
# ---------------------------------------------------------------------------


class TestRecall:
    async def test_recall_empty_bank(self) -> None:
        strategy = ReasoningBankStrategy()
        result = await strategy.recall("anything")
        assert result == []

    async def test_recall_with_embeddings(self) -> None:
        embeddings = MockEmbeddings(vectors={
            "Python basics Introduction to Python Python is a programming language": [1.0, 0.0, 0.0],
            "JavaScript basics Introduction to JavaScript JavaScript runs in the browser": [0.0, 1.0, 0.0],
            "Rust basics Introduction to Rust Rust is a systems programming language": [0.0, 0.0, 1.0],
            "Python": [0.9, 0.1, 0.0],
        })
        strategy = ReasoningBankStrategy(embeddings=embeddings)
        items = [
            _make_structured_item(
                "Python basics", "Introduction to Python",
                "Python is a programming language", "m1",
            ),
            _make_structured_item(
                "JavaScript basics", "Introduction to JavaScript",
                "JavaScript runs in the browser", "m2",
            ),
            _make_structured_item(
                "Rust basics", "Introduction to Rust",
                "Rust is a systems programming language", "m3",
            ),
        ]
        await strategy.evolve(items)

        results = await strategy.recall("Python", top_k=2)
        assert len(results) <= 2
        assert results[0].item_id == "m1"

    async def test_recall_keyword_fallback(self) -> None:
        strategy = ReasoningBankStrategy()
        items = [
            _make_structured_item(
                "Python basics", "Learn Python", "Python is great", "m1",
            ),
            _make_structured_item(
                "Rust basics", "Learn Rust", "Rust is fast", "m2",
            ),
        ]
        await strategy.evolve(items)

        results = await strategy.recall("Python")
        assert len(results) == 2
        # Python item should rank higher (more keyword overlap)
        assert results[0].item_id == "m1"

    async def test_recall_top_k(self) -> None:
        strategy = ReasoningBankStrategy()
        items = [
            _make_item("alpha", "i1"),
            _make_item("beta", "i2"),
            _make_item("gamma", "i3"),
        ]
        await strategy.evolve(items)
        results = await strategy.recall("alpha", top_k=1)
        assert len(results) == 1

    async def test_recall_returns_reasoning_entries(self) -> None:
        strategy = ReasoningBankStrategy()
        items = [_make_structured_item("Title", "Desc", "Content", "m1")]
        await strategy.evolve(items)

        results = await strategy.recall("Title")
        assert len(results) == 1
        assert isinstance(results[0], ReasoningEntry)
        assert results[0].title == "Title"
        assert results[0].description == "Desc"
        assert results[0].content == "Content"
        assert results[0].item_id == "m1"

    async def test_recall_after_dedup(self) -> None:
        """Recall only searches non-duplicated entries."""
        same_vec = [1.0, 0.0, 0.0]
        other_vec = [0.0, 1.0, 0.0]
        embeddings = MockEmbeddings(vectors={
            "A A": same_vec,
            "A longer version A": same_vec,
            "B B": other_vec,
            "query": [0.9, 0.1, 0.0],
        })
        items = [
            _make_structured_item("A", "", "A", "d1"),
            _make_structured_item("A longer version", "", "A", "d2"),
            _make_structured_item("B", "", "B", "d3"),
        ]
        strategy = ReasoningBankStrategy(
            embeddings=embeddings, similarity_threshold=0.85,
        )
        await strategy.evolve(items)
        results = await strategy.recall("query")
        ids = [r.item_id for r in results]
        # d1 was deduped (shorter), only d2 and d3 remain
        assert "d1" not in ids
        assert "d2" in ids
        assert "d3" in ids


# ---------------------------------------------------------------------------
# Composition with pipeline
# ---------------------------------------------------------------------------


class TestComposition:
    async def test_compose_sequential(
        self, sample_items: list[MemoryItem]
    ) -> None:
        from orbiter.memory.evolution import MemoryEvolutionPipeline  # pyright: ignore[reportMissingImports]

        strategy = ReasoningBankStrategy()
        pipeline = strategy >> strategy
        assert isinstance(pipeline, MemoryEvolutionPipeline)
        result = await pipeline.evolve(sample_items)
        assert len(result) == 3

    async def test_compose_parallel(
        self, sample_items: list[MemoryItem]
    ) -> None:
        from orbiter.memory.evolution import MemoryEvolutionPipeline  # pyright: ignore[reportMissingImports]

        strategy = ReasoningBankStrategy()
        pipeline = strategy | strategy
        assert isinstance(pipeline, MemoryEvolutionPipeline)
        result = await pipeline.evolve(sample_items)
        assert len(result) == 3
