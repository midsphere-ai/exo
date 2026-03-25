"""Tests for KnowledgeStore, chunking, and Workspace integration."""

from __future__ import annotations

import pytest

from exo.context._internal.knowledge import (  # pyright: ignore[reportMissingImports]
    Chunk,
    KnowledgeError,
    KnowledgeStore,
    SearchResult,
    chunk_text,
)
from exo.context.workspace import Workspace  # pyright: ignore[reportMissingImports]

# ── chunk_text ──────────────────────────────────────────────────────


class TestChunkText:
    def test_short_text_single_chunk(self) -> None:
        result = chunk_text("hello world", chunk_size=100)
        assert result == ["hello world"]

    def test_empty_text(self) -> None:
        assert chunk_text("") == []

    def test_exact_chunk_size(self) -> None:
        text = "a" * 512
        result = chunk_text(text, chunk_size=512)
        assert len(result) == 1
        assert result[0] == text

    def test_overlapping_chunks(self) -> None:
        text = "a" * 100
        result = chunk_text(text, chunk_size=40, chunk_overlap=10)
        # Step = 40 - 10 = 30, positions: 0, 30, 60 → chunk[60:100] reaches end
        assert len(result) == 3
        assert len(result[0]) == 40
        assert len(result[1]) == 40
        assert len(result[2]) == 40  # last chunk reaches exactly end

    def test_no_overlap(self) -> None:
        text = "a" * 100
        result = chunk_text(text, chunk_size=30, chunk_overlap=0)
        # Step = 30, positions: 0, 30, 60, 90
        assert len(result) == 4
        assert result[3] == "a" * 10

    def test_invalid_chunk_size(self) -> None:
        with pytest.raises(KnowledgeError, match="chunk_size must be positive"):
            chunk_text("hello", chunk_size=0)

    def test_invalid_overlap(self) -> None:
        with pytest.raises(KnowledgeError, match="chunk_overlap"):
            chunk_text("hello", chunk_size=10, chunk_overlap=10)

    def test_negative_overlap(self) -> None:
        with pytest.raises(KnowledgeError, match="chunk_overlap"):
            chunk_text("hello", chunk_size=10, chunk_overlap=-1)


# ── Chunk dataclass ─────────────────────────────────────────────────


class TestChunk:
    def test_creation(self) -> None:
        c = Chunk(artifact_name="doc", index=0, content="hello", start=0, end=5)
        assert c.artifact_name == "doc"
        assert c.index == 0
        assert c.content == "hello"
        assert c.start == 0
        assert c.end == 5

    def test_immutable(self) -> None:
        c = Chunk(artifact_name="doc", index=0, content="hello", start=0, end=5)
        with pytest.raises(AttributeError):
            c.content = "world"  # type: ignore[misc]


# ── SearchResult ────────────────────────────────────────────────────


class TestSearchResult:
    def test_creation(self) -> None:
        c = Chunk(artifact_name="doc", index=0, content="hello", start=0, end=5)
        sr = SearchResult(chunk=c, score=1.5)
        assert sr.chunk is c
        assert sr.score == 1.5


# ── KnowledgeStore ──────────────────────────────────────────────────


class TestKnowledgeStoreInit:
    def test_defaults(self) -> None:
        ks = KnowledgeStore()
        assert ks.chunk_size == 512
        assert ks.chunk_overlap == 64
        assert len(ks) == 0
        assert ks.total_chunks() == 0

    def test_custom_params(self) -> None:
        ks = KnowledgeStore(chunk_size=100, chunk_overlap=20)
        assert ks.chunk_size == 100
        assert ks.chunk_overlap == 20

    def test_repr(self) -> None:
        ks = KnowledgeStore()
        assert "KnowledgeStore" in repr(ks)


class TestKnowledgeStoreAdd:
    def test_add_short_content(self) -> None:
        ks = KnowledgeStore()
        chunks = ks.add("doc1", "hello world")
        assert len(chunks) == 1
        assert chunks[0].artifact_name == "doc1"
        assert chunks[0].content == "hello world"
        assert len(ks) == 1

    def test_add_long_content_creates_multiple_chunks(self) -> None:
        ks = KnowledgeStore(chunk_size=50, chunk_overlap=10)
        content = "word " * 100  # 500 chars
        chunks = ks.add("doc1", content)
        assert len(chunks) > 1
        assert all(c.artifact_name == "doc1" for c in chunks)

    def test_add_empty_name_raises(self) -> None:
        ks = KnowledgeStore()
        with pytest.raises(KnowledgeError, match="artifact name"):
            ks.add("", "content")

    def test_re_add_replaces(self) -> None:
        ks = KnowledgeStore()
        ks.add("doc1", "hello")
        ks.add("doc1", "world")
        chunks = ks.get("doc1")
        assert len(chunks) == 1
        assert chunks[0].content == "world"

    def test_multiple_artifacts(self) -> None:
        ks = KnowledgeStore()
        ks.add("doc1", "hello")
        ks.add("doc2", "world")
        assert len(ks) == 2
        assert set(ks.artifact_names) == {"doc1", "doc2"}


class TestKnowledgeStoreRemove:
    def test_remove_existing(self) -> None:
        ks = KnowledgeStore()
        ks.add("doc1", "hello")
        assert ks.remove("doc1") is True
        assert len(ks) == 0

    def test_remove_missing(self) -> None:
        ks = KnowledgeStore()
        assert ks.remove("nope") is False


class TestKnowledgeStoreGet:
    def test_get_existing(self) -> None:
        ks = KnowledgeStore()
        ks.add("doc1", "hello")
        chunks = ks.get("doc1")
        assert len(chunks) == 1

    def test_get_missing(self) -> None:
        ks = KnowledgeStore()
        assert ks.get("nope") == []


class TestKnowledgeStoreGetRange:
    def test_range_query(self) -> None:
        ks = KnowledgeStore(chunk_size=20, chunk_overlap=5)
        content = "a" * 100
        ks.add("doc1", content)
        # Get chunks overlapping with chars 30-50
        result = ks.get_range("doc1", 30, 50)
        assert len(result) > 0
        for c in result:
            assert c.end > 30
            assert c.start < 50

    def test_range_query_missing_artifact(self) -> None:
        ks = KnowledgeStore()
        assert ks.get_range("nope", 0, 100) == []

    def test_range_query_no_overlap(self) -> None:
        ks = KnowledgeStore(chunk_size=20, chunk_overlap=0)
        ks.add("doc1", "a" * 40)
        # Chunks: [0,20), [20,40) — range [50,60) has no overlap
        result = ks.get_range("doc1", 50, 60)
        assert result == []


class TestKnowledgeStoreSearch:
    def test_keyword_search(self) -> None:
        ks = KnowledgeStore()
        ks.add("doc1", "The quick brown fox jumps over the lazy dog")
        ks.add("doc2", "Python is a great programming language")
        results = ks.search("fox")
        assert len(results) >= 1
        assert results[0].chunk.artifact_name == "doc1"
        assert results[0].score > 0

    def test_search_multiple_terms(self) -> None:
        ks = KnowledgeStore()
        ks.add("doc1", "machine learning algorithms for natural language processing")
        ks.add("doc2", "cooking recipes with fresh ingredients")
        results = ks.search("machine learning language")
        assert len(results) >= 1
        assert results[0].chunk.artifact_name == "doc1"

    def test_search_no_results(self) -> None:
        ks = KnowledgeStore()
        ks.add("doc1", "hello world")
        results = ks.search("zebra")
        assert results == []

    def test_search_empty_query(self) -> None:
        ks = KnowledgeStore()
        ks.add("doc1", "hello world")
        assert ks.search("") == []
        assert ks.search("   ") == []

    def test_search_top_k(self) -> None:
        ks = KnowledgeStore()
        ks.add("doc1", "apple banana cherry")
        ks.add("doc2", "apple orange grape")
        ks.add("doc3", "apple pear melon")
        results = ks.search("apple", top_k=2)
        assert len(results) <= 2

    def test_search_ranking(self) -> None:
        ks = KnowledgeStore()
        # doc1 has "python" 3 times, doc2 has it once
        ks.add("doc1", "python python python is great")
        ks.add("doc2", "python is a language")
        results = ks.search("python")
        assert len(results) == 2
        assert results[0].chunk.artifact_name == "doc1"
        assert results[0].score > results[1].score

    def test_search_empty_store(self) -> None:
        ks = KnowledgeStore()
        assert ks.search("hello") == []


# ── Workspace + KnowledgeStore integration ──────────────────────────


class TestWorkspaceKnowledgeIntegration:
    async def test_write_auto_indexes(self) -> None:
        ks = KnowledgeStore()
        ws = Workspace("test-ws", knowledge_store=ks)
        await ws.write("doc1", "The quick brown fox jumps over the lazy dog")
        assert len(ks) == 1
        results = ks.search("fox")
        assert len(results) == 1
        assert results[0].chunk.artifact_name == "doc1"

    async def test_update_re_indexes(self) -> None:
        ks = KnowledgeStore()
        ws = Workspace("test-ws", knowledge_store=ks)
        await ws.write("doc1", "original content about cats")
        await ws.write("doc1", "updated content about dogs")
        results = ks.search("dogs")
        assert len(results) == 1
        # Old content should not match
        results_old = ks.search("cats")
        assert len(results_old) == 0

    async def test_delete_removes_from_index(self) -> None:
        ks = KnowledgeStore()
        ws = Workspace("test-ws", knowledge_store=ks)
        await ws.write("doc1", "hello world")
        assert len(ks) == 1
        await ws.delete("doc1")
        assert len(ks) == 0
        assert ks.search("hello") == []

    async def test_no_knowledge_store_no_error(self) -> None:
        ws = Workspace("test-ws")
        await ws.write("doc1", "hello")
        await ws.delete("doc1")

    async def test_round_trip_write_search(self) -> None:
        """Full artifact → chunk → search round-trip."""
        ks = KnowledgeStore(chunk_size=100, chunk_overlap=20)
        ws = Workspace("test-ws", knowledge_store=ks)

        # Write a large artifact that will be chunked
        content = (
            "Machine learning is a branch of artificial intelligence. "
            "It uses algorithms to learn from data. "
            "Deep learning is a subset of machine learning. "
            "Neural networks have many layers. "
            "Python is commonly used for machine learning tasks. "
            "Libraries like TensorFlow and PyTorch are popular."
        )
        await ws.write("ml_guide", content)

        # Search should find relevant chunks
        results = ks.search("machine learning")
        assert len(results) >= 1
        assert any("machine" in r.chunk.content.lower() for r in results)

        # Verify chunks are from our artifact
        chunks = ks.get("ml_guide")
        assert len(chunks) >= 1

        # Range query should work
        range_chunks = ks.get_range("ml_guide", 0, 100)
        assert len(range_chunks) >= 1

    async def test_multiple_artifacts_search(self) -> None:
        """Search across multiple workspace artifacts."""
        ks = KnowledgeStore()
        ws = Workspace("test-ws", knowledge_store=ks)

        await ws.write("readme", "This project uses Python for backend development")
        await ws.write("guide", "JavaScript is used for the frontend")
        await ws.write("notes", "Python and JavaScript work together via REST APIs")

        results = ks.search("Python")
        assert len(results) >= 2
        artifact_names = {r.chunk.artifact_name for r in results}
        assert "readme" in artifact_names
        assert "notes" in artifact_names

    async def test_knowledge_store_property(self) -> None:
        ks = KnowledgeStore()
        ws = Workspace("test-ws", knowledge_store=ks)
        assert ws.knowledge_store is ks

    async def test_knowledge_store_none_by_default(self) -> None:
        ws = Workspace("test-ws")
        assert ws.knowledge_store is None
