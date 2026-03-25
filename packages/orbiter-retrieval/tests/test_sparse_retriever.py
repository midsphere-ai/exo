"""Tests for SparseRetriever (BM25)."""

from __future__ import annotations

from typing import Any

import pytest

from orbiter.retrieval.retriever import Retriever
from orbiter.retrieval.sparse_retriever import SparseRetriever
from orbiter.retrieval.types import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    document_id: str = "doc-1",
    index: int = 0,
    content: str = "hello",
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    """Build a Chunk with sensible defaults."""
    return Chunk(
        document_id=document_id,
        index=index,
        content=content,
        start=0,
        end=len(content),
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# ABC conformance
# ---------------------------------------------------------------------------


class TestSparseRetrieverABC:
    def test_is_retriever_subclass(self) -> None:
        retriever = SparseRetriever()
        assert isinstance(retriever, Retriever)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


class TestSparseRetrieverIndex:
    def test_index_populates_internal_state(self) -> None:
        retriever = SparseRetriever()
        chunks = [_chunk(content="hello world"), _chunk(content="foo bar", index=1)]
        retriever.index(chunks)

        assert len(retriever._chunks) == 2
        assert retriever._avg_dl > 0

    def test_index_replaces_previous_data(self) -> None:
        retriever = SparseRetriever()
        retriever.index([_chunk(content="first")])
        retriever.index([_chunk(content="second")])

        assert len(retriever._chunks) == 1
        assert retriever._chunks[0].content == "second"

    def test_index_empty_list(self) -> None:
        retriever = SparseRetriever()
        retriever.index([])
        assert retriever._chunks == []
        assert retriever._avg_dl == 0.0


# ---------------------------------------------------------------------------
# Basic retrieval
# ---------------------------------------------------------------------------


class TestSparseRetrieverBasic:
    @pytest.mark.asyncio
    async def test_retrieve_returns_results(self) -> None:
        retriever = SparseRetriever()
        retriever.index([
            _chunk(content="the cat sat on the mat", index=0),
            _chunk(content="the dog played in the park", index=1),
        ])

        results = await retriever.retrieve("cat mat")

        assert len(results) >= 1
        assert results[0].chunk.content == "the cat sat on the mat"

    @pytest.mark.asyncio
    async def test_retrieve_returns_retrieval_result_objects(self) -> None:
        retriever = SparseRetriever()
        retriever.index([_chunk(content="hello world")])

        results = await retriever.retrieve("hello")

        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)
        assert isinstance(results[0].chunk, Chunk)

    @pytest.mark.asyncio
    async def test_retrieve_empty_index(self) -> None:
        retriever = SparseRetriever()
        results = await retriever.retrieve("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_no_matching_terms(self) -> None:
        retriever = SparseRetriever()
        retriever.index([_chunk(content="alpha beta gamma")])

        results = await retriever.retrieve("delta epsilon")

        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_empty_query(self) -> None:
        retriever = SparseRetriever()
        retriever.index([_chunk(content="hello world")])

        results = await retriever.retrieve("")

        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_scores_are_positive(self) -> None:
        retriever = SparseRetriever()
        retriever.index([
            _chunk(content="python programming language", index=0),
            _chunk(content="java programming language", index=1),
        ])

        results = await retriever.retrieve("python")

        for r in results:
            assert r.score > 0


# ---------------------------------------------------------------------------
# Keyword relevance ranking
# ---------------------------------------------------------------------------


class TestSparseRetrieverRanking:
    @pytest.mark.asyncio
    async def test_exact_keyword_match_ranks_higher(self) -> None:
        retriever = SparseRetriever()
        retriever.index([
            _chunk(content="machine learning algorithms for data", index=0),
            _chunk(content="machine learning machine learning deep learning", index=1),
        ])

        results = await retriever.retrieve("machine learning")

        # Chunk with more occurrences of query terms should rank higher
        assert results[0].chunk.index == 1
        assert results[0].score > results[1].score

    @pytest.mark.asyncio
    async def test_rare_term_ranks_higher_than_common(self) -> None:
        """IDF should boost rare terms over common ones."""
        retriever = SparseRetriever()
        retriever.index([
            _chunk(content="the common word appears here", index=0),
            _chunk(content="the common word also here", index=1),
            _chunk(content="the common word plus rare", index=2),
        ])

        results = await retriever.retrieve("rare")

        # Only the chunk with "rare" should appear
        assert len(results) == 1
        assert results[0].chunk.index == 2

    @pytest.mark.asyncio
    async def test_multi_term_query_sums_scores(self) -> None:
        """A chunk matching multiple query terms should score higher."""
        retriever = SparseRetriever()
        retriever.index([
            _chunk(content="apple", index=0),
            _chunk(content="banana", index=1),
            _chunk(content="apple banana", index=2),
        ])

        results = await retriever.retrieve("apple banana")

        assert results[0].chunk.index == 2
        assert results[0].score > results[1].score


# ---------------------------------------------------------------------------
# top_k
# ---------------------------------------------------------------------------


class TestSparseRetrieverTopK:
    @pytest.mark.asyncio
    async def test_respects_top_k(self) -> None:
        retriever = SparseRetriever()
        retriever.index([
            _chunk(content=f"word{i} common", index=i) for i in range(10)
        ])

        results = await retriever.retrieve("common", top_k=3)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_top_k_larger_than_results(self) -> None:
        retriever = SparseRetriever()
        retriever.index([_chunk(content="hello world")])

        results = await retriever.retrieve("hello", top_k=10)

        assert len(results) == 1


# ---------------------------------------------------------------------------
# BM25 parameters
# ---------------------------------------------------------------------------


class TestSparseRetrieverParams:
    @pytest.mark.asyncio
    async def test_custom_k1_and_b(self) -> None:
        retriever = SparseRetriever(k1=2.0, b=0.5)
        retriever.index([
            _chunk(content="test document one", index=0),
            _chunk(content="test document two", index=1),
        ])

        results = await retriever.retrieve("test")

        assert len(results) == 2
        assert all(r.score > 0 for r in results)

    @pytest.mark.asyncio
    async def test_case_insensitive(self) -> None:
        retriever = SparseRetriever()
        retriever.index([_chunk(content="Python Programming")])

        results = await retriever.retrieve("python programming")

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_punctuation_ignored(self) -> None:
        retriever = SparseRetriever()
        retriever.index([_chunk(content="hello, world! foo-bar.")])

        results = await retriever.retrieve("hello world foo bar")

        assert len(results) == 1
