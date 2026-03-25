"""Tests for Retriever ABC and VectorRetriever."""

from __future__ import annotations

from typing import Any

import pytest

from orbiter.retrieval.embeddings import Embeddings
from orbiter.retrieval.retriever import Retriever, VectorRetriever
from orbiter.retrieval.types import Chunk, RetrievalResult
from orbiter.retrieval.vector_store import InMemoryVectorStore

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


class MockEmbeddings(Embeddings):
    """Deterministic embeddings for testing.

    Maps each query to a fixed 2-d vector based on a simple hash.
    """

    def __init__(self, mapping: dict[str, list[float]] | None = None) -> None:
        self._mapping = mapping or {}

    async def embed(self, text: str) -> list[float]:
        if text in self._mapping:
            return self._mapping[text]
        # Default: use character sum to produce a deterministic vector
        val = float(sum(ord(c) for c in text) % 100) / 100.0
        return [val, 1.0 - val]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return 2


# ---------------------------------------------------------------------------
# Retriever ABC
# ---------------------------------------------------------------------------


class TestRetrieverABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            Retriever()  # type: ignore[abstract]

    def test_concrete_subclass(self) -> None:
        class Dummy(Retriever):
            async def retrieve(
                self,
                query: str,
                *,
                top_k: int = 5,
                **kwargs: Any,
            ) -> list[RetrievalResult]:
                return []

        d = Dummy()
        assert isinstance(d, Retriever)


# ---------------------------------------------------------------------------
# VectorRetriever — basic retrieval
# ---------------------------------------------------------------------------


class TestVectorRetrieverBasic:
    @pytest.mark.asyncio
    async def test_retrieve_returns_results(self) -> None:
        embeddings = MockEmbeddings(mapping={"query": [1.0, 0.0]})
        store = InMemoryVectorStore()
        await store.add(
            [_chunk(content="close", index=0), _chunk(content="far", index=1)],
            [[0.9, 0.1], [0.0, 1.0]],
        )

        retriever = VectorRetriever(embeddings, store)
        results = await retriever.retrieve("query")

        assert len(results) == 2
        assert results[0].chunk.content == "close"
        assert results[0].score > results[1].score

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(self) -> None:
        embeddings = MockEmbeddings(mapping={"query": [1.0, 0.0]})
        store = InMemoryVectorStore()
        chunks = [_chunk(content=f"c{i}", index=i) for i in range(10)]
        vecs = [[float(i) / 10, 1.0 - float(i) / 10] for i in range(10)]
        await store.add(chunks, vecs)

        retriever = VectorRetriever(embeddings, store)
        results = await retriever.retrieve("query", top_k=3)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_retrieve_empty_store(self) -> None:
        embeddings = MockEmbeddings(mapping={"query": [1.0, 0.0]})
        store = InMemoryVectorStore()

        retriever = VectorRetriever(embeddings, store)
        results = await retriever.retrieve("query")

        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_returns_retrieval_result_objects(self) -> None:
        embeddings = MockEmbeddings(mapping={"query": [1.0, 0.0]})
        store = InMemoryVectorStore()
        await store.add([_chunk()], [[1.0, 0.0]])

        retriever = VectorRetriever(embeddings, store)
        results = await retriever.retrieve("query")

        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)
        assert isinstance(results[0].chunk, Chunk)

    @pytest.mark.asyncio
    async def test_retrieve_is_instance_of_retriever(self) -> None:
        retriever = VectorRetriever(MockEmbeddings(), InMemoryVectorStore())
        assert isinstance(retriever, Retriever)


# ---------------------------------------------------------------------------
# VectorRetriever — score threshold
# ---------------------------------------------------------------------------


class TestVectorRetrieverScoreThreshold:
    @pytest.mark.asyncio
    async def test_threshold_filters_low_scores(self) -> None:
        embeddings = MockEmbeddings(mapping={"query": [1.0, 0.0]})
        store = InMemoryVectorStore()
        await store.add(
            [
                _chunk(content="high", index=0),
                _chunk(content="low", index=1),
            ],
            [
                [1.0, 0.0],  # cosine sim = 1.0
                [0.0, 1.0],  # cosine sim = 0.0
            ],
        )

        retriever = VectorRetriever(embeddings, store, score_threshold=0.5)
        results = await retriever.retrieve("query")

        assert len(results) == 1
        assert results[0].chunk.content == "high"
        assert results[0].score >= 0.5

    @pytest.mark.asyncio
    async def test_threshold_zero_keeps_all(self) -> None:
        embeddings = MockEmbeddings(mapping={"query": [1.0, 0.0]})
        store = InMemoryVectorStore()
        await store.add(
            [_chunk(content="a", index=0), _chunk(content="b", index=1)],
            [[1.0, 0.0], [0.0, 1.0]],
        )

        retriever = VectorRetriever(embeddings, store, score_threshold=0.0)
        results = await retriever.retrieve("query")

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_threshold_too_high_returns_empty(self) -> None:
        embeddings = MockEmbeddings(mapping={"query": [0.5, 0.5]})
        store = InMemoryVectorStore()
        await store.add(
            [_chunk(content="a", index=0)],
            [[0.0, 1.0]],
        )

        retriever = VectorRetriever(embeddings, store, score_threshold=0.99)
        results = await retriever.retrieve("query")

        assert results == []

    @pytest.mark.asyncio
    async def test_no_threshold_returns_all(self) -> None:
        embeddings = MockEmbeddings(mapping={"query": [1.0, 0.0]})
        store = InMemoryVectorStore()
        await store.add(
            [
                _chunk(content="high", index=0),
                _chunk(content="low", index=1),
            ],
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )

        retriever = VectorRetriever(embeddings, store)
        results = await retriever.retrieve("query")

        assert len(results) == 2


# ---------------------------------------------------------------------------
# VectorRetriever — kwargs passthrough
# ---------------------------------------------------------------------------


class TestVectorRetrieverKwargs:
    @pytest.mark.asyncio
    async def test_filter_kwarg_passed_to_store(self) -> None:
        embeddings = MockEmbeddings(mapping={"query": [1.0, 0.0]})
        store = InMemoryVectorStore()
        await store.add(
            [
                _chunk(content="web", index=0, metadata={"source": "web"}),
                _chunk(content="pdf", index=1, metadata={"source": "pdf"}),
            ],
            [[1.0, 0.0], [1.0, 0.0]],
        )

        retriever = VectorRetriever(embeddings, store)
        results = await retriever.retrieve("query", filter={"source": "web"})

        assert len(results) == 1
        assert results[0].chunk.metadata["source"] == "web"
