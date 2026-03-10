"""Tests for HybridRetriever with RRF fusion."""

from __future__ import annotations

from typing import Any

import pytest

from orbiter.retrieval.hybrid_retriever import HybridRetriever
from orbiter.retrieval.retriever import Retriever
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


class StubRetriever(Retriever):
    """Returns a fixed list of results for any query."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        return self._results[:top_k]


def _result(
    document_id: str = "doc-1",
    index: int = 0,
    content: str = "hello",
    score: float = 1.0,
) -> RetrievalResult:
    return RetrievalResult(
        chunk=_chunk(document_id=document_id, index=index, content=content),
        score=score,
    )


# ---------------------------------------------------------------------------
# HybridRetriever — basic behaviour
# ---------------------------------------------------------------------------


class TestHybridRetrieverBasic:
    @pytest.mark.asyncio
    async def test_is_retriever_subclass(self) -> None:
        hybrid = HybridRetriever(
            StubRetriever([]), StubRetriever([])
        )
        assert isinstance(hybrid, Retriever)

    @pytest.mark.asyncio
    async def test_empty_retrievers_return_empty(self) -> None:
        hybrid = HybridRetriever(
            StubRetriever([]), StubRetriever([])
        )
        results = await hybrid.retrieve("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_vector_only_results(self) -> None:
        """When sparse returns nothing, only vector results appear."""
        vec_results = [_result(index=0, content="a"), _result(index=1, content="b")]
        hybrid = HybridRetriever(
            StubRetriever(vec_results), StubRetriever([])
        )
        results = await hybrid.retrieve("q")
        assert len(results) == 2
        assert results[0].chunk.content == "a"
        assert results[1].chunk.content == "b"

    @pytest.mark.asyncio
    async def test_sparse_only_results(self) -> None:
        """When vector returns nothing, only sparse results appear."""
        sp_results = [_result(index=0, content="x"), _result(index=1, content="y")]
        hybrid = HybridRetriever(
            StubRetriever([]), StubRetriever(sp_results)
        )
        results = await hybrid.retrieve("q")
        assert len(results) == 2
        assert results[0].chunk.content == "x"

    @pytest.mark.asyncio
    async def test_respects_top_k(self) -> None:
        vec = [_result(index=i, content=f"v{i}") for i in range(5)]
        sp = [_result(index=i + 10, content=f"s{i}") for i in range(5)]
        hybrid = HybridRetriever(StubRetriever(vec), StubRetriever(sp))
        results = await hybrid.retrieve("q", top_k=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# HybridRetriever — RRF fusion ranking
# ---------------------------------------------------------------------------


class TestHybridRetrieverFusion:
    @pytest.mark.asyncio
    async def test_shared_result_scores_higher(self) -> None:
        """A chunk present in both lists should score higher than one in only one."""
        shared = _result(document_id="shared", index=0, content="shared")
        vec_only = _result(document_id="vec", index=0, content="vec-only")
        sp_only = _result(document_id="sp", index=0, content="sp-only")

        hybrid = HybridRetriever(
            StubRetriever([shared, vec_only]),
            StubRetriever([shared, sp_only]),
        )
        results = await hybrid.retrieve("q", top_k=10)

        # The shared chunk should be ranked first since it gets RRF from both.
        assert results[0].chunk.content == "shared"

    @pytest.mark.asyncio
    async def test_rrf_scores_are_correct(self) -> None:
        """Verify the exact RRF scores for a known configuration."""
        k = 60
        # Rank 1 in vector, rank 1 in sparse
        a = _result(document_id="a", index=0, content="a")
        # Rank 2 in vector only
        b = _result(document_id="b", index=0, content="b")
        # Rank 2 in sparse only
        c = _result(document_id="c", index=0, content="c")

        hybrid = HybridRetriever(
            StubRetriever([a, b]),
            StubRetriever([a, c]),
            k=k,
            vector_weight=0.5,
        )
        results = await hybrid.retrieve("q", top_k=10)

        expected_a = 0.5 * (1 / (k + 1)) + 0.5 * (1 / (k + 1))
        expected_b = 0.5 * (1 / (k + 2))
        expected_c = 0.5 * (1 / (k + 2))

        assert abs(results[0].score - expected_a) < 1e-10
        assert results[0].chunk.content == "a"

        # b and c have equal scores; just verify the values match
        b_or_c = {r.chunk.content: r.score for r in results[1:]}
        assert abs(b_or_c["b"] - expected_b) < 1e-10
        assert abs(b_or_c["c"] - expected_c) < 1e-10

    @pytest.mark.asyncio
    async def test_vector_weight_favours_vector(self) -> None:
        """Higher vector_weight makes vector-only results rank above sparse-only."""
        vec_item = _result(document_id="vec", index=0, content="vec-item")
        sp_item = _result(document_id="sp", index=0, content="sp-item")

        hybrid = HybridRetriever(
            StubRetriever([vec_item]),
            StubRetriever([sp_item]),
            vector_weight=0.9,
        )
        results = await hybrid.retrieve("q", top_k=10)

        assert results[0].chunk.content == "vec-item"
        assert results[1].chunk.content == "sp-item"
        assert results[0].score > results[1].score

    @pytest.mark.asyncio
    async def test_low_vector_weight_favours_sparse(self) -> None:
        """Lower vector_weight makes sparse-only results rank above vector-only."""
        vec_item = _result(document_id="vec", index=0, content="vec-item")
        sp_item = _result(document_id="sp", index=0, content="sp-item")

        hybrid = HybridRetriever(
            StubRetriever([vec_item]),
            StubRetriever([sp_item]),
            vector_weight=0.1,
        )
        results = await hybrid.retrieve("q", top_k=10)

        assert results[0].chunk.content == "sp-item"
        assert results[1].chunk.content == "vec-item"

    @pytest.mark.asyncio
    async def test_custom_k_changes_scores(self) -> None:
        """Different k values produce different absolute scores."""
        a = _result(document_id="a", index=0, content="a")
        k_small = HybridRetriever(
            StubRetriever([a]), StubRetriever([a]), k=10
        )
        k_large = HybridRetriever(
            StubRetriever([a]), StubRetriever([a]), k=100
        )

        results_small = await k_small.retrieve("q")
        results_large = await k_large.retrieve("q")

        # Smaller k → larger scores (rank matters more)
        assert results_small[0].score > results_large[0].score


# ---------------------------------------------------------------------------
# HybridRetriever — deduplication
# ---------------------------------------------------------------------------


class TestHybridRetrieverDedup:
    @pytest.mark.asyncio
    async def test_duplicate_chunks_merged(self) -> None:
        """Same chunk from both retrievers should appear only once."""
        chunk = _result(document_id="doc-1", index=0, content="same")

        hybrid = HybridRetriever(
            StubRetriever([chunk]),
            StubRetriever([chunk]),
        )
        results = await hybrid.retrieve("q", top_k=10)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_different_doc_ids_not_merged(self) -> None:
        """Chunks with different document_ids are distinct."""
        a = _result(document_id="doc-a", index=0, content="same-text")
        b = _result(document_id="doc-b", index=0, content="same-text")

        hybrid = HybridRetriever(
            StubRetriever([a]),
            StubRetriever([b]),
        )
        results = await hybrid.retrieve("q", top_k=10)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_different_indices_not_merged(self) -> None:
        """Chunks with different indices from the same doc are distinct."""
        a = _result(document_id="doc-1", index=0, content="part-a")
        b = _result(document_id="doc-1", index=1, content="part-b")

        hybrid = HybridRetriever(
            StubRetriever([a]),
            StubRetriever([b]),
        )
        results = await hybrid.retrieve("q", top_k=10)

        assert len(results) == 2


# ---------------------------------------------------------------------------
# HybridRetriever — defaults
# ---------------------------------------------------------------------------


class TestHybridRetrieverDefaults:
    def test_default_k(self) -> None:
        hybrid = HybridRetriever(StubRetriever([]), StubRetriever([]))
        assert hybrid.k == 60

    def test_default_vector_weight(self) -> None:
        hybrid = HybridRetriever(StubRetriever([]), StubRetriever([]))
        assert hybrid.vector_weight == 0.5
