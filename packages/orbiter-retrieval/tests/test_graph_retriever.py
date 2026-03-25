"""Tests for GraphRetriever with knowledge graph traversal."""

from __future__ import annotations

from typing import Any

import pytest

from orbiter.retrieval.graph_retriever import GraphRetriever
from orbiter.retrieval.retriever import Retriever
from orbiter.retrieval.triple_extractor import Triple
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


def _triple(
    subject: str,
    predicate: str,
    obj: str,
    confidence: float = 0.9,
    source_chunk_id: str = "doc-1:0",
) -> Triple:
    return Triple(
        subject=subject,
        predicate=predicate,
        object=obj,
        confidence=confidence,
        source_chunk_id=source_chunk_id,
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


# ---------------------------------------------------------------------------
# GraphRetriever — ABC conformance
# ---------------------------------------------------------------------------


class TestGraphRetrieverABC:
    def test_is_retriever_subclass(self) -> None:
        gr = GraphRetriever(StubRetriever([]), [])
        assert isinstance(gr, Retriever)

    def test_default_beam_width(self) -> None:
        gr = GraphRetriever(StubRetriever([]), [])
        assert gr.beam_width == 3

    def test_default_max_hops(self) -> None:
        gr = GraphRetriever(StubRetriever([]), [])
        assert gr.max_hops == 2

    def test_custom_parameters(self) -> None:
        gr = GraphRetriever(
            StubRetriever([]), [], beam_width=5, max_hops=4
        )
        assert gr.beam_width == 5
        assert gr.max_hops == 4


# ---------------------------------------------------------------------------
# GraphRetriever — basic behaviour
# ---------------------------------------------------------------------------


class TestGraphRetrieverBasic:
    @pytest.mark.asyncio
    async def test_empty_base_results(self) -> None:
        """No base results means no graph expansion."""
        gr = GraphRetriever(
            StubRetriever([]),
            [_triple("Python", "is", "language")],
        )
        results = await gr.retrieve("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_no_triples_returns_base(self) -> None:
        """With no triples, base results are returned as-is."""
        base = [_result(content="base")]
        gr = GraphRetriever(StubRetriever(base), [])
        results = await gr.retrieve("q")
        assert len(results) == 1
        assert results[0].chunk.content == "base"

    @pytest.mark.asyncio
    async def test_base_results_preserved(self) -> None:
        """Base results should always appear in output."""
        base = [_result(document_id="doc-1", index=0, content="original", score=1.0)]
        triples = [
            _triple("Python", "is", "language", source_chunk_id="doc-1:0"),
            _triple("language", "used_in", "AI", source_chunk_id="doc-2:0"),
        ]
        gr = GraphRetriever(StubRetriever(base), triples)
        results = await gr.retrieve("q", top_k=10)
        contents = [r.chunk.content for r in results]
        assert "original" in contents


# ---------------------------------------------------------------------------
# GraphRetriever — graph expansion
# ---------------------------------------------------------------------------


class TestGraphRetrieverExpansion:
    @pytest.mark.asyncio
    async def test_one_hop_expansion(self) -> None:
        """Triples linked to base result entities add new chunks."""
        base = [_result(document_id="doc-1", index=0, content="Python is great", score=1.0)]
        triples = [
            # This triple comes from the base result chunk
            _triple("Python", "is", "language", source_chunk_id="doc-1:0"),
            # This triple references "Python" but from a different chunk
            _triple("Python", "created_by", "Guido", confidence=0.8, source_chunk_id="doc-2:0"),
        ]
        gr = GraphRetriever(StubRetriever(base), triples, max_hops=1)
        results = await gr.retrieve("q", top_k=10)

        assert len(results) == 2
        # doc-2:0 should be expanded
        doc_ids = [(r.chunk.document_id, r.chunk.index) for r in results]
        assert ("doc-2", 0) in doc_ids

    @pytest.mark.asyncio
    async def test_expansion_adds_graph_metadata(self) -> None:
        """Expanded results include graph traversal metadata."""
        base = [_result(document_id="doc-1", index=0, content="base", score=1.0)]
        triples = [
            _triple("Python", "is", "language", source_chunk_id="doc-1:0"),
            _triple("Python", "runs_on", "Linux", confidence=0.85, source_chunk_id="doc-3:0"),
        ]
        gr = GraphRetriever(StubRetriever(base), triples, max_hops=1)
        results = await gr.retrieve("q", top_k=10)

        expanded = [r for r in results if r.metadata.get("graph_hop")]
        assert len(expanded) >= 1
        meta = expanded[0].metadata
        assert meta["graph_hop"] == 1
        assert "graph_triple" in meta
        assert "graph_source_entity" in meta

    @pytest.mark.asyncio
    async def test_two_hop_expansion(self) -> None:
        """Entities discovered at hop 1 expand further at hop 2."""
        base = [_result(document_id="doc-1", index=0, content="base", score=1.0)]
        triples = [
            # Hop 0: seed entities from doc-1 → Python, language
            _triple("Python", "is", "language", source_chunk_id="doc-1:0"),
            # Hop 1: "language" connects to "AI" via doc-2
            _triple("language", "used_in", "AI", confidence=0.8, source_chunk_id="doc-2:0"),
            # Hop 2: "AI" connects to "ML" via doc-3
            _triple("AI", "includes", "ML", confidence=0.7, source_chunk_id="doc-3:0"),
        ]
        gr = GraphRetriever(StubRetriever(base), triples, max_hops=2)
        results = await gr.retrieve("q", top_k=10)

        doc_keys = {(r.chunk.document_id, r.chunk.index) for r in results}
        assert ("doc-1", 0) in doc_keys  # base
        assert ("doc-2", 0) in doc_keys  # hop 1
        assert ("doc-3", 0) in doc_keys  # hop 2

    @pytest.mark.asyncio
    async def test_max_hops_limits_depth(self) -> None:
        """Expansion stops at max_hops even if more triples exist."""
        base = [_result(document_id="doc-1", index=0, content="base", score=1.0)]
        triples = [
            _triple("A", "rel", "B", source_chunk_id="doc-1:0"),
            _triple("B", "rel", "C", confidence=0.8, source_chunk_id="doc-2:0"),
            _triple("C", "rel", "D", confidence=0.7, source_chunk_id="doc-3:0"),
            _triple("D", "rel", "E", confidence=0.6, source_chunk_id="doc-4:0"),
        ]
        gr = GraphRetriever(StubRetriever(base), triples, max_hops=1)
        results = await gr.retrieve("q", top_k=10)

        doc_keys = {(r.chunk.document_id, r.chunk.index) for r in results}
        # Only doc-1 (base) and doc-2 (hop 1 via B) should be present
        assert ("doc-1", 0) in doc_keys
        assert ("doc-2", 0) in doc_keys
        assert ("doc-3", 0) not in doc_keys
        assert ("doc-4", 0) not in doc_keys


# ---------------------------------------------------------------------------
# GraphRetriever — beam width
# ---------------------------------------------------------------------------


class TestGraphRetrieverBeamWidth:
    @pytest.mark.asyncio
    async def test_beam_width_limits_triples_per_entity(self) -> None:
        """Only beam_width highest-confidence triples per entity are expanded."""
        base = [_result(document_id="doc-1", index=0, content="base", score=1.0)]
        triples = [
            # Seed triple with low confidence so it doesn't consume beam slots
            _triple("X", "is", "Y", confidence=0.1, source_chunk_id="doc-1:0"),
            # Multiple triples for entity X with different confidence
            _triple("X", "a", "P", confidence=0.9, source_chunk_id="doc-a:0"),
            _triple("X", "b", "Q", confidence=0.8, source_chunk_id="doc-b:0"),
            _triple("X", "c", "R", confidence=0.7, source_chunk_id="doc-c:0"),
            _triple("X", "d", "S", confidence=0.6, source_chunk_id="doc-d:0"),
        ]
        gr = GraphRetriever(
            StubRetriever(base), triples, beam_width=2, max_hops=1
        )
        results = await gr.retrieve("q", top_k=10)

        expanded_keys = {
            (r.chunk.document_id, r.chunk.index)
            for r in results
            if r.metadata.get("graph_hop")
        }
        # beam_width=2: only the 2 highest-confidence triples per entity expand
        # Top 2 for entity X: doc-a (0.9) and doc-b (0.8)
        assert ("doc-a", 0) in expanded_keys
        assert ("doc-b", 0) in expanded_keys
        # doc-d should NOT be expanded (below beam_width cutoff)
        assert ("doc-d", 0) not in expanded_keys


# ---------------------------------------------------------------------------
# GraphRetriever — scoring
# ---------------------------------------------------------------------------


class TestGraphRetrieverScoring:
    @pytest.mark.asyncio
    async def test_base_results_score_higher_than_expanded(self) -> None:
        """Base results with score 1.0 should outrank expanded results."""
        base = [_result(document_id="doc-1", index=0, content="base", score=1.0)]
        triples = [
            _triple("A", "rel", "B", source_chunk_id="doc-1:0"),
            _triple("A", "rel", "C", confidence=0.9, source_chunk_id="doc-2:0"),
        ]
        gr = GraphRetriever(StubRetriever(base), triples, max_hops=1)
        results = await gr.retrieve("q", top_k=10)

        assert results[0].chunk.document_id == "doc-1"
        assert results[0].score > results[1].score

    @pytest.mark.asyncio
    async def test_hop_decay_reduces_scores(self) -> None:
        """Scores decay with each hop."""
        base = [_result(document_id="doc-1", index=0, content="base", score=1.0)]
        triples = [
            _triple("A", "rel", "B", source_chunk_id="doc-1:0"),
            _triple("B", "rel", "C", confidence=0.9, source_chunk_id="doc-2:0"),
            _triple("C", "rel", "D", confidence=0.9, source_chunk_id="doc-3:0"),
        ]
        gr = GraphRetriever(StubRetriever(base), triples, max_hops=2)
        results = await gr.retrieve("q", top_k=10)

        scores_by_doc = {r.chunk.document_id: r.score for r in results}
        # Hop 1 score: 0.9 * 0.8 = 0.72
        # Hop 2 score: 0.9 * 0.64 = 0.576
        assert scores_by_doc["doc-2"] > scores_by_doc["doc-3"]

    @pytest.mark.asyncio
    async def test_results_sorted_by_score(self) -> None:
        """Results are sorted descending by score."""
        base = [_result(document_id="doc-1", index=0, content="base", score=0.5)]
        triples = [
            _triple("A", "rel", "B", source_chunk_id="doc-1:0"),
            _triple("A", "rel", "C", confidence=0.3, source_chunk_id="doc-2:0"),
            _triple("A", "rel", "D", confidence=0.9, source_chunk_id="doc-3:0"),
        ]
        gr = GraphRetriever(StubRetriever(base), triples, max_hops=1)
        results = await gr.retrieve("q", top_k=10)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# GraphRetriever — deduplication
# ---------------------------------------------------------------------------


class TestGraphRetrieverDedup:
    @pytest.mark.asyncio
    async def test_no_duplicate_chunks(self) -> None:
        """Same chunk should not appear multiple times."""
        base = [_result(document_id="doc-1", index=0, content="base", score=1.0)]
        triples = [
            _triple("A", "rel", "B", source_chunk_id="doc-1:0"),
            _triple("B", "rel", "A", confidence=0.8, source_chunk_id="doc-1:0"),
        ]
        gr = GraphRetriever(StubRetriever(base), triples, max_hops=2)
        results = await gr.retrieve("q", top_k=10)

        keys = [(r.chunk.document_id, r.chunk.index) for r in results]
        assert len(keys) == len(set(keys))

    @pytest.mark.asyncio
    async def test_expansion_does_not_replace_base(self) -> None:
        """If expansion finds a chunk already in base results, base version is kept."""
        base = [_result(document_id="doc-1", index=0, content="original", score=1.0)]
        triples = [
            _triple("A", "rel", "B", source_chunk_id="doc-1:0"),
            # This would expand to doc-1:0, but it's already in base
            _triple("B", "rel", "C", confidence=0.5, source_chunk_id="doc-1:0"),
        ]
        gr = GraphRetriever(StubRetriever(base), triples, max_hops=2)
        results = await gr.retrieve("q", top_k=10)

        doc1 = [r for r in results if r.chunk.document_id == "doc-1" and r.chunk.index == 0]
        assert len(doc1) == 1
        assert doc1[0].chunk.content == "original"  # base version kept
        assert doc1[0].score == 1.0


# ---------------------------------------------------------------------------
# GraphRetriever — top_k
# ---------------------------------------------------------------------------


class TestGraphRetrieverTopK:
    @pytest.mark.asyncio
    async def test_respects_top_k(self) -> None:
        """Output is limited to top_k results."""
        base = [_result(document_id="doc-1", index=0, content="base", score=1.0)]
        triples = [
            _triple("A", "rel", "B", source_chunk_id="doc-1:0"),
            _triple("A", "r1", "C", confidence=0.9, source_chunk_id="doc-2:0"),
            _triple("A", "r2", "D", confidence=0.8, source_chunk_id="doc-3:0"),
            _triple("A", "r3", "E", confidence=0.7, source_chunk_id="doc-4:0"),
        ]
        gr = GraphRetriever(StubRetriever(base), triples, max_hops=1)
        results = await gr.retrieve("q", top_k=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_top_k_passed_to_base(self) -> None:
        """top_k is forwarded to the base retriever."""
        base = [
            _result(document_id="doc-1", index=i, content=f"c{i}", score=1.0 - i * 0.1)
            for i in range(5)
        ]
        gr = GraphRetriever(StubRetriever(base), [], max_hops=1)
        results = await gr.retrieve("q", top_k=2)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# GraphRetriever — case insensitivity
# ---------------------------------------------------------------------------


class TestGraphRetrieverCaseInsensitive:
    @pytest.mark.asyncio
    async def test_entity_matching_is_case_insensitive(self) -> None:
        """Entity matching ignores case."""
        base = [_result(document_id="doc-1", index=0, content="base", score=1.0)]
        triples = [
            _triple("python", "is", "Language", source_chunk_id="doc-1:0"),
            _triple("PYTHON", "used_for", "web", confidence=0.8, source_chunk_id="doc-2:0"),
        ]
        gr = GraphRetriever(StubRetriever(base), triples, max_hops=1)
        results = await gr.retrieve("q", top_k=10)

        doc_keys = {(r.chunk.document_id, r.chunk.index) for r in results}
        assert ("doc-2", 0) in doc_keys
