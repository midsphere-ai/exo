"""Tests for AgenticRetriever."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from exo.retrieval.agentic_retriever import AgenticRetriever
from exo.retrieval.query_rewriter import QueryRewriter
from exo.retrieval.retriever import Retriever
from exo.retrieval.types import Chunk, RetrievalResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(doc_id: str = "doc1", index: int = 0, content: str = "text") -> Chunk:
    return Chunk(
        document_id=doc_id,
        index=index,
        content=content,
        start=0,
        end=len(content),
    )


def _result(
    doc_id: str = "doc1",
    index: int = 0,
    content: str = "text",
    score: float = 0.9,
) -> RetrievalResult:
    return RetrievalResult(chunk=_chunk(doc_id, index, content), score=score)


class MockRetriever(Retriever):
    """Returns preconfigured results, optionally varying per call."""

    def __init__(self, results: list[list[RetrievalResult]]) -> None:
        self._results = results
        self.call_count = 0
        self.queries: list[str] = []

    async def retrieve(self, query: str, *, top_k: int = 5, **kwargs: Any) -> list[RetrievalResult]:
        self.queries.append(query)
        idx = min(self.call_count, len(self._results) - 1)
        self.call_count += 1
        return self._results[idx]


class DualMockProvider:
    """Mock provider that returns different responses for rewrite vs judge calls.

    Distinguishes calls by checking if the prompt contains 'sufficiency'
    (judge prompt) vs not (rewriter prompt).
    """

    def __init__(
        self,
        rewrite_responses: list[str],
        judge_responses: list[str],
    ) -> None:
        self._rewrite_responses = rewrite_responses
        self._judge_responses = judge_responses
        self._rewrite_count = 0
        self._judge_count = 0
        self.judge_call_count = 0

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        from exo.models.types import ModelResponse

        prompt = messages[0].content if messages else ""
        if "sufficiency" in prompt.lower() or "retrieval quality judge" in prompt.lower():
            idx = min(self._judge_count, len(self._judge_responses) - 1)
            content = self._judge_responses[idx]
            self._judge_count += 1
            self.judge_call_count += 1
        else:
            idx = min(self._rewrite_count, len(self._rewrite_responses) - 1)
            content = self._rewrite_responses[idx]
            self._rewrite_count += 1
        return ModelResponse(content=content)

    async def stream(self, messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield  # pragma: no cover


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestAgenticRetrieverConfig:
    def test_stores_parameters(self) -> None:
        base = MockRetriever([[]])
        rewriter = QueryRewriter("openai:gpt-4o")
        ar = AgenticRetriever(
            base, rewriter, "openai:gpt-4o", max_rounds=5, sufficiency_threshold=0.8
        )
        assert ar.base_retriever is base
        assert ar.rewriter is rewriter
        assert ar.model == "openai:gpt-4o"
        assert ar.max_rounds == 5
        assert ar.sufficiency_threshold == 0.8

    def test_defaults(self) -> None:
        ar = AgenticRetriever(MockRetriever([[]]), QueryRewriter("openai:gpt-4o"), "openai:gpt-4o")
        assert ar.max_rounds == 3
        assert ar.sufficiency_threshold == 0.7


# ---------------------------------------------------------------------------
# Sufficiency parsing
# ---------------------------------------------------------------------------


class TestSufficiencyParsing:
    def test_parse_json_response(self) -> None:
        content = '{"score": 0.85, "reason": "good results"}'
        assert AgenticRetriever._parse_sufficiency(content) == 0.85

    def test_parse_json_with_surrounding_text(self) -> None:
        content = 'Here is my assessment:\n{"score": 0.6, "reason": "partial"}\nDone.'
        assert AgenticRetriever._parse_sufficiency(content) == 0.6

    def test_clamps_above_one(self) -> None:
        content = '{"score": 1.5, "reason": "over"}'
        assert AgenticRetriever._parse_sufficiency(content) == 1.0

    def test_clamps_below_zero(self) -> None:
        content = '{"score": -0.3, "reason": "under"}'
        assert AgenticRetriever._parse_sufficiency(content) == 0.0

    def test_fallback_bare_float(self) -> None:
        content = "The sufficiency score is 0.75"
        assert AgenticRetriever._parse_sufficiency(content) == 0.75

    def test_fallback_unparseable_returns_zero(self) -> None:
        content = "I cannot evaluate this."
        assert AgenticRetriever._parse_sufficiency(content) == 0.0

    def test_invalid_json_falls_back(self) -> None:
        content = "{score: broken}"
        assert AgenticRetriever._parse_sufficiency(content) == 0.0


# ---------------------------------------------------------------------------
# Retrieve — early stop on sufficient results
# ---------------------------------------------------------------------------


class TestAgenticRetrieverRetrieve:
    @pytest.mark.asyncio
    async def test_stops_on_sufficient_first_round(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should stop after round 1 when sufficiency >= threshold."""
        results = [_result("d1", 0, "answer", 0.95)]
        base = MockRetriever([results])

        provider = DualMockProvider(
            rewrite_responses=["rewritten query"],
            judge_responses=['{"score": 0.9, "reason": "sufficient"}'],
        )
        monkeypatch.setattr("exo.models.get_provider", lambda *a, **kw: provider)

        rewriter = QueryRewriter("openai:gpt-4o")
        ar = AgenticRetriever(base, rewriter, "openai:gpt-4o")
        out = await ar.retrieve("test query")

        assert len(out) == 1
        assert out[0].chunk.document_id == "d1"
        assert base.call_count == 1  # Only 1 round

    @pytest.mark.asyncio
    async def test_retries_on_insufficient_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should retry when sufficiency is below threshold."""
        round1 = [_result("d1", 0, "partial answer", 0.5)]
        round2 = [_result("d1", 0, "partial answer", 0.5), _result("d2", 0, "better", 0.8)]
        base = MockRetriever([round1, round2])

        provider = DualMockProvider(
            rewrite_responses=["rewritten v1", "rewritten v2"],
            judge_responses=[
                '{"score": 0.3, "reason": "insufficient"}',
                '{"score": 0.9, "reason": "sufficient"}',
            ],
        )
        monkeypatch.setattr("exo.models.get_provider", lambda *a, **kw: provider)

        rewriter = QueryRewriter("openai:gpt-4o")
        ar = AgenticRetriever(base, rewriter, "openai:gpt-4o")
        out = await ar.retrieve("test query")

        assert base.call_count == 2
        assert len(out) == 2  # Deduplicated results from both rounds

    @pytest.mark.asyncio
    async def test_max_rounds_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should stop after max_rounds even if never sufficient."""
        results = [_result("d1", 0, "answer")]
        base = MockRetriever([results])

        provider = DualMockProvider(
            rewrite_responses=["rewritten"],
            judge_responses=['{"score": 0.1, "reason": "bad"}'],
        )
        monkeypatch.setattr("exo.models.get_provider", lambda *a, **kw: provider)

        rewriter = QueryRewriter("openai:gpt-4o")
        ar = AgenticRetriever(base, rewriter, "openai:gpt-4o", max_rounds=2)
        out = await ar.retrieve("test query")

        assert base.call_count == 2
        assert len(out) == 1

    @pytest.mark.asyncio
    async def test_deduplication_keeps_highest_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same chunk appearing in multiple rounds keeps highest score."""
        round1 = [_result("d1", 0, "text", 0.5)]
        round2 = [_result("d1", 0, "text", 0.9)]
        base = MockRetriever([round1, round2])

        provider = DualMockProvider(
            rewrite_responses=["rewritten"],
            judge_responses=[
                '{"score": 0.3, "reason": "retry"}',
                '{"score": 0.9, "reason": "ok"}',
            ],
        )
        monkeypatch.setattr("exo.models.get_provider", lambda *a, **kw: provider)

        rewriter = QueryRewriter("openai:gpt-4o")
        ar = AgenticRetriever(base, rewriter, "openai:gpt-4o", max_rounds=2)
        out = await ar.retrieve("test")

        assert len(out) == 1
        assert out[0].score == 0.9  # Kept the higher score

    @pytest.mark.asyncio
    async def test_top_k_limits_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Output should be limited to top_k results."""
        results = [_result(f"d{i}", 0, f"text {i}", 1.0 - i * 0.1) for i in range(5)]
        base = MockRetriever([results])

        provider = DualMockProvider(
            rewrite_responses=["rewritten"],
            judge_responses=['{"score": 0.9, "reason": "good"}'],
        )
        monkeypatch.setattr("exo.models.get_provider", lambda *a, **kw: provider)

        rewriter = QueryRewriter("openai:gpt-4o")
        ar = AgenticRetriever(base, rewriter, "openai:gpt-4o")
        out = await ar.retrieve("test", top_k=3)

        assert len(out) == 3
        # Sorted by score descending
        assert out[0].score >= out[1].score >= out[2].score

    @pytest.mark.asyncio
    async def test_empty_retrieval_no_judge_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When base retriever returns nothing, judge is not called."""
        base = MockRetriever([[]])

        provider = DualMockProvider(
            rewrite_responses=["rewritten"],
            judge_responses=["should not be called"],
        )
        monkeypatch.setattr("exo.models.get_provider", lambda *a, **kw: provider)

        rewriter = QueryRewriter("openai:gpt-4o")
        ar = AgenticRetriever(base, rewriter, "openai:gpt-4o")
        out = await ar.retrieve("test")

        assert out == []
        assert provider.judge_call_count == 0

    @pytest.mark.asyncio
    async def test_results_sorted_by_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Results across rounds should be sorted by score descending."""
        round1 = [_result("d1", 0, "low", 0.3), _result("d2", 0, "mid", 0.6)]
        round2 = [_result("d3", 0, "high", 0.95)]
        base = MockRetriever([round1, round2])

        provider = DualMockProvider(
            rewrite_responses=["r1", "r2"],
            judge_responses=[
                '{"score": 0.4, "reason": "retry"}',
                '{"score": 0.9, "reason": "ok"}',
            ],
        )
        monkeypatch.setattr("exo.models.get_provider", lambda *a, **kw: provider)

        rewriter = QueryRewriter("openai:gpt-4o")
        ar = AgenticRetriever(base, rewriter, "openai:gpt-4o", max_rounds=2)
        out = await ar.retrieve("test")

        assert len(out) == 3
        assert out[0].score == 0.95
        assert out[1].score == 0.6
        assert out[2].score == 0.3

    @pytest.mark.asyncio
    async def test_custom_sufficiency_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Custom threshold should be respected."""
        results = [_result("d1", 0, "text")]
        base = MockRetriever([results])

        provider = DualMockProvider(
            rewrite_responses=["rewritten"],
            # Score of 0.5 is below default 0.7 but above custom 0.4
            judge_responses=['{"score": 0.5, "reason": "ok for low bar"}'],
        )
        monkeypatch.setattr("exo.models.get_provider", lambda *a, **kw: provider)

        rewriter = QueryRewriter("openai:gpt-4o")
        ar = AgenticRetriever(base, rewriter, "openai:gpt-4o", sufficiency_threshold=0.4)
        out = await ar.retrieve("test")

        assert base.call_count == 1  # Stopped after 1 round
        assert len(out) == 1
