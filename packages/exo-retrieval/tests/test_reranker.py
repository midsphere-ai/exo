"""Tests for Reranker ABC and LLMReranker."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from exo.retrieval.reranker import LLMReranker, Reranker
from exo.retrieval.types import Chunk, RetrievalResult

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


class MockProvider:
    """Mock LLM provider that returns a configurable response."""

    def __init__(self, content: str = "[]") -> None:
        self._content = content

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        from exo.models.types import ModelResponse

        return ModelResponse(content=self._content)

    async def stream(self, messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield  # pragma: no cover


# ---------------------------------------------------------------------------
# Reranker ABC
# ---------------------------------------------------------------------------


class TestRerankerABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            Reranker()  # type: ignore[abstract]

    def test_concrete_subclass(self) -> None:
        class Dummy(Reranker):
            async def rerank(
                self,
                query: str,
                results: list[RetrievalResult],
                *,
                top_k: int = 5,
            ) -> list[RetrievalResult]:
                return results[:top_k]

        d = Dummy()
        assert isinstance(d, Reranker)


# ---------------------------------------------------------------------------
# LLMReranker — basic behaviour
# ---------------------------------------------------------------------------


class TestLLMRerankerBasic:
    @pytest.mark.asyncio
    async def test_empty_results_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reranker = LLMReranker("openai:gpt-4o")
        result = await reranker.rerank("query", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_reorders_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM returns [2, 0, 1] so results should be reordered."""
        provider = MockProvider(content="[2, 0, 1]")
        monkeypatch.setattr(
            "exo.models.get_provider",
            lambda *a, **kw: provider,
        )

        results = [
            _result(index=0, content="first", score=0.9),
            _result(index=1, content="second", score=0.8),
            _result(index=2, content="third", score=0.7),
        ]
        reranker = LLMReranker("openai:gpt-4o")
        reranked = await reranker.rerank("test query", results)

        assert len(reranked) == 3
        assert reranked[0].chunk.content == "third"
        assert reranked[1].chunk.content == "first"
        assert reranked[2].chunk.content == "second"

    @pytest.mark.asyncio
    async def test_rerank_respects_top_k(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = MockProvider(content="[0, 1, 2, 3, 4]")
        monkeypatch.setattr(
            "exo.models.get_provider",
            lambda *a, **kw: provider,
        )

        results = [_result(index=i, content=f"doc-{i}") for i in range(5)]
        reranker = LLMReranker("openai:gpt-4o")
        reranked = await reranker.rerank("query", results, top_k=2)

        assert len(reranked) == 2

    @pytest.mark.asyncio
    async def test_rerank_assigns_descending_scores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = MockProvider(content="[1, 0]")
        monkeypatch.setattr(
            "exo.models.get_provider",
            lambda *a, **kw: provider,
        )

        results = [
            _result(index=0, content="a", score=0.5),
            _result(index=1, content="b", score=0.9),
        ]
        reranker = LLMReranker("openai:gpt-4o")
        reranked = await reranker.rerank("query", results)

        assert reranked[0].score > reranked[1].score

    @pytest.mark.asyncio
    async def test_rerank_preserves_original_score_in_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = MockProvider(content="[0, 1]")
        monkeypatch.setattr(
            "exo.models.get_provider",
            lambda *a, **kw: provider,
        )

        results = [
            _result(index=0, content="a", score=0.7),
            _result(index=1, content="b", score=0.3),
        ]
        reranker = LLMReranker("openai:gpt-4o")
        reranked = await reranker.rerank("query", results)

        assert reranked[0].metadata["original_score"] == 0.7
        assert reranked[1].metadata["original_score"] == 0.3


# ---------------------------------------------------------------------------
# LLMReranker — configuration
# ---------------------------------------------------------------------------


class TestLLMRerankerConfig:
    def test_default_prompt_template(self) -> None:
        reranker = LLMReranker("openai:gpt-4o")
        assert "{query}" in reranker.prompt_template
        assert "{passages}" in reranker.prompt_template

    def test_custom_prompt_template(self) -> None:
        template = "Rank these for {query}: {passages}"
        reranker = LLMReranker("openai:gpt-4o", prompt_template=template)
        assert reranker.prompt_template == template

    def test_model_stored(self) -> None:
        reranker = LLMReranker("anthropic:claude-sonnet-4-20250514")
        assert reranker.model == "anthropic:claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_custom_prompt_template_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify the custom template is actually sent to the LLM."""
        captured_messages: list[Any] = []

        class CapturingProvider:
            async def complete(self, messages: Any, **kwargs: Any) -> Any:
                captured_messages.extend(messages)
                from exo.models.types import ModelResponse

                return ModelResponse(content="[0]")

        monkeypatch.setattr(
            "exo.models.get_provider",
            lambda *a, **kw: CapturingProvider(),
        )

        template = "Custom: {query} | {passages}"
        reranker = LLMReranker("openai:gpt-4o", prompt_template=template)
        await reranker.rerank("my query", [_result(content="passage text")])

        assert len(captured_messages) == 1
        assert "Custom: my query" in captured_messages[0].content
        assert "passage text" in captured_messages[0].content


# ---------------------------------------------------------------------------
# LLMReranker — parse_ranking edge cases
# ---------------------------------------------------------------------------


class TestParseRanking:
    def test_valid_json_array(self) -> None:
        result = LLMReranker._parse_ranking("[2, 0, 1]", 3)
        assert result == [2, 0, 1]

    def test_json_with_surrounding_text(self) -> None:
        result = LLMReranker._parse_ranking("Here is the ranking: [1, 0, 2]", 3)
        assert result == [1, 0, 2]

    def test_invalid_json_falls_back(self) -> None:
        result = LLMReranker._parse_ranking("not json at all", 3)
        assert result == [0, 1, 2]

    def test_out_of_range_indices_falls_back(self) -> None:
        result = LLMReranker._parse_ranking("[5, 10, 20]", 3)
        assert result == [0, 1, 2]

    def test_partial_indices_appends_missing(self) -> None:
        result = LLMReranker._parse_ranking("[2, 0]", 3)
        assert result == [2, 0, 1]

    def test_duplicate_indices_deduped(self) -> None:
        result = LLMReranker._parse_ranking("[1, 1, 0]", 2)
        assert result == [1, 0]

    def test_empty_array_falls_back(self) -> None:
        result = LLMReranker._parse_ranking("[]", 3)
        # Empty valid array → all missing indices appended
        assert result == [0, 1, 2]

    def test_single_result(self) -> None:
        result = LLMReranker._parse_ranking("[0]", 1)
        assert result == [0]
