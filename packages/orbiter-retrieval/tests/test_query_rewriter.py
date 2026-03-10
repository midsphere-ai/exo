"""Tests for QueryRewriter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from orbiter.retrieval.query_rewriter import QueryRewriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockProvider:
    """Mock LLM provider that returns a configurable response."""

    def __init__(self, content: str = "") -> None:
        self._content = content

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        from orbiter.models.types import ModelResponse

        return ModelResponse(content=self._content)

    async def stream(self, messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield  # pragma: no cover


class CapturingProvider:
    """Mock provider that captures messages sent to it."""

    def __init__(self) -> None:
        self.captured_messages: list[Any] = []

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        self.captured_messages.extend(messages)
        from orbiter.models.types import ModelResponse

        return ModelResponse(content="rewritten query")

    async def stream(self, messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield  # pragma: no cover


# ---------------------------------------------------------------------------
# Construction and configuration
# ---------------------------------------------------------------------------


class TestQueryRewriterConfig:
    def test_model_stored(self) -> None:
        rewriter = QueryRewriter("openai:gpt-4o")
        assert rewriter.model == "openai:gpt-4o"

    def test_default_prompt_template(self) -> None:
        rewriter = QueryRewriter("openai:gpt-4o")
        assert "{query}" in rewriter.prompt_template

    def test_custom_prompt_template(self) -> None:
        template = "Rewrite: {query}"
        rewriter = QueryRewriter("openai:gpt-4o", prompt_template=template)
        assert rewriter.prompt_template == template


# ---------------------------------------------------------------------------
# Basic rewriting
# ---------------------------------------------------------------------------


class TestQueryRewriterRewrite:
    @pytest.mark.asyncio
    async def test_rewrite_returns_llm_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = MockProvider(content="expanded search query with synonyms")
        monkeypatch.setattr(
            "orbiter.models.get_provider",
            lambda *a, **kw: provider,
        )

        rewriter = QueryRewriter("openai:gpt-4o")
        result = await rewriter.rewrite("machine learning")

        assert result == "expanded search query with synonyms"

    @pytest.mark.asyncio
    async def test_rewrite_strips_whitespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = MockProvider(content="  rewritten query  \n")
        monkeypatch.setattr(
            "orbiter.models.get_provider",
            lambda *a, **kw: provider,
        )

        rewriter = QueryRewriter("openai:gpt-4o")
        result = await rewriter.rewrite("test")

        assert result == "rewritten query"

    @pytest.mark.asyncio
    async def test_rewrite_falls_back_on_empty_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = MockProvider(content="   ")
        monkeypatch.setattr(
            "orbiter.models.get_provider",
            lambda *a, **kw: provider,
        )

        rewriter = QueryRewriter("openai:gpt-4o")
        result = await rewriter.rewrite("original query")

        assert result == "original query"

    @pytest.mark.asyncio
    async def test_rewrite_sends_query_in_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        capturing = CapturingProvider()
        monkeypatch.setattr(
            "orbiter.models.get_provider",
            lambda *a, **kw: capturing,
        )

        rewriter = QueryRewriter("openai:gpt-4o")
        await rewriter.rewrite("python web frameworks")

        assert len(capturing.captured_messages) == 1
        assert "python web frameworks" in capturing.captured_messages[0].content


# ---------------------------------------------------------------------------
# History compression
# ---------------------------------------------------------------------------


class TestQueryRewriterHistory:
    @pytest.mark.asyncio
    async def test_history_included_in_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        capturing = CapturingProvider()
        monkeypatch.setattr(
            "orbiter.models.get_provider",
            lambda *a, **kw: capturing,
        )

        rewriter = QueryRewriter("openai:gpt-4o")
        await rewriter.rewrite(
            "what about performance?",
            history=["Tell me about Django", "How does it compare to Flask?"],
        )

        prompt = capturing.captured_messages[0].content
        assert "what about performance?" in prompt
        assert "Django" in prompt
        assert "Flask" in prompt

    @pytest.mark.asyncio
    async def test_history_none_uses_default_template(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        capturing = CapturingProvider()
        monkeypatch.setattr(
            "orbiter.models.get_provider",
            lambda *a, **kw: capturing,
        )

        rewriter = QueryRewriter("openai:gpt-4o")
        await rewriter.rewrite("simple query", history=None)

        prompt = capturing.captured_messages[0].content
        # Should NOT contain history-related instructions
        assert "Conversation history" not in prompt

    @pytest.mark.asyncio
    async def test_custom_template_with_history_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        capturing = CapturingProvider()
        monkeypatch.setattr(
            "orbiter.models.get_provider",
            lambda *a, **kw: capturing,
        )

        template = "Context: {history}\nRewrite: {query}"
        rewriter = QueryRewriter("openai:gpt-4o", prompt_template=template)
        await rewriter.rewrite(
            "the query",
            history=["first message"],
        )

        prompt = capturing.captured_messages[0].content
        assert "Context: - first message" in prompt
        assert "Rewrite: the query" in prompt

    @pytest.mark.asyncio
    async def test_custom_template_without_history_placeholder_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Custom template lacking {history} falls back to built-in history template."""
        capturing = CapturingProvider()
        monkeypatch.setattr(
            "orbiter.models.get_provider",
            lambda *a, **kw: capturing,
        )

        template = "Just rewrite: {query}"
        rewriter = QueryRewriter("openai:gpt-4o", prompt_template=template)
        await rewriter.rewrite(
            "the query",
            history=["some context"],
        )

        prompt = capturing.captured_messages[0].content
        # Falls back to _HISTORY_TEMPLATE which includes history
        assert "some context" in prompt

    @pytest.mark.asyncio
    async def test_empty_history_list_uses_default_template(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        capturing = CapturingProvider()
        monkeypatch.setattr(
            "orbiter.models.get_provider",
            lambda *a, **kw: capturing,
        )

        rewriter = QueryRewriter("openai:gpt-4o")
        await rewriter.rewrite("simple query", history=[])

        prompt = capturing.captured_messages[0].content
        # Empty list is falsy, so default template used
        assert "Conversation history" not in prompt
