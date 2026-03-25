"""Tests for Triple dataclass and TripleExtractor."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from orbiter.retrieval.triple_extractor import Triple, TripleExtractor
from orbiter.retrieval.types import Chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    document_id: str = "doc-1",
    index: int = 0,
    content: str = "hello",
) -> Chunk:
    """Build a Chunk with sensible defaults."""
    return Chunk(
        document_id=document_id,
        index=index,
        content=content,
        start=0,
        end=len(content),
    )


class MockProvider:
    """Mock LLM provider that returns a configurable response."""

    def __init__(self, content: str = "[]") -> None:
        self._content = content

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        from orbiter.models.types import ModelResponse

        return ModelResponse(content=self._content)

    async def stream(self, messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield  # pragma: no cover


class MultiResponseProvider:
    """Mock provider that returns different responses per call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        from orbiter.models.types import ModelResponse

        content = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return ModelResponse(content=content)

    async def stream(self, messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield  # pragma: no cover


# ---------------------------------------------------------------------------
# Triple dataclass
# ---------------------------------------------------------------------------


class TestTriple:
    def test_creation(self) -> None:
        t = Triple(
            subject="Python",
            predicate="is",
            object="language",
            confidence=0.9,
            source_chunk_id="doc-1:0",
        )
        assert t.subject == "Python"
        assert t.predicate == "is"
        assert t.object == "language"
        assert t.confidence == 0.9
        assert t.source_chunk_id == "doc-1:0"

    def test_frozen(self) -> None:
        t = Triple(
            subject="A",
            predicate="B",
            object="C",
            confidence=1.0,
            source_chunk_id="doc-1:0",
        )
        with pytest.raises(AttributeError):
            t.subject = "X"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = Triple("A", "B", "C", 0.5, "doc:0")
        b = Triple("A", "B", "C", 0.5, "doc:0")
        assert a == b

    def test_inequality(self) -> None:
        a = Triple("A", "B", "C", 0.5, "doc:0")
        b = Triple("X", "B", "C", 0.5, "doc:0")
        assert a != b


# ---------------------------------------------------------------------------
# TripleExtractor — configuration
# ---------------------------------------------------------------------------


class TestTripleExtractorConfig:
    def test_default_prompt_template(self) -> None:
        extractor = TripleExtractor("openai:gpt-4o")
        assert "{text}" in extractor.prompt_template

    def test_custom_prompt_template(self) -> None:
        template = "Extract triples from: {text}"
        extractor = TripleExtractor("openai:gpt-4o", prompt_template=template)
        assert extractor.prompt_template == template

    def test_model_stored(self) -> None:
        extractor = TripleExtractor("anthropic:claude-sonnet-4-20250514")
        assert extractor.model == "anthropic:claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# TripleExtractor — extraction
# ---------------------------------------------------------------------------


class TestTripleExtractorExtract:
    @pytest.mark.asyncio
    async def test_empty_chunks_returns_empty(self) -> None:
        extractor = TripleExtractor("openai:gpt-4o")
        result = await extractor.extract([])
        assert result == []

    @pytest.mark.asyncio
    async def test_extracts_triples_from_chunk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = '[{"subject": "Python", "predicate": "is", "object": "language", "confidence": 0.95}]'
        provider = MockProvider(content=response)
        monkeypatch.setattr(
            "orbiter.models.get_provider",
            lambda *a, **kw: provider,
        )

        chunk = _chunk(content="Python is a programming language.")
        extractor = TripleExtractor("openai:gpt-4o")
        triples = await extractor.extract([chunk])

        assert len(triples) == 1
        assert triples[0].subject == "Python"
        assert triples[0].predicate == "is"
        assert triples[0].object == "language"
        assert triples[0].confidence == 0.95
        assert triples[0].source_chunk_id == "doc-1:0"

    @pytest.mark.asyncio
    async def test_extracts_multiple_triples(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = """[
            {"subject": "A", "predicate": "relates", "object": "B", "confidence": 0.9},
            {"subject": "C", "predicate": "causes", "object": "D", "confidence": 0.8}
        ]"""
        provider = MockProvider(content=response)
        monkeypatch.setattr(
            "orbiter.models.get_provider",
            lambda *a, **kw: provider,
        )

        extractor = TripleExtractor("openai:gpt-4o")
        triples = await extractor.extract([_chunk()])

        assert len(triples) == 2
        assert triples[0].subject == "A"
        assert triples[1].subject == "C"

    @pytest.mark.asyncio
    async def test_multiple_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        responses = [
            '[{"subject": "X", "predicate": "r1", "object": "Y", "confidence": 0.9}]',
            '[{"subject": "M", "predicate": "r2", "object": "N", "confidence": 0.8}]',
        ]
        provider = MultiResponseProvider(responses)
        monkeypatch.setattr(
            "orbiter.models.get_provider",
            lambda *a, **kw: provider,
        )

        chunks = [
            _chunk(document_id="doc-1", index=0, content="text one"),
            _chunk(document_id="doc-2", index=3, content="text two"),
        ]
        extractor = TripleExtractor("openai:gpt-4o")
        triples = await extractor.extract(chunks)

        assert len(triples) == 2
        assert triples[0].source_chunk_id == "doc-1:0"
        assert triples[1].source_chunk_id == "doc-2:3"

    @pytest.mark.asyncio
    async def test_source_chunk_id_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = '[{"subject": "A", "predicate": "B", "object": "C", "confidence": 0.5}]'
        provider = MockProvider(content=response)
        monkeypatch.setattr(
            "orbiter.models.get_provider",
            lambda *a, **kw: provider,
        )

        chunk = _chunk(document_id="my-doc", index=7)
        extractor = TripleExtractor("openai:gpt-4o")
        triples = await extractor.extract([chunk])

        assert triples[0].source_chunk_id == "my-doc:7"

    @pytest.mark.asyncio
    async def test_custom_template_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_messages: list[Any] = []

        class CapturingProvider:
            async def complete(self, messages: Any, **kwargs: Any) -> Any:
                captured_messages.extend(messages)
                from orbiter.models.types import ModelResponse

                return ModelResponse(content="[]")

        monkeypatch.setattr(
            "orbiter.models.get_provider",
            lambda *a, **kw: CapturingProvider(),
        )

        template = "Custom extraction: {text}"
        extractor = TripleExtractor("openai:gpt-4o", prompt_template=template)
        await extractor.extract([_chunk(content="sample text")])

        assert len(captured_messages) == 1
        assert "Custom extraction: sample text" in captured_messages[0].content


# ---------------------------------------------------------------------------
# TripleExtractor — parse_triples edge cases
# ---------------------------------------------------------------------------


class TestParseTriples:
    def test_valid_json(self) -> None:
        content = '[{"subject": "A", "predicate": "B", "object": "C", "confidence": 0.9}]'
        triples = TripleExtractor._parse_triples(content, "doc:0")
        assert len(triples) == 1
        assert triples[0].subject == "A"

    def test_json_with_surrounding_text(self) -> None:
        content = 'Here are the triples: [{"subject": "A", "predicate": "B", "object": "C", "confidence": 0.8}] done'
        triples = TripleExtractor._parse_triples(content, "doc:0")
        assert len(triples) == 1

    def test_no_json_returns_empty(self) -> None:
        triples = TripleExtractor._parse_triples("no json here", "doc:0")
        assert triples == []

    def test_invalid_json_returns_empty(self) -> None:
        triples = TripleExtractor._parse_triples("[not valid json]", "doc:0")
        assert triples == []

    def test_missing_required_fields_skipped(self) -> None:
        content = '[{"subject": "A", "predicate": "B"}, {"subject": "X", "predicate": "Y", "object": "Z", "confidence": 0.5}]'
        triples = TripleExtractor._parse_triples(content, "doc:0")
        assert len(triples) == 1
        assert triples[0].subject == "X"

    def test_non_string_fields_skipped(self) -> None:
        content = '[{"subject": 123, "predicate": "B", "object": "C", "confidence": 0.5}]'
        triples = TripleExtractor._parse_triples(content, "doc:0")
        assert triples == []

    def test_default_confidence(self) -> None:
        content = '[{"subject": "A", "predicate": "B", "object": "C"}]'
        triples = TripleExtractor._parse_triples(content, "doc:0")
        assert len(triples) == 1
        assert triples[0].confidence == 0.5

    def test_confidence_clamped_high(self) -> None:
        content = '[{"subject": "A", "predicate": "B", "object": "C", "confidence": 5.0}]'
        triples = TripleExtractor._parse_triples(content, "doc:0")
        assert triples[0].confidence == 1.0

    def test_confidence_clamped_low(self) -> None:
        content = '[{"subject": "A", "predicate": "B", "object": "C", "confidence": -0.5}]'
        triples = TripleExtractor._parse_triples(content, "doc:0")
        assert triples[0].confidence == 0.0

    def test_non_numeric_confidence_defaults(self) -> None:
        content = '[{"subject": "A", "predicate": "B", "object": "C", "confidence": "high"}]'
        triples = TripleExtractor._parse_triples(content, "doc:0")
        assert triples[0].confidence == 0.5

    def test_non_dict_items_skipped(self) -> None:
        content = '["not a dict", {"subject": "A", "predicate": "B", "object": "C", "confidence": 0.7}]'
        triples = TripleExtractor._parse_triples(content, "doc:0")
        assert len(triples) == 1

    def test_empty_array(self) -> None:
        triples = TripleExtractor._parse_triples("[]", "doc:0")
        assert triples == []

    def test_source_chunk_id_assigned(self) -> None:
        content = '[{"subject": "A", "predicate": "B", "object": "C", "confidence": 0.5}]'
        triples = TripleExtractor._parse_triples(content, "my-doc:42")
        assert triples[0].source_chunk_id == "my-doc:42"
