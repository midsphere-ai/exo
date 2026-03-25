"""Tests for exo.models.types — provider-agnostic model response types."""

import pytest
from pydantic import ValidationError

from exo.models.types import (  # pyright: ignore[reportMissingImports]
    ModelError,
    ModelResponse,
    StreamChunk,
    ToolCallDelta,
)
from exo.types import ExoError, ToolCall, Usage

# ---------------------------------------------------------------------------
# TestModelError
# ---------------------------------------------------------------------------


class TestModelError:
    def test_is_exo_error(self) -> None:
        err = ModelError("fail")
        assert isinstance(err, ExoError)
        assert isinstance(err, Exception)

    def test_raise_and_catch(self) -> None:
        with pytest.raises(ModelError, match="fail"):
            raise ModelError("fail")

    def test_default_model(self) -> None:
        err = ModelError("timeout")
        assert err.model == ""
        assert str(err) == "timeout"

    def test_custom_model(self) -> None:
        err = ModelError("rate limited", model="openai:gpt-4o")
        assert err.model == "openai:gpt-4o"
        assert "[openai:gpt-4o]" in str(err)
        assert "rate limited" in str(err)


# ---------------------------------------------------------------------------
# TestModelResponse
# ---------------------------------------------------------------------------


class TestModelResponse:
    def test_defaults(self) -> None:
        r = ModelResponse()
        assert r.id == ""
        assert r.model == ""
        assert r.content == ""
        assert r.tool_calls == []
        assert r.usage == Usage()
        assert r.finish_reason == "stop"
        assert r.reasoning_content == ""

    def test_text_response(self) -> None:
        r = ModelResponse(
            id="resp_1",
            model="openai:gpt-4o",
            content="Hello!",
            finish_reason="stop",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        assert r.content == "Hello!"
        assert r.model == "openai:gpt-4o"
        assert r.usage.total_tokens == 15

    def test_tool_call_response(self) -> None:
        tc = ToolCall(id="tc_1", name="search", arguments='{"q": "weather"}')
        r = ModelResponse(
            content="",
            tool_calls=[tc],
            finish_reason="tool_calls",
        )
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "search"
        assert r.finish_reason == "tool_calls"

    def test_reasoning_content(self) -> None:
        r = ModelResponse(
            content="The answer is 42.",
            reasoning_content="Let me think step by step...",
        )
        assert r.reasoning_content == "Let me think step by step..."

    def test_frozen(self) -> None:
        r = ModelResponse(content="hi")
        with pytest.raises(ValidationError):
            r.content = "bye"  # type: ignore[misc]

    def test_independent_lists(self) -> None:
        """Default list factory produces independent instances."""
        r1 = ModelResponse()
        r2 = ModelResponse()
        assert r1.tool_calls is not r2.tool_calls

    def test_roundtrip(self) -> None:
        tc = ToolCall(id="tc_1", name="calc", arguments='{"x": 1}')
        r = ModelResponse(
            id="resp_42",
            model="anthropic:claude-sonnet-4-5-20250929",
            content="done",
            tool_calls=[tc],
            usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150),
            finish_reason="tool_calls",
            reasoning_content="thinking...",
        )
        data = r.model_dump()
        r2 = ModelResponse.model_validate(data)
        assert r == r2

    def test_reuses_core_types(self) -> None:
        """ModelResponse uses ToolCall and Usage from exo-core."""
        r = ModelResponse(
            tool_calls=[ToolCall(id="t1", name="x")],
            usage=Usage(input_tokens=1),
        )
        assert isinstance(r.tool_calls[0], ToolCall)
        assert isinstance(r.usage, Usage)


# ---------------------------------------------------------------------------
# TestToolCallDelta
# ---------------------------------------------------------------------------


class TestToolCallDelta:
    def test_defaults(self) -> None:
        d = ToolCallDelta()
        assert d.index == 0
        assert d.id is None
        assert d.name is None
        assert d.arguments == ""

    def test_create(self) -> None:
        d = ToolCallDelta(
            index=1,
            id="tc_1",
            name="search",
            arguments='{"q": "w',
        )
        assert d.index == 1
        assert d.id == "tc_1"
        assert d.name == "search"
        assert d.arguments == '{"q": "w'

    def test_frozen(self) -> None:
        d = ToolCallDelta(arguments="partial")
        with pytest.raises(ValidationError):
            d.arguments = "changed"  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        d = ToolCallDelta(index=2, id="tc_3", name="run", arguments='{"cmd":')
        data = d.model_dump()
        d2 = ToolCallDelta.model_validate(data)
        assert d == d2


# ---------------------------------------------------------------------------
# TestStreamChunk
# ---------------------------------------------------------------------------


class TestStreamChunk:
    def test_defaults(self) -> None:
        c = StreamChunk()
        assert c.delta == ""
        assert c.tool_call_deltas == []
        assert c.finish_reason is None
        assert c.usage == Usage()

    def test_text_chunk(self) -> None:
        c = StreamChunk(delta="Hello")
        assert c.delta == "Hello"
        assert c.finish_reason is None

    def test_tool_call_chunk(self) -> None:
        td = ToolCallDelta(index=0, id="tc_1", name="search", arguments="")
        c = StreamChunk(tool_call_deltas=[td])
        assert len(c.tool_call_deltas) == 1
        assert c.tool_call_deltas[0].name == "search"

    def test_final_chunk(self) -> None:
        c = StreamChunk(
            finish_reason="stop",
            usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
        )
        assert c.finish_reason == "stop"
        assert c.usage.total_tokens == 30

    def test_frozen(self) -> None:
        c = StreamChunk(delta="hi")
        with pytest.raises(ValidationError):
            c.delta = "bye"  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        td = ToolCallDelta(index=0, arguments='{"x":1}')
        c = StreamChunk(
            delta="partial",
            tool_call_deltas=[td],
            finish_reason="tool_calls",
            usage=Usage(input_tokens=5, output_tokens=10, total_tokens=15),
        )
        data = c.model_dump()
        c2 = StreamChunk.model_validate(data)
        assert c == c2
