"""Tests for the Google Vertex AI LLM provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orbiter.config import ModelConfig  # pyright: ignore[reportMissingImports]
from orbiter.models.vertex import (  # pyright: ignore[reportMissingImports]
    VertexProvider,
    _build_config,
    _convert_tools,
    _map_finish_reason,
    _parse_response,
    _parse_stream_chunk,
    _to_google_contents,
)
from orbiter.models.provider import model_registry  # pyright: ignore[reportMissingImports]
from orbiter.models.types import ModelError  # pyright: ignore[reportMissingImports]
from orbiter.types import (  # pyright: ignore[reportMissingImports]
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> ModelConfig:
    defaults: dict[str, Any] = {"provider": "vertex", "model_name": "gemini-2.0-flash"}
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_part(
    *,
    text: str | None = None,
    function_call: Any | None = None,
    thought: bool | None = None,
    thought_signature: bytes | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        function_call=function_call,
        thought=thought,
        thought_signature=thought_signature,
    )


def _make_function_call(
    name: str = "search",
    args: dict[str, Any] | None = None,
    id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(name=name, args=args or {}, id=id)


def _make_response(
    *,
    content: str = "hello",
    parts: list[Any] | None = None,
    finish_reason: str = "STOP",
    prompt_token_count: int = 10,
    candidates_token_count: int = 5,
    total_token_count: int = 15,
) -> SimpleNamespace:
    if parts is None:
        parts = [_make_part(text=content)]
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=parts),
        finish_reason=finish_reason,
    )
    usage_metadata = SimpleNamespace(
        prompt_token_count=prompt_token_count,
        candidates_token_count=candidates_token_count,
        total_token_count=total_token_count,
    )
    return SimpleNamespace(
        candidates=[candidate],
        usage_metadata=usage_metadata,
    )


def _make_stream_chunk(
    *,
    parts: list[Any] | None = None,
    finish_reason: str | None = None,
    usage_metadata: Any = None,
) -> SimpleNamespace:
    if parts is None:
        parts = []
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=parts),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(
        candidates=[candidate],
        usage_metadata=usage_metadata,
    )


def _make_usage_metadata(
    *, prompt_token_count: int = 10, candidates_token_count: int = 5, total_token_count: int = 15
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_count=prompt_token_count,
        candidates_token_count=candidates_token_count,
        total_token_count=total_token_count,
    )


# ---------------------------------------------------------------------------
# Finish reason mapping
# ---------------------------------------------------------------------------


class TestFinishReasonMapping:
    def test_stop(self) -> None:
        assert _map_finish_reason("STOP") == "stop"

    def test_max_tokens(self) -> None:
        assert _map_finish_reason("MAX_TOKENS") == "length"

    def test_safety(self) -> None:
        assert _map_finish_reason("SAFETY") == "content_filter"

    def test_recitation(self) -> None:
        assert _map_finish_reason("RECITATION") == "content_filter"

    def test_blocklist(self) -> None:
        assert _map_finish_reason("BLOCKLIST") == "content_filter"

    def test_malformed_function_call(self) -> None:
        assert _map_finish_reason("MALFORMED_FUNCTION_CALL") == "stop"

    def test_other(self) -> None:
        assert _map_finish_reason("OTHER") == "stop"

    def test_none_defaults_to_stop(self) -> None:
        assert _map_finish_reason(None) == "stop"

    def test_unknown_defaults_to_stop(self) -> None:
        assert _map_finish_reason("unknown_reason") == "stop"


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


class TestToGoogleContents:
    def test_system_extracted(self) -> None:
        contents, system = _to_google_contents([SystemMessage(content="be helpful")])
        assert system == "be helpful"
        assert contents == []

    def test_multiple_system_joined(self) -> None:
        _, system = _to_google_contents(
            [
                SystemMessage(content="rule 1"),
                SystemMessage(content="rule 2"),
            ]
        )
        assert system == "rule 1\nrule 2"

    def test_user_message(self) -> None:
        contents, _ = _to_google_contents([UserMessage(content="hi")])
        assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]

    def test_assistant_with_text(self) -> None:
        contents, _ = _to_google_contents([AssistantMessage(content="hello")])
        assert contents[0]["role"] == "model"
        assert contents[0]["parts"] == [{"text": "hello"}]

    def test_assistant_with_tool_calls(self) -> None:
        msg = AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="tc1", name="search", arguments='{"q":"x"}')],
        )
        contents, _ = _to_google_contents([msg])
        assert contents[0]["role"] == "model"
        parts = contents[0]["parts"]
        assert len(parts) == 1
        assert parts[0]["function_call"]["name"] == "search"
        assert parts[0]["function_call"]["args"] == {"q": "x"}

    def test_assistant_with_text_and_tool_calls(self) -> None:
        msg = AssistantMessage(
            content="Let me search",
            tool_calls=[ToolCall(id="tc1", name="search", arguments='{"q":"x"}')],
        )
        contents, _ = _to_google_contents([msg])
        parts = contents[0]["parts"]
        assert len(parts) == 2
        assert parts[0] == {"text": "Let me search"}
        assert parts[1]["function_call"]["name"] == "search"

    def test_assistant_empty(self) -> None:
        contents, _ = _to_google_contents([AssistantMessage()])
        assert contents[0]["parts"] == [{"text": ""}]

    def test_tool_result(self) -> None:
        contents, _ = _to_google_contents(
            [
                ToolResult(tool_call_id="tc1", tool_name="search", content="found it"),
            ]
        )
        assert contents[0]["role"] == "user"
        fr = contents[0]["parts"][0]["function_response"]
        assert fr["name"] == "search"
        assert fr["response"]["content"] == "found it"

    def test_tool_result_with_error(self) -> None:
        contents, _ = _to_google_contents(
            [
                ToolResult(tool_call_id="tc1", tool_name="search", content="", error="failed"),
            ]
        )
        fr = contents[0]["parts"][0]["function_response"]
        assert fr["response"]["content"] == "failed"


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


class TestConvertTools:
    def test_basic_conversion(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                },
            }
        ]
        result = _convert_tools(tools)
        assert len(result) == 1
        decls = result[0]["function_declarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "search"
        assert decls[0]["description"] == "Search the web"
        assert decls[0]["parameters"]["properties"]["q"]["type"] == "string"

    def test_empty_tools(self) -> None:
        assert _convert_tools([]) == []


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


class TestBuildConfig:
    def test_all_params(self) -> None:
        tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
        config = _build_config(tools, 0.5, 100, "be helpful")
        assert config["system_instruction"] == "be helpful"
        assert config["temperature"] == 0.5
        assert config["max_output_tokens"] == 100
        assert "tools" in config

    def test_no_optional_params(self) -> None:
        config = _build_config(None, None, None, "")
        assert config == {}


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_text_response(self) -> None:
        raw = _make_response(content="Hello!")
        resp = _parse_response(raw, "gemini-2.0-flash")
        assert resp.content == "Hello!"
        assert resp.finish_reason == "stop"
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5
        assert resp.usage.total_tokens == 15

    def test_tool_call_response(self) -> None:
        fc = _make_function_call("search", {"q": "test"}, id="call_0")
        parts = [_make_part(function_call=fc)]
        raw = _make_response(parts=parts, finish_reason="STOP")
        resp = _parse_response(raw, "gemini-2.0-flash")
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[0].arguments == '{"q": "test"}'
        assert resp.tool_calls[0].id == "call_0"

    def test_tool_call_no_id_generates_one(self) -> None:
        fc = _make_function_call("search", {"q": "test"})
        parts = [_make_part(function_call=fc)]
        raw = _make_response(parts=parts)
        resp = _parse_response(raw, "gemini-2.0-flash")
        assert resp.tool_calls[0].id == "call_0"

    def test_no_usage(self) -> None:
        raw = _make_response()
        raw.usage_metadata = None
        resp = _parse_response(raw, "gemini-2.0-flash")
        assert resp.usage.input_tokens == 0

    def test_finish_reason_mapping(self) -> None:
        raw = _make_response(finish_reason="MAX_TOKENS")
        resp = _parse_response(raw, "gemini-2.0-flash")
        assert resp.finish_reason == "length"

    def test_thought_parts_extracted_as_reasoning(self) -> None:
        """Thought parts populate reasoning_content, not content."""
        sig = b"\x01\x02\x03"
        parts = [
            _make_part(text="thinking...", thought=True, thought_signature=sig),
            _make_part(text="Hello!"),
        ]
        raw = _make_response(parts=parts)
        resp = _parse_response(raw, "gemini-2.5-pro")
        assert resp.content == "Hello!"
        assert resp.reasoning_content == "thinking..."
        assert resp.thought_signatures == [sig]

    def test_no_thought_parts(self) -> None:
        """Standard response without thought parts has empty reasoning fields."""
        raw = _make_response(content="Hello!")
        resp = _parse_response(raw, "gemini-2.0-flash")
        assert resp.reasoning_content == ""
        assert resp.thought_signatures == []


# ---------------------------------------------------------------------------
# Stream chunk parsing
# ---------------------------------------------------------------------------


class TestParseStreamChunk:
    def test_text_delta(self) -> None:
        chunk = _make_stream_chunk(parts=[_make_part(text="Hi")])
        result = _parse_stream_chunk(chunk)
        assert result.delta == "Hi"
        assert result.finish_reason is None

    def test_finish_reason(self) -> None:
        chunk = _make_stream_chunk(finish_reason="STOP")
        result = _parse_stream_chunk(chunk)
        assert result.finish_reason == "stop"

    def test_tool_call_delta(self) -> None:
        fc = _make_function_call("search", {"q": "test"}, id="call_0")
        chunk = _make_stream_chunk(parts=[_make_part(function_call=fc)])
        result = _parse_stream_chunk(chunk)
        assert len(result.tool_call_deltas) == 1
        assert result.tool_call_deltas[0].id == "call_0"
        assert result.tool_call_deltas[0].name == "search"

    def test_usage_on_chunk(self) -> None:
        chunk = _make_stream_chunk(
            finish_reason="STOP",
            usage_metadata=_make_usage_metadata(
                prompt_token_count=20, candidates_token_count=30, total_token_count=50
            ),
        )
        result = _parse_stream_chunk(chunk)
        assert result.usage.input_tokens == 20
        assert result.usage.output_tokens == 30

    def test_empty_candidates(self) -> None:
        chunk = SimpleNamespace(candidates=[], usage_metadata=None)
        result = _parse_stream_chunk(chunk)
        assert result.delta == ""
        assert result.usage.input_tokens == 0

    def test_thought_delta(self) -> None:
        """Thought parts in stream chunks produce reasoning_delta, not delta."""
        sig = b"\xab\xcd"
        chunk = _make_stream_chunk(
            parts=[
                _make_part(text="thinking", thought=True, thought_signature=sig),
                _make_part(text="answer"),
            ]
        )
        result = _parse_stream_chunk(chunk)
        assert result.delta == "answer"
        assert result.reasoning_delta == "thinking"
        assert result.thought_signatures == [sig]


# ---------------------------------------------------------------------------
# Thought signature round-trip in message conversion
# ---------------------------------------------------------------------------


class TestThoughtSignatureRoundTrip:
    def test_thought_signatures_prepended_in_contents(self) -> None:
        """AssistantMessage with thought_signatures produces thought parts."""
        sig = b"\x01\x02\x03"
        msg = AssistantMessage(
            content="answer",
            thought_signatures=[sig],
        )
        contents, _ = _to_google_contents([msg])
        parts = contents[0]["parts"]
        assert parts[0] == {"thought": True, "thought_signature": sig}
        assert parts[1] == {"text": "answer"}

    def test_no_signatures_unchanged(self) -> None:
        """AssistantMessage without thought_signatures behaves normally."""
        msg = AssistantMessage(content="hello")
        contents, _ = _to_google_contents([msg])
        parts = contents[0]["parts"]
        assert len(parts) == 1
        assert parts[0] == {"text": "hello"}


# ---------------------------------------------------------------------------
# Provider: complete()
# ---------------------------------------------------------------------------


class TestVertexProviderComplete:
    @patch("orbiter.models.vertex.genai")
    async def test_basic_complete(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = VertexProvider(config)
        raw = _make_response(content="world")
        provider._client = MagicMock()
        provider._client.aio.models.generate_content = AsyncMock(return_value=raw)

        result = await provider.complete([UserMessage(content="hello")])

        assert result.content == "world"
        provider._client.aio.models.generate_content.assert_awaited_once()

    @patch("orbiter.models.vertex.genai")
    async def test_complete_with_tools(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = VertexProvider(config)
        raw = _make_response()
        provider._client = MagicMock()
        provider._client.aio.models.generate_content = AsyncMock(return_value=raw)

        tools = [
            {"type": "function", "function": {"name": "t", "description": "d", "parameters": {}}}
        ]
        await provider.complete([UserMessage(content="hi")], tools=tools)

        call_kwargs = provider._client.aio.models.generate_content.call_args[1]
        assert "tools" in call_kwargs["config"]

    @patch("orbiter.models.vertex.genai")
    async def test_complete_with_temperature_and_max_tokens(
        self, mock_genai: MagicMock
    ) -> None:
        config = _make_config()
        provider = VertexProvider(config)
        raw = _make_response()
        provider._client = MagicMock()
        provider._client.aio.models.generate_content = AsyncMock(return_value=raw)

        await provider.complete([UserMessage(content="hi")], temperature=0.5, max_tokens=100)

        call_kwargs = provider._client.aio.models.generate_content.call_args[1]
        assert call_kwargs["config"]["temperature"] == 0.5
        assert call_kwargs["config"]["max_output_tokens"] == 100

    @patch("orbiter.models.vertex.genai")
    async def test_complete_omits_none_params(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = VertexProvider(config)
        raw = _make_response()
        provider._client = MagicMock()
        provider._client.aio.models.generate_content = AsyncMock(return_value=raw)

        await provider.complete([UserMessage(content="hi")])

        call_kwargs = provider._client.aio.models.generate_content.call_args[1]
        cfg = call_kwargs["config"]
        assert "tools" not in cfg
        assert "temperature" not in cfg
        assert "max_output_tokens" not in cfg

    @patch("orbiter.models.vertex.genai")
    async def test_complete_api_error(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = VertexProvider(config)
        provider._client = MagicMock()
        provider._client.aio.models.generate_content = AsyncMock(
            side_effect=RuntimeError("rate limited")
        )

        with pytest.raises(ModelError, match="rate limited"):
            await provider.complete([UserMessage(content="hi")])

    @patch("orbiter.models.vertex.genai")
    async def test_error_message_prefix(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = VertexProvider(config)
        provider._client = MagicMock()
        provider._client.aio.models.generate_content = AsyncMock(
            side_effect=RuntimeError("fail")
        )

        with pytest.raises(ModelError) as exc_info:
            await provider.complete([UserMessage(content="hi")])
        assert "vertex:" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Provider: stream()
# ---------------------------------------------------------------------------


class TestVertexProviderStream:
    @patch("orbiter.models.vertex.genai")
    async def test_basic_stream(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = VertexProvider(config)

        chunks = [
            _make_stream_chunk(parts=[_make_part(text="Hel")]),
            _make_stream_chunk(parts=[_make_part(text="lo")]),
            _make_stream_chunk(
                finish_reason="STOP",
                usage_metadata=_make_usage_metadata(
                    prompt_token_count=5, candidates_token_count=2, total_token_count=7
                ),
            ),
        ]

        async def mock_stream(**kwargs: Any) -> Any:
            async def gen():
                for c in chunks:
                    yield c

            return gen()

        provider._client = MagicMock()
        provider._client.aio.models.generate_content_stream = AsyncMock(
            side_effect=mock_stream
        )

        collected = []
        async for chunk in provider.stream([UserMessage(content="hi")]):
            collected.append(chunk)

        assert len(collected) == 3
        assert collected[0].delta == "Hel"
        assert collected[1].delta == "lo"
        assert collected[2].finish_reason == "stop"
        assert collected[2].usage.input_tokens == 5

    @patch("orbiter.models.vertex.genai")
    async def test_stream_api_error(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = VertexProvider(config)
        provider._client = MagicMock()
        provider._client.aio.models.generate_content_stream = AsyncMock(
            side_effect=RuntimeError("server error")
        )

        with pytest.raises(ModelError, match="server error"):
            async for _ in provider.stream([UserMessage(content="hi")]):
                pass

    @patch("orbiter.models.vertex.genai")
    async def test_stream_error_prefix(self, mock_genai: MagicMock) -> None:
        config = _make_config()
        provider = VertexProvider(config)
        provider._client = MagicMock()
        provider._client.aio.models.generate_content_stream = AsyncMock(
            side_effect=RuntimeError("fail")
        )

        with pytest.raises(ModelError) as exc_info:
            async for _ in provider.stream([UserMessage(content="hi")]):
                pass
        assert "vertex:" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestVertexRegistration:
    def test_registered(self) -> None:
        assert model_registry.get("vertex") is VertexProvider

    @patch("orbiter.models.vertex.genai")
    def test_get_provider_creates_instance(self, mock_genai: MagicMock) -> None:
        from orbiter.models.provider import get_provider  # pyright: ignore[reportMissingImports]

        provider = get_provider("vertex:gemini-2.0-flash")
        assert isinstance(provider, VertexProvider)
        assert provider.config.model_name == "gemini-2.0-flash"
