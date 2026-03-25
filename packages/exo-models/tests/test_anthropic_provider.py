"""Tests for the Anthropic LLM provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from exo.config import ModelConfig  # pyright: ignore[reportMissingImports]
from exo.models.anthropic import (  # pyright: ignore[reportMissingImports]
    _DEFAULT_MAX_TOKENS,
    AnthropicProvider,
    _build_messages,
    _convert_tools,
    _map_stop_reason,
    _parse_response,
)
from exo.models.provider import model_registry  # pyright: ignore[reportMissingImports]
from exo.models.types import ModelError  # pyright: ignore[reportMissingImports]
from exo.types import (  # pyright: ignore[reportMissingImports]
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
    defaults: dict[str, Any] = {"provider": "anthropic", "model_name": "claude-sonnet-4-5-20250929"}
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_text_block(text: str = "hello") -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _make_tool_use_block(
    id: str = "tu1", name: str = "search", input: dict[str, Any] | None = None
) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input or {})


def _make_thinking_block(thinking: str = "let me think") -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking=thinking)


def _make_response(
    *,
    content: list[Any] | None = None,
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 5,
    response_id: str = "msg-abc",
    model: str = "claude-sonnet-4-5-20250929",
) -> SimpleNamespace:
    if content is None:
        content = [_make_text_block()]
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(
        id=response_id,
        model=model,
        content=content,
        stop_reason=stop_reason,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Stop reason mapping
# ---------------------------------------------------------------------------


class TestStopReasonMapping:
    def test_end_turn(self) -> None:
        assert _map_stop_reason("end_turn") == "stop"

    def test_tool_use(self) -> None:
        assert _map_stop_reason("tool_use") == "tool_calls"

    def test_max_tokens(self) -> None:
        assert _map_stop_reason("max_tokens") == "length"

    def test_stop_sequence(self) -> None:
        assert _map_stop_reason("stop_sequence") == "stop"

    def test_none_defaults_to_stop(self) -> None:
        assert _map_stop_reason(None) == "stop"

    def test_unknown_defaults_to_stop(self) -> None:
        assert _map_stop_reason("unknown") == "stop"


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_system_extracted(self) -> None:
        system, msgs = _build_messages([SystemMessage(content="be helpful")])
        assert system == "be helpful"
        assert msgs == []

    def test_multiple_system_joined(self) -> None:
        system, _msgs = _build_messages(
            [
                SystemMessage(content="rule 1"),
                SystemMessage(content="rule 2"),
            ]
        )
        assert system == "rule 1\nrule 2"

    def test_user_message(self) -> None:
        _, msgs = _build_messages([UserMessage(content="hi")])
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_assistant_with_text(self) -> None:
        _, msgs = _build_messages([AssistantMessage(content="hello")])
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] == [{"type": "text", "text": "hello"}]

    def test_assistant_with_tool_calls(self) -> None:
        msg = AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="tc1", name="search", arguments='{"q":"x"}')],
        )
        _, msgs = _build_messages([msg])
        content = msgs[0]["content"]
        # Empty text block + tool_use block
        assert len(content) == 1
        assert content[0]["type"] == "tool_use"
        assert content[0]["input"] == {"q": "x"}

    def test_assistant_with_text_and_tool_calls(self) -> None:
        msg = AssistantMessage(
            content="Let me search",
            tool_calls=[ToolCall(id="tc1", name="search", arguments='{"q":"x"}')],
        )
        _, msgs = _build_messages([msg])
        content = msgs[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "tool_use"

    def test_assistant_empty(self) -> None:
        _, msgs = _build_messages([AssistantMessage()])
        assert msgs[0]["content"] == [{"type": "text", "text": ""}]

    def test_tool_result(self) -> None:
        _, msgs = _build_messages(
            [
                ToolResult(tool_call_id="tc1", tool_name="search", content="found it"),
            ]
        )
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][0]["type"] == "tool_result"
        assert msgs[0]["content"][0]["content"] == "found it"

    def test_tool_result_with_error(self) -> None:
        _, msgs = _build_messages(
            [
                ToolResult(tool_call_id="tc1", tool_name="search", error="failed"),
            ]
        )
        block = msgs[0]["content"][0]
        assert block["content"] == "failed"
        assert block["is_error"] is True

    def test_consecutive_tool_results_merged(self) -> None:
        _, msgs = _build_messages(
            [
                ToolResult(tool_call_id="tc1", tool_name="search", content="r1"),
                ToolResult(tool_call_id="tc2", tool_name="calc", content="r2"),
            ]
        )
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert len(msgs[0]["content"]) == 2

    def test_tool_result_after_user_not_merged(self) -> None:
        _, msgs = _build_messages(
            [
                UserMessage(content="hi"),
                ToolResult(tool_call_id="tc1", tool_name="search", content="r1"),
            ]
        )
        assert len(msgs) == 2


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
        assert result[0]["name"] == "search"
        assert result[0]["description"] == "Search the web"
        assert result[0]["input_schema"]["properties"]["q"]["type"] == "string"

    def test_empty_tools(self) -> None:
        assert _convert_tools([]) == []


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_text_response(self) -> None:
        raw = _make_response(content=[_make_text_block("Hello!")])
        resp = _parse_response(raw, "claude-sonnet-4-5-20250929")
        assert resp.content == "Hello!"
        assert resp.finish_reason == "stop"
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5
        assert resp.usage.total_tokens == 15

    def test_tool_use_response(self) -> None:
        raw = _make_response(
            content=[_make_tool_use_block("tu1", "search", {"q": "test"})],
            stop_reason="tool_use",
        )
        resp = _parse_response(raw, "claude-sonnet-4-5-20250929")
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[0].arguments == '{"q": "test"}'
        assert resp.finish_reason == "tool_calls"

    def test_mixed_text_and_tools(self) -> None:
        raw = _make_response(
            content=[
                _make_text_block("Let me search"),
                _make_tool_use_block("tu1", "search", {"q": "test"}),
            ],
            stop_reason="tool_use",
        )
        resp = _parse_response(raw, "claude-sonnet-4-5-20250929")
        assert resp.content == "Let me search"
        assert len(resp.tool_calls) == 1

    def test_thinking_block(self) -> None:
        raw = _make_response(
            content=[_make_thinking_block("I need to..."), _make_text_block("answer")],
        )
        resp = _parse_response(raw, "claude-sonnet-4-5-20250929")
        assert resp.reasoning_content == "I need to..."
        assert resp.content == "answer"

    def test_model_id_passthrough(self) -> None:
        raw = _make_response(response_id="msg-123", model="claude-opus-4-6")
        resp = _parse_response(raw, "claude-sonnet-4-5-20250929")
        assert resp.id == "msg-123"
        assert resp.model == "claude-opus-4-6"


# ---------------------------------------------------------------------------
# Provider: complete()
# ---------------------------------------------------------------------------


class TestAnthropicProviderComplete:
    async def test_basic_complete(self) -> None:
        config = _make_config()
        provider = AnthropicProvider(config)
        raw = _make_response(content=[_make_text_block("world")])
        provider._client.messages.create = AsyncMock(return_value=raw)

        result = await provider.complete([UserMessage(content="hello")])

        assert result.content == "world"
        provider._client.messages.create.assert_awaited_once()

    async def test_complete_with_system(self) -> None:
        config = _make_config()
        provider = AnthropicProvider(config)
        raw = _make_response()
        provider._client.messages.create = AsyncMock(return_value=raw)

        await provider.complete(
            [
                SystemMessage(content="be helpful"),
                UserMessage(content="hi"),
            ]
        )

        call_kwargs = provider._client.messages.create.call_args[1]
        assert call_kwargs["system"] == "be helpful"

    async def test_complete_no_system_omits_kwarg(self) -> None:
        config = _make_config()
        provider = AnthropicProvider(config)
        raw = _make_response()
        provider._client.messages.create = AsyncMock(return_value=raw)

        await provider.complete([UserMessage(content="hi")])

        call_kwargs = provider._client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    async def test_complete_with_tools(self) -> None:
        config = _make_config()
        provider = AnthropicProvider(config)
        raw = _make_response()
        provider._client.messages.create = AsyncMock(return_value=raw)

        tools = [
            {"type": "function", "function": {"name": "t", "description": "d", "parameters": {}}}
        ]
        await provider.complete([UserMessage(content="hi")], tools=tools)

        call_kwargs = provider._client.messages.create.call_args[1]
        assert call_kwargs["tools"][0]["name"] == "t"
        assert "input_schema" in call_kwargs["tools"][0]

    async def test_complete_default_max_tokens(self) -> None:
        config = _make_config()
        provider = AnthropicProvider(config)
        raw = _make_response()
        provider._client.messages.create = AsyncMock(return_value=raw)

        await provider.complete([UserMessage(content="hi")])

        call_kwargs = provider._client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == _DEFAULT_MAX_TOKENS

    async def test_complete_custom_max_tokens(self) -> None:
        config = _make_config()
        provider = AnthropicProvider(config)
        raw = _make_response()
        provider._client.messages.create = AsyncMock(return_value=raw)

        await provider.complete([UserMessage(content="hi")], max_tokens=1000)

        call_kwargs = provider._client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 1000

    async def test_complete_with_temperature(self) -> None:
        config = _make_config()
        provider = AnthropicProvider(config)
        raw = _make_response()
        provider._client.messages.create = AsyncMock(return_value=raw)

        await provider.complete([UserMessage(content="hi")], temperature=0.7)

        call_kwargs = provider._client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    async def test_complete_api_error(self) -> None:
        import anthropic as anthropic_mod

        config = _make_config()
        provider = AnthropicProvider(config)
        provider._client.messages.create = AsyncMock(
            side_effect=anthropic_mod.APIError(
                message="rate limited",
                request=MagicMock(),
                body=None,
            )
        )

        with pytest.raises(ModelError, match="rate limited"):
            await provider.complete([UserMessage(content="hi")])


# ---------------------------------------------------------------------------
# Provider: stream()
# ---------------------------------------------------------------------------


class TestAnthropicProviderStream:
    async def test_text_stream(self) -> None:
        config = _make_config()
        provider = AnthropicProvider(config)

        events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage=SimpleNamespace(input_tokens=10)),
            ),
            SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(type="text"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text="Hel"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text="lo"),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(output_tokens=5),
            ),
        ]

        async def mock_stream(**kwargs: Any) -> Any:
            async def gen():
                for e in events:
                    yield e

            return gen()

        provider._client.messages.create = AsyncMock(side_effect=mock_stream)

        collected = []
        async for chunk in provider.stream([UserMessage(content="hi")]):
            collected.append(chunk)

        # text_delta x2 + message_delta x1 = 3 chunks
        assert len(collected) == 3
        assert collected[0].delta == "Hel"
        assert collected[1].delta == "lo"
        assert collected[2].finish_reason == "stop"
        assert collected[2].usage.input_tokens == 10
        assert collected[2].usage.output_tokens == 5

    async def test_tool_use_stream(self) -> None:
        config = _make_config()
        provider = AnthropicProvider(config)

        events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage=SimpleNamespace(input_tokens=10)),
            ),
            SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(type="tool_use", id="tu1", name="search"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="input_json_delta", partial_json='{"q":'),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="input_json_delta", partial_json='"test"}'),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="tool_use"),
                usage=SimpleNamespace(output_tokens=8),
            ),
        ]

        async def mock_stream(**kwargs: Any) -> Any:
            async def gen():
                for e in events:
                    yield e

            return gen()

        provider._client.messages.create = AsyncMock(side_effect=mock_stream)

        collected = []
        async for chunk in provider.stream([UserMessage(content="search")]):
            collected.append(chunk)

        assert len(collected) == 4
        # First: tool_call start
        assert collected[0].tool_call_deltas[0].id == "tu1"
        assert collected[0].tool_call_deltas[0].name == "search"
        # Second + third: argument deltas
        assert collected[1].tool_call_deltas[0].arguments == '{"q":'
        assert collected[2].tool_call_deltas[0].arguments == '"test"}'
        # Fourth: final
        assert collected[3].finish_reason == "tool_calls"

    async def test_stream_passes_stream_flag(self) -> None:
        config = _make_config()
        provider = AnthropicProvider(config)

        async def mock_stream(**kwargs: Any) -> Any:
            assert kwargs["stream"] is True

            async def gen():
                return
                yield  # make it an async generator

            return gen()

        provider._client.messages.create = AsyncMock(side_effect=mock_stream)

        async for _ in provider.stream([UserMessage(content="hi")]):
            pass

    async def test_stream_api_error(self) -> None:
        import anthropic as anthropic_mod

        config = _make_config()
        provider = AnthropicProvider(config)
        provider._client.messages.create = AsyncMock(
            side_effect=anthropic_mod.APIError(
                message="server error",
                request=MagicMock(),
                body=None,
            )
        )

        with pytest.raises(ModelError, match="server error"):
            async for _ in provider.stream([UserMessage(content="hi")]):
                pass


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestAnthropicRegistration:
    def test_registered(self) -> None:
        assert model_registry.get("anthropic") is AnthropicProvider

    def test_get_provider_creates_instance(self) -> None:
        from exo.models.provider import get_provider  # pyright: ignore[reportMissingImports]

        provider = get_provider("anthropic:claude-sonnet-4-5-20250929", api_key="test-key")
        assert isinstance(provider, AnthropicProvider)
        assert provider.config.model_name == "claude-sonnet-4-5-20250929"
