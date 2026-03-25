"""Anthropic LLM provider implementation.

Wraps the ``anthropic`` SDK to implement ``ModelProvider.complete()`` and
``ModelProvider.stream()`` with normalized response types.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from exo.config import ModelConfig
from exo.types import (
    AssistantMessage,
    AudioBlock,
    ContentBlock,
    DocumentBlock,
    ImageDataBlock,
    ImageURLBlock,
    Message,
    MessageContent,
    SystemMessage,
    TextBlock,
    ToolCall,
    ToolResult,
    Usage,
    UserMessage,
    VideoBlock,
)

from .provider import ModelProvider, model_registry
from .types import (
    FinishReason,
    ModelError,
    ModelResponse,
    StreamChunk,
    ToolCallDelta,
)

_log = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# Stop reason mapping
# ---------------------------------------------------------------------------

_STOP_REASON_MAP: dict[str | None, FinishReason] = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
    None: "stop",
}


def _map_stop_reason(raw: str | None) -> FinishReason:
    """Normalize an Anthropic stop reason to a ``FinishReason``."""
    return _STOP_REASON_MAP.get(raw, "stop")


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def _content_blocks_to_anthropic(blocks: list[ContentBlock]) -> list[dict[str, Any]]:
    """Convert a list of ContentBlock objects to Anthropic content parts.

    Args:
        blocks: List of ContentBlock objects.

    Returns:
        List of Anthropic-format content part dicts.
    """
    parts: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageURLBlock):
            parts.append({"type": "image", "source": {"type": "url", "url": block.url}})
        elif isinstance(block, ImageDataBlock):
            parts.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.media_type,
                        "data": block.data,
                    },
                }
            )
        elif isinstance(block, DocumentBlock):
            doc: dict[str, Any] = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": block.media_type,
                    "data": block.data,
                },
            }
            if block.title:
                doc["title"] = block.title
            parts.append(doc)
        elif isinstance(block, AudioBlock):
            _log.warning("Anthropic does not support audio input; skipping AudioBlock")
        elif isinstance(block, VideoBlock):
            _log.warning("Anthropic does not support video input; skipping VideoBlock")
    return parts


def _message_content_to_anthropic(content: MessageContent) -> str | list[dict[str, Any]]:
    """Convert MessageContent to an Anthropic-compatible content value.

    Args:
        content: A string or list of ContentBlock objects.

    Returns:
        A string (for plain text) or list of content part dicts.
    """
    if isinstance(content, str):
        return content
    return _content_blocks_to_anthropic(content)


def _build_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Convert Exo messages to Anthropic format.

    Extracts system messages into a single system string (Anthropic takes
    ``system=`` as a separate kwarg). Consecutive tool results are merged
    into one ``user`` message to maintain strict alternation.

    Args:
        messages: Exo message sequence.

    Returns:
        A ``(system, messages)`` tuple for the Anthropic API.
    """
    system_parts: list[str] = []
    result: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append(msg.content)
        elif isinstance(msg, UserMessage):
            result.append({"role": "user", "content": _message_content_to_anthropic(msg.content)})
        elif isinstance(msg, AssistantMessage):
            content: list[dict[str, Any]] = []
            if msg.content:
                if isinstance(msg.content, str):
                    content.append({"type": "text", "text": msg.content})
                else:
                    content.extend(_content_blocks_to_anthropic(msg.content))
            for tc in msg.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": json.loads(tc.arguments) if tc.arguments else {},
                    }
                )
            if not content:
                content.append({"type": "text", "text": ""})
            result.append({"role": "assistant", "content": content})
        elif isinstance(msg, ToolResult):
            # Anthropic natively supports image blocks inside tool_result content
            if isinstance(msg.content, list):
                tool_content: str | list[dict[str, Any]] = _content_blocks_to_anthropic(msg.content)
            else:
                tool_content = msg.error if msg.error else msg.content
            tool_block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": tool_content,
            }
            if msg.error:
                tool_block["is_error"] = True
            # Merge consecutive tool results into one user message
            if result and result[-1]["role"] == "user" and isinstance(result[-1]["content"], list):
                result[-1]["content"].append(tool_block)
            else:
                result.append({"role": "user", "content": [tool_block]})

    return "\n".join(system_parts), result


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-format tool schemas to Anthropic format.

    Args:
        tools: List of OpenAI-style tool definitions.

    Returns:
        List of Anthropic-style tool definitions.
    """
    converted: list[dict[str, Any]] = []
    for t in tools:
        fn = t.get("function", {})
        converted.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return converted


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_response(raw: Any, model_name: str) -> ModelResponse:
    """Convert an Anthropic Message to a ``ModelResponse``.

    Args:
        raw: The raw Anthropic API response object.
        model_name: The model name for context.

    Returns:
        A normalized ``ModelResponse``.
    """
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    reasoning_content = ""

    for block in raw.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            content_parts.append(block.text)
        elif block_type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=json.dumps(block.input) if block.input else "",
                )
            )
        elif block_type == "thinking":
            reasoning_content = getattr(block, "thinking", "")

    usage = Usage()
    if raw.usage:
        usage = Usage(
            input_tokens=raw.usage.input_tokens,
            output_tokens=raw.usage.output_tokens,
            total_tokens=raw.usage.input_tokens + raw.usage.output_tokens,
        )

    return ModelResponse(
        id=raw.id or "",
        model=raw.model or model_name,
        content="".join(content_parts),
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=_map_stop_reason(raw.stop_reason),
        reasoning_content=reasoning_content,
    )


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class AnthropicProvider(ModelProvider):
    """Anthropic LLM provider.

    Wraps the ``anthropic.AsyncAnthropic`` client for message completions.

    Args:
        config: Provider connection configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._client = AsyncAnthropic(
            api_key=config.api_key or "dummy",
            base_url=config.base_url,
            max_retries=config.max_retries,
            timeout=config.timeout,
        )

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        """Send a message completion request to Anthropic.

        Args:
            messages: Conversation history.
            tools: JSON-schema tool definitions (OpenAI format, auto-converted).
            temperature: Sampling temperature override.
            max_tokens: Maximum output tokens override.

        Returns:
            Normalized model response.

        Raises:
            ModelError: If the API call fails.
        """
        kwargs = self._build_kwargs(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )
        _log.debug(
            "anthropic complete: model=%s, messages=%d, tools=%d",
            self.config.model_name,
            len(messages),
            len(tools or []),
        )
        try:
            response = await self._client.messages.create(**kwargs)
        except anthropic.APIError as exc:
            _log.error(
                "anthropic complete failed: model=%s, error=%s",
                self.config.model_name,
                exc,
                exc_info=True,
            )
            raise ModelError(str(exc), model=f"anthropic:{self.config.model_name}") from exc
        return _parse_response(response, self.config.model_name)

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a message completion from Anthropic.

        Args:
            messages: Conversation history.
            tools: JSON-schema tool definitions (OpenAI format, auto-converted).
            temperature: Sampling temperature override.
            max_tokens: Maximum output tokens override.

        Yields:
            Incremental response chunks.

        Raises:
            ModelError: If the API call fails.
        """
        kwargs = self._build_kwargs(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )
        kwargs["stream"] = True
        input_tokens = 0
        _log.debug(
            "anthropic stream: model=%s, messages=%d, tools=%d",
            self.config.model_name,
            len(messages),
            len(tools or []),
        )
        try:
            response = await self._client.messages.create(**kwargs)
            async for event in response:
                event_type = event.type
                if event_type == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        input_tokens = event.message.usage.input_tokens
                elif event_type == "content_block_start":
                    block = event.content_block
                    if getattr(block, "type", None) == "tool_use":
                        yield StreamChunk(
                            tool_call_deltas=[
                                ToolCallDelta(
                                    index=event.index,
                                    id=block.id,
                                    name=block.name,
                                )
                            ]
                        )
                elif event_type == "content_block_delta":
                    delta = event.delta
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "text_delta":
                        yield StreamChunk(delta=delta.text)
                    elif delta_type == "input_json_delta":
                        yield StreamChunk(
                            tool_call_deltas=[
                                ToolCallDelta(
                                    index=event.index,
                                    arguments=delta.partial_json,
                                )
                            ]
                        )
                elif event_type == "message_delta":
                    output_tokens = 0
                    if hasattr(event, "usage") and event.usage:
                        output_tokens = event.usage.output_tokens
                    total = input_tokens + output_tokens
                    stop_reason = getattr(event.delta, "stop_reason", None)
                    yield StreamChunk(
                        finish_reason=_map_stop_reason(stop_reason),
                        usage=Usage(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=total,
                        ),
                    )
        except anthropic.APIError as exc:
            _log.error(
                "anthropic stream failed: model=%s, error=%s",
                self.config.model_name,
                exc,
                exc_info=True,
            )
            raise ModelError(str(exc), model=f"anthropic:{self.config.model_name}") from exc

    def _build_kwargs(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Build keyword arguments for the Anthropic API call.

        Args:
            messages: Conversation history.
            tools: JSON-schema tool definitions.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.

        Returns:
            Dict of kwargs for ``messages.create()``.
        """
        system, converted = _build_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": converted,
            "max_tokens": max_tokens or _DEFAULT_MAX_TOKENS,
        }
        if system:
            kwargs["system"] = system
        if tools is not None:
            kwargs["tools"] = _convert_tools(tools)
        if temperature is not None:
            kwargs["temperature"] = temperature
        return kwargs


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

model_registry.register("anthropic", AnthropicProvider)
