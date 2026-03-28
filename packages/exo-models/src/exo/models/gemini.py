"""Google Gemini LLM provider implementation.

Wraps the ``google-genai`` SDK to implement ``ModelProvider.complete()`` and
``ModelProvider.stream()`` with normalized response types.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from google import genai

from exo.config import ModelConfig
from exo.types import (
    AssistantMessage,
    Message,
    MessageContent,
    SystemMessage,
    ToolCall,
    ToolResult,
    Usage,
    UserMessage,
)

from ._media import content_blocks_to_google
from .provider import ModelProvider, model_registry
from .types import (
    FinishReason,
    ModelError,
    ModelResponse,
    StreamChunk,
    ToolCallDelta,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Finish reason mapping
# ---------------------------------------------------------------------------

_FINISH_REASON_MAP: dict[str | None, FinishReason] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "MALFORMED_FUNCTION_CALL": "stop",
    "OTHER": "stop",
    None: "stop",
}


def _map_finish_reason(raw: str | None) -> FinishReason:
    """Normalize a Google finish reason to a ``FinishReason``."""
    return _FINISH_REASON_MAP.get(raw, "stop")


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def _user_parts_from_content(content: MessageContent) -> list[dict[str, Any]]:
    """Convert MessageContent to Google API parts for a user message.

    Args:
        content: A string or list of ContentBlock objects.

    Returns:
        List of Google-format part dicts.
    """
    if isinstance(content, str):
        return [{"text": content}]
    return content_blocks_to_google(content)


def _to_google_contents(messages: list[Message]) -> tuple[list[dict[str, Any]], str]:
    """Convert Exo messages to Google API format.

    Extracts system messages into a separate system instruction string.

    Args:
        messages: Exo message sequence.

    Returns:
        A ``(contents, system_instruction)`` tuple.
    """
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append(msg.content)
        elif isinstance(msg, UserMessage):
            contents.append({"role": "user", "parts": _user_parts_from_content(msg.content)})
        elif isinstance(msg, AssistantMessage):
            parts: list[dict[str, Any]] = []
            if msg.content:
                if isinstance(msg.content, str):
                    parts.append({"text": msg.content})
                else:
                    parts.extend(content_blocks_to_google(msg.content))
            for tc in msg.tool_calls:
                args = json.loads(tc.arguments) if tc.arguments else {}
                fc_part: dict[str, Any] = {"function_call": {"name": tc.name, "args": args}}
                if tc.thought_signature is not None:
                    fc_part["thought_signature"] = tc.thought_signature
                parts.append(fc_part)
            if not parts:
                parts.append({"text": ""})
            contents.append({"role": "model", "parts": parts})
        elif isinstance(msg, ToolResult):
            # Build function_response part
            if isinstance(msg.content, list):
                response_data: Any = content_blocks_to_google(msg.content)
            else:
                response_data = msg.error if msg.error else msg.content
            function_response_part: dict[str, Any] = {
                "function_response": {
                    "name": msg.tool_name,
                    "response": {"content": response_data},
                },
            }
            # Include any media parts alongside the function response
            media_parts: list[dict[str, Any]] = []
            if isinstance(msg.content, list):
                media_parts = content_blocks_to_google(msg.content)
            all_parts = [function_response_part, *media_parts]
            contents.append({"role": "user", "parts": all_parts})

    return contents, "\n".join(system_parts)


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-format tool schemas to Google format.

    Args:
        tools: List of OpenAI-style tool definitions.

    Returns:
        List of Google-style tool definitions with function_declarations.
    """
    declarations: list[dict[str, Any]] = []
    for t in tools:
        fn = t.get("function", {})
        declarations.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return [{"function_declarations": declarations}] if declarations else []


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _build_config(
    tools: list[dict[str, Any]] | None,
    temperature: float | None,
    max_tokens: int | None,
    system_instruction: str,
) -> dict[str, Any]:
    """Build a config dict for ``generate_content()``.

    Args:
        tools: OpenAI-format tool definitions (will be converted).
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        system_instruction: System prompt text.

    Returns:
        Dict suitable for the ``config`` parameter.
    """
    config: dict[str, Any] = {}
    if system_instruction:
        config["system_instruction"] = system_instruction
    if temperature is not None:
        config["temperature"] = temperature
    if max_tokens is not None:
        config["max_output_tokens"] = max_tokens
    if tools is not None:
        config["tools"] = _convert_tools(tools)
    return config


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_response(raw: Any, model_name: str) -> ModelResponse:
    """Convert a Google GenerateContentResponse to a ``ModelResponse``.

    Args:
        raw: The raw Google API response object.
        model_name: The model name for context.

    Returns:
        A normalized ``ModelResponse``.
    """
    candidate = raw.candidates[0]
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for i, part in enumerate(candidate.content.parts):
        text = getattr(part, "text", None)
        if text and not getattr(part, "thought", False):
            content_parts.append(text)
        fc = getattr(part, "function_call", None)
        if fc:
            call_id = getattr(fc, "id", None) or f"call_{i}"
            thought_sig = getattr(part, "thought_signature", None)
            tool_calls.append(
                ToolCall(
                    id=call_id,
                    name=fc.name,
                    arguments=json.dumps(fc.args) if fc.args else "{}",
                    thought_signature=thought_sig,
                )
            )

    finish_reason_raw = str(candidate.finish_reason) if candidate.finish_reason else None

    usage = Usage()
    if raw.usage_metadata:
        usage = Usage(
            input_tokens=raw.usage_metadata.prompt_token_count or 0,
            output_tokens=raw.usage_metadata.candidates_token_count or 0,
            total_tokens=raw.usage_metadata.total_token_count or 0,
        )

    return ModelResponse(
        id="",
        model=model_name,
        content="".join(content_parts),
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=_map_finish_reason(finish_reason_raw),
    )


def _parse_stream_chunk(chunk: Any) -> StreamChunk:
    """Convert a single Google streaming chunk to a ``StreamChunk``.

    Args:
        chunk: A single chunk from the streaming response.

    Returns:
        A normalized ``StreamChunk``.
    """
    if not chunk.candidates:
        usage = Usage()
        if chunk.usage_metadata:
            usage = Usage(
                input_tokens=chunk.usage_metadata.prompt_token_count or 0,
                output_tokens=chunk.usage_metadata.candidates_token_count or 0,
                total_tokens=chunk.usage_metadata.total_token_count or 0,
            )
        return StreamChunk(usage=usage)

    candidate = chunk.candidates[0]
    text_parts: list[str] = []
    tool_call_deltas: list[ToolCallDelta] = []

    for i, part in enumerate(candidate.content.parts):
        text = getattr(part, "text", None)
        if text and not getattr(part, "thought", False):
            text_parts.append(text)
        fc = getattr(part, "function_call", None)
        if fc:
            call_id = getattr(fc, "id", None) or f"call_{i}"
            thought_sig = getattr(part, "thought_signature", None)
            tool_call_deltas.append(
                ToolCallDelta(
                    index=i,
                    id=call_id,
                    name=fc.name,
                    arguments=json.dumps(fc.args) if fc.args else "{}",
                    thought_signature=thought_sig,
                )
            )

    finish_reason_raw = str(candidate.finish_reason) if candidate.finish_reason else None
    finish = _map_finish_reason(finish_reason_raw) if finish_reason_raw else None

    usage = Usage()
    if chunk.usage_metadata:
        usage = Usage(
            input_tokens=chunk.usage_metadata.prompt_token_count or 0,
            output_tokens=chunk.usage_metadata.candidates_token_count or 0,
            total_tokens=chunk.usage_metadata.total_token_count or 0,
        )

    return StreamChunk(
        delta="".join(text_parts),
        tool_call_deltas=tool_call_deltas,
        finish_reason=finish,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class GeminiProvider(ModelProvider):
    """Google Gemini LLM provider.

    Wraps the ``google.genai.Client`` for Gemini API completions.

    Args:
        config: Provider connection configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._client = genai.Client(api_key=config.api_key or "dummy")

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        """Send a completion request to Google Gemini.

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
        _log.debug(
            "gemini complete: model=%s, messages=%d, tools=%d",
            self.config.model_name,
            len(messages),
            len(tools or []),
        )
        contents, system_instruction = _to_google_contents(messages)
        config = _build_config(tools, temperature, max_tokens, system_instruction)
        try:
            response = await self._client.aio.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            _log.error(
                "gemini complete failed: model=%s, error=%s",
                self.config.model_name,
                exc,
                exc_info=True,
            )
            raise ModelError(str(exc), model=f"gemini:{self.config.model_name}") from exc
        return _parse_response(response, self.config.model_name)

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion from Google Gemini.

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
        _log.debug(
            "gemini stream: model=%s, messages=%d, tools=%d",
            self.config.model_name,
            len(messages),
            len(tools or []),
        )
        contents, system_instruction = _to_google_contents(messages)
        config = _build_config(tools, temperature, max_tokens, system_instruction)
        try:
            async for chunk in await self._client.aio.models.generate_content_stream(
                model=self.config.model_name,
                contents=contents,
                config=config,
            ):
                yield _parse_stream_chunk(chunk)
        except ModelError:
            raise
        except Exception as exc:
            _log.error(
                "gemini stream failed: model=%s, error=%s",
                self.config.model_name,
                exc,
                exc_info=True,
            )
            raise ModelError(str(exc), model=f"gemini:{self.config.model_name}") from exc


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

model_registry.register("gemini", GeminiProvider)
