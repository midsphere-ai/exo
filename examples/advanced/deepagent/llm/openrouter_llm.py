#!/usr/bin/env python
"""OpenRouter LLM provider for Exo framework.

Implements the ``ModelProvider`` protocol using the OpenRouter API,
which is OpenAI-compatible. Supports cache control, provider routing,
token tracking, and context limit management.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import tiktoken  # pyright: ignore[reportMissingImports]
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

from exo.config import ModelConfig  # pyright: ignore[reportMissingImports]
from exo.models.provider import ModelProvider  # pyright: ignore[reportMissingImports]
from exo.models.types import (  # pyright: ignore[reportMissingImports]
    ModelError,
    ModelResponse,
    StreamChunk,
)
from exo.types import (  # pyright: ignore[reportMissingImports]
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolResult,
    Usage,
    UserMessage,
)

logger = logging.getLogger("deepagent")


class ContextLimitError(Exception):
    """Exception raised when context limit is exceeded."""


class TokenUsage(BaseModel):
    """Token usage tracking."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_input_tokens: int = 0
    total_cache_creation_input_tokens: int = 0


class OpenRouterConfig(BaseModel):
    """OpenRouter specific configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: str
    api_base: str = "https://openrouter.ai/api/v1"
    model_name: str
    max_retries: int = 3
    timeout: int = 600
    temperature: float = 0.1
    top_p: float = 1.0
    max_tokens: int = 4096
    max_context_length: int = 200000

    # Pricing (per million tokens)
    input_token_price: float = 3.0
    output_token_price: float = 15.0
    cache_input_token_price: float = 0.3

    # OpenRouter provider preference (google, anthropic, amazon)
    openrouter_provider: str | None = None

    # Cache control
    disable_cache_control: bool = False


# ---------------------------------------------------------------------------
# Finish reason mapping
# ---------------------------------------------------------------------------

_FINISH_REASON_MAP: dict[str | None, str] = {
    "stop": "stop",
    "tool_calls": "tool_calls",
    "length": "length",
    "content_filter": "content_filter",
    None: "stop",
}


def _map_finish_reason(raw: str | None) -> str:
    """Normalize an OpenAI finish reason string."""
    return _FINISH_REASON_MAP.get(raw, "stop")


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def _to_openai_messages(messages: list[Message] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Exo ``Message`` objects or raw dicts to OpenAI chat message dicts.

    Args:
        messages: Exo message sequence or raw message dicts.

    Returns:
        List of dicts suitable for the OpenAI ``messages`` parameter.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        # Handle raw dict messages (e.g. from ContextManager)
        if isinstance(msg, dict):
            result.append(msg)
            continue
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, UserMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AssistantMessage):
            entry: dict[str, Any] = {"role": "assistant"}
            if msg.content:
                entry["content"] = msg.content
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            if not msg.content and not msg.tool_calls:
                entry["content"] = ""
            result.append(entry)
        elif isinstance(msg, ToolResult):
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.error if msg.error else msg.content,
            })
    return result


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class OpenRouterLLM(ModelProvider):
    """OpenRouter LLM provider implementing the Exo ``ModelProvider`` protocol.

    Wraps the OpenAI-compatible OpenRouter API with support for cache control,
    provider routing preferences, token usage tracking, and context limit
    management.

    Args:
        api_key: OpenRouter API key.
        api_base: OpenRouter API base URL.
        model_name: Model identifier (e.g. ``"anthropic/claude-3.5-sonnet"``).
        max_retries: Maximum retries on transient failures.
        timeout: Request timeout in seconds.
        **kwargs: Additional fields forwarded to ``OpenRouterConfig``.
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://openrouter.ai/api/v1",
        model_name: str = "anthropic/claude-3.5-sonnet",
        max_retries: int = 3,
        timeout: int = 600,
        **kwargs: Any,
    ):
        # Build Exo ModelConfig for the parent ABC
        exo_config = ModelConfig(
            provider="openrouter",
            model_name=model_name,
            api_key=api_key,
            base_url=api_base,
            max_retries=max_retries,
            timeout=float(timeout),
        )
        super().__init__(exo_config)

        # OpenRouter-specific config (pricing, cache control, provider prefs)
        self.or_config = OpenRouterConfig(
            api_key=api_key,
            api_base=api_base,
            model_name=model_name,
            max_retries=max_retries,
            timeout=timeout,
            **kwargs,
        )

        self._client = AsyncOpenAI(
            api_key=self.or_config.api_key,
            base_url=self.or_config.api_base,
            timeout=self.or_config.timeout,
        )

        # Token usage tracking
        self.token_usage = TokenUsage()
        self.last_call_tokens: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

        # tiktoken for token estimation
        self.encoding: tiktoken.Encoding | None = None

    # ------------------------------------------------------------------
    # ModelProvider interface
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        """Send a completion request to OpenRouter and return the full response.

        Converts Exo ``Message`` objects to OpenAI format, applies cache
        control if enabled, and normalizes the response to ``ModelResponse``.

        Args:
            messages: Conversation history as Exo message objects.
            tools: JSON-schema tool definitions in OpenAI format.
            temperature: Sampling temperature override.
            max_tokens: Maximum output tokens override.

        Returns:
            Normalized ``ModelResponse`` with content, tool calls, and usage.

        Raises:
            ContextLimitError: If the request exceeds the model's context window.
            ModelError: If the API call fails for other reasons.
        """
        try:
            openai_messages = _to_openai_messages(messages)
            processed = self._prepare_messages(openai_messages)

            params: dict[str, Any] = {
                "model": self.or_config.model_name,
                "messages": processed,
                "temperature": temperature if temperature is not None else self.or_config.temperature,
                "max_tokens": max_tokens or self.or_config.max_tokens,
                "stream": False,
            }

            if tools:  # Only add non-empty tool lists (some APIs reject tools=[])
                params["tools"] = tools
                logger.debug("Added %d tools to API call", len(tools))

            if self.or_config.openrouter_provider:
                params["extra_body"] = self._get_provider_config(
                    self.or_config.openrouter_provider
                )

            response = await self._client.chat.completions.create(**params)

            if response.usage:
                self._update_token_usage(response.usage)

            return self._parse_response(response)

        except ContextLimitError:
            raise
        except Exception as e:
            error_str = str(e)
            if any(
                phrase in error_str
                for phrase in [
                    "Input is too long",
                    "context limit",
                    "maximum context length",
                ]
            ):
                logger.error("Context limit exceeded: %s", error_str)
                raise ContextLimitError(f"Context limit exceeded: {error_str}") from e

            logger.error("OpenRouter LLM call failed: %s", error_str)
            raise ModelError(
                error_str, model=f"openrouter:{self.or_config.model_name}"
            ) from e

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion from OpenRouter, yielding chunks incrementally.

        Currently delegates to ``complete()`` and yields a single chunk.
        Full streaming support can be added in a future iteration.

        Args:
            messages: Conversation history as Exo message objects.
            tools: JSON-schema tool definitions in OpenAI format.
            temperature: Sampling temperature override.
            max_tokens: Maximum output tokens override.

        Yields:
            A single ``StreamChunk`` containing the full response.

        Raises:
            ContextLimitError: If the request exceeds the model's context window.
            ModelError: If the API call fails.
        """
        result = await self.complete(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )
        yield StreamChunk(
            delta=result.content,
            finish_reason=result.finish_reason,
            usage=result.usage,
        )

    # ------------------------------------------------------------------
    # Message preparation
    # ------------------------------------------------------------------

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prepare raw message dicts, applying cache control if enabled."""
        if self.or_config.disable_cache_control:
            return messages
        return self._apply_cache_control(messages)

    def _apply_cache_control(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply ephemeral cache control to system and last user message."""
        cached_messages: list[dict[str, Any]] = []
        user_turns_processed = 0

        for turn in reversed(messages):
            if (turn["role"] == "user" and user_turns_processed < 1) or turn["role"] == "system":
                new_content: list[dict[str, Any]] = []
                processed_text = False

                if isinstance(turn.get("content"), list):
                    for item in turn["content"]:
                        if (
                            item.get("type") == "text"
                            and len(item.get("text", "")) > 0
                            and not processed_text
                        ):
                            text_item = item.copy()
                            text_item["cache_control"] = {"type": "ephemeral"}
                            new_content.append(text_item)
                            processed_text = True
                        else:
                            new_content.append(item.copy())

                    cached_messages.append({"role": turn["role"], "content": new_content})
                else:
                    cached_messages.append(turn)

                if turn["role"] == "user":
                    user_turns_processed += 1
            else:
                cached_messages.append(turn)

        return list(reversed(cached_messages))

    # ------------------------------------------------------------------
    # Provider routing
    # ------------------------------------------------------------------

    def _get_provider_config(self, provider: str) -> dict[str, Any]:
        """Get OpenRouter provider-specific routing configuration.

        Args:
            provider: Provider preference string (``"google"``, ``"anthropic"``,
                or ``"amazon"``).

        Returns:
            Extra body dict for the OpenRouter API request.
        """
        provider_map: dict[str, dict[str, Any]] = {
            "google": {
                "provider": {
                    "only": ["google-vertex/us", "google-vertex/europe", "google-vertex/global"]
                }
            },
            "anthropic": {"provider": {"only": ["anthropic"]}},
            "amazon": {"provider": {"only": ["amazon-bedrock"]}},
        }
        return provider_map.get(provider.strip().lower(), {})

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def _update_token_usage(self, usage_data: Any) -> None:
        """Update cumulative token usage from an API response.

        Args:
            usage_data: The ``usage`` object from the OpenAI response.
        """
        if not usage_data:
            return

        input_tokens = getattr(usage_data, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage_data, "completion_tokens", 0) or 0

        prompt_tokens_details = getattr(usage_data, "prompt_tokens_details", None)
        cached_tokens = 0
        if prompt_tokens_details:
            cached_tokens = getattr(prompt_tokens_details, "cached_tokens", 0) or 0

        self.last_call_tokens = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
        }

        self.token_usage.total_input_tokens += input_tokens
        self.token_usage.total_output_tokens += output_tokens
        self.token_usage.total_cache_read_input_tokens += cached_tokens

        logger.debug(
            "Token usage - Input: %d, Output: %d, Cache: %d",
            self.token_usage.total_input_tokens,
            self.token_usage.total_output_tokens,
            self.token_usage.total_cache_read_input_tokens,
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: Any) -> ModelResponse:
        """Parse an OpenRouter/OpenAI API response into a ``ModelResponse``.

        Args:
            response: The raw API response from the OpenAI SDK.

        Returns:
            A normalized ``ModelResponse``.

        Raises:
            ValueError: If the response has no valid choices.
        """
        if not response or not getattr(response, "choices", None):
            raise ValueError("LLM did not return a valid response")

        choice = response.choices[0]
        message = choice.message

        content = getattr(message, "content", "") or ""

        tool_calls: list[ToolCall] = []
        raw_tool_calls = getattr(message, "tool_calls", None)

        if raw_tool_calls:
            for tc in raw_tool_calls:
                fn = getattr(tc, "function", None)
                if fn is not None:
                    name = getattr(fn, "name", "") or ""
                    arguments = getattr(fn, "arguments", None)
                else:
                    name = getattr(tc, "name", "") or ""
                    arguments = getattr(tc, "arguments", None)

                tool_calls.append(
                    ToolCall(
                        id=getattr(tc, "id", "") or "",
                        name=name,
                        arguments=arguments if arguments is not None else "{}",
                    )
                )

        usage = Usage()
        if getattr(response, "usage", None):
            usage = Usage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        return ModelResponse(
            id=getattr(response, "id", "") or "",
            model=getattr(response, "model", "") or self.or_config.model_name,
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=_map_finish_reason(getattr(choice, "finish_reason", None)),
        )

    # ------------------------------------------------------------------
    # Token estimation & context management
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken.

        Args:
            text: The text to estimate tokens for.

        Returns:
            Estimated number of tokens.
        """
        if not self.encoding:
            try:
                self.encoding = tiktoken.get_encoding("o200k_base")
            except Exception:
                self.encoding = tiktoken.get_encoding("cl100k_base")

        try:
            return len(self.encoding.encode(text))  # pyright: ignore[reportOptionalMemberAccess]
        except Exception:
            # Fallback: ~4 chars per token
            return len(text) // 4

    def ensure_summary_context(
        self,
        message_history: list[dict[str, Any]],
        summary_prompt: str,
    ) -> bool:
        """Check if message history plus a summary prompt would exceed the context limit.

        If the estimated total exceeds ``max_context_length``, the last
        assistant-user message pair is removed from *message_history* (mutated
        in-place).

        Args:
            message_history: Mutable list of raw message dicts.
            summary_prompt: The summary prompt text to estimate.

        Returns:
            ``True`` if context is within limits, ``False`` if messages were removed.
        """
        last_prompt_tokens = self.last_call_tokens.get("prompt_tokens", 0)
        last_completion_tokens = self.last_call_tokens.get("completion_tokens", 0)
        buffer_factor = 1.2

        summary_tokens = self._estimate_tokens(summary_prompt) * buffer_factor

        last_user_tokens = 0.0
        if message_history and message_history[-1]["role"] == "user":
            content = message_history[-1]["content"]
            if isinstance(content, list):
                text = " ".join(
                    item.get("text", "") for item in content if item.get("type") == "text"
                )
            else:
                text = content
            last_user_tokens = self._estimate_tokens(text) * buffer_factor

        estimated_total = (
            last_prompt_tokens
            + last_completion_tokens
            + last_user_tokens
            + summary_tokens
            + self.or_config.max_tokens
        )

        if estimated_total >= self.or_config.max_context_length:
            logger.warning(
                "Context + summary would exceed limit: %s >= %s",
                estimated_total,
                self.or_config.max_context_length,
            )

            if message_history and message_history[-1]["role"] == "user":
                message_history.pop()

            if message_history and message_history[-1]["role"] == "assistant":
                message_history.pop()

            logger.info(
                "Removed last assistant-user pair, current history length: %d",
                len(message_history),
            )
            return False

        logger.debug(
            "Context check passed: %s/%s",
            estimated_total,
            self.or_config.max_context_length,
        )
        return True

    # ------------------------------------------------------------------
    # Usage reporting
    # ------------------------------------------------------------------

    def get_token_usage(self) -> dict[str, int]:
        """Get current cumulative token usage as a dict.

        Returns:
            Dict with ``total_input_tokens``, ``total_output_tokens``, etc.
        """
        return self.token_usage.model_dump()

    def format_token_usage_summary(self) -> tuple[list[str], str]:
        """Format token usage statistics and cost estimation.

        Returns:
            A tuple of (summary_lines, log_string) for display and logging.
        """
        usage = self.token_usage

        total_input = usage.total_input_tokens
        total_output = usage.total_output_tokens
        cache_input = usage.total_cache_read_input_tokens

        cost = (
            ((total_input - cache_input) / 1_000_000 * self.or_config.input_token_price)
            + (cache_input / 1_000_000 * self.or_config.cache_input_token_price)
            + (total_output / 1_000_000 * self.or_config.output_token_price)
        )

        summary_lines = [
            "\n" + "-" * 20 + " Token Usage & Cost " + "-" * 20,
            f"Total Input Tokens: {total_input}",
            f"Total Cache Input Tokens: {cache_input}",
            f"Total Output Tokens: {total_output}",
            "-" * 60,
            f"Input Token Price: ${self.or_config.input_token_price:.4f} USD",
            f"Output Token Price: ${self.or_config.output_token_price:.4f} USD",
            f"Cache Input Token Price: ${self.or_config.cache_input_token_price:.4f} USD",
            "-" * 60,
            f"Estimated Cost (with cache): ${cost:.4f} USD",
            "-" * 60,
        ]

        log_string = (
            f"[OpenRouter/{self.or_config.model_name}] "
            f"Total Input: {total_input}, Cache Input: {cache_input}, "
            f"Output: {total_output}, Cost: ${cost:.4f} USD"
        )

        return summary_lines, log_string
