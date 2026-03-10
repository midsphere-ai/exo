"""LLM utility functions — bridges Orbiter's ModelProvider to SkyworkAI's model_manager pattern."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel

from orbiter.models.provider import ModelProvider, get_provider
from orbiter.types import Message, SystemMessage, UserMessage

logger = logging.getLogger("deepagent")


class LLMResponse:
    """Response from an LLM call, mirroring SkyworkAI's model_manager response."""

    def __init__(
        self,
        message: str,
        success: bool = True,
        parsed_model: BaseModel | None = None,
    ) -> None:
        self.message = message
        self.success = success
        self.parsed_model = parsed_model


def _extract_json_from_text(text: str) -> str:
    """Extract JSON from text that may contain markdown code blocks or extra text."""
    # Try to extract from markdown code blocks first
    json_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_block:
        return json_block.group(1).strip()

    # Try to find a JSON object or array
    # Look for the first { and last } or first [ and last ]
    text = text.strip()
    if text.startswith("{"):
        depth = 0
        for i, ch in enumerate(text):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[: i + 1]
    if text.startswith("["):
        depth = 0
        for i, ch in enumerate(text):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[: i + 1]

    return text


async def call_llm(
    model: str,
    messages: list[Message],
    *,
    response_format: type[BaseModel] | None = None,
    provider: ModelProvider | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMResponse:
    """Call an LLM model, optionally with structured output parsing.

    This bridges Orbiter's ModelProvider.complete() to match the SkyworkAI
    model_manager(model=..., messages=..., response_format=...) pattern.

    Args:
        model: Model string, e.g. "openai:gpt-4o".
        messages: Conversation messages.
        response_format: Optional Pydantic model for structured output.
        provider: Pre-built provider (avoids re-creation).
        temperature: Sampling temperature override.
        max_tokens: Maximum output tokens override.

    Returns:
        LLMResponse with .message (text), .success, and .parsed_model.
    """
    if provider is None:
        provider = get_provider(model)

    # If structured output requested, add JSON schema instruction
    actual_messages = list(messages)
    if response_format is not None:
        schema = response_format.model_json_schema()
        # Remove $defs and other meta-keys for cleaner prompts
        clean_schema = {k: v for k, v in schema.items() if k not in ("$defs", "definitions")}
        schema_str = json.dumps(clean_schema, indent=2)

        json_instruction = (
            f"\n\nYou MUST respond with valid JSON matching this schema:\n"
            f"```json\n{schema_str}\n```\n"
            f"Return ONLY the JSON object, no other text or explanation."
        )

        # Append instruction to the last user message
        if actual_messages and isinstance(actual_messages[-1], UserMessage):
            last_msg = actual_messages[-1]
            actual_messages[-1] = UserMessage(content=last_msg.content + json_instruction)
        else:
            actual_messages.append(UserMessage(content=json_instruction))

    try:
        response = await provider.complete(
            actual_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.content.strip()

        parsed_model = None
        if response_format is not None and text:
            try:
                json_str = _extract_json_from_text(text)
                parsed_model = response_format.model_validate_json(json_str)
            except Exception as e:
                logger.warning(f"Failed to parse structured output: {e}")
                # Try with model_validate (dict) as fallback
                try:
                    data = json.loads(_extract_json_from_text(text))
                    parsed_model = response_format.model_validate(data)
                except Exception:
                    pass

        return LLMResponse(
            message=text,
            success=True,
            parsed_model=parsed_model,
        )

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return LLMResponse(
            message=str(e),
            success=False,
            parsed_model=None,
        )
