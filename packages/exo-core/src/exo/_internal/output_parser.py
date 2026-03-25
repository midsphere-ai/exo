"""Parse LLM responses into agent-level output types.

The output parser bridges the model layer (``ModelResponse``) to the agent
layer (``AgentOutput``, ``ActionModel``).  It extracts text and tool calls,
parses JSON-encoded tool arguments, and optionally validates structured
output against a Pydantic model.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from exo.types import (
    ActionModel,
    AgentOutput,
    ExoError,
    ToolCall,
    Usage,
)

_T = TypeVar("_T", bound=BaseModel)


class OutputParseError(ExoError):
    """Raised when LLM output cannot be parsed as expected."""


# ---------------------------------------------------------------------------
# ModelResponse → AgentOutput
# ---------------------------------------------------------------------------


def parse_response(
    *,
    content: str,
    tool_calls: list[ToolCall],
    usage: Usage,
) -> AgentOutput:
    """Convert raw model fields into an ``AgentOutput``.

    This is a lightweight mapping — it copies text, tool calls, and usage
    without transforming the data.  Use :func:`parse_tool_arguments` and
    :func:`parse_structured_output` for deeper parsing.

    Args:
        content: Text output from the model.
        tool_calls: Tool invocations from the model.
        usage: Token usage statistics.

    Returns:
        An ``AgentOutput`` with the same data.
    """
    return AgentOutput(text=content, tool_calls=tool_calls, usage=usage)


# ---------------------------------------------------------------------------
# ToolCall → ActionModel (JSON argument parsing)
# ---------------------------------------------------------------------------


def parse_tool_arguments(tool_calls: list[ToolCall]) -> list[ActionModel]:
    """Parse JSON-encoded arguments from tool calls into ``ActionModel`` objects.

    Each ``ToolCall.arguments`` is a JSON string.  This function decodes
    it into a dict and wraps the result in an ``ActionModel`` ready for
    tool execution.

    Args:
        tool_calls: Tool calls with JSON-encoded arguments.

    Returns:
        A list of ``ActionModel`` objects with parsed arguments.

    Raises:
        OutputParseError: If any tool call has invalid JSON arguments.
    """
    actions: list[ActionModel] = []
    for tc in tool_calls:
        args = _parse_json_arguments(tc.arguments, tool_name=tc.name)
        actions.append(
            ActionModel(
                tool_call_id=tc.id,
                tool_name=tc.name,
                arguments=args,
            )
        )
    return actions


def _parse_json_arguments(raw: str, *, tool_name: str) -> dict[str, Any]:
    """Decode a JSON string into a dict of tool arguments.

    Empty strings are treated as no arguments (returns ``{}``).

    Args:
        raw: JSON-encoded arguments string.
        tool_name: Tool name for error context.

    Returns:
        Parsed argument dict.

    Raises:
        OutputParseError: If *raw* is non-empty and not valid JSON, or
            if the parsed value is not a JSON object.
    """
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OutputParseError(f"Invalid JSON in arguments for tool '{tool_name}': {exc}") from exc
    if not isinstance(parsed, dict):
        raise OutputParseError(
            f"Tool '{tool_name}' arguments must be a JSON object, got {type(parsed).__name__}"
        )
    return parsed


# ---------------------------------------------------------------------------
# Structured output validation
# ---------------------------------------------------------------------------

_MARKDOWN_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences wrapping JSON output.

    Some LLMs (e.g. Gemini) wrap structured output in triple-backtick fences.
    This strips ````` ```json ... ``` ````` or ````` ``` ... ``` ````` wrappers
    so that the inner content can be parsed as plain JSON.

    Args:
        text: Raw text from the LLM, possibly wrapped in a code fence.

    Returns:
        The original text with any surrounding markdown fence removed.
    """
    match = _MARKDOWN_FENCE_RE.match(text)
    if match:
        return match.group(1)
    return text


def parse_structured_output(
    text: str,
    output_type: type[_T],
) -> _T:
    """Validate LLM text output against a Pydantic model.

    Attempts to parse *text* as JSON and validate it against *output_type*.

    Args:
        text: Raw text from the LLM (expected to be JSON).
        output_type: The Pydantic model class to validate against.

    Returns:
        A validated instance of *output_type*.

    Raises:
        OutputParseError: If *text* is not valid JSON or fails validation.
    """
    try:
        data = json.loads(_strip_markdown_fences(text))
    except json.JSONDecodeError as exc:
        raise OutputParseError(f"Structured output is not valid JSON: {exc}") from exc
    try:
        return output_type.model_validate(data)
    except ValidationError as exc:
        raise OutputParseError(
            f"Structured output failed {output_type.__name__} validation: {exc}"
        ) from exc
