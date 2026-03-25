"""Build the message list for LLM provider calls.

The message builder constructs a correctly ordered sequence of messages
from system instructions, conversation history, and pending tool results.
It ensures API compatibility: no dangling tool calls without results,
and proper system → user → assistant → tool cycling.
"""

from __future__ import annotations

from collections.abc import Sequence

from exo.types import (
    AssistantMessage,
    Message,
    MessageContent,
    SystemMessage,
    ToolResult,
    UserMessage,
)


def _content_is_empty(content: MessageContent) -> bool:
    """Return True when a MessageContent value is semantically empty.

    Args:
        content: A string or list of ContentBlock objects.

    Returns:
        True if the content is an empty string or an empty list.
    """
    if isinstance(content, str):
        return not content
    return len(content) == 0


def build_messages(
    instructions: str,
    history: Sequence[Message],
    *,
    tool_results: Sequence[ToolResult] | None = None,
) -> list[Message]:
    """Build the message list for an LLM call.

    Constructs a correctly ordered message sequence from system instructions,
    conversation history, and any pending tool results.

    Args:
        instructions: The system prompt. If empty, no system message is added.
        history: Previous conversation messages.
        tool_results: Results from tool calls to append at the end.

    Returns:
        Ordered list of messages ready for the LLM provider.
    """
    messages: list[Message] = []

    if instructions:
        messages.append(SystemMessage(content=instructions))

    messages.extend(history)

    if tool_results:
        messages.extend(tool_results)

    return messages


def validate_message_order(messages: Sequence[Message]) -> list[str]:
    """Check message ordering for common provider API issues.

    Detects problems like dangling tool calls (assistant requested tool
    calls but no corresponding tool results follow) that would cause
    provider API errors.

    Args:
        messages: The message list to validate.

    Returns:
        A list of warning strings. Empty if no issues found.
    """
    warnings: list[str] = []
    pending_tool_call_ids: set[str] = set()

    for msg in messages:
        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                pending_tool_call_ids.add(tc.id)
        elif isinstance(msg, ToolResult):
            pending_tool_call_ids.discard(msg.tool_call_id)

    if pending_tool_call_ids:
        ids = ", ".join(sorted(pending_tool_call_ids))
        warnings.append(f"Dangling tool calls without results: {ids}")

    return warnings


def extract_last_assistant_tool_calls(
    messages: Sequence[Message],
) -> list[str]:
    """Get tool call IDs from the last assistant message, if any.

    Useful for checking whether the conversation is mid-tool-execution
    (the assistant requested tools but results haven't been appended yet).

    Args:
        messages: The message list to inspect.

    Returns:
        List of tool call IDs from the final assistant message,
        or empty list if the last message isn't an assistant with tool calls.
    """
    for msg in reversed(messages):
        if isinstance(msg, AssistantMessage):
            return [tc.id for tc in msg.tool_calls]
        # Stop at user messages — don't look past the current turn
        if isinstance(msg, UserMessage):
            break
    return []


def merge_usage(
    current_input: int,
    current_output: int,
    new_input: int,
    new_output: int,
) -> tuple[int, int, int]:
    """Accumulate token usage across multiple LLM calls.

    Args:
        current_input: Running total of input tokens.
        current_output: Running total of output tokens.
        new_input: Input tokens from the latest call.
        new_output: Output tokens from the latest call.

    Returns:
        Tuple of (total_input, total_output, total) tokens.
    """
    total_in = current_input + new_input
    total_out = current_output + new_output
    return total_in, total_out, total_in + total_out
