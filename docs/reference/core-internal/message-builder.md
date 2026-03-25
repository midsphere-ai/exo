# exo._internal.message_builder

Build the message list for LLM provider calls. The message builder constructs a correctly ordered sequence of messages from system instructions, conversation history, and pending tool results. It ensures API compatibility: no dangling tool calls without results, and proper message cycling.

> **Internal API** -- subject to change without notice.

**Module:** `exo._internal.message_builder`

```python
from exo._internal.message_builder import (
    build_messages,
    validate_message_order,
    extract_last_assistant_tool_calls,
    merge_usage,
)
```

---

## build_messages()

```python
def build_messages(
    instructions: str,
    history: Sequence[Message],
    *,
    tool_results: Sequence[ToolResult] | None = None,
) -> list[Message]
```

Build the message list for an LLM call. Constructs a correctly ordered message sequence from system instructions, conversation history, and any pending tool results.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `instructions` | `str` | *(required)* | The system prompt. If empty, no system message is added. |
| `history` | `Sequence[Message]` | *(required)* | Previous conversation messages. |
| `tool_results` | `Sequence[ToolResult] \| None` | `None` | Results from tool calls to append at the end. |

### Returns

Ordered `list[Message]` ready for the LLM provider. The order is:
1. `SystemMessage` (if `instructions` is non-empty)
2. All messages from `history`
3. All messages from `tool_results` (if provided)

### Example

```python
from exo._internal.message_builder import build_messages
from exo.types import UserMessage

messages = build_messages(
    "You are a helpful assistant.",
    [UserMessage(content="Hello")],
)
# [SystemMessage(content="You are..."), UserMessage(content="Hello")]
```

---

## validate_message_order()

```python
def validate_message_order(messages: Sequence[Message]) -> list[str]
```

Check message ordering for common provider API issues. Detects problems like dangling tool calls (assistant requested tool calls but no corresponding tool results follow) that would cause provider API errors.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `messages` | `Sequence[Message]` | *(required)* | The message list to validate. |

### Returns

A list of warning strings. Empty if no issues found.

### Example

```python
from exo._internal.message_builder import validate_message_order
from exo.types import AssistantMessage, ToolCall

messages = [
    AssistantMessage(
        content="",
        tool_calls=[ToolCall(id="call_1", name="search", arguments="{}")]
    ),
    # Missing ToolResult for call_1
]

warnings = validate_message_order(messages)
# ["Dangling tool calls without results: call_1"]
```

---

## extract_last_assistant_tool_calls()

```python
def extract_last_assistant_tool_calls(messages: Sequence[Message]) -> list[str]
```

Get tool call IDs from the last assistant message, if any. Useful for checking whether the conversation is mid-tool-execution (the assistant requested tools but results haven't been appended yet).

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `messages` | `Sequence[Message]` | *(required)* | The message list to inspect. |

### Returns

List of tool call IDs from the final assistant message, or empty list if the last message is not an assistant message with tool calls. Stops searching at user messages (does not look past the current turn).

### Example

```python
from exo._internal.message_builder import extract_last_assistant_tool_calls
from exo.types import AssistantMessage, ToolCall, UserMessage

messages = [
    UserMessage(content="Search for X"),
    AssistantMessage(
        content="",
        tool_calls=[ToolCall(id="call_1", name="search", arguments='{"q":"X"}')]
    ),
]

ids = extract_last_assistant_tool_calls(messages)
# ["call_1"]
```

---

## merge_usage()

```python
def merge_usage(
    current_input: int,
    current_output: int,
    new_input: int,
    new_output: int,
) -> tuple[int, int, int]
```

Accumulate token usage across multiple LLM calls.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `current_input` | `int` | *(required)* | Running total of input tokens. |
| `current_output` | `int` | *(required)* | Running total of output tokens. |
| `new_input` | `int` | *(required)* | Input tokens from the latest call. |
| `new_output` | `int` | *(required)* | Output tokens from the latest call. |

### Returns

Tuple of `(total_input, total_output, total)` tokens.

### Example

```python
from exo._internal.message_builder import merge_usage

total_in, total_out, total = merge_usage(100, 50, 80, 30)
# total_in=180, total_out=80, total=260
```
