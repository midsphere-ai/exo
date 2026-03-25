# exo.models.types

Provider-agnostic model response types. These define the contract between LLM provider implementations and the agent core.

## Module Path

```python
from exo.models.types import (
    ModelError,
    FinishReason,
    ModelResponse,
    ToolCallDelta,
    StreamChunk,
)
```

---

## ModelError

Exception raised when an LLM provider call fails.

**Inherits:** `exo.types.ExoError`

### Constructor

```python
ModelError(message: str, *, model: str = "", code: str = "")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `message` | `str` | *(required)* | Human-readable error description |
| `model` | `str` | `""` | Model identifier that caused the error |
| `code` | `str` | `""` | Error code for classification (e.g. `"context_length"`, `"rate_limit"`) |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `model` | `str` | The model identifier |
| `code` | `str` | Error classification code |

### Example

```python
from exo.models.types import ModelError

try:
    response = await provider.complete(messages)
except ModelError as e:
    print(f"Model: {e.model}")
    print(f"Code: {e.code}")
    print(f"Message: {e}")
```

---

## FinishReason

```python
FinishReason = Literal["stop", "tool_calls", "length", "content_filter"]
```

Why the model stopped generating. Providers normalize their native values to these four categories:

| Value | Description | Anthropic Equivalent |
|---|---|---|
| `"stop"` | Natural completion | `"end_turn"`, `"stop_sequence"` |
| `"tool_calls"` | Model wants to invoke tools | `"tool_use"` |
| `"length"` | Hit max tokens limit | `"max_tokens"` |
| `"content_filter"` | Content was filtered by the provider | -- |

---

## ModelResponse

Response from a non-streaming LLM completion call. Returned by `ModelProvider.complete()`.

**Inherits:** `pydantic.BaseModel` (frozen)

### Constructor

```python
ModelResponse(
    id: str = "",
    model: str = "",
    content: str = "",
    tool_calls: list[ToolCall] = [],
    usage: Usage = Usage(),
    finish_reason: FinishReason = "stop",
    reasoning_content: str = "",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | `""` | Provider-assigned correlation ID |
| `model` | `str` | `""` | Which model produced this response |
| `content` | `str` | `""` | Text output from the model |
| `tool_calls` | `list[ToolCall]` | `[]` | Tool invocations requested by the model |
| `usage` | `Usage` | `Usage()` | Token usage statistics |
| `finish_reason` | `FinishReason` | `"stop"` | Why the model stopped generating |
| `reasoning_content` | `str` | `""` | Chain-of-thought content for reasoning models (o1/o3) |

### Example

```python
response = await provider.complete(messages)

print(response.content)        # "The answer is 42."
print(response.finish_reason)  # "stop"
print(response.usage.total_tokens)  # 150

if response.tool_calls:
    for tc in response.tool_calls:
        print(f"Tool: {tc.name}, Args: {tc.arguments}")

if response.reasoning_content:
    print(f"Reasoning: {response.reasoning_content}")
```

---

## ToolCallDelta

An incremental fragment of a streamed tool call. Yielded inside `StreamChunk.tool_call_deltas`.

**Inherits:** `pydantic.BaseModel` (frozen)

### Constructor

```python
ToolCallDelta(
    index: int = 0,
    id: str | None = None,
    name: str | None = None,
    arguments: str = "",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `index` | `int` | `0` | Position in multi-tool-call responses |
| `id` | `str \| None` | `None` | Tool call ID, present only in the first chunk |
| `name` | `str \| None` | `None` | Tool name, present only in the first chunk |
| `arguments` | `str` | `""` | Incremental JSON fragment of arguments |

### Usage Pattern

Tool call deltas arrive across multiple stream chunks. The `id` and `name` are set only on the first delta for each tool call. Subsequent deltas for the same tool call carry only `arguments` fragments that must be concatenated.

```python
tool_calls: dict[int, dict] = {}

async for chunk in await provider.stream(messages, tools=tool_defs):
    for delta in chunk.tool_call_deltas:
        if delta.index not in tool_calls:
            tool_calls[delta.index] = {"id": delta.id, "name": delta.name, "args": ""}
        tool_calls[delta.index]["args"] += delta.arguments
```

---

## StreamChunk

A single chunk yielded during streaming LLM completion. Yielded by `ModelProvider.stream()`.

**Inherits:** `pydantic.BaseModel` (frozen)

### Constructor

```python
StreamChunk(
    delta: str = "",
    tool_call_deltas: list[ToolCallDelta] = [],
    finish_reason: FinishReason | None = None,
    usage: Usage = Usage(),
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `delta` | `str` | `""` | Incremental text content |
| `tool_call_deltas` | `list[ToolCallDelta]` | `[]` | Incremental tool call fragments |
| `finish_reason` | `FinishReason \| None` | `None` | Non-None only on the final chunk |
| `usage` | `Usage` | `Usage()` | Token usage, typically only on the final chunk |

### Example

```python
full_text = ""
async for chunk in await provider.stream(messages):
    if chunk.delta:
        full_text += chunk.delta
        print(chunk.delta, end="", flush=True)

    if chunk.finish_reason:
        print(f"\nFinished: {chunk.finish_reason}")
        print(f"Tokens used: {chunk.usage.total_tokens}")
```

---

## Related Types (from exo.types)

These types are used in the models API but defined in `exo-core`:

- **`Usage`** -- Token usage statistics with `input_tokens`, `output_tokens`, `total_tokens` fields
- **`ToolCall`** -- A tool invocation request with `id`, `name`, `arguments` fields
- **`Message`** -- Union type of `SystemMessage | UserMessage | AssistantMessage | ToolResult`
