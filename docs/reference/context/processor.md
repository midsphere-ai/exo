# exo.context.processor

Event-driven context processing pipeline. Processors intervene at specific points in the LLM execution cycle to dynamically transform context.

## Module Path

```python
from exo.context.processor import (
    ContextProcessor,
    ProcessorPipeline,
    ProcessorError,
    SummarizeProcessor,
    ToolResultOffloader,
)
```

---

## ProcessorError

Exception raised for processor pipeline errors.

```python
class ProcessorError(Exception): ...
```

---

## ContextProcessor (ABC)

Abstract base for context processors. Each processor handles a single event type.

### Constructor

```python
ContextProcessor(event: str, *, name: str | None = None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `event` | `str` | *(required)* | The event type this processor handles (must be non-empty) |
| `name` | `str \| None` | `None` | Human-readable name (defaults to class name) |

**Raises:** `ProcessorError` if `event` is empty.

### Properties

| Property | Type | Description |
|---|---|---|
| `event` | `str` | The event type this processor handles |
| `name` | `str` | Human-readable processor name |

### Abstract Methods

#### process()

```python
async def process(self, ctx: Context, payload: dict[str, Any]) -> None
```

Process the event with context and payload. Implementations may mutate `ctx.state` to transform context.

| Parameter | Type | Description |
|---|---|---|
| `ctx` | `Context` | The current context |
| `payload` | `dict[str, Any]` | Event-specific data |

---

## ProcessorPipeline

Registers and fires context processors by event type. Processors are called sequentially in registration order for each event. Errors propagate immediately.

### Constructor

```python
ProcessorPipeline()
```

No parameters.

### Methods

#### register()

```python
def register(self, processor: ContextProcessor) -> ProcessorPipeline
```

Register a processor for its declared event type. Returns `self` for chaining.

#### unregister()

```python
def unregister(self, processor: ContextProcessor) -> None
```

Remove a processor. Silently does nothing if not registered.

#### fire()

```python
async def fire(
    self,
    event: str,
    ctx: Context,
    payload: dict[str, Any] | None = None,
) -> None
```

Fire all processors registered for the event, in registration order.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `event` | `str` | *(required)* | The event type to fire |
| `ctx` | `Context` | *(required)* | The context passed to each processor |
| `payload` | `dict[str, Any] \| None` | `None` | Optional event-specific data (defaults to `{}`) |

#### has_processors()

```python
def has_processors(self, event: str) -> bool
```

Check whether any processors are registered for the event.

#### list_processors()

```python
def list_processors(self, event: str | None = None) -> list[ContextProcessor]
```

List processors, optionally filtered by event. If `event` is `None`, returns all processors.

#### clear()

```python
def clear(self) -> None
```

Remove all processors.

### Dunder Methods

| Method | Description |
|---|---|
| `__len__` | Total number of registered processors across all events |
| `__repr__` | `ProcessorPipeline(events=['pre_llm_call', 'post_tool_call'], total=2)` |

---

## SummarizeProcessor

Marks context for summarization when history exceeds a threshold. Fires on `"pre_llm_call"`.

**Inherits:** `ContextProcessor`

### Constructor

```python
SummarizeProcessor(*, name: str = "summarize")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"summarize"` | Processor name |

### Behavior

Checks the `history` list in `ctx.state` against `ctx.config.summary_threshold`. When exceeded:

1. Sets `needs_summary = True` in state
2. Stores the oldest excess messages under `summary_candidates` in state

### State Effects

| Key | Type | Description |
|---|---|---|
| `needs_summary` | `bool` | Set to `True` when summarization needed |
| `summary_candidates` | `list[dict]` | Messages to be summarized |

---

## ToolResultOffloader

Offloads large tool results to workspace. Fires on `"post_tool_call"`.

**Inherits:** `ContextProcessor`

### Constructor

```python
ToolResultOffloader(*, max_size: int = 5000, name: str = "tool_result_offloader")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_size` | `int` | `5000` | Maximum character length before offloading |
| `name` | `str` | `"tool_result_offloader"` | Processor name |

### Properties

| Property | Type | Description |
|---|---|---|
| `max_size` | `int` | Maximum content size before offloading |

### Behavior

When a tool result's content exceeds `max_size` characters:

1. Stores full content under `offloaded_results` in state
2. Replaces `payload["tool_result"]` with a truncated reference

### Expected Payload Keys

| Key | Type | Description |
|---|---|---|
| `tool_result` | `str` | The tool result content |
| `tool_name` | `str` | Name of the tool |
| `tool_call_id` | `str` | ID of the tool call |

---

## Example

```python
import asyncio
from exo.context import (
    Context, ContextConfig,
    ProcessorPipeline, SummarizeProcessor, ToolResultOffloader,
)

async def main():
    ctx = Context("task-1", config=ContextConfig(summary_threshold=3))

    # Set up history
    ctx.state.set("history", [
        {"role": "user", "content": "msg1"},
        {"role": "assistant", "content": "reply1"},
        {"role": "user", "content": "msg2"},
        {"role": "assistant", "content": "reply2"},
    ])

    # Build pipeline
    pipeline = ProcessorPipeline()
    pipeline.register(SummarizeProcessor())
    pipeline.register(ToolResultOffloader(max_size=100))

    # Fire pre_llm_call
    await pipeline.fire("pre_llm_call", ctx)
    print(ctx.state.get("needs_summary"))  # True
    print(len(ctx.state.get("summary_candidates")))  # 1 (excess messages)

    # Fire post_tool_call with a large result
    payload = {
        "tool_result": "x" * 200,
        "tool_name": "search",
        "tool_call_id": "tc-1",
    }
    await pipeline.fire("post_tool_call", ctx, payload)
    print(len(payload["tool_result"]))  # truncated

asyncio.run(main())
```
