---
name: exo:streaming
description: "Use when consuming Exo agent output via streaming — run.stream(), StreamEvent types (TextEvent, ToolCallEvent, StepEvent, ToolResultEvent, UsageEvent, StatusEvent, ErrorEvent, ContextEvent, MCPProgressEvent, ReasoningEvent, MessageInjectedEvent), event filtering, detailed mode. Triggers on: run.stream, streaming, TextEvent, ToolCallEvent, StepEvent, stream events, event_types, detailed, real-time output."
---

> **Branch:** These skills are written for the `rename/orbiter-to-exo` branch. The Exo APIs referenced here may differ on other branches.

# Exo Streaming — Streaming Execution & Events

## When To Use This Skill

Use this skill when the developer needs to:
- Stream agent output in real-time using `run.stream()`
- Understand the 11 event types and when each is emitted
- Filter events by type
- Use `detailed=True` mode for rich execution telemetry
- Stream from Swarms with per-agent event attribution
- Build real-time UIs or logging from streamed events

## Decision Guide

1. **Just want text output in real-time?** → `run.stream(agent, input, provider=p)` — yields `TextEvent` and `ToolCallEvent` by default
2. **Want full execution telemetry?** → Add `detailed=True` — also emits `StepEvent`, `UsageEvent`, `ToolResultEvent`, `StatusEvent`
3. **Only care about specific events?** → Pass `event_types={"text", "tool_call"}` to filter
4. **Streaming a Swarm?** → Use `swarm.stream()` — all events include `agent_name` showing which agent produced them
5. **Building a UI?** → Use `detailed=True` with `event_types` filter for the events you need

## Reference

### run.stream() Signature

```python
from exo import run

async for event in run.stream(
    agent,                          # Agent or Swarm instance
    "User query",                   # Input string or list[ContentBlock]
    messages=None,                  # Optional prior conversation history
    provider=provider,              # LLM provider
    max_steps=None,                 # Override agent's max_steps
    detailed=False,                 # When True, emit rich event types
    event_types=None,               # Set of event type strings to filter
    conversation_id=None,           # Memory/conversation tracking ID
):
    ...
```

### All 11 Event Types

#### TextEvent (always emitted)

```python
class TextEvent:
    type: Literal["text"] = "text"
    text: str                       # Text delta from LLM stream
    agent_name: str = ""
```

Emitted for each text chunk from the LLM. Concatenate all `text` fields to reconstruct full output.

#### ToolCallEvent (always emitted)

```python
class ToolCallEvent:
    type: Literal["tool_call"] = "tool_call"
    tool_name: str                  # Name of tool being called
    tool_call_id: str               # Unique ID for this call
    agent_name: str = ""
```

Emitted when the LLM requests a tool execution. One event per tool call.

#### ErrorEvent (always emitted)

```python
class ErrorEvent:
    type: Literal["error"] = "error"
    error: str                      # Error message
    error_type: str                 # Exception class name
    agent_name: str = ""
    step_number: int | None = None
    recoverable: bool = False
```

Emitted before an exception is re-raised. The exception still propagates to the caller.

#### ContextEvent (always emitted when context operations occur)

```python
class ContextEvent:
    type: Literal["context"] = "context"
    action: Literal["offload", "summarize", "window", "token_budget"]
    agent_name: str = ""
    before_count: int = 0           # Message count before operation
    after_count: int = 0            # Message count after operation
    details: dict[str, Any] = {}    # Action-specific metadata
```

**Actions:**
- `"offload"` — Aggressive trimming past offload threshold
- `"summarize"` — LLM-based conversation summarization
- `"window"` — History rounds trimming
- `"token_budget"` — Token fill ratio exceeded threshold

#### MCPProgressEvent (always emitted when MCP progress arrives)

```python
class MCPProgressEvent:
    type: Literal["mcp_progress"] = "mcp_progress"
    tool_name: str
    progress: int
    total: int | None = None
    message: str = ""
    agent_name: str = ""
```

Emitted after tool execution when MCP tools report progress.

#### MessageInjectedEvent (always emitted when messages injected)

```python
class MessageInjectedEvent:
    type: Literal["message_injected"] = "message_injected"
    content: str                    # The injected message text
    agent_name: str = ""
```

Emitted when `agent.inject_message()` pushes content into the running agent.

#### ReasoningEvent (emitted when model supports reasoning)

```python
class ReasoningEvent:
    type: Literal["reasoning"] = "reasoning"
    text: str                       # Extended thinking/reasoning content
    agent_name: str = ""
```

Emitted for models that support extended thinking (e.g., o1-style reasoning).

#### StepEvent (detailed=True only)

```python
class StepEvent:
    type: Literal["step"] = "step"
    step_number: int
    agent_name: str
    status: Literal["started", "completed"]
    started_at: float               # time.monotonic() timestamp
    completed_at: float | None = None
    usage: Usage | None = None      # Token usage (on completion)
```

Emitted at the start and end of each LLM call step.

#### ToolResultEvent (detailed=True only)

```python
class ToolResultEvent:
    type: Literal["tool_result"] = "tool_result"
    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any] = {}  # Parsed arguments dict
    result: str | list = ""         # Tool return value
    error: str | None = None        # Error message if tool failed
    success: bool = True
    duration_ms: float = 0.0        # Execution time in milliseconds
    agent_name: str = ""
```

Emitted after each tool execution with full details.

#### UsageEvent (detailed=True only)

```python
class UsageEvent:
    type: Literal["usage"] = "usage"
    usage: Usage                    # Usage(input_tokens, output_tokens, total_tokens)
    agent_name: str = ""
    step_number: int = 0
    model: str = ""
```

Emitted once per LLM call step with token usage.

#### StatusEvent (detailed=True only)

```python
class StatusEvent:
    type: Literal["status"] = "status"
    status: Literal["starting", "running", "waiting_for_tool", "completed", "cancelled", "error"]
    agent_name: str = ""
    message: str = ""
```

Emitted for execution status changes. Particularly useful in Swarm mode for handoff/delegation tracking.

### Event Emission Summary

| Event Type | `detailed=False` | `detailed=True` |
|-----------|-----------------|-----------------|
| `TextEvent` | Yes | Yes |
| `ToolCallEvent` | Yes | Yes |
| `ErrorEvent` | Yes | Yes |
| `ContextEvent` | Yes (when triggered) | Yes (when triggered) |
| `MCPProgressEvent` | Yes (when triggered) | Yes (when triggered) |
| `MessageInjectedEvent` | Yes (when triggered) | Yes (when triggered) |
| `ReasoningEvent` | Yes (when model supports) | Yes (when model supports) |
| `StepEvent` | No | Yes |
| `ToolResultEvent` | No | Yes |
| `UsageEvent` | No | Yes |
| `StatusEvent` | No | Yes |

### Event Filtering

```python
# Only text and tool calls
async for event in run.stream(
    agent, "Hi",
    provider=provider,
    event_types={"text", "tool_call"},
):
    ...

# Only detailed telemetry
async for event in run.stream(
    agent, "Hi",
    provider=provider,
    detailed=True,
    event_types={"step", "usage", "tool_result"},
):
    ...
```

**Valid event_types strings:**
```python
{
    "text", "tool_call", "step", "tool_result", "reasoning",
    "error", "status", "usage", "mcp_progress", "context",
    "message_injected"
}
```

When `event_types=None` (default), all events pass. When `event_types=set()`, no events are yielded.

### Event Sequence: Text-Only Response

```
detailed=False:                    detailed=True:
  TextEvent("Hello")                StatusEvent(starting)
  TextEvent(" world")              StepEvent(started, step=1)
  TextEvent("!")                    TextEvent("Hello")
                                    TextEvent(" world")
                                    TextEvent("!")
                                    UsageEvent(tokens=...)
                                    StepEvent(completed, step=1)
                                    StatusEvent(completed)
```

### Event Sequence: With Tool Call

```
detailed=True:
  StatusEvent(starting)
  StepEvent(started, step=1)
  ToolCallEvent(search, tc_1)
  UsageEvent(...)
  StepEvent(completed, step=1)       ← LLM call done
  ToolResultEvent(search, tc_1, result="...", duration_ms=150)
  StepEvent(started, step=2)         ← Next LLM call with tool result
  TextEvent("Based on the search...")
  TextEvent("...")
  UsageEvent(...)
  StepEvent(completed, step=2)
  StatusEvent(completed)
```

### Streaming from Swarms

```python
async for event in swarm.stream(
    "Query",
    provider=provider,
    detailed=True,
    event_types={"text", "status"},
):
    if isinstance(event, StatusEvent):
        print(f"\n--- {event.message} ---")
    elif isinstance(event, TextEvent):
        print(event.text, end="", flush=True)
```

**Swarm-specific StatusEvents:**
- Workflow: `"Workflow executing agent 'writer'"`
- Handoff: `"Handoff from 'triage' to 'billing'"`
- Team: `"Team lead 'lead' starting execution"`
- Branch: `"Branch 'router' routing to 'specialist'"`

All events include `agent_name` — track which agent produced each event.

## Patterns

### Simple Text Streaming (CLI)

```python
async for event in run.stream(agent, "Tell me about AI", provider=provider):
    if isinstance(event, TextEvent):
        print(event.text, end="", flush=True)
print()  # Final newline
```

### Rich Progress Display

```python
from exo.types import TextEvent, ToolCallEvent, ToolResultEvent, StepEvent, UsageEvent

async for event in run.stream(agent, query, provider=provider, detailed=True):
    match event:
        case TextEvent(text=t):
            print(t, end="", flush=True)
        case ToolCallEvent(tool_name=name):
            print(f"\n[Calling {name}...]")
        case ToolResultEvent(tool_name=name, duration_ms=ms, success=ok):
            status = "done" if ok else "FAILED"
            print(f"[{name} {status} in {ms:.0f}ms]")
        case StepEvent(step_number=n, status="completed", usage=u):
            if u:
                print(f"\n[Step {n}: {u.total_tokens} tokens]")
        case UsageEvent(usage=u, model=m):
            total_tokens += u.total_tokens
```

### Collecting Full Output from Stream

```python
text_parts: list[str] = []
tool_calls: list[ToolCallEvent] = []

async for event in run.stream(agent, query, provider=provider):
    if isinstance(event, TextEvent):
        text_parts.append(event.text)
    elif isinstance(event, ToolCallEvent):
        tool_calls.append(event)

full_output = "".join(text_parts)
```

### Server-Sent Events (SSE) Endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.get("/stream")
async def stream_endpoint(query: str):
    async def generate():
        async for event in run.stream(agent, query, provider=provider, detailed=True):
            data = json.dumps({"type": event.type, **event.model_dump()})
            yield f"data: {data}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Token Budget Monitoring

```python
async for event in run.stream(agent, query, provider=provider, detailed=True):
    if isinstance(event, ContextEvent):
        if event.action == "token_budget":
            print(f"[Warning: {event.details.get('fill_ratio', 0):.0%} context used]")
        elif event.action == "summarize":
            print(f"[Context compressed: {event.before_count} → {event.after_count} messages]")
    elif isinstance(event, TextEvent):
        print(event.text, end="", flush=True)
```

## Gotchas

- **All events are frozen Pydantic models** — `event.text = "new"` raises `ValidationError`
- **ErrorEvent is emitted before exception propagates** — the exception still reaches the caller
- **`event_types` filter applies to ALL events including errors** — if you filter out `"error"`, you won't see ErrorEvents (but the exception still raises)
- **`agent_name` defaults to `""`** — always set by the runtime for agent/swarm execution, but default is empty string
- **Tool call accumulation** — tool calls are built incrementally from stream deltas. The `ToolCallEvent` is emitted after accumulation is complete, not during.
- **MCP progress is drained after tool execution** — not during. Progress events arrive in a batch after tools complete.
- **Hooks still fire during streaming** — `START`, `POST_LLM_CALL`, `FINISHED`, `ERROR` hooks all fire normally
- **Swarm.stream() delegates to run.stream()** — each sub-agent is streamed via `run.stream()` internally
- **Usage comes from the final stream chunk** — if the provider doesn't include usage in stream chunks, `UsageEvent` may have zero values
