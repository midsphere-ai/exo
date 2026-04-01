---
name: exo:streaming
description: "Use when consuming Exo agent output via streaming — run.stream(), StreamEvent types (TextEvent, ToolCallEvent, StepEvent, ToolResultEvent, UsageEvent, StatusEvent, ErrorEvent, ContextEvent, MCPProgressEvent, ReasoningEvent, MessageInjectedEvent, RalphIterationEvent, RalphStopEvent), ToolContext event forwarding from nested agents, event filtering, detailed mode. Triggers on: run.stream, streaming, TextEvent, ToolCallEvent, StepEvent, stream events, event_types, detailed, real-time output, ToolContext, nested streaming, inner agent events, RalphIterationEvent, RalphStopEvent."
---

> **Branch:** These skills are written for the `rename/orbiter-to-exo` branch. The Exo APIs referenced here may differ on other branches.

# Exo Streaming — Streaming Execution & Events

## When To Use This Skill

Use this skill when the developer needs to:
- Stream agent output in real-time using `run.stream()`
- Understand the 13 event types and when each is emitted
- Filter events by type
- Use `detailed=True` mode for rich execution telemetry
- Stream from Swarms with per-agent event attribution
- Forward inner agent events through tools via `ToolContext`
- Build real-time UIs or logging from streamed events

## Decision Guide

1. **Just want text output in real-time?** → `run.stream(agent, input, provider=p)` — yields `TextEvent` and `ToolCallEvent` by default
2. **Want full execution telemetry?** → Add `detailed=True` — also emits `StepEvent`, `UsageEvent`, `ToolResultEvent`, `StatusEvent`
3. **Only care about specific events?** → Pass `event_types={"text", "tool_call"}` to filter
4. **Streaming a Swarm?** → Use `swarm.stream()` — all events include `agent_name` showing which agent produced them
5. **Building a UI?** → Use `detailed=True` with `event_types` filter for the events you need
6. **Tool runs an inner agent and you want its events?** → Declare `ctx: ToolContext` in the tool, call `ctx.emit(event)` for each inner event
7. **Streaming a Ralph loop?** → Use `RalphRunner.stream()` — yields `RalphIterationEvent` + inner events + `RalphStopEvent`

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

### All 13 Event Types

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

#### RalphIterationEvent (always emitted during Ralph streaming)

```python
class RalphIterationEvent:
    type: Literal["ralph_iteration"] = "ralph_iteration"
    iteration: int                  # 1-based iteration number
    status: Literal["started", "completed", "failed"]
    scores: dict[str, float] = {}   # Scorer results (on completed)
    agent_name: str = ""
```

Emitted at the start and end of each Ralph loop iteration. On `"completed"`, includes the scorer results for that iteration.

#### RalphStopEvent (always emitted when Ralph loop terminates)

```python
class RalphStopEvent:
    type: Literal["ralph_stop"] = "ralph_stop"
    stop_type: str                  # e.g. "max_iterations", "score_threshold"
    reason: str                     # Human-readable stop reason
    iterations: int                 # Total iterations executed
    final_scores: dict[str, float] = {}
    agent_name: str = ""
```

Emitted once when the Ralph loop terminates, with the stop reason and final scores.

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
| `RalphIterationEvent` | Yes (during Ralph streaming) | Yes (during Ralph streaming) |
| `RalphStopEvent` | Yes (during Ralph streaming) | Yes (during Ralph streaming) |
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
    "message_injected", "ralph_iteration", "ralph_stop"
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

### Forwarding Inner Agent Events via ToolContext

When a tool runs an inner agent, use `ToolContext` to forward its events to the parent stream:

```python
from exo import tool, Agent, run, ToolContext
from exo.types import TextEvent

@tool
async def research(query: str, ctx: ToolContext) -> str:
    """Run a research agent and forward its events.

    Args:
        query: The research query.
    """
    inner_agent = Agent(name="researcher", instructions="Research deeply.")
    parts = []
    async for event in run.stream(inner_agent, query, provider=provider):
        ctx.emit(event)  # Forward to parent stream
        if isinstance(event, TextEvent):
            parts.append(event.text)
    return "".join(parts)

# Parent agent — tool events appear in its stream
parent = Agent(name="orchestrator", tools=[research])
async for event in run.stream(parent, "Analyze AI trends", provider=provider):
    print(f"[{event.agent_name}] {event.type}: ", end="")
    if isinstance(event, TextEvent):
        print(event.text)
```

**How it works:**
1. Tool declares `ctx: ToolContext` — auto-injected, excluded from LLM schema
2. `ctx.emit(event)` pushes events to the parent agent's `_event_queue`
3. After tool execution, `run.stream()` drains the queue and yields all buffered events
4. Inner events retain their original `agent_name` — consumers can distinguish them from outer agent events

**Zero cost when not streaming:** The queue exists but is never drained during `run()` (non-streaming). Events are discarded on GC.

### Streaming a Ralph Loop

```python
from exo.eval.ralph import RalphRunner, RalphIterationEvent, RalphStopEvent
from exo.types import TextEvent

runner = RalphRunner.from_agent(agent, scorers=[quality_scorer])

async for event in runner.stream("Analyze market trends", name="research_loop"):
    match event:
        case RalphIterationEvent(iteration=n, status="started"):
            print(f"\n--- Iteration {n} ---")
        case TextEvent(text=t):
            print(t, end="", flush=True)
        case RalphIterationEvent(iteration=n, status="completed", scores=s):
            print(f"\n[Scores: {s}]")
        case RalphStopEvent(stop_type=st, reason=r, iterations=n):
            print(f"\n=== Stopped ({st}): {r} after {n} iterations ===")
```

**Event sequence per iteration:**
```
RalphIterationEvent(status="started", iteration=1)
TextEvent(...)           ← inner agent events
ToolCallEvent(...)       ← inner agent events
TextEvent(...)           ← inner agent events
RalphIterationEvent(status="completed", iteration=1, scores={...})
... (next iteration or stop)
RalphStopEvent(stop_type="max_iterations", iterations=3)
```

## Gotchas

- **All events are frozen Pydantic models** — `event.text = "new"` raises `ValidationError`
- **ErrorEvent is emitted before exception propagates** — the exception still reaches the caller
- **`event_types` filter applies to ALL events including errors** — if you filter out `"error"`, you won't see ErrorEvents (but the exception still raises)
- **`agent_name` defaults to `""`** — always set by the runtime for agent/swarm execution, but default is empty string
- **Tool call accumulation** — tool calls are built incrementally from stream deltas. The `ToolCallEvent` is emitted after accumulation is complete, not during.
- **MCP progress is drained after tool execution** — not during. Progress events arrive in a batch after tools complete.
- **ToolContext events are also drained after tool execution** — inner agent events are buffered and yielded after all tools complete, not during.
- **ToolContext only works with `FunctionTool`** — custom `Tool` ABC subclasses and MCP tools don't support ToolContext injection. Use `FunctionTool` or `@tool` decorator.
- **ToolContext.emit() must be called from the event loop thread** — sync tools wrapped via `asyncio.to_thread()` cannot safely call `emit()`. Use async tools for inner agent streaming.
- **Hooks still fire during streaming** — `START`, `POST_LLM_CALL`, `FINISHED`, `ERROR` hooks all fire normally
- **Swarm.stream() delegates to run.stream()** — each sub-agent is streamed via `run.stream()` internally
- **Usage comes from the final stream chunk** — if the provider doesn't include usage in stream chunks, `UsageEvent` may have zero values
