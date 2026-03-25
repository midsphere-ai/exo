# Streaming Events

Exo provides a rich streaming event system that gives real-time visibility into agent execution. Events are emitted via `run.stream()` and can be consumed as an async iterator.

## Quick Start

```python
from exo import Agent, run

agent = Agent(name="assistant", model="gpt-4o", instructions="You are helpful.")

# Basic streaming — TextEvent and ToolCallEvent only
async for event in run.stream(agent, "What is 2 + 2?"):
    if event.type == "text":
        print(event.text, end="", flush=True)

# Detailed streaming — all event types
async for event in run.stream(agent, "What is 2 + 2?", detailed=True):
    match event.type:
        case "text":
            print(event.text, end="", flush=True)
        case "step":
            print(f"\n[Step {event.step_number} {event.status}]")
        case "tool_call":
            print(f"\n[Calling tool: {event.tool_name}]")
        case "tool_result":
            print(f"\n[Tool result: {event.result[:100]}]")
        case "error":
            print(f"\n[Error: {event.error}]")
```

## The `detailed` Parameter

By default, `run.stream()` only emits `TextEvent` and `ToolCallEvent` for backward compatibility. Set `detailed=True` to enable all rich event types:

| Parameter | Events Emitted |
|-----------|---------------|
| `detailed=False` (default) | `TextEvent`, `ToolCallEvent` |
| `detailed=True` | All event types |

`ErrorEvent` and `MessageInjectedEvent` are always emitted regardless of the `detailed` flag.

## Event Types

All events are frozen Pydantic `BaseModel` instances. Every event has a `type` field (string literal) that acts as a discriminator, and an `agent_name` field identifying which agent produced the event.

The `StreamEvent` union type includes all event types:

```python
StreamEvent = (
    TextEvent | ToolCallEvent | StepEvent | ToolResultEvent
    | ReasoningEvent | ErrorEvent | StatusEvent | UsageEvent
    | MCPProgressEvent | ContextEvent | MessageInjectedEvent
)
```

### TextEvent

Emitted for each text chunk from the LLM response stream.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal["text"]` | Always `"text"` |
| `text` | `str` | The text chunk |
| `agent_name` | `str` | Agent name (default `""`) |

```python
async for event in run.stream(agent, "Hello"):
    if event.type == "text":
        print(event.text, end="", flush=True)
```

### ToolCallEvent

Emitted when the LLM requests a tool invocation.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal["tool_call"]` | Always `"tool_call"` |
| `tool_name` | `str` | Name of the tool being called |
| `tool_call_id` | `str` | Unique identifier for this call |
| `agent_name` | `str` | Agent name (default `""`) |

```python
if event.type == "tool_call":
    print(f"Calling tool: {event.tool_name} (id: {event.tool_call_id})")
```

### StepEvent

Emitted at the start and end of each agent step (LLM call + tool execution cycle). Only emitted when `detailed=True`.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal["step"]` | Always `"step"` |
| `step_number` | `int` | Step index (1-based) |
| `agent_name` | `str` | Agent executing this step |
| `status` | `Literal["started", "completed"]` | Step phase |
| `started_at` | `float` | Timestamp when step started |
| `completed_at` | `float \| None` | Timestamp when step completed (`None` if still running) |
| `usage` | `Usage \| None` | Token usage for this step (`None` if not yet available) |

```python
if event.type == "step":
    if event.status == "started":
        print(f"Step {event.step_number} starting...")
    else:
        duration = event.completed_at - event.started_at
        print(f"Step {event.step_number} completed in {duration:.2f}s")
```

### ToolResultEvent

Emitted after each tool execution with the result and timing. Only emitted when `detailed=True`.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal["tool_result"]` | Always `"tool_result"` |
| `tool_name` | `str` | Name of the tool that was executed |
| `tool_call_id` | `str` | Links back to the originating `ToolCallEvent` |
| `arguments` | `dict[str, Any]` | Arguments passed to the tool |
| `result` | `str` | String result from the tool (default `""`) |
| `error` | `str \| None` | Error message if the tool failed (`None` on success) |
| `success` | `bool` | Whether the tool succeeded (default `True`) |
| `duration_ms` | `float` | Execution time in milliseconds (default `0.0`) |
| `agent_name` | `str` | Agent that invoked the tool (default `""`) |

```python
if event.type == "tool_result":
    status = "OK" if event.success else "FAILED"
    print(f"[{status}] {event.tool_name}: {event.result[:100]}")
    if event.error:
        print(f"  Error: {event.error}")
```

### ReasoningEvent

Emitted when the model returns reasoning/thinking content (supported by some providers). Only emitted when `detailed=True`.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal["reasoning"]` | Always `"reasoning"` |
| `text` | `str` | The reasoning text content |
| `agent_name` | `str` | Agent name (default `""`) |

```python
if event.type == "reasoning":
    print(f"[Thinking] {event.text}")
```

### ErrorEvent

Emitted when an error occurs during execution. **Always emitted regardless of the `detailed` flag.**

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal["error"]` | Always `"error"` |
| `error` | `str` | The error message |
| `error_type` | `str` | The exception class name (e.g., `"ValueError"`) |
| `agent_name` | `str` | Agent that encountered the error (default `""`) |
| `step_number` | `int \| None` | Step during which the error occurred (`None` if not step-specific) |
| `recoverable` | `bool` | Whether the error is recoverable (default `False`) |

```python
if event.type == "error":
    print(f"[ERROR] {event.error_type}: {event.error}")
    if event.step_number:
        print(f"  Occurred at step {event.step_number}")
```

### StatusEvent

Emitted for agent lifecycle status changes. Only emitted when `detailed=True` (except for error status).

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal["status"]` | Always `"status"` |
| `status` | `Literal["starting", "running", "waiting_for_tool", "completed", "cancelled", "error"]` | Current agent status |
| `agent_name` | `str` | Agent whose status changed (default `""`) |
| `message` | `str` | Human-readable description (default `""`) |

Status values:
- `"starting"` — Agent beginning execution
- `"running"` — Agent actively processing (used in Swarm mode for agent transitions)
- `"waiting_for_tool"` — Agent waiting for tool execution
- `"completed"` — Agent finished successfully
- `"cancelled"` — Agent execution was cancelled (distributed mode)
- `"error"` — Agent encountered an error

```python
if event.type == "status":
    print(f"[{event.status.upper()}] {event.agent_name}: {event.message}")
```

### UsageEvent

Emitted after each LLM call with token usage statistics. Only emitted when `detailed=True`.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal["usage"]` | Always `"usage"` |
| `usage` | `Usage` | Token usage (input_tokens, output_tokens, total_tokens) |
| `agent_name` | `str` | Agent that consumed the tokens (default `""`) |
| `step_number` | `int` | Step this usage is associated with (default `0`) |
| `model` | `str` | Model identifier used (default `""`) |

```python
if event.type == "usage":
    u = event.usage
    print(f"Tokens: {u.input_tokens} in / {u.output_tokens} out / {u.total_tokens} total")
    print(f"  Model: {event.model}, Step: {event.step_number}")
```

### MessageInjectedEvent

Emitted when a message is injected into a running agent via `agent.inject_message()`. This allows external code to push additional context ("Also check X", "Actually use Y instead") into the agent's conversation without cancelling the run. Always emitted (not gated by `detailed`).

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal["message_injected"]` | Always `"message_injected"` |
| `content` | `str` | The injected message text |
| `agent_name` | `str` | Agent name (default `""`) |

```python
if event.type == "message_injected":
    print(f"[Injected] {event.content}")
```

See [Live Message Injection](#live-message-injection) below for full usage examples.

## Event Emission Order

When `detailed=True`, events are emitted in this order for each step:

```
StatusEvent(status="starting")           # Once at start
├── MessageInjectedEvent(...)            # Per injected message (if any)
├── StepEvent(status="started")          # Per step
├── TextEvent(text="...")                # Text chunks from LLM
├── UsageEvent(...)                      # After LLM stream completes
├── ToolCallEvent(...)                   # Per tool call (if any)
├── ToolResultEvent(...)                 # Per tool result (if detailed)
├── StepEvent(status="completed")        # Step done
├── [next step repeats...]
StatusEvent(status="completed")          # Once at end
```

On error at any point:
```
ErrorEvent(...)                          # Always emitted
StatusEvent(status="error")              # Only when detailed=True
```

## Event Filtering

Filter which event types you receive using the `event_types` parameter:

```python
# Only receive text and tool results
async for event in run.stream(
    agent, "Search for Python docs",
    detailed=True,
    event_types={"text", "tool_result"},
):
    match event.type:
        case "text":
            print(event.text, end="")
        case "tool_result":
            print(f"\n[Tool: {event.tool_name}] {event.result[:200]}")

# Only receive status updates
async for event in run.stream(
    agent, "Do the task",
    detailed=True,
    event_types={"status", "error"},
):
    print(f"[{event.type}] {getattr(event, 'status', event.error)}")
```

When `event_types` is `None` (default), all events pass through (respecting the `detailed` flag). An empty set `set()` filters out everything.

Available type strings for filtering: `"text"`, `"tool_call"`, `"step"`, `"tool_result"`, `"reasoning"`, `"error"`, `"status"`, `"usage"`, `"mcp_progress"`, `"context"`, `"message_injected"`.

## Multi-Agent Streaming (Swarm)

When streaming a `Swarm`, all events include the correct `agent_name` of the sub-agent that produced them. Additional `StatusEvent` events are emitted for agent transitions:

```python
from exo import Agent, Swarm, run

researcher = Agent(name="researcher", model="gpt-4o", instructions="Research topics.")
writer = Agent(name="writer", model="gpt-4o", instructions="Write articles.")

swarm = Swarm(
    agents=[researcher, writer],
    flow="researcher >> writer",
    mode="workflow",
)

async for event in run.stream(swarm, "Write about Python", detailed=True):
    match event.type:
        case "status":
            print(f"\n--- [{event.agent_name}] {event.message} ---")
        case "text":
            print(event.text, end="", flush=True)
        case "step":
            print(f"\n[{event.agent_name}] Step {event.step_number} {event.status}")
```

### Swarm Modes and Events

**Workflow mode** (`mode="workflow"`): Agents run sequentially. A `StatusEvent(status="running")` is emitted for each agent transition. Text output from each agent is collected and passed as input to the next agent.

**Handoff mode** (`mode="handoff"`): Agents delegate dynamically. A `StatusEvent` with `"Handoff from 'X' to 'Y'"` message is emitted for each transition.

**Team mode** (`mode="team"`): The lead agent delegates to workers. A `StatusEvent(status="running")` is emitted when the lead starts.

## SSE Integration (FastAPI)

Stream events to a frontend via Server-Sent Events:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from exo import Agent, run

app = FastAPI()
agent = Agent(name="assistant", model="gpt-4o", instructions="You are helpful.")

@app.get("/chat")
async def chat(message: str):
    async def event_generator():
        async for event in run.stream(agent, message, detailed=True):
            data = event.model_dump_json()
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
```

### SSE with Event Type Routing

Use the event `type` field as the SSE event name for client-side routing:

```python
@app.get("/chat")
async def chat(message: str):
    async def event_generator():
        async for event in run.stream(agent, message, detailed=True):
            data = event.model_dump_json()
            yield f"event: {event.type}\ndata: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
```

### Django SSE

```python
from django.http import StreamingHttpResponse
from exo import Agent, run
import asyncio

agent = Agent(name="assistant", model="gpt-4o", instructions="You are helpful.")

async def chat_view(request):
    message = request.GET.get("message", "")

    async def event_generator():
        async for event in run.stream(agent, message, detailed=True):
            data = event.model_dump_json()
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingHttpResponse(
        event_generator(),
        content_type="text/event-stream",
    )
```

## Frontend Consumption

### JavaScript EventSource

```javascript
const source = new EventSource("/chat?message=Hello");

source.addEventListener("text", (e) => {
  const event = JSON.parse(e.data);
  document.getElementById("output").textContent += event.text;
});

source.addEventListener("tool_call", (e) => {
  const event = JSON.parse(e.data);
  console.log(`Calling tool: ${event.tool_name}`);
});

source.addEventListener("step", (e) => {
  const event = JSON.parse(e.data);
  console.log(`Step ${event.step_number}: ${event.status}`);
});

source.addEventListener("error", (e) => {
  const event = JSON.parse(e.data);
  console.error(`Error: ${event.error}`);
  source.close();
});

source.addEventListener("status", (e) => {
  const event = JSON.parse(e.data);
  if (event.status === "completed") {
    source.close();
  }
});
```

### Fetch API with ReadableStream

```javascript
async function streamChat(message) {
  const response = await fetch(`/chat?message=${encodeURIComponent(message)}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n\n");
    buffer = lines.pop(); // Keep incomplete chunk

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const data = line.slice(6);
        if (data === "[DONE]") return;
        const event = JSON.parse(data);
        handleEvent(event);
      }
    }
  }
}

function handleEvent(event) {
  switch (event.type) {
    case "text":
      process.stdout.write(event.text);
      break;
    case "tool_call":
      console.log(`\nTool: ${event.tool_name}`);
      break;
    case "tool_result":
      console.log(`Result: ${event.result.slice(0, 100)}`);
      break;
    case "status":
      console.log(`\n[${event.status}] ${event.message}`);
      break;
    case "error":
      console.error(`Error: ${event.error}`);
      break;
  }
}
```

## JSON Serialization

All events are Pydantic models and serialize cleanly to JSON:

```python
event = TextEvent(text="Hello", agent_name="assistant")

# To dict
event.model_dump()
# {'type': 'text', 'text': 'Hello', 'agent_name': 'assistant'}

# To JSON string
event.model_dump_json()
# '{"type":"text","text":"Hello","agent_name":"assistant"}'
```

The `type` field acts as a discriminator for deserializing events back to the correct type:

```python
import json
from exo.types import (
    TextEvent, ToolCallEvent, StepEvent, ToolResultEvent,
    ReasoningEvent, ErrorEvent, StatusEvent, UsageEvent,
)

EVENT_TYPES = {
    "text": TextEvent,
    "tool_call": ToolCallEvent,
    "step": StepEvent,
    "tool_result": ToolResultEvent,
    "reasoning": ReasoningEvent,
    "error": ErrorEvent,
    "status": StatusEvent,
    "usage": UsageEvent,
}

def deserialize_event(data: dict) -> StreamEvent:
    event_cls = EVENT_TYPES[data["type"]]
    return event_cls(**data)

# Example
raw = json.loads('{"type":"text","text":"Hello","agent_name":"assistant"}')
event = deserialize_event(raw)  # TextEvent(type='text', text='Hello', agent_name='assistant')
```

## Distributed Streaming

When using distributed execution via `exo-distributed`, events are published to Redis and can be consumed in real-time or replayed:

```python
from exo.distributed import distributed

# Submit task and stream events
handle = await distributed(agent, "Hello", redis_url="redis://localhost", detailed=True)

# Live streaming via Redis Pub/Sub
async for event in handle.stream():
    match event.type:
        case "text":
            print(event.text, end="")
        case "status" if event.status == "completed":
            print("\nDone!")

# Or wait for result
result = await handle.result()
```

Events in distributed mode are identical to local streaming — the same `StreamEvent` types are used. The `EventPublisher` on the worker side serializes events to JSON and publishes to both Redis Pub/Sub (for live streaming) and Redis Streams (for replay). See the [distributed architecture docs](distributed/architecture.md) for details.

## Live Message Injection

The `inject_message()` method on `Agent` lets external code push additional user messages into a running agent's context. Injected messages are picked up before the next LLM call, allowing you to steer, correct, or augment the agent mid-run without cancelling and restarting.

### Basic Usage

```python
import asyncio
from exo import Agent, run, tool

@tool
def slow_search(query: str) -> str:
    """A tool that takes time to execute."""
    return f"Results for {query}"

agent = Agent(name="researcher", tools=[slow_search])

async def main():
    # Start the agent in a background task
    task = asyncio.create_task(run(agent, "Search for Python tutorials"))

    # Wait a moment, then inject additional context
    await asyncio.sleep(0.1)
    agent.inject_message("Also check the official Python docs")
    agent.inject_message("Focus on Python 3.12 features")

    result = await task
    print(result.output)
```

### Streaming with Injection Events

When using `run.stream()`, each injected message emits a `MessageInjectedEvent`:

```python
async for event in run.stream(agent, "Search for Python tutorials"):
    match event.type:
        case "text":
            print(event.text, end="", flush=True)
        case "message_injected":
            print(f"\n[Injected: {event.content}]")
```

### HTTP Injection via exo-server

The `exo-server` provides a `POST /inject` endpoint for pushing messages into a running agent over HTTP:

```bash
# Inject a message into the default agent
curl -X POST http://localhost:8000/inject \
  -H "Content-Type: application/json" \
  -d '{"message": "Also check the weather"}'

# Inject into a specific agent
curl -X POST http://localhost:8000/inject \
  -H "Content-Type: application/json" \
  -d '{"message": "Use metric units", "agent_name": "weather_bot"}'
```

### Message Ordering Guarantee

Injected messages are appended as `UserMessage` entries **after** any pending tool results and **before** the next LLM call. This preserves provider message compliance -- injected messages never break the required `AssistantMessage(tool_calls) → ToolResult` adjacency:

```
[..., AssistantMessage(tool_calls=[...]), ToolResult, ToolResult, UserMessage(injected), ...] → LLM call
```

Multiple messages injected between steps are drained in FIFO order.

### API Reference

```python
agent.inject_message(content: str) -> None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `content` | `str` | The message text to inject. Must be non-empty. |

**Raises:** `ValueError` if `content` is empty.

**Thread-safety:** Uses `asyncio.Queue` internally. Safe to call from any coroutine in the same event loop.
