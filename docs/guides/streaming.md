# Streaming

Exo provides first-class streaming support via `run.stream()`, an async generator that yields events in real time as the agent processes a request. This enables responsive UIs, progress indicators, and incremental output.

## Basic Usage

```python
import asyncio
from exo import Agent, run, tool


@tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city to check.
    """
    return f"Sunny, 22°C in {city}."


agent = Agent(
    name="weather-bot",
    model="openai:gpt-4o-mini",
    instructions="You are a helpful weather assistant.",
    tools=[get_weather],
)


async def main():
    async for event in run.stream(agent, "What's the weather in Tokyo?"):
        if event.type == "text":
            print(event.text, end="", flush=True)
        elif event.type == "tool_call":
            print(f"\n[Calling tool: {event.tool_name}]")
    print()  # final newline


asyncio.run(main())
```

## Stream Event Types

`run.stream()` yields `StreamEvent` objects, which is a union of two types:

### TextEvent

A chunk of text from the LLM. These arrive incrementally as the model generates tokens.

```python
from exo.types import TextEvent

# event.type == "text"
# event.text -- the text delta (a few words or a partial sentence)
# event.agent_name -- which agent produced this event
```

### ToolCallEvent

Notification that the agent is invoking a tool.

```python
from exo.types import ToolCallEvent

# event.type == "tool_call"
# event.tool_name -- name of the tool being called
# event.tool_call_id -- unique ID for this invocation
# event.agent_name -- which agent produced this event
```

### The StreamEvent Union

```python
from exo.types import StreamEvent

# StreamEvent = TextEvent | ToolCallEvent
```

## How the Tool Loop Works with Streaming

When the LLM requests a tool call during streaming, the stream handles the full tool cycle automatically:

1. The LLM begins generating a response
2. If the LLM decides to call a tool, a `ToolCallEvent` is yielded
3. The tool executes (behind the scenes)
4. The tool result is fed back to the LLM
5. The LLM resumes generating, yielding more `TextEvent` chunks
6. Steps 2-5 repeat until the LLM produces a final response without tool calls

This means you do not need to handle tool execution manually -- the stream takes care of it.

```python
async for event in run.stream(agent, "Search for Python and summarize"):
    if event.type == "text":
        # Incremental text from the LLM
        print(event.text, end="")
    elif event.type == "tool_call":
        # The agent decided to call a tool
        print(f"\n[Tool: {event.tool_name}]")
        # Tool runs automatically, then more TextEvents follow
```

## Streaming with Multi-Agent Swarms

Streaming works with swarms. When a swarm is in workflow mode, each agent streams its output in sequence. The `agent_name` field on events tells you which agent is currently producing output:

```python
from exo import Agent, Swarm, run

researcher = Agent(name="researcher", model="openai:gpt-4o-mini", ...)
writer = Agent(name="writer", model="openai:gpt-4o-mini", ...)

swarm = Swarm(
    agents=[researcher, writer],
    flow="researcher >> writer",
    mode="workflow",
)

async for event in run.stream(swarm, "Write about quantum computing"):
    if event.type == "text":
        agent = event.agent_name
        print(f"[{agent}] {event.text}", end="")
```

## Stream Parameters

`run.stream()` accepts the same core parameters as `run()`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `Agent` or `Swarm` | *(required)* | The agent or swarm to execute |
| `input` | `str` | *(required)* | User query string |
| `messages` | `Sequence[Message] \| None` | `None` | Prior conversation history |
| `provider` | `Any` | `None` | LLM provider (auto-resolved if None) |
| `max_steps` | `int \| None` | `None` | Max LLM-tool round-trips (defaults to `agent.max_steps`) |

## Collecting the Full Result

If you need both streaming output and the final `RunResult`, collect events while streaming:

```python
chunks = []
async for event in run.stream(agent, "Hello!"):
    if event.type == "text":
        chunks.append(event.text)
        print(event.text, end="")

full_output = "".join(chunks)
print(f"\n\nFull output: {full_output}")
```

## Server-Sent Events (SSE)

When using `exo-server`, streaming is exposed as Server-Sent Events over HTTP. The server converts `StreamEvent` objects to SSE format automatically:

```python
# Server side (using exo-server)
from exo_server import create_app

app = create_app(agents=[agent])
# Clients can POST to /chat/stream for SSE responses
```

```bash
# Client side
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello!", "agent": "assistant"}'
```

Each SSE event has a `data` field containing JSON with the event type and payload.

## WebSocket Streaming

For bidirectional communication, `exo-server` also supports WebSocket connections:

```python
import websockets
import json

async with websockets.connect("ws://localhost:8000/ws/assistant") as ws:
    await ws.send(json.dumps({"input": "Hello!"}))

    async for message in ws:
        event = json.loads(message)
        if event["type"] == "text":
            print(event["text"], end="")
        elif event["type"] == "done":
            break
```

## Tips

- **Flush output** -- use `flush=True` with `print()` for immediate display.
- **UI integration** -- yield events to your frontend via SSE or WebSocket for real-time UIs.
- **Error handling** -- wrap the stream in a try/except. If the LLM call fails, the stream raises the exception.
- **Multi-turn streaming** -- pass `messages` from a previous run to continue a conversation with streaming.
