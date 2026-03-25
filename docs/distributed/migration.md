# Migration Guide: Local to Distributed Execution

This guide walks you through moving from local `run()` / `run.stream()` to distributed execution with `distributed()`. The API is designed so the transition requires minimal code changes.

## Side-by-Side Comparison

### Running an Agent

**Local execution:**

```python
from exo import Agent, run

agent = Agent(name="assistant", model="openai:gpt-4o")

result = await run(agent, "What is the capital of France?")
print(result.output)
```

**Distributed execution:**

```python
from exo import Agent
from exo.distributed import distributed

agent = Agent(name="assistant", model="openai:gpt-4o")

handle = await distributed(agent, "What is the capital of France?")
result = await handle.result()
print(result)
```

Key differences:
- `distributed()` returns a `TaskHandle` instead of a `RunResult` directly
- Call `handle.result()` to block until the task completes on a remote worker
- The result is a `dict` (JSON-serialized), not a `RunResult` object

### Streaming Events

**Local streaming:**

```python
from exo import Agent, run

agent = Agent(name="assistant", model="openai:gpt-4o")

async for event in run.stream(agent, "Tell me a story", detailed=True):
    match event.type:
        case "text":
            print(event.text, end="", flush=True)
        case "tool_call":
            print(f"Calling tool: {event.tool_name}")
        case "step":
            print(f"Step {event.step_number} {event.status}")
        case "status":
            print(f"Status: {event.status}")
```

**Distributed streaming:**

```python
from exo import Agent
from exo.distributed import distributed

agent = Agent(name="assistant", model="openai:gpt-4o")

handle = await distributed(agent, "Tell me a story", detailed=True)

async for event in handle.stream():
    match event.type:
        case "text":
            print(event.text, end="", flush=True)
        case "tool_call":
            print(f"Calling tool: {event.tool_name}")
        case "step":
            print(f"Step {event.step_number} {event.status}")
        case "status":
            print(f"Status: {event.status}")
```

Key differences:
- Submit with `distributed()`, then call `handle.stream()` for live events
- Events are delivered via Redis Pub/Sub — same `StreamEvent` types, same `match` patterns
- The `detailed=True` flag works the same in both modes

### Streaming a Swarm

**Local Swarm streaming:**

```python
from exo import Agent, Swarm, run

researcher = Agent(name="researcher", model="openai:gpt-4o")
writer = Agent(name="writer", model="openai:gpt-4o")
swarm = Swarm(agents=[researcher, writer], mode="workflow")

async for event in run.stream(swarm, "Research and write about AI"):
    print(f"[{event.agent_name}] {event.type}: ...", end="")
```

**Distributed Swarm streaming:**

```python
from exo import Agent, Swarm
from exo.distributed import distributed

researcher = Agent(name="researcher", model="openai:gpt-4o")
writer = Agent(name="writer", model="openai:gpt-4o")
swarm = Swarm(agents=[researcher, writer], mode="workflow")

handle = await distributed(swarm, "Research and write about AI", detailed=True)

async for event in handle.stream():
    print(f"[{event.agent_name}] {event.type}: ...", end="")
```

Swarms work the same way — `distributed()` accepts both `Agent` and `Swarm` instances.

## Step-by-Step Migration

### 1. Install exo-distributed

```bash
pip install exo-distributed
```

### 2. Start Redis

```bash
docker run -d --name redis -p 6379:6379 redis:7
```

### 3. Set the Redis URL

```bash
export EXO_REDIS_URL=redis://localhost:6379
```

Or pass it explicitly:

```python
handle = await distributed(agent, "Hello", redis_url="redis://localhost:6379")
```

### 4. Start a Worker

Workers execute tasks from the queue. At least one worker must be running:

```bash
exo start worker
```

For higher throughput:

```bash
exo start worker --concurrency 4
```

### 5. Replace `run()` with `distributed()`

```python
# Before
result = await run(agent, input_text)

# After
handle = await distributed(agent, input_text)
result = await handle.result()
```

### 6. Replace `run.stream()` with `distributed()` + `handle.stream()`

```python
# Before
async for event in run.stream(agent, input_text, detailed=True):
    process(event)

# After
handle = await distributed(agent, input_text, detailed=True)
async for event in handle.stream():
    process(event)
```

## Event Consumption Patterns

### Local: Direct Async Iterator

With local execution, `run.stream()` yields events directly as the agent runs in your process:

```python
async for event in run.stream(agent, "Hello", detailed=True):
    # Events arrive as the agent executes in this process
    handle_event(event)
```

### Distributed: Submit + Subscribe

With distributed execution, submission and consumption are decoupled:

```python
# Submit — returns immediately
handle = await distributed(agent, "Hello", detailed=True)

# Option 1: Stream live events via Redis Pub/Sub
async for event in handle.stream():
    handle_event(event)

# Option 2: Wait for the final result (no streaming)
result = await handle.result()

# Option 3: Poll status periodically
status = await handle.status()
print(f"Task {handle.task_id}: {status.status}")
```

### Distributed: Replay Past Events

Events are persisted in Redis Streams (1-hour TTL by default), so you can replay them after the fact:

```python
from exo.distributed import EventSubscriber

subscriber = EventSubscriber("redis://localhost:6379")
await subscriber.connect()

async for event in subscriber.replay(task_id):
    handle_event(event)
```

## Error Handling Differences

### Local Errors

Errors raise exceptions directly in your process:

```python
try:
    result = await run(agent, "Hello")
except Exception as e:
    print(f"Agent failed: {e}")
```

With streaming, `ErrorEvent` is yielded and then the exception is re-raised:

```python
try:
    async for event in run.stream(agent, "Hello", detailed=True):
        if event.type == "error":
            print(f"Error: {event.error}")
except Exception:
    pass  # Exception re-raised after ErrorEvent
```

### Distributed Errors

Errors are captured by the worker and stored in the task result:

```python
handle = await distributed(agent, "Hello")

try:
    result = await handle.result()
except RuntimeError as e:
    # "Task <id> failed: <error message>"
    print(f"Task failed: {e}")
```

With streaming, `ErrorEvent` is delivered via Redis just like other events:

```python
handle = await distributed(agent, "Hello", detailed=True)

async for event in handle.stream():
    if event.type == "error":
        print(f"Error from worker: {event.error}")
        print(f"  Type: {event.error_type}")
        print(f"  Recoverable: {event.recoverable}")
```

You can also check status directly:

```python
status = await handle.status()
if status and status.status == "FAILED":
    print(f"Error: {status.error}")
```

### Task Cancellation

Distributed execution supports cooperative cancellation — not available in local mode:

```python
handle = await distributed(agent, "Long running task")

# Cancel after 10 seconds
await asyncio.sleep(10)
await handle.cancel()

# Worker emits StatusEvent(status='cancelled') and stops execution
```

## When to Use Local vs Distributed

| Scenario | Recommended | Why |
|----------|------------|-----|
| Development / prototyping | Local `run()` | No infrastructure needed |
| Single-user CLI tool | Local `run()` | Simpler, lower latency |
| Web API serving many users | Distributed | Scales horizontally with workers |
| Long-running agent tasks | Distributed | Task survives client disconnect |
| Multi-agent pipelines in production | Distributed | Worker pool shares load |
| Need task cancellation | Distributed | Cooperative cancellation built in |
| Need event replay / audit | Distributed | Events persisted in Redis Streams |
| Need durable execution | Distributed + Temporal | Tasks survive worker crashes |

## Parameter Mapping

| `run()` / `run.stream()` | `distributed()` | Notes |
|--------------------------|-----------------|-------|
| `agent` | `agent` | Same — Agent or Swarm |
| `input` | `input` | Same — input string |
| `messages` | `messages` | Same — conversation history |
| `provider` | — | Workers resolve providers automatically |
| `detailed` | `detailed` | Same — enables rich events |
| `event_types` | — | Filter events client-side after `handle.stream()` |
| `max_steps` | — | Set in Agent config (serialized via `to_dict()`) |
| — | `redis_url` | Redis connection (or `EXO_REDIS_URL` env var) |
| — | `timeout` | Task timeout in seconds (default 300) |
| — | `metadata` | Arbitrary metadata attached to the task |

## Event Filtering

Local mode supports server-side event filtering:

```python
# Local: only receive text and tool_result events
async for event in run.stream(agent, "Hello", detailed=True, event_types={"text", "tool_result"}):
    process(event)
```

In distributed mode, filter client-side:

```python
handle = await distributed(agent, "Hello", detailed=True)

async for event in handle.stream():
    if event.type in {"text", "tool_result"}:
        process(event)
```
