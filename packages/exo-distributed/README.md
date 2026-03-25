# exo-distributed

Distributed execution for Exo agents — Redis task queue, workers, and event streaming. Run agents across multiple worker processes with real-time event streaming, task cancellation, and optional Temporal integration for durable execution.

## Installation

```bash
pip install exo-distributed
```

For durable execution with Temporal:

```bash
pip install exo-distributed[temporal]
```

## Quick Start

### 1. Start Redis

```bash
docker run -d --name redis -p 6379:6379 redis:7
```

### 2. Start a Worker

```bash
export EXO_REDIS_URL=redis://localhost:6379

# Via CLI
exo start worker

# With options
exo start worker --concurrency 4 --queue my-queue
```

### 3. Submit a Task

```python
import asyncio
from exo import Agent
from exo.distributed import distributed

agent = Agent(name="assistant", model="openai:gpt-4o")

async def main():
    handle = await distributed(agent, "What is the capital of France?")

    # Wait for the result
    result = await handle.result()
    print(result["output"])

asyncio.run(main())
```

## Usage

### Agent Execution

Submit a single agent for distributed execution:

```python
from exo import Agent
from exo.distributed import distributed

agent = Agent(
    name="researcher",
    model="openai:gpt-4o",
    instructions="You are a research assistant.",
    tools=[search_web, read_page],
)

handle = await distributed(
    agent,
    "Research the latest advances in quantum computing",
    redis_url="redis://localhost:6379",
    detailed=True,       # Enable rich streaming events
    timeout=600.0,       # 10-minute timeout
    metadata={"user_id": "u-123"},
)
```

### Swarm Execution

Submit multi-agent Swarm for distributed execution:

```python
from exo import Agent
from exo.swarm import Swarm
from exo.distributed import distributed

researcher = Agent(name="researcher", model="openai:gpt-4o", tools=[search_web])
writer = Agent(name="writer", model="openai:gpt-4o")

swarm = Swarm(
    agents=[researcher, writer],
    flow="researcher >> writer",
    mode="workflow",
)

handle = await distributed(swarm, "Write a report on climate change")
result = await handle.result()
```

### Streaming Events

Subscribe to live events as the agent executes on a remote worker:

```python
handle = await distributed(agent, "Analyze this dataset", detailed=True)

async for event in handle.stream():
    match event.type:
        case "text":
            print(event.text, end="", flush=True)
        case "tool_call":
            print(f"\nCalling tool: {event.name}")
        case "tool_result":
            print(f"Tool {event.tool_name}: {'OK' if event.success else 'FAILED'}")
        case "step":
            print(f"\nStep {event.step_number} {event.status}")
        case "status":
            print(f"Status: {event.status} - {event.message}")
        case "error":
            print(f"Error: {event.error}")
```

### Task Management

```python
# Check task status
status = await handle.status()
print(f"Task {handle.task_id}: {status.status}")

# Cancel a running task
await handle.cancel()

# Wait for result (blocks until completion)
try:
    result = await handle.result()
except RuntimeError as e:
    print(f"Task failed: {e}")
```

### CLI Commands

```bash
# Start a worker
exo start worker --redis-url redis://localhost:6379 --concurrency 4

# List tasks
exo task list
exo task list --status RUNNING

# Check task status
exo task status <task-id>

# Cancel a task
exo task cancel <task-id>

# List active workers
exo worker list
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `EXO_REDIS_URL` | Redis connection URL | *(required)* |
| `TEMPORAL_HOST` | Temporal server address | `localhost:7233` |
| `TEMPORAL_NAMESPACE` | Temporal namespace | `default` |

## Temporal Integration

For durable execution that survives worker crashes:

```python
# Start worker with Temporal backend
exo start worker --executor temporal
```

```python
# Client code is identical — just start workers with Temporal
handle = await distributed(agent, "Long-running analysis task", timeout=3600.0)
result = await handle.result()
```

Temporal wraps each task in a durable workflow with heartbeating activities. If a worker crashes mid-execution, Temporal automatically retries the task on another worker.

Requires a running Temporal server:

```bash
# Via Docker
docker run -d --name temporal -p 7233:7233 temporalio/auto-setup:latest
```

## Architecture

```
Client ──distributed()──> Redis Stream (exo:tasks)
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
                Worker 1  Worker 2  Worker 3
                    │
              ┌─────┴─────┐
              ▼            ▼
         Redis Pub/Sub  Redis Stream
         (live events)  (replay/persist)
              │
              ▼
         Client.stream()
```

- **TaskBroker**: Redis Streams-backed task queue with consumer groups
- **Worker**: Claims tasks, reconstructs agents, executes via `run.stream()`
- **EventPublisher/Subscriber**: Dual-channel event delivery (Pub/Sub + Streams)
- **TaskStore**: Redis hash-backed task state with TTL auto-cleanup
- **TaskHandle**: Client-side handle for result retrieval, streaming, and cancellation

## Worker Features

### Provider Factory

By default, the worker auto-resolves an LLM provider from the agent's model string. For custom provider logic (token refresh, custom endpoints, per-request credentials), pass a `provider_factory`:

```python
from exo.distributed.worker import Worker

def my_factory(model: str):
    """Return a provider for the given model string."""
    from exo.models.openai import OpenAIProvider
    return OpenAIProvider(api_key=get_fresh_token(), model=model)

worker = Worker(
    "redis://localhost:6379",
    provider_factory=my_factory,
)
```

The factory receives the model string (e.g., `"openai:gpt-4o"`) and returns a provider instance. When `None`, the standard auto-resolution is used.

### Post-Task Callback (on_task_done)

Subclass `Worker` and override `on_task_done()` for post-task cleanup, billing, or notifications:

```python
from exo.distributed.worker import Worker
from exo.distributed.models import TaskPayload, TaskStatus

class BillingWorker(Worker):
    async def on_task_done(self, task, status, result, error):
        if status == TaskStatus.COMPLETED:
            await bill_user(task.metadata.get("user_id"), result)
        elif status == TaskStatus.FAILED:
            await alert_ops(task.task_id, error)
```

The callback fires in a `finally` block, so it runs on success, failure, and cancellation. Exceptions in `on_task_done` are logged but never crash the worker.

### Memory Hydration

Workers can automatically set up memory for tasks. Pass a `memory` config in the task metadata:

```python
handle = await distributed(
    agent,
    "Continue our conversation",
    metadata={
        "memory": {
            "backend": "short_term",  # or "sqlite", "postgres"
            "scope": {
                "user_id": "u-1",
                "session_id": "s-1",
            },
        },
    },
)
```

When a memory config is present, the worker:
1. Creates a memory store from the config
2. Attaches `MemoryPersistence` to auto-save LLM and tool results
3. Saves the user input as `HumanMemory`
4. Loads prior conversation history from the store (for multi-turn sessions)
5. Tears down the store in the `finally` block

This requires `exo-memory` to be installed (`pip install exo-distributed[memory]`).
