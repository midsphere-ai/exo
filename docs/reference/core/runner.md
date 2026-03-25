# exo.runner

Public entry point for running agents. Provides `run()` (async), `run.sync()` (blocking), and `run.stream()` (async generator) as the primary API for executing an Agent.

**Module:** `exo.runner`

```python
from exo.runner import run
# or
from exo import run
```

---

## run()

```python
async def run(
    agent: Any,
    input: str,
    *,
    messages: Sequence[Message] | None = None,
    provider: Any = None,
    max_retries: int = 3,
    loop_threshold: int = 3,
) -> RunResult
```

Execute an agent (or swarm) and return the result. This is the primary async API for running agents.

If `provider` is `None`, a default provider is resolved from the agent's `provider_name` using the model registry (if `exo-models` is installed).

For Swarm instances (detected by the presence of a `flow_order` attribute), delegates to the swarm's own `run()` method.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `agent` | `Any` | *(required)* | An `Agent` (or `Swarm`) instance. |
| `input` | `str` | *(required)* | User query string. |
| `messages` | `Sequence[Message] \| None` | `None` | Prior conversation history to continue from. |
| `provider` | `Any` | `None` | LLM provider with `async complete()` method. When `None`, auto-resolved from the agent's model string. |
| `max_retries` | `int` | `3` | Retry attempts for transient LLM errors. |
| `loop_threshold` | `int` | `3` | Consecutive identical tool-call patterns before raising a loop error. |

### Returns

`RunResult` with the agent's output, message history, usage stats, and step count.

### Example

```python
import asyncio
from exo import Agent, run

agent = Agent(name="bot", model="openai:gpt-4o")
result = asyncio.run(run(agent, "Hello!"))
print(result.output)
print(result.steps)
print(result.usage.total_tokens)
```

---

## run.sync()

```python
def run.sync(
    agent: Any,
    input: str,
    *,
    messages: Sequence[Message] | None = None,
    provider: Any = None,
    max_retries: int = 3,
    loop_threshold: int = 3,
) -> RunResult
```

Execute an agent synchronously (blocking wrapper). Calls `run()` via `asyncio.run()`. This is a convenience for scripts and notebooks where an event loop is not already running.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `agent` | `Any` | *(required)* | An `Agent` (or `Swarm`) instance. |
| `input` | `str` | *(required)* | User query string. |
| `messages` | `Sequence[Message] \| None` | `None` | Prior conversation history to continue from. |
| `provider` | `Any` | `None` | LLM provider with `async complete()` method. |
| `max_retries` | `int` | `3` | Retry attempts for transient LLM errors. |
| `loop_threshold` | `int` | `3` | Consecutive identical tool-call patterns before raising a loop error. |

### Returns

`RunResult` with the agent's output, message history, usage stats, and step count.

### Example

```python
from exo import Agent, run

agent = Agent(name="bot", model="openai:gpt-4o")
result = run.sync(agent, "Hello!")
print(result.output)
```

---

## run.stream()

```python
async def run.stream(
    agent: Any,
    input: str,
    *,
    messages: Sequence[Message] | None = None,
    provider: Any = None,
    max_steps: int | None = None,
) -> AsyncIterator[StreamEvent]
```

Stream agent execution, yielding events in real-time. Uses the provider's `stream()` method to deliver text deltas as `TextEvent` objects and emit `ToolCallEvent` for each tool invocation.

When tool calls are detected, tools are executed and the LLM is re-streamed with the results -- looping until a text-only response or `max_steps` is reached.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `agent` | `Any` | *(required)* | An `Agent` instance. |
| `input` | `str` | *(required)* | User query string. |
| `messages` | `Sequence[Message] \| None` | `None` | Prior conversation history. |
| `provider` | `Any` | `None` | LLM provider with an `async stream()` method. When `None`, auto-resolved from the agent's model string. |
| `max_steps` | `int \| None` | `None` | Maximum LLM-tool round-trips. Defaults to `agent.max_steps`. |

### Yields

- `TextEvent` -- for text chunks (contains `text` and `agent_name` fields).
- `ToolCallEvent` -- for tool invocations (contains `tool_name`, `tool_call_id`, and `agent_name` fields).
- `MessageInjectedEvent` -- when a message is injected via `agent.inject_message()` (contains `content` and `agent_name` fields).

### Example

```python
import asyncio
from exo import Agent, run
from exo.types import TextEvent, ToolCallEvent

agent = Agent(name="bot", model="openai:gpt-4o")

async def main():
    async for event in run.stream(agent, "Tell me a story"):
        if isinstance(event, TextEvent):
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolCallEvent):
            print(f"\n[Tool: {event.tool_name}]")

asyncio.run(main())
```

---

## Provider Auto-Resolution

When `provider=None`, the runner attempts to auto-resolve a provider:

1. Imports `exo.models.provider.get_provider`
2. Calls `get_provider(agent.provider_name)` to get a registered provider
3. If auto-resolution fails (import error, no provider registered), returns `None` and lets `Agent.run()` raise its own error

This means if you have `exo-models` installed and configured, you can omit the `provider` argument entirely.
