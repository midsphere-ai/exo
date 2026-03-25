# exo._internal.call_runner

Core execution loop for running an agent with state tracking. The call runner orchestrates the LLM-to-tool-to-LLM cycle, integrating with `RunState` for message tracking, node lifecycle, and usage accumulation. It also detects endless loops where the agent repeatedly produces the same tool calls without making progress.

> **Internal API** -- subject to change without notice.

**Module:** `exo._internal.call_runner`

```python
from exo._internal.call_runner import call_runner, CallRunnerError
```

---

## CallRunnerError

```python
class CallRunnerError(ExoError)
```

Raised for call runner errors (loop detection, state errors, agent failures). Inherits from `ExoError`.

---

## call_runner()

```python
async def call_runner(
    agent: Any,
    input: str,
    *,
    state: RunState | None = None,
    messages: Sequence[Message] | None = None,
    provider: Any = None,
    max_retries: int = 3,
    loop_threshold: int = 3,
) -> RunResult
```

Run an agent's full LLM-tool loop with state tracking. Orchestrates message building, LLM calls, tool execution, and result aggregation via `Agent.run()`. Wraps each step in a `RunNode` for lifecycle tracking and detects endless loops where the same tool calls repeat without progress.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `agent` | `Any` | *(required)* | An `Agent` instance with a `run()` method. |
| `input` | `str` | *(required)* | User query string. |
| `state` | `RunState \| None` | `None` | Optional pre-existing `RunState`. A new one is created if not provided. |
| `messages` | `Sequence[Message] \| None` | `None` | Prior conversation history. |
| `provider` | `Any` | `None` | LLM provider with an `async complete()` method. |
| `max_retries` | `int` | `3` | Retry attempts passed to `Agent.run()`. |
| `loop_threshold` | `int` | `3` | Number of consecutive identical tool-call patterns before raising `CallRunnerError`. |

### Returns

`RunResult` with the final output, full message history, aggregated usage, and step count.

### Raises

- `CallRunnerError` -- on endless loop detection (same tool calls repeated `loop_threshold` times).
- `CallRunnerError` -- if the agent raises any exception (wraps the original error).

### Execution Flow

1. Creates a `RunState` if not provided.
2. Creates a `RunNode` and marks it as started.
3. Calls `agent.run()` with the input, messages, provider, and retry config.
4. Records token usage on both the node and the state.
5. Checks for endless loops by comparing tool call signatures.
6. Builds the final message list from the instructions and history.
7. Returns a `RunResult` with aggregated data.

### Loop Detection

The call runner detects endless loops by computing a deterministic signature for each step's tool calls (sorted by name and arguments). If the same signature appears `loop_threshold` consecutive times, a `CallRunnerError` is raised. The signature is stored in the node's `metadata["tool_signature"]`.

### Example

```python
import asyncio
from exo._internal.call_runner import call_runner
from exo.agent import Agent

agent = Agent(name="bot", model="openai:gpt-4o")

# result = asyncio.run(call_runner(agent, "Hello!", provider=my_provider))
# print(result.output)
# print(result.steps)
```
