# exo_cli.executor

Local agent executor for Exo CLI. Wraps `exo.runner.run` and `exo.runner.run.stream` with Rich output formatting, error handling, timeout support, and provider resolution.

```python
from exo_cli.executor import ExecutionResult, ExecutorError, LocalExecutor
```

---

## ExecutorError

```python
class ExecutorError(Exception)
```

Raised for execution-level errors (timeout, agent failure, streaming not available).

---

## ExecutionResult

```python
class ExecutionResult(
    *,
    output: str,
    steps: int = 0,
    elapsed: float = 0.0,
    usage: dict[str, int] | None = None,
    raw: Any = None,
)
```

Wraps a run result with CLI-friendly accessors.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `output` | `str` | *(required)* | Agent output text |
| `steps` | `int` | `0` | Number of LLM call steps |
| `elapsed` | `float` | `0.0` | Wall-clock seconds |
| `usage` | `dict[str, int] \| None` | `None` | Token usage dict (prompt_tokens, output_tokens, total_tokens) |
| `raw` | `Any` | `None` | The underlying `RunResult` object (if available) |

### Properties

| Property | Type | Description |
|---|---|---|
| `output` | `str` | Agent output text |
| `steps` | `int` | Number of LLM call steps |
| `elapsed` | `float` | Wall-clock seconds |
| `usage` | `dict[str, int]` | Copy of token usage dict |
| `raw` | `Any` | The underlying `RunResult` object |

### Methods

#### summary

```python
def summary(self) -> str
```

Human-readable one-line summary including step count, elapsed time, and token count.

### Example

```python
result = ExecutionResult(output="Hello!", steps=2, elapsed=1.5, usage={"total_tokens": 150})
print(result.output)    # "Hello!"
print(result.summary()) # "2 step(s), 1.5s, 150 tokens"
```

---

## LocalExecutor

```python
class LocalExecutor(
    *,
    agent: Any,
    provider: Any = None,
    timeout: float = 0.0,
    max_retries: int = 3,
    console: RichConsole | None = None,
    verbose: bool = False,
)
```

Executes agents locally via `exo.runner.run`.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `agent` | `Any` | *(required)* | An `Agent` (or `Swarm`) instance |
| `provider` | `Any` | `None` | LLM provider. When `None`, auto-resolved |
| `timeout` | `float` | `0.0` | Per-execution timeout in seconds (0 = no timeout) |
| `max_retries` | `int` | `3` | Retry attempts for transient LLM errors |
| `console` | `RichConsole \| None` | `None` | Rich console for formatted output (default: stderr) |
| `verbose` | `bool` | `False` | When `True`, print timing and usage details |

### Properties

| Property | Type | Description |
|---|---|---|
| `agent` | `Any` | The agent instance |
| `timeout` | `float` | Per-execution timeout in seconds |
| `verbose` | `bool` | Whether verbose output is enabled |

### Methods

#### execute

```python
async def execute(
    self,
    input: str,
    *,
    messages: Sequence[Any] | None = None,
) -> ExecutionResult
```

Run the agent and return an `ExecutionResult`. Calls `exo.runner.run()` with the configured agent, provider, and retry settings.

| Name | Type | Default | Description |
|---|---|---|---|
| `input` | `str` | *(required)* | User input text |
| `messages` | `Sequence[Any] \| None` | `None` | Optional conversation history |

**Returns:** `ExecutionResult` with output, steps, elapsed time, usage, and raw result.

**Raises:** `ExecutorError` -- On timeout or agent failure.

#### stream

```python
async def stream(
    self,
    input: str,
    *,
    messages: Sequence[Any] | None = None,
) -> AsyncIterator[str]
```

Stream agent output, yielding text chunks. Uses `exo.runner.run.stream` if available.

| Name | Type | Default | Description |
|---|---|---|---|
| `input` | `str` | *(required)* | User input text |
| `messages` | `Sequence[Any] \| None` | `None` | Optional conversation history |

**Yields:** `str` -- Text chunks from the agent.

**Raises:** `ExecutorError` -- If streaming is not available or fails during execution.

#### print_result

```python
def print_result(self, result: ExecutionResult) -> None
```

Pretty-print an execution result to the console using a Rich `Panel`.

#### print_error

```python
def print_error(self, error: Exception) -> None
```

Display an error to the console in red.

### Example

```python
from exo_cli import LocalExecutor

executor = LocalExecutor(agent=my_agent, timeout=30.0, verbose=True)

# Non-streaming execution
result = await executor.execute("What is Python?")
executor.print_result(result)

# Streaming execution
async for chunk in executor.stream("Explain decorators"):
    print(chunk, end="")
```
