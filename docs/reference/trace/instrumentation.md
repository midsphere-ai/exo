# exo.trace.instrumentation

Agent and tool metrics instrumentation using OpenTelemetry.

```python
from exo.trace.instrumentation import (
    Timer,
    build_agent_attributes,
    build_tool_attributes,
    record_agent_run,
    record_tool_step,
)
```

---

## build_agent_attributes

```python
def build_agent_attributes(
    *,
    agent_name: str,
    task_id: str = "",
    session_id: str = "",
    user_id: str = "",
    step: int | None = None,
) -> dict[str, str | int]
```

Build an attribute dict for agent metrics.

| Name | Type | Default | Description |
|---|---|---|---|
| `agent_name` | `str` | *(required)* | Name of the agent |
| `task_id` | `str` | `""` | Task identifier |
| `session_id` | `str` | `""` | Session identifier |
| `user_id` | `str` | `""` | User identifier |
| `step` | `int \| None` | `None` | Current step number (omitted from dict if None) |

**Returns:** Dict with keys from the semantic conventions: `exo.agent.name`, `exo.task.id`, `exo.session.id`, `exo.user.id`, and optionally `exo.agent.step`.

---

## build_tool_attributes

```python
def build_tool_attributes(
    *,
    tool_name: str,
    agent_name: str = "",
    task_id: str = "",
) -> dict[str, str]
```

Build an attribute dict for tool metrics.

| Name | Type | Default | Description |
|---|---|---|---|
| `tool_name` | `str` | *(required)* | Name of the tool |
| `agent_name` | `str` | `""` | Name of the invoking agent |
| `task_id` | `str` | `""` | Task identifier |

**Returns:** Dict with keys `exo.tool.name`, `exo.agent.name`, `exo.task.id`.

---

## record_agent_run

```python
def record_agent_run(
    *,
    duration: float,
    success: bool,
    attributes: dict[str, Any] | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None
```

Record agent run metrics (duration, counter, token usage). Creates instruments from the current `MeterProvider` on each call so that metrics are recorded to whatever provider is active at call time.

| Name | Type | Default | Description |
|---|---|---|---|
| `duration` | `float` | *(required)* | Run duration in seconds |
| `success` | `bool` | *(required)* | Whether the run succeeded |
| `attributes` | `dict[str, Any] \| None` | `None` | Additional metric attributes |
| `input_tokens` | `int` | `0` | Number of input tokens consumed |
| `output_tokens` | `int` | `0` | Number of output tokens produced |

### Recorded metrics

| Metric | Type | Unit | Description |
|---|---|---|---|
| `agent_run_duration` | Histogram | `s` | Agent run duration in seconds |
| `agent_run_counter` | Counter | `1` | Number of agent run invocations |
| `agent_token_usage` | Histogram | `token` | Agent token usage per run (only if tokens > 0) |

### Example

```python
from exo.trace import record_agent_run, build_agent_attributes

attrs = build_agent_attributes(agent_name="assistant", task_id="task-123")
record_agent_run(
    duration=2.5,
    success=True,
    attributes=attrs,
    input_tokens=150,
    output_tokens=80,
)
```

---

## record_tool_step

```python
def record_tool_step(
    *,
    duration: float,
    success: bool,
    attributes: dict[str, Any] | None = None,
) -> None
```

Record tool step metrics (duration, counter). Creates instruments from the current `MeterProvider` on each call.

| Name | Type | Default | Description |
|---|---|---|---|
| `duration` | `float` | *(required)* | Step duration in seconds |
| `success` | `bool` | *(required)* | Whether the step succeeded |
| `attributes` | `dict[str, Any] \| None` | `None` | Additional metric attributes |

### Recorded metrics

| Metric | Type | Unit | Description |
|---|---|---|---|
| `tool_step_duration` | Histogram | `s` | Tool step execution duration in seconds |
| `tool_step_counter` | Counter | `1` | Number of tool step invocations |

---

## Timer

```python
class Timer()
```

Simple timer for measuring durations using `time.monotonic()`.

### Methods

#### start

```python
def start(self) -> Timer
```

Start the timer. Returns self for chaining.

#### stop

```python
def stop(self) -> float
```

Stop the timer and return elapsed seconds.

### Properties

| Property | Type | Description |
|---|---|---|
| `elapsed` | `float` | The last recorded elapsed time |

### Example

```python
from exo.trace import Timer

timer = Timer()
timer.start()

# ... do work ...

elapsed = timer.stop()
print(f"Took {elapsed:.3f}s")
print(timer.elapsed)  # same value
```

---

## Metric name constants

| Constant | Value |
|---|---|
| `METRIC_AGENT_RUN_DURATION` | `"agent_run_duration"` |
| `METRIC_AGENT_RUN_COUNTER` | `"agent_run_counter"` |
| `METRIC_AGENT_TOKEN_USAGE` | `"agent_token_usage"` |
| `METRIC_TOOL_STEP_DURATION` | `"tool_step_duration"` |
| `METRIC_TOOL_STEP_COUNTER` | `"tool_step_counter"` |
