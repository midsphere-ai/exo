# exo._internal.state

Internal run state tracking for agent execution. Provides `RunState` and `RunNode` for tracking the full lifecycle of an agent run, including per-step timing, status transitions, token usage, and error recording.

> **Internal API** -- subject to change without notice.

**Module:** `exo._internal.state`

```python
from exo._internal.state import RunNodeStatus, RunNode, RunState
```

---

## RunNodeStatus

```python
class RunNodeStatus(StrEnum)
```

Status of an execution node (agent step or tool call).

### Values

| Member | Value | Description |
|--------|-------|-------------|
| `INIT` | `"init"` | Node has been created but not yet started. |
| `RUNNING` | `"running"` | Node is currently executing. |
| `SUCCESS` | `"success"` | Node completed successfully. |
| `FAILED` | `"failed"` | Node failed with an error. |
| `TIMEOUT` | `"timeout"` | Node exceeded its time limit. |

---

## RunNode

```python
class RunNode(BaseModel)
```

A single execution step within a run. Tracks one LLM call or tool execution with timing and status.

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `agent_name` | `str` | *(required)* | Name of the agent that owns this step. |
| `step_index` | `int` | `0` | Zero-based step index within the run. |
| `status` | `RunNodeStatus` | `RunNodeStatus.INIT` | Current execution status. |
| `group_id` | `str \| None` | `None` | Optional group identifier for parallel/serial groups. |
| `created_at` | `float` | `time.time()` | Timestamp when the node was created. |
| `started_at` | `float \| None` | `None` | Timestamp when execution started. |
| `ended_at` | `float \| None` | `None` | Timestamp when execution finished. |
| `tool_calls` | `list[ToolCall]` | `[]` | Tool calls produced during this step. |
| `usage` | `Usage` | `Usage()` | Token usage for this step. |
| `error` | `str \| None` | `None` | Error message if the step failed. |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary key-value metadata. |

### Methods

#### start()

```python
def start(self) -> None
```

Transition to `RUNNING` status. Sets `started_at` to current time.

#### succeed()

```python
def succeed(self, usage: Usage | None = None) -> None
```

Transition to `SUCCESS` status. Sets `ended_at` to current time.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `usage` | `Usage \| None` | `None` | Optional token usage to record. |

#### fail()

```python
def fail(self, error: str) -> None
```

Transition to `FAILED` status. Sets `ended_at` to current time.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `error` | `str` | *(required)* | Error message. |

#### timeout()

```python
def timeout(self) -> None
```

Transition to `TIMEOUT` status. Sets `ended_at` to current time.

### Properties

#### duration

```python
@property
def duration(self) -> float | None
```

Elapsed time in seconds, or `None` if not yet finished.

### Example

```python
from exo._internal.state import RunNode, RunNodeStatus
from exo.types import Usage

node = RunNode(agent_name="bot")
node.start()
print(node.status)  # RunNodeStatus.RUNNING

node.succeed(usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150))
print(node.status)    # RunNodeStatus.SUCCESS
print(node.duration)  # e.g. 0.5 (seconds)
```

---

## RunState

```python
class RunState
```

Mutable execution state for a single run. Tracks the full message history, tool calls, iteration count, current agent, and per-step nodes.

### Constructor

```python
def __init__(self, agent_name: str) -> None
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `agent_name` | `str` | *(required)* | Name of the initial agent. |

### Instance Attributes

| Name | Type | Description |
|------|------|-------------|
| `agent_name` | `str` | Name of the agent. |
| `status` | `RunNodeStatus` | Current run status (starts as `INIT`). |
| `messages` | `list[Message]` | Full message history. |
| `nodes` | `list[RunNode]` | Per-step execution nodes. |
| `iterations` | `int` | Number of iterations completed. |
| `total_usage` | `Usage` | Accumulated token usage. |

### Methods

#### start()

```python
def start(self) -> None
```

Mark the run as started (transitions to `RUNNING`).

#### add_message()

```python
def add_message(self, message: Message) -> None
```

Append a message to the run history.

#### add_messages()

```python
def add_messages(self, messages: Sequence[Message]) -> None
```

Append multiple messages to the run history.

#### new_node()

```python
def new_node(self, agent_name: str | None = None, group_id: str | None = None) -> RunNode
```

Create and track a new execution node. Increments the iteration count.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `agent_name` | `str \| None` | `None` | Agent name for this node (defaults to the run's agent). |
| `group_id` | `str \| None` | `None` | Optional group identifier. |

**Returns:** The newly created `RunNode`.

#### record_usage()

```python
def record_usage(self, usage: Usage) -> None
```

Accumulate token usage into the run total.

#### succeed()

```python
def succeed(self) -> None
```

Mark the run as successful.

#### fail()

```python
def fail(self, error: str | None = None) -> None
```

Mark the run as failed.

#### timeout()

```python
def timeout(self) -> None
```

Mark the run as timed out.

### Properties

#### current_node

```python
@property
def current_node(self) -> RunNode | None
```

The most recently created node, or `None` if no nodes exist.

#### is_running

```python
@property
def is_running(self) -> bool
```

Whether the run is currently in progress (`status == RUNNING`).

#### is_terminal

```python
@property
def is_terminal(self) -> bool
```

Whether the run has reached a terminal state (`SUCCESS`, `FAILED`, or `TIMEOUT`).

### Example

```python
from exo._internal.state import RunState
from exo.types import Usage, UserMessage

state = RunState(agent_name="bot")
state.start()
assert state.is_running

# Create a step
node = state.new_node()
node.start()

# Record usage
usage = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
state.record_usage(usage)
node.succeed(usage=usage)

# Add messages
state.add_message(UserMessage(content="Hello"))

# Complete
state.succeed()
assert state.is_terminal
print(state.iterations)  # 1
print(state.total_usage.total_tokens)  # 150
```
