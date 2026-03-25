# exo._internal.handlers

Handler abstractions for composable agent execution. Provides `Handler[IN, OUT]` as the base abstraction for processing units that transform inputs to outputs via async generators, and concrete handlers for agent routing, tool execution, and group orchestration in multi-agent swarms.

> **Internal API** -- subject to change without notice.

**Module:** `exo._internal.handlers`

```python
from exo._internal.handlers import (
    Handler,
    HandlerError,
    SwarmMode,
    AgentHandler,
    ToolHandler,
    GroupHandler,
)
```

---

## HandlerError

```python
class HandlerError(ExoError)
```

Raised for handler-level errors (routing, dispatch, stop checks). Inherits from `ExoError`.

---

## SwarmMode

```python
class SwarmMode(StrEnum)
```

Swarm topology modes for agent orchestration.

### Values

| Member | Value | Description |
|--------|-------|-------------|
| `WORKFLOW` | `"workflow"` | Sequential pipeline execution. |
| `HANDOFF` | `"handoff"` | Agent-driven delegation chains. |
| `TEAM` | `"team"` | Lead-worker delegation pattern. |

---

## Handler[IN, OUT] (ABC)

```python
class Handler(ABC, Generic[IN, OUT])
```

Abstract base for composable processing units. Handlers receive an input and yield zero or more outputs via an async generator. This enables streaming, backpressure, and composable pipelines.

### Methods

#### handle() *(abstract)*

```python
@abstractmethod
def handle(self, input: IN, **kwargs: Any) -> AsyncIterator[OUT]
```

Process input and yield outputs.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input` | `IN` | *(required)* | The input to process. |
| `**kwargs` | `Any` | | Additional context passed through the pipeline. |

**Yields:** Processed output items.

---

## AgentHandler

```python
class AgentHandler(Handler[str, RunResult])
```

Routes execution between agents in a swarm with topology-aware stops. Manages agent dispatch, handoff detection, and stop condition checks for workflow, handoff, and team modes.

### Constructor

```python
def __init__(
    self,
    *,
    agents: dict[str, Any],
    mode: SwarmMode = SwarmMode.WORKFLOW,
    flow_order: list[str] | None = None,
    provider: Any = None,
    max_handoffs: int = 10,
) -> None
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `agents` | `dict[str, Any]` | *(required)* | Dict mapping agent name to agent instance. |
| `mode` | `SwarmMode` | `SwarmMode.WORKFLOW` | Swarm topology mode. |
| `flow_order` | `list[str] \| None` | `None` | Ordered list of agent names for workflow mode. Defaults to dict key order. |
| `provider` | `Any` | `None` | LLM provider for agent execution. |
| `max_handoffs` | `int` | `10` | Maximum handoff count before stopping (handoff mode). |

### Methods

#### handle()

```python
async def handle(self, input: str, **kwargs: Any) -> AsyncIterator[RunResult]
```

Execute agents according to the swarm topology.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input` | `str` | *(required)* | User query string. |
| `**kwargs` | `Any` | | Additional context (`messages`, `state`, etc.). |

**Yields:** `RunResult` from each agent execution.

**Raises:** `HandlerError` if an agent is not found or max handoffs exceeded.

### Stop Condition Methods

The AgentHandler provides topology-specific stop condition checks:

```python
def _check_workflow_stop(self, agent_name: str) -> bool
```
Returns `True` if `agent_name` is the last in `flow_order`.

```python
def _check_handoff_stop(self, result: RunResult, agent: Any) -> bool
```
Returns `True` if no handoff target is detected in the result.

```python
def _check_team_stop(self, agent_name: str) -> bool
```
Returns `True` after the lead agent completes (first in `flow_order`).

---

## ToolHandler

```python
class ToolHandler(Handler[dict[str, Any], ToolResult])
```

Handles dynamic tool loading, execution, and result aggregation. Accepts a dict of tool arguments keyed by tool call ID, resolves tools from a registry, executes them in parallel, and yields `ToolResult` objects.

### Constructor

```python
def __init__(self, *, tools: dict[str, Tool] | None = None) -> None
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tools` | `dict[str, Tool] \| None` | `None` | Dict mapping tool name to `Tool` instance. |

### Methods

#### register()

```python
def register(self, tool: Tool) -> None
```

Register a tool for execution.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tool` | `Tool` | *(required)* | The tool to register. |

**Raises:** `HandlerError` if a tool with the same name already exists.

#### register_many()

```python
def register_many(self, tools: Sequence[Tool]) -> None
```

Register multiple tools at once.

#### handle()

```python
async def handle(self, input: dict[str, Any], **kwargs: Any) -> AsyncIterator[ToolResult]
```

Execute tool calls described by the input dict. Tools are executed in parallel via `asyncio.TaskGroup`.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input` | `dict[str, Any]` | *(required)* | Mapping of `tool_call_id` to `{"name": str, "arguments": dict}`. |

**Yields:** `ToolResult` for each tool call (in order of tool_call_ids).

#### aggregate()

```python
def aggregate(self, results: Sequence[ToolResult]) -> dict[str, str]
```

Aggregate tool results into a summary dict.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `results` | `Sequence[ToolResult]` | *(required)* | Sequence of tool results. |

**Returns:** Mapping of `tool_call_id` to content or error string.

---

## GroupHandler

```python
class GroupHandler(Handler[str, RunResult])
```

Orchestrates parallel and sequential agent/tool group execution. Groups can be run in parallel (all at once) or serial (with output-to-input chaining and dependency resolution).

### Constructor

```python
def __init__(
    self,
    *,
    agents: dict[str, Any],
    provider: Any = None,
    parallel: bool = True,
    dependencies: dict[str, list[str]] | None = None,
) -> None
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `agents` | `dict[str, Any]` | *(required)* | Dict mapping agent name to agent instance. |
| `provider` | `Any` | `None` | LLM provider. |
| `parallel` | `bool` | `True` | If `True`, run agents concurrently; otherwise serially. |
| `dependencies` | `dict[str, list[str]] \| None` | `None` | Mapping of agent name to list of agent names it depends on (serial mode only). |

### Methods

#### handle()

```python
async def handle(self, input: str, **kwargs: Any) -> AsyncIterator[RunResult]
```

Execute agent group in parallel or serial mode.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input` | `str` | *(required)* | User query string. |
| `**kwargs` | `Any` | | Additional context (`messages`, etc.). |

**Yields:** `RunResult` from each agent execution.

**Raises:** `HandlerError` if a dependency cycle is detected (serial mode) or an agent is not found.

### Example

```python
from exo._internal.handlers import ToolHandler, GroupHandler, SwarmMode

# ToolHandler example
tool_handler = ToolHandler()
tool_handler.register(my_tool)

# GroupHandler example (parallel)
group = GroupHandler(
    agents={"a": agent_a, "b": agent_b},
    provider=my_provider,
    parallel=True,
)
```
