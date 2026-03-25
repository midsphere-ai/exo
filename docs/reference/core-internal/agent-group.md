# exo._internal.agent_group

Parallel and serial agent group primitives for expressing concurrent-then-sequential execution patterns within a Swarm flow.

> **Internal API** -- `ParallelGroup` and `SerialGroup` are re-exported as public API from `exo.__init__`.

**Module:** `exo._internal.agent_group`

```python
from exo._internal.agent_group import ParallelGroup, SerialGroup, GroupError
# or (public API)
from exo import ParallelGroup, SerialGroup
```

---

## GroupError

```python
class GroupError(ExoError)
```

Raised for agent group errors (empty agent list). Inherits from `ExoError`.

---

## ParallelGroup

```python
class ParallelGroup
```

Concurrent execution of multiple agents. All agents receive the same input and run concurrently via `asyncio.TaskGroup`. Results are aggregated by joining outputs with the specified separator, or via a custom aggregation function.

Groups behave like agents from the Swarm's perspective -- they have a `name` attribute and can be placed in the agent list and flow DSL.

### Constructor

```python
def __init__(
    self,
    *,
    name: str,
    agents: list[Any],
    separator: str = "\n\n",
    aggregate_fn: Any = None,
) -> None
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | *(required)* | Group name (used as a node in flow DSL). |
| `agents` | `list[Any]` | *(required)* | List of agents to run concurrently. |
| `separator` | `str` | `"\n\n"` | String used to join agent outputs. |
| `aggregate_fn` | `Any` | `None` | Optional custom `(list[RunResult]) -> str` aggregation function. Overrides `separator` when provided. |

**Raises:** `GroupError` if `agents` is empty.

### Instance Attributes

| Name | Type | Description |
|------|------|-------------|
| `name` | `str` | Group name. |
| `agents` | `dict[str, Any]` | Agents indexed by name. |
| `agent_order` | `list[str]` | Agent names in original order. |
| `separator` | `str` | Output join separator. |
| `aggregate_fn` | `Any` | Custom aggregation function. |
| `is_group` | `bool` | Always `True`. Marker so Swarm can detect groups vs regular agents. |

### Methods

#### run()

```python
async def run(
    self,
    input: str,
    *,
    messages: Sequence[Message] | None = None,
    provider: Any = None,
    max_retries: int = 3,
) -> RunResult
```

Run all agents concurrently and aggregate results.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input` | `str` | *(required)* | User query string (sent to every agent). |
| `messages` | `Sequence[Message] \| None` | `None` | Prior conversation history. |
| `provider` | `Any` | `None` | LLM provider for all agents. |
| `max_retries` | `int` | `3` | Retry attempts for transient errors. |

**Returns:** `RunResult` with aggregated output, merged usage (summed across all agents), and combined step count.

#### describe()

```python
def describe(self) -> dict[str, Any]
```

Return a summary of the group's configuration.

**Returns:** Dict with keys `type` (`"parallel"`), `name`, `agents`.

### Example

```python
from exo import Agent, Swarm, ParallelGroup

researcher = Agent(name="researcher", instructions="Research the topic.")
analyst = Agent(name="analyst", instructions="Analyze the data.")
synthesizer = Agent(name="synthesizer", instructions="Synthesize the findings.")

# Run researcher and analyst in parallel, then synthesizer
parallel = ParallelGroup(
    name="research_team",
    agents=[researcher, analyst],
    separator="\n---\n",
)

swarm = Swarm(
    agents=[parallel, synthesizer],
    flow="research_team >> synthesizer",
)
```

### Custom Aggregation

```python
from exo.types import RunResult

def pick_best(results: list[RunResult]) -> str:
    """Pick the longest output as the 'best' result."""
    return max(results, key=lambda r: len(r.output)).output

parallel = ParallelGroup(
    name="voters",
    agents=[agent_a, agent_b, agent_c],
    aggregate_fn=pick_best,
)
```

---

## SerialGroup

```python
class SerialGroup
```

Sequential execution of agents with output-to-input chaining. Agents execute in order; each agent's output becomes the next agent's input. The final agent's output is the group output.

### Constructor

```python
def __init__(
    self,
    *,
    name: str,
    agents: list[Any],
) -> None
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | *(required)* | Group name (used as a node in flow DSL). |
| `agents` | `list[Any]` | *(required)* | List of agents to run sequentially (in given order). |

**Raises:** `GroupError` if `agents` is empty.

### Instance Attributes

| Name | Type | Description |
|------|------|-------------|
| `name` | `str` | Group name. |
| `agents` | `dict[str, Any]` | Agents indexed by name. |
| `agent_order` | `list[str]` | Agent names in execution order. |
| `is_group` | `bool` | Always `True`. Marker so Swarm can detect groups. |

### Methods

#### run()

```python
async def run(
    self,
    input: str,
    *,
    messages: Sequence[Message] | None = None,
    provider: Any = None,
    max_retries: int = 3,
) -> RunResult
```

Run agents sequentially, chaining output to input.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input` | `str` | *(required)* | User query string for the first agent. |
| `messages` | `Sequence[Message] \| None` | `None` | Prior conversation history. |
| `provider` | `Any` | `None` | LLM provider for all agents. |
| `max_retries` | `int` | `3` | Retry attempts for transient errors. |

**Returns:** `RunResult` from the last agent, with accumulated usage and step count from all agents.

#### describe()

```python
def describe(self) -> dict[str, Any]
```

Return a summary of the group's configuration.

**Returns:** Dict with keys `type` (`"serial"`), `name`, `agents`.

### Example

```python
from exo import Agent, Swarm, SerialGroup

drafter = Agent(name="drafter", instructions="Write a first draft.")
reviewer = Agent(name="reviewer", instructions="Review and improve the draft.")
finalizer = Agent(name="finalizer", instructions="Finalize the document.")

pipeline = SerialGroup(
    name="writing_pipeline",
    agents=[drafter, reviewer, finalizer],
)

# Use in a swarm with other agents
swarm = Swarm(
    agents=[pipeline, publisher],
    flow="writing_pipeline >> publisher",
)
```
