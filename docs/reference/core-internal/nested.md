# exo._internal.nested

Support for nested swarms -- using a Swarm as a node in another Swarm.

> **Internal API** -- `SwarmNode` is re-exported as public API from `exo.__init__`.

**Module:** `exo._internal.nested`

```python
from exo._internal.nested import SwarmNode, NestedSwarmError
# or (public API)
from exo import SwarmNode
```

---

## NestedSwarmError

```python
class NestedSwarmError(ExoError)
```

Raised for nested swarm errors (invalid swarm instance). Inherits from `ExoError`.

---

## SwarmNode

```python
class SwarmNode
```

Wraps a `Swarm` so it can be used as a node in another Swarm. Provides the same interface that Swarm expects from agent nodes -- `name` attribute and `run()` method. An `is_swarm` marker allows the outer Swarm to detect nested swarms via duck-typing.

**Context isolation:** The inner Swarm creates its own `RunState` and message history on each `run()` invocation. No mutable state leaks between inner and outer executions.

### Constructor

```python
def __init__(
    self,
    *,
    swarm: Any,
    name: str | None = None,
) -> None
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `swarm` | `Any` | *(required)* | The inner Swarm to wrap. Must have a `flow_order` attribute. |
| `name` | `str \| None` | `None` | Node name for the outer Swarm's flow DSL. Defaults to the inner Swarm's `name` attribute. |

**Raises:** `NestedSwarmError` if `swarm` does not have a `flow_order` attribute.

### Instance Attributes

| Name | Type | Description |
|------|------|-------------|
| `name` | `str` | Node name in the outer swarm. |
| `is_swarm` | `bool` | Always `True`. Marker for outer Swarm detection. |

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

Execute the inner swarm with context isolation. Each call creates a fresh execution context -- the inner Swarm builds its own `RunState` and does not share mutable state with the outer Swarm.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input` | `str` | *(required)* | User query string. |
| `messages` | `Sequence[Message] \| None` | `None` | Not forwarded to inner swarm (context isolation). The inner swarm starts with a clean message history. |
| `provider` | `Any` | `None` | LLM provider, forwarded to inner swarm. |
| `max_retries` | `int` | `3` | Retry attempts, forwarded to inner swarm. |

**Returns:** `RunResult` from the inner swarm's execution.

#### describe()

```python
def describe(self) -> dict[str, Any]
```

Return a summary including the inner swarm's description.

**Returns:** Dict with keys:
- `type` (`"nested_swarm"`)
- `name` (str)
- `inner` (dict) -- the inner swarm's `describe()` output

#### \_\_repr\_\_()

```python
def __repr__(self) -> str
```

E.g. `SwarmNode(name='inner_pipeline', inner=Swarm(...))`.

### Example

```python
from exo import Agent, Swarm, SwarmNode, run

# Build an inner pipeline
researcher = Agent(name="researcher", instructions="Research the topic.")
writer = Agent(name="writer", instructions="Write based on research.")
inner = Swarm(agents=[researcher, writer], flow="researcher >> writer")

# Wrap as a node in an outer swarm
inner_node = SwarmNode(swarm=inner, name="research_pipeline")

# Build the outer swarm
editor = Agent(name="editor", instructions="Edit the article.")
publisher = Agent(name="publisher", instructions="Publish the article.")

outer = Swarm(
    agents=[inner_node, editor, publisher],
    flow="research_pipeline >> editor >> publisher",
)

# Execute
result = run.sync(outer, "Write an article about quantum computing")
```

### Context Isolation

The key feature of `SwarmNode` is context isolation. The inner swarm:

- Gets a fresh message history (outer messages are NOT forwarded)
- Creates its own `RunState`
- Does not leak any mutable state back to the outer swarm
- Only its final `RunResult.output` is used as input for the next node in the outer flow
