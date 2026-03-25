# exo.context.context

Core context lifecycle with fork/merge for hierarchical task decomposition.

## Module Path

```python
from exo.context.context import Context, ContextError
```

---

## ContextError

Exception raised for context lifecycle errors (e.g., empty task_id, invalid merge).

```python
class ContextError(Exception): ...
```

---

## Context

Per-task context with hierarchical state and fork/merge lifecycle. Holds configuration, state, token usage tracking, and checkpointing.

### Constructor

```python
Context(
    task_id: str,
    *,
    config: ContextConfig | None = None,
    parent: Context | None = None,
    state: ContextState | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task_id` | `str` | *(required)* | Unique identifier for the task (must be non-empty) |
| `config` | `ContextConfig \| None` | `None` | Immutable configuration. Inherits from parent if `None`, else defaults to `ContextConfig()` |
| `parent` | `Context \| None` | `None` | Optional parent context. Forked contexts set this automatically |
| `state` | `ContextState \| None` | `None` | Initial state. If omitted, creates empty state (with parent chain if parent is set) |

**Raises:** `ContextError` if `task_id` is empty.

### Properties

| Property | Type | Description |
|---|---|---|
| `task_id` | `str` | The task identifier |
| `config` | `ContextConfig` | Immutable context configuration |
| `parent` | `Context \| None` | Parent context, if any |
| `state` | `ContextState` | Hierarchical state object |
| `children` | `list[Context]` | Direct child contexts (copy) |
| `token_usage` | `dict[str, int]` | Current token usage counters (copy) |
| `checkpoints` | `CheckpointStore` | Checkpoint store for this context |

### Methods

#### add_tokens()

```python
def add_tokens(self, usage: dict[str, int]) -> None
```

Add token counts to this context's usage tracker.

| Parameter | Type | Description |
|---|---|---|
| `usage` | `dict[str, int]` | Token counts to add (keys are metric names) |

#### fork()

```python
def fork(self, task_id: str) -> Context
```

Create a child context for a sub-task. The child inherits:

- **config** -- shared reference (immutable)
- **state** -- via `ContextState` parent chain (reads inherit, writes isolate)
- **token_usage** -- snapshot for net-delta calculation on merge

The child is registered in the parent's `children` list.

| Parameter | Type | Description |
|---|---|---|
| `task_id` | `str` | Identifier for the child task |

**Returns:** A new child `Context`.

#### merge()

```python
def merge(self, child: Context) -> None
```

Consolidate a child context back into this context.

Merges:
1. Child's **local** state entries into parent state
2. **Net** token usage (child current - parent snapshot at fork time)

| Parameter | Type | Description |
|---|---|---|
| `child` | `Context` | The child context to merge |

**Raises:** `ContextError` if `child` is not a direct child of this context.

#### snapshot()

```python
def snapshot(self, *, metadata: dict[str, Any] | None = None) -> Checkpoint
```

Save a checkpoint of the current context state. Captures a deep copy of state values and token usage.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metadata` | `dict[str, Any] \| None` | `None` | Optional metadata (e.g., description, step number) |

**Returns:** The created `Checkpoint`.

#### restore() (classmethod)

```python
@classmethod
def restore(cls, checkpoint: Checkpoint, *, config: ContextConfig | None = None) -> Context
```

Restore a context from a checkpoint. Creates a new `Context` with state and token usage reconstructed from the checkpoint data.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `checkpoint` | `Checkpoint` | *(required)* | The checkpoint to restore from |
| `config` | `ContextConfig \| None` | `None` | Optional config override |

**Returns:** A new `Context` with the restored state.

**Raises:** `CheckpointError` if the checkpoint data is invalid.

### Example

```python
import asyncio
from exo.context import Context, ContextConfig, AutomationMode

async def main():
    # Create root context
    ctx = Context("main-task", config=ContextConfig(mode=AutomationMode.COPILOT))

    # Set state
    ctx.state.set("task_input", "Build a web server")
    ctx.add_tokens({"input_tokens": 100, "output_tokens": 50})

    # Fork for sub-task
    child = ctx.fork("subtask-search")
    child.state.set("search_results", ["result1", "result2"])
    child.add_tokens({"input_tokens": 200, "output_tokens": 80})

    # Merge child back
    ctx.merge(child)
    print(ctx.state.get("search_results"))  # ["result1", "result2"]
    print(ctx.token_usage)  # includes net child usage

    # Checkpoint and restore
    cp = ctx.snapshot(metadata={"step": "after search"})
    restored = Context.restore(cp)
    print(restored.state.get("search_results"))  # ["result1", "result2"]

asyncio.run(main())
```

### Fork/Merge Lifecycle

```
Root Context (task_id="main")
  |
  |-- fork("sub-1") --> Child Context
  |     state: inherits from parent (reads go up the chain)
  |     writes: isolated to child's local state
  |     tokens: snapshot of parent's usage at fork time
  |
  |-- merge(child) <-- child.local_dict() -> parent.state.update()
  |     net tokens: child_current - snapshot_at_fork -> added to parent
```
