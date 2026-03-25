# State Management

`ContextState` provides a hierarchical key-value store with parent inheritance. When you `fork()` a context for a sub-task, the child automatically sees the parent's state while maintaining its own local overrides.

## Basic Usage

```python
from exo.context import ContextState

state = ContextState()

# Set and get values
state.set("user_id", "u-123")
state.set("task", {"name": "research", "priority": 1})

print(state.get("user_id"))        # "u-123"
print(state.get("missing"))         # None
print(state.get("missing", "fallback"))  # "fallback"
```

## Parent Inheritance

Child states inherit values from their parent. Local values override inherited ones:

```python
parent = ContextState()
parent.set("model", "gpt-4o")
parent.set("temperature", 0.7)

child = ContextState(parent=parent)
child.set("temperature", 0.2)  # override parent

print(child.get("model"))       # "gpt-4o" (inherited from parent)
print(child.get("temperature")) # 0.2 (local override)
```

The lookup chain walks up through all ancestors until a value is found:

```python
grandparent = ContextState()
grandparent.set("org_id", "org-1")

parent = ContextState(parent=grandparent)
child = ContextState(parent=parent)

print(child.get("org_id"))  # "org-1" (inherited two levels up)
```

## Context Fork and Merge

The `Context` class uses `ContextState` for its state management. Forking creates a child context with an inherited state:

```python
from exo.context import Context, ContextConfig

ctx = Context(task_id="main", config=ContextConfig())
ctx.state.set("plan", ["step1", "step2"])

child = ctx.fork("sub-task")
child.state.set("result", "done")

# Parent doesn't see child's local state yet
print(ctx.state.get("result"))  # None

# Merge child back into parent
ctx.merge(child)
print(ctx.state.get("result"))  # "done"
```

When `ctx.merge(child)` is called:

1. The child's **local** state keys are copied into the parent's state.
2. The child's **net token delta** is consolidated into the parent's token tracker.

## Mutation Methods

`ContextState` supports the following operations:

```python
state = ContextState()

# Set a value
state.set("key", "value")

# Update multiple keys at once
state.update({"a": 1, "b": 2, "c": 3})

# Delete a key (from local store only)
state.delete("a")

# Pop a key -- returns value and removes it
val = state.pop("b", "default")

# Clear all local keys
state.clear()
```

## Inspecting State

```python
state = ContextState()
state.set("x", 10)
state.set("y", 20)

# Local keys only (excludes inherited)
print(state.local_dict())  # {"x": 10, "y": 20}

# All keys including inherited
print(state.to_dict())     # {"x": 10, "y": 20} (plus parent keys)

# Available keys
print(state.keys())        # {"x", "y"} (plus parent keys)

# Access the parent
parent_state = state.parent  # ContextState | None
```

## Advanced Patterns

### Scoped Sub-Tasks

Use fork/merge to isolate sub-task state and avoid polluting the parent:

```python
async def run_subtask(parent_ctx: Context, task_id: str, input: str):
    child = parent_ctx.fork(task_id)
    child.state.set("input", input)

    # ... run agent steps using child context ...
    child.state.set("output", "result of subtask")

    # Only merge back the keys you want the parent to see
    parent_ctx.merge(child)
```

### State as Agent Memory

Store conversation history or accumulated facts in state for prompt building:

```python
ctx.state.set("facts", [
    "User prefers Python",
    "Project uses FastAPI",
])

# A neuron can read this during prompt assembly:
facts = ctx.state.get("facts", [])
```

### Snapshot Before Risky Operations

Use checkpoints to save state before potentially destructive operations:

```python
checkpoint = ctx.snapshot(metadata={"reason": "before tool execution"})

try:
    # ... risky operation ...
    pass
except Exception:
    ctx.restore(checkpoint)  # roll back to saved state
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `ContextState` | `exo.context` | Hierarchical key-value store with parent inheritance |
| `ContextState.get(key, default)` | | Lookup with parent chain fallback |
| `ContextState.set(key, value)` | | Set a local key-value pair |
| `ContextState.update(mapping)` | | Set multiple keys at once |
| `ContextState.delete(key)` | | Remove a key from local store |
| `ContextState.pop(key, default)` | | Remove and return a key |
| `ContextState.clear()` | | Clear all local keys |
| `ContextState.local_dict()` | | Dict of local keys only |
| `ContextState.to_dict()` | | Dict of all keys (local + inherited) |
| `ContextState.keys()` | | Set of all available keys |
| `ContextState.parent` | | Parent state or `None` |
| `Context.fork(task_id)` | `exo.context` | Create child context with inherited state |
| `Context.merge(child)` | `exo.context` | Consolidate child's local state into parent |
