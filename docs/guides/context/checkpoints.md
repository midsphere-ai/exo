# Checkpoints

Checkpoints provide snapshot save and restore for context state. They capture the full state dictionary, token usage, and arbitrary metadata at a specific point in time. Use checkpoints to implement rollback after failed operations, save progress on long-running tasks, or maintain multiple branches of an agent's execution.

## Basic Usage

```python
from exo.context import Context, ContextConfig, Checkpoint, CheckpointStore

ctx = Context(task_id="task-1", config=ContextConfig())
ctx.state.set("progress", "step-3")
ctx.state.set("data", {"results": [1, 2, 3]})

# Save a snapshot
checkpoint = ctx.snapshot(metadata={"reason": "before risky tool call"})
print(checkpoint.task_id)   # "task-1"
print(checkpoint.version)   # 1
print(checkpoint.values)    # {"progress": "step-3", "data": {"results": [1, 2, 3]}}
```

## Checkpoint Structure

A `Checkpoint` is a frozen (immutable) dataclass:

```python
@dataclass(frozen=True)
class Checkpoint:
    task_id: str                    # Context task ID
    version: int                    # Sequential version number
    values: dict[str, Any]          # State snapshot
    token_usage: dict[str, Any]     # Token tracker snapshot
    metadata: dict[str, Any]        # User-provided metadata
    created_at: str                 # ISO timestamp
```

Checkpoints are serializable via `to_dict()` and `from_dict()`:

```python
# Serialize
data = checkpoint.to_dict()

# Restore from dict
restored = Checkpoint.from_dict(data)
```

## Save and Restore

```python
ctx = Context(task_id="task-1", config=ContextConfig())
ctx.state.set("step", 1)

# Save checkpoint
cp1 = ctx.snapshot(metadata={"step": 1})

# Continue working
ctx.state.set("step", 2)
ctx.state.set("result", "intermediate")

# Something goes wrong -- roll back
ctx.restore(cp1)
print(ctx.state.get("step"))    # 1
print(ctx.state.get("result"))  # None (was not in checkpoint)
```

## Checkpoint Store

The `CheckpointStore` manages multiple checkpoints with version tracking:

```python
from exo.context import CheckpointStore

store = CheckpointStore()

# Save checkpoints
store.save(checkpoint_1)
store.save(checkpoint_2)
store.save(checkpoint_3)

# Get the latest checkpoint
latest = store.latest
print(latest.version)  # 3

# Get a specific version
cp = store.get(version=1)

# List all versions
versions = store.list_versions()  # [1, 2, 3]
```

## Advanced Patterns

### Rollback on Error

Wrap risky operations in a try/except with automatic rollback:

```python
async def safe_tool_execution(ctx: Context, tool, args):
    checkpoint = ctx.snapshot(metadata={"before_tool": tool.name})

    try:
        result = await tool.execute(**args)
        ctx.state.set("last_tool_result", result)
        return result
    except Exception as e:
        ctx.restore(checkpoint)
        ctx.state.set("last_error", str(e))
        raise
```

### Long-Running Task Progress

Save checkpoints periodically so a task can resume from the last good state:

```python
store = CheckpointStore()

async def run_multi_step_task(ctx: Context, steps: list[str]):
    for i, step in enumerate(steps):
        # Save progress before each step
        cp = ctx.snapshot(metadata={"step_index": i, "step_name": step})
        store.save(cp)

        try:
            await execute_step(ctx, step)
        except Exception:
            # Resume from last checkpoint on restart
            latest = store.latest
            if latest:
                ctx.restore(latest)
            raise
```

### Branching Execution

Use checkpoints to explore multiple execution paths:

```python
# Save the decision point
branch_point = ctx.snapshot(metadata={"branch": "decision_point"})

# Try approach A
ctx.state.set("approach", "A")
result_a = await try_approach_a(ctx)

if not result_a.success:
    # Roll back and try approach B
    ctx.restore(branch_point)
    ctx.state.set("approach", "B")
    result_b = await try_approach_b(ctx)
```

### Checkpoint Persistence

Serialize checkpoints for external storage (database, filesystem):

```python
import json

# Save to disk
checkpoint = ctx.snapshot(metadata={"reason": "daily backup"})
with open("checkpoint.json", "w") as f:
    json.dump(checkpoint.to_dict(), f)

# Load from disk
with open("checkpoint.json") as f:
    data = json.load(f)
restored = Checkpoint.from_dict(data)
ctx.restore(restored)
```

### Comparing Checkpoints

Track state changes between checkpoints:

```python
store = CheckpointStore()

cp_before = ctx.snapshot()
store.save(cp_before)

# ... agent does work ...

cp_after = ctx.snapshot()
store.save(cp_after)

# Compare
added = set(cp_after.values.keys()) - set(cp_before.values.keys())
removed = set(cp_before.values.keys()) - set(cp_after.values.keys())
changed = {
    k for k in cp_before.values.keys() & cp_after.values.keys()
    if cp_before.values[k] != cp_after.values[k]
}
print(f"Added: {added}, Removed: {removed}, Changed: {changed}")
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `Checkpoint` | `exo.context` | Frozen snapshot of context state and token usage |
| `Checkpoint.to_dict()` | | Serialize to dictionary |
| `Checkpoint.from_dict(data)` | | Deserialize from dictionary (class method) |
| `CheckpointStore` | `exo.context` | Manages versioned checkpoint history |
| `CheckpointStore.save(checkpoint)` | | Save a checkpoint |
| `CheckpointStore.get(version)` | | Retrieve a specific version |
| `CheckpointStore.latest` | | Property: the most recent checkpoint |
| `CheckpointStore.list_versions()` | | List all saved version numbers |
| `Context.snapshot(metadata)` | `exo.context` | Create a checkpoint from current state |
| `Context.restore(checkpoint)` | `exo.context` | Restore state from a checkpoint |
