# exo.context.checkpoint

Save and restore complete execution state for long-running tasks.

## Module Path

```python
from exo.context.checkpoint import Checkpoint, CheckpointStore, CheckpointError
```

---

## CheckpointError

Exception raised for checkpoint lifecycle errors.

```python
class CheckpointError(Exception): ...
```

---

## Checkpoint

Immutable snapshot of context state at a point in time.

**Decorator:** `@dataclass(frozen=True, slots=True)`

### Constructor

```python
Checkpoint(
    task_id: str,
    version: int,
    values: dict[str, Any],
    token_usage: dict[str, int],
    metadata: dict[str, Any] = {},
    created_at: datetime = datetime.now(UTC),
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task_id` | `str` | *(required)* | The task this checkpoint belongs to |
| `version` | `int` | *(required)* | Monotonically increasing version number |
| `values` | `dict[str, Any]` | *(required)* | Snapshot of context state key-value pairs |
| `token_usage` | `dict[str, int]` | *(required)* | Snapshot of token usage counters |
| `metadata` | `dict[str, Any]` | `{}` | Optional user-provided metadata |
| `created_at` | `datetime` | `datetime.now(UTC)` | UTC creation timestamp |

### Methods

#### to_dict()

```python
def to_dict(self) -> dict[str, Any]
```

Serialize checkpoint to a dictionary.

**Returns:** A dict with keys: `task_id`, `version`, `values`, `token_usage`, `metadata`, `created_at` (ISO string).

#### from_dict() (classmethod)

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> Checkpoint
```

Deserialize checkpoint from a dictionary. Handles ISO-8601 `created_at` string conversion.

---

## CheckpointStore

Per-session checkpoint store with version tracking. Manages a sequence of checkpoints for a given task.

### Constructor

```python
CheckpointStore(task_id: str)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task_id` | `str` | *(required)* | Task identifier (must be non-empty) |

**Raises:** `CheckpointError` if `task_id` is empty.

### Properties

| Property | Type | Description |
|---|---|---|
| `task_id` | `str` | The task identifier |
| `version` | `int` | Current version (number of checkpoints taken) |
| `latest` | `Checkpoint \| None` | Most recent checkpoint, or `None` if empty |

### Methods

#### save()

```python
def save(
    self,
    values: dict[str, Any],
    token_usage: dict[str, int],
    *,
    metadata: dict[str, Any] | None = None,
) -> Checkpoint
```

Create a new checkpoint from the given state.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `values` | `dict[str, Any]` | *(required)* | Context state to snapshot |
| `token_usage` | `dict[str, int]` | *(required)* | Token usage counters to snapshot |
| `metadata` | `dict[str, Any] \| None` | `None` | Optional metadata |

**Returns:** The created `Checkpoint`.

#### get()

```python
def get(self, version: int) -> Checkpoint
```

Retrieve a checkpoint by version number (1-based).

**Raises:** `CheckpointError` if the version does not exist.

#### list_versions()

```python
def list_versions(self) -> list[int]
```

List all available checkpoint version numbers.

### Dunder Methods

| Method | Description |
|---|---|
| `__len__` | Number of checkpoints |
| `__repr__` | `CheckpointStore(task_id='task-1', checkpoints=3)` |

### Example

```python
from exo.context.checkpoint import CheckpointStore, Checkpoint

# Create store
store = CheckpointStore("task-1")

# Save checkpoints
cp1 = store.save(
    values={"step": 1, "data": "initial"},
    token_usage={"input_tokens": 100, "output_tokens": 50},
    metadata={"description": "After step 1"},
)
assert cp1.version == 1

cp2 = store.save(
    values={"step": 2, "data": "updated"},
    token_usage={"input_tokens": 250, "output_tokens": 120},
)
assert cp2.version == 2

# Retrieve
assert store.latest == cp2
assert store.get(1) == cp1

# List versions
assert store.list_versions() == [1, 2]

# Serialize/deserialize
data = cp1.to_dict()
restored = Checkpoint.from_dict(data)
assert restored.task_id == "task-1"
assert restored.version == 1
```

### Integration with Context

The `Context` class uses `CheckpointStore` internally:

```python
from exo.context import Context

ctx = Context("task-1")
ctx.state.set("progress", "step 1")
ctx.add_tokens({"input_tokens": 100})

# Snapshot via Context
cp = ctx.snapshot(metadata={"step": 1})

# Restore via Context
restored_ctx = Context.restore(cp)
assert restored_ctx.state.get("progress") == "step 1"
```
