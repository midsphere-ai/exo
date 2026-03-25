# exo.train.trajectory

Trajectory capture with strategy patterns and export to JSON/CSV.

```python
from exo.train.trajectory import (
    DefaultStrategy,
    TrajectoryDataset,
    TrajectoryError,
    TrajectoryItem,
    TrajectoryStrategy,
)
```

---

## TrajectoryError

```python
class TrajectoryError(Exception)
```

Error during trajectory operations.

---

## TrajectoryItem

```python
@dataclass(frozen=True, slots=True)
class TrajectoryItem
```

A single step in an agent execution trajectory. Follows the State-Action-Reward (SAR) pattern.

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | *(auto: 12-char hex)* | Unique step identifier |
| `task_id` | `str` | `""` | Task this step belongs to |
| `agent_id` | `str` | `""` | Agent that produced this step |
| `step` | `int` | `0` | Step number within the task |
| `timestamp` | `float` | *(auto: time.time())* | When the step occurred |
| `input` | `str` | `""` | Input state (user message) |
| `messages` | `tuple[dict[str, Any], ...]` | `()` | Full message history |
| `context` | `dict[str, Any]` | `{}` | Context state |
| `output` | `str` | `""` | Agent output text |
| `tool_calls` | `tuple[dict[str, Any], ...]` | `()` | Tool calls made |
| `score` | `float \| None` | `None` | Reward score |
| `status` | `str` | `"success"` | Step status |
| `metadata` | `dict[str, Any]` | `{}` | Additional metadata |

### Methods

#### to_dict

```python
def to_dict(self) -> dict[str, Any]
```

Serialise to a plain dict.

#### from_dict (classmethod)

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> TrajectoryItem
```

Create a `TrajectoryItem` from a plain dict.

---

## TrajectoryStrategy

```python
class TrajectoryStrategy(ABC)
```

Pluggable strategy for generating trajectory items from messages.

### Abstract methods

#### build_item

```python
def build_item(
    self,
    messages: Sequence[dict[str, Any]],
    *,
    task_id: str = "",
    agent_id: str = "",
    step: int = 0,
    **kwargs: Any,
) -> TrajectoryItem
```

Build a single trajectory item from a message list.

### Methods

#### validate

```python
def validate(self, items: Sequence[TrajectoryItem]) -> bool
```

Return `True` if the trajectory is valid. Override for custom checks. Default implementation returns `True` if `len(items) > 0`.

---

## DefaultStrategy

```python
class DefaultStrategy(TrajectoryStrategy)
```

Default strategy: extracts input/output/tool_calls from message dicts. Scans for the first `user` message as input and the last `assistant` message as output.

---

## TrajectoryDataset

```python
class TrajectoryDataset(
    *,
    strategy: TrajectoryStrategy | None = None,
)
```

Dataset of trajectory items with capture, strategy, and export.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `strategy` | `TrajectoryStrategy \| None` | `None` | Strategy for building items. Defaults to `DefaultStrategy` |

### Properties

| Property | Type | Description |
|---|---|---|
| `items` | `list[TrajectoryItem]` | Copy of all trajectory items |
| `strategy` | `TrajectoryStrategy` | The active strategy |

### Methods

#### append_trajectory

```python
def append_trajectory(self, item: TrajectoryItem) -> None
```

Append a pre-built trajectory item.

#### from_messages

```python
def from_messages(
    self,
    messages: Sequence[dict[str, Any]],
    *,
    task_id: str = "",
    agent_id: str = "",
    **kwargs: Any,
) -> TrajectoryItem
```

Build a trajectory item from messages via strategy and append it. Automatically increments the step counter per task/agent pair.

**Returns:** The built and appended `TrajectoryItem`.

#### save_task_trajectory

```python
def save_task_trajectory(
    self,
    task_id: str,
    items: Sequence[TrajectoryItem],
) -> int
```

Bulk-append items for a given task. Returns count added.

#### get_task_trajectory

```python
def get_task_trajectory(self, task_id: str) -> list[TrajectoryItem]
```

Return all items matching *task_id*.

#### validate

```python
def validate(self) -> bool
```

Delegate validation to the strategy.

#### to_json

```python
def to_json(self) -> str
```

Export all items as a JSON string.

#### to_csv

```python
def to_csv(self) -> str
```

Export all items as CSV text. Complex fields (`messages`, `tool_calls`, `context`, `metadata`) are serialized to JSON strings within the CSV.

#### clear

```python
def clear(self) -> None
```

Remove all items and reset step counters.

### Special methods

- `__len__()` -- Returns the number of items.

### Example

```python
from exo.train import TrajectoryDataset

dataset = TrajectoryDataset()

messages = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
]

item = dataset.from_messages(messages, task_id="task-1", agent_id="tutor")
print(item.input)   # "What is Python?"
print(item.output)  # "Python is a programming language."
print(len(dataset)) # 1

# Export
json_str = dataset.to_json()
csv_str = dataset.to_csv()
```
