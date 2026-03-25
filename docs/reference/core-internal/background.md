# exo._internal.background

Background task handler with hot-merge and wake-up-merge patterns. Manages long-running background tasks that produce results asynchronously. Results can be merged into the running execution (hot-merge) or queued for later retrieval (wake-up-merge) when the main task has already completed.

> **Internal API** -- subject to change without notice.

**Module:** `exo._internal.background`

```python
from exo._internal.background import (
    BackgroundTask,
    BackgroundTaskError,
    BackgroundTaskHandler,
    MergeMode,
    PendingQueue,
)
```

---

## BackgroundTaskError

```python
class BackgroundTaskError(ExoError)
```

Raised for background task handler errors (duplicate task IDs, unknown tasks). Inherits from `ExoError`.

---

## MergeMode

```python
class MergeMode(StrEnum)
```

How background results are merged into the main execution.

### Values

| Member | Value | Description |
|--------|-------|-------------|
| `HOT` | `"hot"` | Result merged directly into the active execution state. |
| `WAKEUP` | `"wakeup"` | Result queued for later retrieval (main task already completed). |

---

## BackgroundTask

```python
class BackgroundTask
```

Tracks a single background task's lifecycle.

### Constructor

```python
def __init__(
    self,
    task_id: str,
    parent_task_id: str,
    *,
    payload: Any = None,
) -> None
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `task_id` | `str` | *(required)* | Unique identifier for this task. |
| `parent_task_id` | `str` | *(required)* | The task that spawned this background task. |
| `payload` | `Any` | `None` | Arbitrary data associated with the task. |

### Instance Attributes

| Name | Type | Description |
|------|------|-------------|
| `task_id` | `str` | Task identifier. |
| `parent_task_id` | `str` | Parent task identifier. |
| `payload` | `Any` | Associated data. |
| `result` | `Any` | Task result (set by `complete()`). Initially `None`. |
| `error` | `str \| None` | Error message (set by `fail()`). Initially `None`. |
| `status` | `RunNodeStatus` | Current status. Starts as `INIT`. |
| `merge_mode` | `MergeMode \| None` | How the result was merged. Set by the handler. Initially `None`. |

### Methods

#### start()

```python
def start(self) -> None
```

Mark task as running (transitions to `RUNNING`).

#### complete()

```python
def complete(self, result: Any) -> None
```

Mark task as successfully completed with a result (transitions to `SUCCESS`).

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `result` | `Any` | *(required)* | The task's result value. |

#### fail()

```python
def fail(self, error: str) -> None
```

Mark task as failed with an error message (transitions to `FAILED`).

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `error` | `str` | *(required)* | Error description. |

### Properties

#### is_complete

```python
@property
def is_complete(self) -> bool
```

Whether the task has reached a terminal state (`SUCCESS` or `FAILED`).

---

## PendingQueue

```python
class PendingQueue
```

Thread-safe queue for background results awaiting merge. Used in the wake-up-merge pattern: when the main task has already completed, background results are queued here for later retrieval and re-processing.

### Constructor

```python
def __init__(self) -> None
```

Creates an empty queue.

### Methods

#### push()

```python
def push(self, task: BackgroundTask) -> None
```

Add a completed task to the pending queue. Signals any waiters.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `task` | `BackgroundTask` | *(required)* | The completed task. |

#### pop_all()

```python
def pop_all(self) -> list[BackgroundTask]
```

Remove and return all pending tasks. Clears the internal event signal.

**Returns:** List of pending `BackgroundTask` instances.

#### wait()

```python
async def wait(self, timeout: float | None = None) -> bool
```

Wait until at least one item is available.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `timeout` | `float \| None` | `None` | Max seconds to wait, or `None` for no timeout. |

**Returns:** `True` if items are available, `False` on timeout.

### Properties

#### size

```python
@property
def size(self) -> int
```

Number of pending items.

#### empty

```python
@property
def empty(self) -> bool
```

Whether the queue has no pending items.

---

## BackgroundTaskHandler

```python
class BackgroundTaskHandler
```

Manages background tasks with hot-merge and wake-up-merge patterns.

**Hot-merge:** When the main task is still running and a background task completes, the result is merged directly into the active execution state.

**Wake-up-merge:** When the main task has already completed (or paused), background results are queued in a `PendingQueue` for later retrieval. A checkpoint can be restored and the pending results merged in.

### Constructor

```python
def __init__(self, *, state: RunState | None = None) -> None
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `state` | `RunState \| None` | `None` | Optional `RunState` for tracking background task nodes. |

### Methods

#### submit()

```python
def submit(
    self,
    task_id: str,
    parent_task_id: str,
    *,
    payload: Any = None,
) -> BackgroundTask
```

Register a new background task. Creates a `BackgroundTask`, marks it as started, and optionally creates a tracking `RunNode` in the state.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `task_id` | `str` | *(required)* | Unique identifier for the task. |
| `parent_task_id` | `str` | *(required)* | The parent task that spawned this one. |
| `payload` | `Any` | `None` | Arbitrary data for the task. |

**Returns:** The created `BackgroundTask`.

**Raises:** `BackgroundTaskError` if a task with this ID already exists.

#### handle_result()

```python
async def handle_result(
    self,
    task_id: str,
    result: Any,
    *,
    is_main_running: bool = True,
) -> MergeMode
```

Handle a background task's completion. Routes to hot-merge or wake-up-merge based on whether the main task is still running.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `task_id` | `str` | *(required)* | The background task that completed. |
| `result` | `Any` | *(required)* | The task's result value. |
| `is_main_running` | `bool` | `True` | Whether the parent task is still active. |

**Returns:** The `MergeMode` that was applied.

**Raises:** `BackgroundTaskError` if the task ID is not found.

#### handle_error()

```python
def handle_error(self, task_id: str, error: str) -> None
```

Record a background task failure.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `task_id` | `str` | *(required)* | The task that failed. |
| `error` | `str` | *(required)* | Error description. |

**Raises:** `BackgroundTaskError` if the task ID is not found.

#### drain_pending()

```python
async def drain_pending(self) -> AsyncIterator[BackgroundTask]
```

Yield all pending background tasks (wake-up-merge pattern). Used when restoring from a checkpoint to process any background results that arrived while the main task was paused.

**Yields:** Each pending `BackgroundTask` with its result.

#### on_merge()

```python
def on_merge(self, callback: Any) -> None
```

Register a callback for merge events. The callback is called with `(task, merge_mode)` whenever a background result is merged (hot or wake-up).

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `callback` | `Any` | *(required)* | An async callable `(BackgroundTask, MergeMode) -> None`. |

#### get_task()

```python
def get_task(self, task_id: str) -> BackgroundTask | None
```

Retrieve a background task by ID.

**Returns:** The `BackgroundTask`, or `None` if not found.

#### list_tasks()

```python
def list_tasks(self, *, status: RunNodeStatus | None = None) -> list[BackgroundTask]
```

List background tasks, optionally filtered by status.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `status` | `RunNodeStatus \| None` | `None` | If given, only return tasks with this status. |

**Returns:** List of matching tasks.

### Properties

#### pending_queue

```python
@property
def pending_queue(self) -> PendingQueue
```

Access the pending queue for wake-up-merge tasks.

### Example

```python
import asyncio
from exo._internal.background import BackgroundTaskHandler, MergeMode

handler = BackgroundTaskHandler()

# Submit a background task
task = handler.submit("task_1", "main_run", payload={"url": "https://..."})

# Later, when the task completes (hot-merge: main is still running)
mode = asyncio.run(handler.handle_result("task_1", result="fetched data"))
print(mode)  # MergeMode.HOT

# Or if the main task already finished (wake-up-merge)
task2 = handler.submit("task_2", "main_run")
mode = asyncio.run(handler.handle_result("task_2", result="late data", is_main_running=False))
print(mode)  # MergeMode.WAKEUP

# Drain pending results
async def process_pending():
    async for pending_task in handler.drain_pending():
        print(f"Processing late result: {pending_task.result}")

asyncio.run(process_pending())
```

### Merge Callback Example

```python
async def on_bg_merge(task, mode):
    if mode == MergeMode.HOT:
        print(f"Hot-merged result from {task.task_id}")
    else:
        print(f"Wake-up merged result from {task.task_id}")

handler = BackgroundTaskHandler()
handler.on_merge(on_bg_merge)
```
