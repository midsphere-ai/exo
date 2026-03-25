# Task Controller Design — Hierarchical Task Management

**Status:** Proposed
**Epic:** 7 — Task Management Controller
**Date:** 2026-03-10

---

## 1. Motivation

Exo's execution model centers on the `Agent.run()` loop (LLM call →
tool execution → repeat) tracked by `RunState` / `RunNode`. While effective
for single-agent and multi-agent workflows, it lacks:

- **Explicit task decomposition** — there is no way for an agent to break
  a high-level goal into named, trackable sub-tasks with priorities.
- **Task lifecycle management** — `RunNodeStatus` tracks execution steps,
  not user-facing task states like PAUSED, INPUT_REQUIRED, or WAITING.
- **Concurrent scheduling** — `asyncio.TaskGroup` in `_execute_tools()`
  parallelizes tool calls, but there is no scheduler for concurrent
  task execution with configurable limits.
- **Event-driven reactions** — `HookManager` fires lifecycle hooks for
  LLM/tool events, but there is no pub/sub mechanism for task-level
  events (task created, completed, failed).
- **Intent-based routing** — no mechanism to classify user input as
  task actions (create, pause, resume, cancel).

Agent-core's controller modules (`TaskManager`, `TaskScheduler`,
`EventQueue`, `IntentRecognizer`) address all five gaps. This document
proposes porting these concepts into Exo as a new internal module.

---

## 2. Key Decision: Module in exo-core `_internal/task_controller/`

### Option A — New package `exo-task` (rejected)

A separate package with its own pyproject.toml. This adds dependency
management overhead and forces consumers to install an extra package for
basic task management.

### Option B — Top-level module in `exo/` (rejected)

Placing task controller files alongside `agent.py` and `swarm.py`
pollutes the public API surface when task management is an optional,
internal coordination mechanism.

### Option C — Internal module `_internal/task_controller/` (chosen)

The task controller lives in
`packages/exo-core/src/exo/_internal/task_controller/` as a
private implementation module, consistent with existing internal modules
(`_internal/state.py`, `_internal/background.py`, `_internal/loop_node.py`).

Public re-exports from `exo/task_controller.py` expose only the
user-facing API.

**Why Option C:**

- Follows the established `_internal/` pattern for implementation details.
- Task controller is optional — agents work fine without it.
- Keeps `exo/` public API clean; only re-exports are public.
- No new package, no new dependency graph changes.
- Easy to promote to a standalone package later if needed.

---

## 3. Relationship to Existing Architecture

### 3.1 Tasks vs. RunNodes

| Concept | RunNode | Task |
|---------|---------|------|
| **Scope** | Single LLM call or tool execution | User-facing work unit (may span multiple runs) |
| **Lifecycle** | INIT → RUNNING → SUCCESS/FAILED/TIMEOUT | SUBMITTED → WORKING → COMPLETED/FAILED/CANCELED (+ PAUSED, INPUT_REQUIRED, WAITING) |
| **Tracking** | Automatic via `RunState` | Explicit via `TaskManager` |
| **Hierarchy** | Flat list within a run | Parent-child tree (subtasks) |
| **Persistence** | In-memory per run | In-memory dict (extensible to persistent stores) |

Tasks are **higher-level abstractions** that may spawn one or more agent
runs. A single task like "Research competitor pricing" might involve
multiple `Agent.run()` calls, each producing its own `RunState` with
multiple `RunNode`s.

### 3.2 Tasks and Agent Runs

```
Task (WORKING)
  ├─ Agent.run() → RunState [SUCCESS]
  │    ├─ RunNode (LLM call) [SUCCESS]
  │    ├─ RunNode (tool: search) [SUCCESS]
  │    └─ RunNode (LLM call) [SUCCESS]
  │
  └─ Agent.run() → RunState [SUCCESS]  (follow-up)
       ├─ RunNode (LLM call) [SUCCESS]
       └─ RunNode (tool: write_report) [SUCCESS]
```

### 3.3 Tasks and Swarm Workflows

In a Swarm workflow, a parent task can decompose into child tasks, each
assigned to a different agent in the swarm:

```
Parent Task: "Analyze Q4 Results"
  ├─ Child Task: "Gather financial data"  → agent: data_collector
  ├─ Child Task: "Run statistical analysis" → agent: analyst
  └─ Child Task: "Write executive summary" → agent: writer
```

The `TaskScheduler` coordinates concurrent execution of child tasks,
respecting the configured concurrency limit (e.g., `max_concurrent=3`).

### 3.4 Tasks and Background Tasks

`BackgroundTaskHandler` manages async background operations within a
single run. The task controller operates at a higher level — managing
work items across runs. They complement each other:

- **BackgroundTaskHandler**: "This LLM call spawned a background fetch"
- **TaskManager**: "This workflow has 5 sub-tasks, 3 are complete"

---

## 4. Task Model

### 4.1 TaskStatus

```python
class TaskStatus(StrEnum):
    SUBMITTED = "submitted"        # Created, waiting to be picked up
    WORKING = "working"            # Actively being executed
    PAUSED = "paused"              # Execution suspended
    INPUT_REQUIRED = "input_required"  # Blocked on external input
    COMPLETED = "completed"        # Successfully finished
    CANCELED = "canceled"          # Explicitly canceled
    FAILED = "failed"              # Execution failed
    WAITING = "waiting"            # Waiting on dependencies
```

### 4.2 Status Transition Rules

```
SUBMITTED → WORKING, CANCELED
WORKING → PAUSED, INPUT_REQUIRED, COMPLETED, FAILED, CANCELED, WAITING
PAUSED → WORKING (resume), CANCELED
INPUT_REQUIRED → WORKING (input provided), CANCELED
WAITING → WORKING (dependencies met), CANCELED
COMPLETED → (terminal)
CANCELED → (terminal)
FAILED → SUBMITTED (retry)
```

Terminal states (`COMPLETED`, `CANCELED`) cannot transition to
non-terminal states. `FAILED` can transition back to `SUBMITTED` to
allow retry.

### 4.3 Task Model

```python
class Task(BaseModel):
    id: str                          # UUID
    name: str
    description: str = ""
    status: TaskStatus = TaskStatus.SUBMITTED
    priority: int = 0                # Higher = more important
    parent_id: str | None = None     # For hierarchical tasks
    metadata: dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime
```

---

## 5. Component Design

### 5.1 TaskManager

CRUD operations with status-transition enforcement and optional event
emission.

```python
class TaskManager:
    def __init__(self, *, event_bus: TaskEventBus | None = None) -> None:
        self._tasks: dict[str, Task] = {}
        self._event_bus = event_bus

    def create(self, name, *, description="", priority=0,
               parent_id=None, metadata=None) -> Task: ...
    def get(self, task_id: str) -> Task | None: ...
    def update(self, task_id: str, *, status=None, ...) -> Task: ...
    def delete(self, task_id: str) -> bool: ...
    def list(self, *, status: TaskStatus | None = None) -> list[Task]: ...
    def get_children(self, parent_id: str) -> list[Task]: ...
    def get_subtree(self, task_id: str) -> list[Task]: ...
```

`list()` returns tasks sorted by priority (descending) then `created_at`
(ascending).

### 5.2 TaskScheduler

Concurrent execution with `asyncio.Semaphore`:

```python
class TaskScheduler:
    def __init__(self, task_manager: TaskManager,
                 *, max_concurrent: int = 3) -> None:
        self._manager = task_manager
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def schedule(
        self, executor: Callable[[Task], Coroutine]
    ) -> list[Task]: ...

    def pause(self, task_id: str) -> Task: ...
    def resume(self, task_id: str) -> Task: ...
    def cancel(self, task_id: str) -> Task: ...
```

`schedule()` picks all SUBMITTED tasks (sorted by priority), runs them
through the executor up to `max_concurrent` at a time. Each task
transitions: SUBMITTED → WORKING → COMPLETED (or FAILED).

### 5.3 IntentRecognizer

LLM-powered classification of user input into task actions:

```python
@dataclass
class Intent:
    action: str        # create_task, pause_task, resume_task, cancel_task, etc.
    task_id: str | None
    confidence: float
    details: dict[str, Any]

class IntentRecognizer:
    def __init__(self, *, model: str = "openai:gpt-4o-mini") -> None: ...

    async def recognize(
        self, input: str,
        *, available_tasks: list[Task] | None = None,
    ) -> Intent: ...
```

Uses a structured prompt with available task names/IDs for context.

### 5.4 TaskEventBus

In-memory pub/sub for task lifecycle events:

```python
@dataclass
class TaskEvent:
    event_type: str    # task.created, task.started, task.completed, etc.
    task_id: str
    data: dict[str, Any]
    timestamp: float

class TaskEventBus:
    def subscribe(self, event_type: str,
                  handler: Callable[[TaskEvent], Coroutine]) -> None: ...
    def unsubscribe(self, event_type: str,
                    handler: Callable) -> None: ...
    async def emit(self, event: TaskEvent) -> None: ...
```

Event types:

| Event | Emitted when |
|-------|-------------|
| `task.created` | `TaskManager.create()` |
| `task.started` | Status → WORKING |
| `task.completed` | Status → COMPLETED |
| `task.failed` | Status → FAILED |
| `task.paused` | Status → PAUSED |
| `task.canceled` | Status → CANCELED |

---

## 6. File Layout

All new files in `packages/exo-core/src/exo/_internal/task_controller/`:

| File | Contents |
|------|----------|
| `__init__.py` | Package init, re-exports |
| `types.py` | `TaskStatus`, `Task`, `TaskEvent`, `Intent` |
| `manager.py` | `TaskManager` |
| `scheduler.py` | `TaskScheduler` |
| `intent.py` | `IntentRecognizer` |
| `event_bus.py` | `TaskEventBus` |

Public re-export in `packages/exo-core/src/exo/task_controller.py`:

```python
from exo._internal.task_controller import (
    Task,
    TaskStatus,
    TaskManager,
    TaskScheduler,
    TaskEvent,
    TaskEventBus,
    Intent,
    IntentRecognizer,
)
```

Tests in `packages/exo-core/tests/`:

| File | Contents |
|------|----------|
| `test_task_types.py` | Task model, status transitions |
| `test_task_manager.py` | CRUD, hierarchy, priority sorting |
| `test_task_scheduler.py` | Concurrent execution, pause/resume/cancel |
| `test_task_intent.py` | Intent recognition with MockProvider |
| `test_task_event_bus.py` | Subscribe/emit/unsubscribe |
| `test_task_integration.py` | End-to-end with Agent |

---

## 7. Integration Flow

```
User Input
  │
  ├─ IntentRecognizer.recognize(input)
  │    └─ Intent(action="create_task", details={...})
  │
  ├─ TaskManager.create("Research topic", priority=5)
  │    └─ TaskEventBus.emit(task.created)
  │
  ├─ TaskScheduler.schedule(executor=agent_executor)
  │    ├─ Semaphore acquire (max_concurrent=3)
  │    ├─ Task → WORKING
  │    │    └─ TaskEventBus.emit(task.started)
  │    ├─ executor(task) → Agent.run(task.description)
  │    │    └─ RunState with RunNodes (LLM calls, tool calls)
  │    └─ Task → COMPLETED
  │         └─ TaskEventBus.emit(task.completed)
  │
  └─ Parent task auto-completion (when all children complete)
```

---

## 8. Parent-Child Cascading Rules

1. **Cancel cascade**: When a parent is CANCELED, all descendants
   (children, grandchildren, etc.) are also CANCELED.
2. **Auto-completion** (optional, configurable): When all children of a
   parent reach COMPLETED, the parent auto-transitions to COMPLETED.
3. **Failure propagation**: When a child FAILS, the parent stays WORKING
   (other children may still succeed). The parent can be explicitly
   failed or the application can decide policy.

---

## 9. Open Questions

1. **Persistent storage.** The initial implementation uses an in-memory
   dict. Should we define a `TaskStore` ABC for future persistent
   backends (SQLite, Redis)? **Recommendation:** Defer; the in-memory
   implementation is sufficient for the initial port. The dict-based
   store can be swapped later behind the `TaskManager` interface.

2. **Task ↔ RunState linkage.** Should `Task` hold a reference to its
   `RunState`? **Recommendation:** Store `run_id` in `Task.metadata`
   rather than a direct reference, to avoid coupling and serialization
   issues.

3. **Swarm integration depth.** Should `Swarm.run()` auto-create tasks
   for each agent in a workflow? **Recommendation:** Defer; keep the
   task controller opt-in. Users can wrap `Swarm.run()` with task
   creation if needed.

4. **IntentRecognizer provider.** Should `IntentRecognizer` accept a
   `Provider` instance or a model string? **Recommendation:** Model
   string (consistent with `Agent`). The recognizer creates its own
   provider internally.

---

## 10. Summary

- Task controller lives in `_internal/task_controller/` as an optional
  internal module.
- `Task` is a Pydantic model with 8 lifecycle states and enforced
  transitions.
- `TaskManager` provides CRUD with parent-child hierarchy and priority
  indexing.
- `TaskScheduler` runs tasks concurrently via `asyncio.Semaphore`.
- `TaskEventBus` provides pub/sub for task lifecycle events.
- `IntentRecognizer` classifies user input into task actions via LLM.
- Zero changes to existing `Agent`, `Swarm`, `RunState`, or
  `BackgroundTaskHandler` APIs.
- Implementation spans ~6 new files (~800 lines total) + tests.
