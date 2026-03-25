# Task Management — agent-core to Exo Mapping

**Epic:** 7 — Task Management & Controller
**Date:** 2026-03-11

This document maps agent-core's (openJiuwen) controller/task system to
Exo's task management module, covering CRUD operations, scheduling,
intent routing, event queues, and task loop control.

---

## 1. Agent-Core Overview

Agent-core's task management system lives in `openjiuwen/core/controller/`
and provides a full event-driven task lifecycle for long-running agents.

### Key Components

**`TaskManager`** — CRUD operations with priority indexing and hierarchical
parent-child relationships. Uses async locking to coordinate concurrent
access. Supports creating task trees where child tasks roll up to parents.

**`TaskScheduler`** — Concurrent task execution with configurable limits and
pause/cancel support. Picks up submitted tasks in priority order and runs
them through a pluggable executor registry (`TaskExecutor` per task type).

**`TaskState` Enum (8 states)** — Full lifecycle tracking:

| State | Meaning |
|-------|---------|
| `SUBMITTED` | Created, waiting to be picked up |
| `WORKING` | Actively being executed |
| `PAUSED` | Execution suspended |
| `INPUT_REQUIRED` | Blocked on external input |
| `COMPLETED` | Successfully finished (terminal) |
| `CANCELED` | Explicitly canceled (terminal) |
| `FAILED` | Execution failed |
| `WAITING` | Waiting on dependencies |

**`IntentRecognizer`** — LLM-based intent detection that classifies user
input into task actions (`create_task`, `pause_task`, `resume_task`,
`cancel_task`, `list_tasks`, `get_task_status`, `update_task`). Routes
natural-language commands to the appropriate TaskManager operation.

**`EventQueue`** — Pub/sub messaging via `MessageQueueInMemory` with
per-session topic routing. Enables reactive event handling across the
controller system.

---

## 2. Exo Equivalent

Exo's task management lives in `exo._internal.task_controller`
(implementation) with public re-exports from `exo.task_controller`.

### Architecture Difference

Where agent-core uses async-locked managers with a pluggable executor
registry, Exo uses Pydantic models with explicit state-transition
validation and a semaphore-based scheduler. Exo also adds a
`TaskLoopQueue` for mid-execution agent steering — a concept not present
in agent-core's controller.

```python
# Agent-core: async-locked manager + executor registry
task_manager = TaskManager(config)
await task_manager.create_task("Research topic", priority=5)
scheduler = TaskScheduler(task_manager, executors={...})
await scheduler.run()

# Exo: Pydantic models + semaphore scheduler
from exo.task_controller import (
    TaskManager, TaskScheduler, TaskEventBus,
    IntentRecognizer, TaskLoopQueue,
)

bus = TaskEventBus()
manager = TaskManager(event_bus=bus, auto_complete_parent=True)
task = manager.create("Research topic", priority=5)
scheduler = TaskScheduler(manager, max_concurrent=3)
await scheduler.schedule(executor=my_agent_executor)
```

### Component Mapping

| Agent-Core Component | Exo Equivalent | Notes |
|---------------------|-------------------|-------|
| `TaskManager` (async-locked) | `TaskManager` | Dict-based CRUD; sync methods; event emission via optional `TaskEventBus` |
| `TaskScheduler` + executor registry | `TaskScheduler` | `asyncio.Semaphore` throttling; single `executor` callable (not registry) |
| `TaskState` enum | `TaskStatus` enum | Same 8 states; Pydantic-validated transitions via `Task.transition()` |
| `IntentRecognizer` | `IntentRecognizer` | Same LLM-based approach; returns `Intent` dataclass with confidence score |
| `EventQueue` (MessageQueueInMemory) | `TaskEventBus` | Typed `TaskEventType` enum; async handlers called sequentially |
| Task model (dict-based) | `Task` (Pydantic `BaseModel`) | UUID id, priority, parent_id, metadata, timestamps |
| *(no equivalent)* | `TaskLoopQueue` | Thread-safe priority queue for ABORT/STEER/FOLLOWUP events |
| *(no equivalent)* | `TaskLoopEvent` / `TaskLoopEventType` | Prioritized events: ABORT (0) > STEER (1) > FOLLOWUP (2) |
| *(no equivalent)* | `steer_agent_tool` / `abort_agent_tool` | Queue-based tools for external agent loop control |
| *(no equivalent)* | `InvalidTransitionError` | Explicit error for invalid state transitions |
| `TaskExecutor` registry | *(simplified)* | Exo uses a single `executor` callable instead of per-type registry |

### Key Exo Additions Beyond Agent-Core

**Task Loop Queue** — A thread-safe priority queue (`TaskLoopQueue`) that
bridges external threads with the async agent loop. Three event priorities:

| Priority | Type | Purpose |
|----------|------|---------|
| 0 (highest) | `ABORT` | Stop the agent immediately |
| 1 | `STEER` | Redirect the agent mid-task |
| 2 (lowest) | `FOLLOWUP` | Queue additional work |

**Queue-Based Tools** — `steer_agent_tool` and `abort_agent_tool` let an
LLM push events into the `TaskLoopQueue`. The `_QueueTool` wrapper binds
the queue at runtime so the `queue` parameter is hidden from the tool schema.

**Cascading Behaviors** — TaskManager supports:
- **Cancel cascade:** Canceling a parent automatically cancels all non-terminal
  descendants
- **Auto-complete parent:** When all children reach `COMPLETED`, the parent
  auto-transitions (opt-in via `auto_complete_parent=True`)

**Explicit State Transitions** — The `Task` model validates every status
change against a transition table. Invalid transitions raise
`InvalidTransitionError` rather than silently succeeding.

---

## 3. Side-by-Side Examples

### Creating a Task Hierarchy and Scheduling

```python
# Agent-core
task_manager = TaskManager(config)
parent = await task_manager.create_task("Analyze Q4 Results", priority=5)
child1 = await task_manager.create_task(
    "Gather data", priority=3, parent_id=parent.id,
)
child2 = await task_manager.create_task(
    "Run analysis", priority=4, parent_id=parent.id,
)
child3 = await task_manager.create_task(
    "Write summary", priority=2, parent_id=parent.id,
)

scheduler = TaskScheduler(task_manager, max_concurrent=2)
await scheduler.run()  # Picks up SUBMITTED tasks by priority

# Exo
from exo.task_controller import (
    TaskManager, TaskScheduler, TaskEventBus,
)

bus = TaskEventBus()
manager = TaskManager(event_bus=bus, auto_complete_parent=True)

parent = manager.create("Analyze Q4 Results", priority=5)
child1 = manager.create("Gather data", priority=3, parent_id=parent.id)
child2 = manager.create("Run analysis", priority=4, parent_id=parent.id)
child3 = manager.create("Write summary", priority=2, parent_id=parent.id)

scheduler = TaskScheduler(manager, max_concurrent=2)

async def my_executor(task):
    # Your agent logic here
    print(f"Executing: {task.name}")

await scheduler.schedule(executor=my_executor)
# Runs child2 (pri 4) and child1 (pri 3) first, then child3 (pri 2)
# When all children complete, parent auto-transitions to COMPLETED
```

### Intent Recognition

```python
# Agent-core
recognizer = IntentRecognizer(llm=my_llm)
intent = await recognizer.recognize("pause the data gathering task")
# intent.action == "pause_task", intent.task_id == child1.id

# Exo
from exo.task_controller import IntentRecognizer

recognizer = IntentRecognizer(model="openai:gpt-4o")
intent = await recognizer.recognize(
    "pause the data gathering task",
    available_tasks=[
        {"id": child1.id, "name": "Gather data"},
        {"id": child2.id, "name": "Run analysis"},
    ],
)
# intent.action == "pause_task"
# intent.task_id == child1.id
# intent.confidence == 0.95
```

### Event-Driven Reactions

```python
# Agent-core
event_queue = EventQueue(session_id="session-1")
event_queue.subscribe("task.completed", on_task_done)
# Events emitted implicitly by task lifecycle

# Exo
from exo.task_controller import (
    TaskEventBus, TaskEventType, TaskEvent,
)

bus = TaskEventBus()

async def on_task_done(event: TaskEvent) -> None:
    print(f"Task {event.task_id} completed: {event.data}")

bus.subscribe(TaskEventType.COMPLETED, on_task_done)

# Events emitted automatically by TaskManager when event_bus is set
manager = TaskManager(event_bus=bus)
task = manager.create("My task")
manager.update(task.id, status=TaskStatus.WORKING)
manager.update(task.id, status=TaskStatus.COMPLETED)
# on_task_done is called with TaskEvent(event_type=COMPLETED, task_id=task.id)
```

### Task Loop Steering (Exo-only)

```python
from exo.task_controller import (
    TaskLoopQueue, TaskLoopEvent, TaskLoopEventType,
    get_task_loop_tools,
)

queue = TaskLoopQueue()

# Push a steering event from an external thread
queue.push(TaskLoopEvent(
    type=TaskLoopEventType.STEER,
    content="Focus on cost analysis instead",
))

# Agent loop checks queue each iteration
event = queue.pop()
if event and event.type == TaskLoopEventType.ABORT:
    raise SystemExit("Agent aborted")
elif event and event.type == TaskLoopEventType.STEER:
    # Redirect agent behavior
    pass

# Or use the built-in tools (bind queue at runtime)
tools = get_task_loop_tools()
for tool in tools:
    tool.bind(queue)
```

---

## 4. Migration Table

| Agent-Core Path | Exo Import | Symbol |
|----------------|----------------|--------|
| `openjiuwen.core.controller.TaskManager` | `exo.task_controller.TaskManager` | CRUD with priority sort, parent-child hierarchy, cancel cascade, auto-complete parent |
| `openjiuwen.core.controller.TaskScheduler` | `exo.task_controller.TaskScheduler` | Semaphore-based concurrent execution with pause/resume/cancel |
| `openjiuwen.core.controller.TaskState` | `exo.task_controller.TaskStatus` | 8-state enum: SUBMITTED, WORKING, PAUSED, INPUT_REQUIRED, COMPLETED, CANCELED, FAILED, WAITING |
| `openjiuwen.core.controller.IntentRecognizer` | `exo.task_controller.IntentRecognizer` | LLM-powered intent classification returning `Intent` dataclass |
| `openjiuwen.core.controller.EventQueue` | `exo.task_controller.TaskEventBus` | Typed pub/sub with `TaskEventType` enum and async handlers |
| *(task dict)* | `exo.task_controller.Task` | Pydantic `BaseModel` with `transition()` validation and `is_terminal` property |
| *(task events)* | `exo.task_controller.TaskEvent` | Dataclass with `event_type`, `task_id`, `data`, `timestamp` |
| *(event types)* | `exo.task_controller.TaskEventType` | `StrEnum`: `task.created`, `task.started`, `task.completed`, `task.failed`, `task.paused`, `task.canceled` |
| *(intent actions)* | `exo.task_controller.TASK_ACTIONS` | `frozenset` of 8 action strings (create, pause, resume, cancel, list, get_status, update, unknown) |
| *(intent result)* | `exo.task_controller.Intent` | Dataclass with `action`, `task_id`, `confidence`, `details` |
| *(no equivalent)* | `exo.task_controller.TaskLoopQueue` | Thread-safe priority queue for ABORT/STEER/FOLLOWUP events |
| *(no equivalent)* | `exo.task_controller.TaskLoopEvent` | Priority-ordered event with `type`, `content`, `metadata` |
| *(no equivalent)* | `exo.task_controller.TaskLoopEventType` | `IntEnum`: ABORT (0), STEER (1), FOLLOWUP (2) |
| *(no equivalent)* | `exo.task_controller.steer_agent_tool` | Queue-bound tool to redirect agent mid-task |
| *(no equivalent)* | `exo.task_controller.abort_agent_tool` | Queue-bound tool to stop agent execution |
| *(no equivalent)* | `exo.task_controller.get_task_loop_tools` | Returns `[steer_agent_tool, abort_agent_tool]` |
| *(no equivalent)* | `exo.task_controller.TaskError` | Base exception for task controller errors |
| *(no equivalent)* | `exo.task_controller.TaskNotFoundError` | Raised when task ID does not exist |
| *(no equivalent)* | `exo.task_controller.InvalidTransitionError` | Raised on invalid status transitions |
| *(no equivalent)* | `exo.task_controller.TaskEventHandler` | Type alias: `Callable[[TaskEvent], Coroutine[Any, Any, None]]` |

All public symbols are re-exported from `exo.task_controller` (via
`packages/exo-core/src/exo/task_controller.py`), so
`from exo.task_controller import TaskManager` works as a convenience
import.
