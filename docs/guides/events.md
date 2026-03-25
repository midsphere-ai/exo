# Events

The `EventBus` provides decoupled async communication between components. Unlike [hooks](hooks.md), events are designed for cross-cutting concerns like logging and metrics -- handler exceptions are silently absorbed and do not disrupt agent execution.

## Basic Usage

```python
from exo.events import EventBus

bus = EventBus()

async def on_task_complete(**data):
    task = data.get("task")
    print(f"Task completed: {task}")

bus.on("task.complete", on_task_complete)
await bus.emit("task.complete", task="Write report")
# Output: "Task completed: Write report"
```

## EventBus API

### Constructor

```python
bus = EventBus()
```

No parameters. Creates an empty event bus with no handlers.

### Subscribing to Events

```python
bus.on(event: str, handler: EventHandler) -> None
```

Register an async handler for a named event. Multiple handlers can be registered for the same event and are called in registration order.

```python
async def handler_a(**data):
    print("Handler A")

async def handler_b(**data):
    print("Handler B")

bus.on("user.login", handler_a)
bus.on("user.login", handler_b)

await bus.emit("user.login", user="alice")
# Output:
# Handler A
# Handler B
```

### Unsubscribing from Events

```python
bus.off(event: str, handler: EventHandler) -> None
```

Remove a handler from an event. Silently does nothing if the handler is not registered. Removes only the first occurrence.

```python
bus.off("user.login", handler_a)
```

### Emitting Events

```python
await bus.emit(event: str, **data: Any) -> None
```

Emit a named event, calling all registered handlers sequentially. Data is passed as keyword arguments to each handler.

```python
await bus.emit("order.placed", order_id="123", amount=99.99)
```

### Checking for Handlers

```python
bus.has_handlers(event: str) -> bool
```

Returns `True` if at least one handler is registered for the event.

```python
if bus.has_handlers("user.login"):
    print("Login events are being monitored")
```

### Clearing All Handlers

```python
bus.clear() -> None
```

Remove all handlers for all events.

## EventHandler Type

An event handler is any async callable that accepts keyword arguments:

```python
from exo.events import EventHandler
# EventHandler = Callable[..., Coroutine[Any, Any, None]]

async def my_handler(**data):
    # Process event data
    pass
```

## Event Naming Conventions

Event names are plain strings with no enforced convention. A recommended pattern is dot-separated namespaces:

```python
bus.on("agent.started", on_agent_start)
bus.on("agent.finished", on_agent_finish)
bus.on("tool.called", on_tool_call)
bus.on("llm.response", on_llm_response)
bus.on("pipeline.step.complete", on_step)
```

## Practical Examples

### Metrics Collection

```python
class MetricsCollector:
    def __init__(self):
        self.call_count = 0
        self.total_tokens = 0

    async def on_llm_call(self, **data):
        self.call_count += 1

    async def on_usage(self, **data):
        tokens = data.get("tokens", 0)
        self.total_tokens += tokens

metrics = MetricsCollector()
bus.on("llm.call", metrics.on_llm_call)
bus.on("llm.usage", metrics.on_usage)
```

### Audit Logging

```python
import json

async def audit_log(**data):
    event_type = data.pop("_event", "unknown")
    log_entry = {"event": event_type, **data}
    with open("audit.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

bus.on("tool.executed", audit_log)
bus.on("agent.handoff", audit_log)
```

### Multi-System Notification

```python
async def notify_slack(**data):
    message = data.get("message", "")
    # await slack_client.post(channel, message)
    print(f"[Slack] {message}")

async def notify_email(**data):
    message = data.get("message", "")
    # await email_client.send(to, message)
    print(f"[Email] {message}")

bus.on("alert.critical", notify_slack)
bus.on("alert.critical", notify_email)

await bus.emit("alert.critical", message="Agent budget exceeded")
```

## Hooks vs Events

The key difference: **hooks abort on failure**, events do not.

| Feature | Hooks (`HookManager`) | Events (`EventBus`) |
|---------|----------------------|---------------------|
| **Error behavior** | Exceptions propagate and abort | Exceptions are swallowed |
| **Keys** | `HookPoint` enum values | Free-form strings |
| **Scope** | Per-agent lifecycle | Application-wide |
| **Primary use** | Control flow, validation | Observation, logging |
| **Registration** | `manager.add(point, hook)` | `bus.on(event, handler)` |
| **Execution** | `await manager.run(point, ...)` | `await bus.emit(event, ...)` |

**When to use hooks:** When the handler's success is critical to correctness (e.g., budget enforcement, tool approval, input validation).

**When to use events:** When the handler is purely observational and should never disrupt the main flow (e.g., logging, metrics, notifications).

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `EventBus` | `exo.events` | Async event bus for decoupled communication |
| `EventHandler` | `exo.events` | Type alias for async event handler functions |
