# exo.events

Async event bus for decoupled communication between components.

**Module:** `exo.events`

```python
from exo.events import EventBus, EventHandler
```

---

## EventHandler

```python
EventHandler = Callable[..., Coroutine[Any, Any, None]]
```

Type alias for async event handler functions. Handlers must be async callables that accept keyword arguments and return `None`.

---

## EventBus

```python
class EventBus
```

A simple async event bus. Handlers are called sequentially in registration order when an event is emitted. Event names are plain strings.

### Constructor

```python
def __init__(self) -> None
```

Creates a new event bus with no registered handlers.

### Methods

#### on()

```python
def on(self, event: str, handler: EventHandler) -> None
```

Subscribe a handler to an event.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `event` | `str` | *(required)* | The event name to listen for. |
| `handler` | `EventHandler` | *(required)* | Async callable to invoke when the event fires. |

#### off()

```python
def off(self, event: str, handler: EventHandler) -> None
```

Unsubscribe a handler from an event. Silently does nothing if the handler is not registered. Removes the first occurrence only.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `event` | `str` | *(required)* | The event name. |
| `handler` | `EventHandler` | *(required)* | The handler to remove. |

#### emit()

```python
async def emit(self, event: str, **data: Any) -> None
```

Emit an event, calling all registered handlers sequentially.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `event` | `str` | *(required)* | The event name to emit. |
| `**data` | `Any` | | Keyword arguments passed to each handler. |

#### has_handlers()

```python
def has_handlers(self, event: str) -> bool
```

Check whether any handlers are registered for an event.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `event` | `str` | *(required)* | The event name to check. |

**Returns:** `True` if at least one handler is registered.

#### clear()

```python
def clear(self) -> None
```

Remove all handlers for all events.

### Example

```python
import asyncio
from exo.events import EventBus

bus = EventBus()

async def on_step_complete(**data):
    print(f"Step completed: {data}")

# Subscribe
bus.on("step_complete", on_step_complete)

# Emit
asyncio.run(bus.emit("step_complete", step=1, output="done"))

# Check
assert bus.has_handlers("step_complete")

# Unsubscribe
bus.off("step_complete", on_step_complete)
assert not bus.has_handlers("step_complete")

# Clear all
bus.clear()
```
