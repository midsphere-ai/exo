# exo.hooks

Async hook system for intercepting agent lifecycle events.

**Module:** `exo.hooks`

```python
from exo.hooks import HookPoint, Hook, HookManager, run_hooks
```

---

## Hook

```python
Hook = Callable[..., Coroutine[Any, Any, None]]
```

Type alias for async hook functions. Hooks must be async callables that accept keyword arguments and return `None`.

---

## HookPoint

```python
class HookPoint(enum.Enum)
```

Points in the agent lifecycle where hooks can be attached. Each value is a descriptive string for readable debug output.

### Values

| Member | Value | Description |
|--------|-------|-------------|
| `START` | `"start"` | Fired when an agent run begins. |
| `FINISHED` | `"finished"` | Fired when an agent run completes successfully. |
| `ERROR` | `"error"` | Fired when an agent run encounters an error. |
| `PRE_LLM_CALL` | `"pre_llm_call"` | Fired before each LLM API call. |
| `POST_LLM_CALL` | `"post_llm_call"` | Fired after each LLM API call. |
| `PRE_TOOL_CALL` | `"pre_tool_call"` | Fired before each tool execution. |
| `POST_TOOL_CALL` | `"post_tool_call"` | Fired after each tool execution. |

### Example

```python
from exo.hooks import HookPoint

# Iterate over all hook points
for point in HookPoint:
    print(f"{point.name} = {point.value}")
```

---

## HookManager

```python
class HookManager
```

Manages async hooks attached to lifecycle points. Hooks are called sequentially in registration order. Unlike `EventBus`, exceptions from hooks are **not** suppressed -- a failing hook aborts the run.

### Constructor

```python
def __init__(self) -> None
```

Creates a new hook manager with no registered hooks.

### Methods

#### add()

```python
def add(self, point: HookPoint, hook: Hook) -> None
```

Register a hook at a lifecycle point.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `point` | `HookPoint` | *(required)* | The lifecycle point to attach to. |
| `hook` | `Hook` | *(required)* | Async callable to invoke at that point. |

#### remove()

```python
def remove(self, point: HookPoint, hook: Hook) -> None
```

Remove a hook from a lifecycle point. Silently does nothing if the hook is not registered. Removes the first occurrence only.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `point` | `HookPoint` | *(required)* | The lifecycle point. |
| `hook` | `Hook` | *(required)* | The hook to remove. |

#### run()

```python
async def run(self, point: HookPoint, **data: Any) -> None
```

Run all hooks for a lifecycle point sequentially. Exceptions from hooks propagate immediately -- they are never silently swallowed.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `point` | `HookPoint` | *(required)* | The lifecycle point to fire. |
| `**data` | `Any` | | Keyword arguments passed to each hook. |

#### has_hooks()

```python
def has_hooks(self, point: HookPoint) -> bool
```

Check whether any hooks are registered for a point.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `point` | `HookPoint` | *(required)* | The lifecycle point to check. |

**Returns:** `True` if at least one hook is registered.

#### clear()

```python
def clear(self) -> None
```

Remove all hooks for all lifecycle points.

---

## run_hooks()

```python
async def run_hooks(manager: HookManager, point: HookPoint, **data: Any) -> None
```

Convenience function to run hooks on a manager. Equivalent to `await manager.run(point, **data)`.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `manager` | `HookManager` | *(required)* | The hook manager to use. |
| `point` | `HookPoint` | *(required)* | The lifecycle point to fire. |
| `**data` | `Any` | | Keyword arguments passed to each hook. |

---

## Example

```python
import asyncio
from exo.hooks import HookPoint, HookManager, run_hooks

manager = HookManager()

async def log_llm_call(**data):
    agent = data.get("agent")
    print(f"LLM call for agent: {agent.name if agent else 'unknown'}")

async def log_tool_call(**data):
    print(f"Tool called: {data.get('tool_name')}")

# Register hooks
manager.add(HookPoint.PRE_LLM_CALL, log_llm_call)
manager.add(HookPoint.PRE_TOOL_CALL, log_tool_call)

# Run hooks directly
asyncio.run(manager.run(HookPoint.PRE_TOOL_CALL, tool_name="get_weather"))

# Or use the convenience function
asyncio.run(run_hooks(manager, HookPoint.PRE_TOOL_CALL, tool_name="get_weather"))

# Pass hooks when creating an Agent
from exo.agent import Agent
agent = Agent(
    name="my_agent",
    hooks=[
        (HookPoint.PRE_LLM_CALL, log_llm_call),
        (HookPoint.PRE_TOOL_CALL, log_tool_call),
    ],
)
```
