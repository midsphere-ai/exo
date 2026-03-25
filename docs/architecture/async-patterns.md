# Async Patterns

Exo is async-first. This document describes the async design patterns used throughout the framework, how sync functions are bridged, and how parallel execution works.

## Async-First Design

All internal functions in Exo are `async def`. This is a deliberate choice:

```python
# runner.py -- the primary API
async def run(agent, input, ...) -> RunResult:
    ...

# agent.py -- the tool loop
async def run(self, input, ...) -> AgentOutput:
    ...

# tool.py -- tool execution
async def execute(self, **kwargs) -> str | dict:
    ...

# hooks.py -- lifecycle hooks
async def run(self, point, **data) -> None:
    ...
```

There is **no** parallel sync implementation. Every component has exactly one implementation, and it is async.

## The Sync Bridge

The only sync entry point is `run.sync()`, which bridges to async via `asyncio.run()`:

```python
# In exo/runner.py

async def run(agent, input, **kwargs) -> RunResult:
    """Primary async API."""
    ...

def _sync(agent, input, **kwargs) -> RunResult:
    """Blocking wrapper -- the ONLY sync bridge."""
    return asyncio.run(run(agent, input, **kwargs))

# Attach as attribute for clean API
run.sync = _sync
```

Usage:

```python
# Async context (recommended)
result = await run(agent, "Hello!")

# Sync context (scripts, notebooks)
result = run.sync(agent, "Hello!")
```

### Rules

- **Never use `loop.run_until_complete()`** -- always use `asyncio.run()`. The former requires managing the event loop manually and can cause issues with nested loops.
- **Only ONE sync entry point exists** -- `run.sync()`. Everything else is async.
- **Do not create new sync wrappers** for internal functions. If you need sync access, call `run.sync()` at the boundary.

## Sync Function Wrapping

When a sync function is registered as a tool via `@tool`, it is automatically wrapped with `asyncio.to_thread()`:

```python
from exo import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # This sync function is auto-wrapped
    return requests.get(f"https://api.weather.com/{city}").text

@tool
async def search_web(query: str) -> str:
    """Search the web."""
    # This async function runs natively
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://api.search.com?q={query}")
        return resp.text
```

The wrapping happens in `FunctionTool.__init__()`:

```python
class FunctionTool(Tool):
    def __init__(self, fn, *, name=None, description=None):
        self._fn = fn
        self._is_async = asyncio.iscoroutinefunction(fn)
        ...

    async def execute(self, **kwargs):
        if self._is_async:
            return await self._fn(**kwargs)
        return await asyncio.to_thread(self._fn, **kwargs)
```

`asyncio.to_thread()` runs the sync function in a thread pool, so it does not block the event loop. This means:

- Sync tools work transparently alongside async tools
- Multiple sync tools can run in parallel (each gets its own thread)
- Users do not need to worry about whether their tool function is sync or async

## Parallel Tool Execution

When an LLM requests multiple tool calls in a single response, Exo executes them in parallel using `asyncio.TaskGroup` (Python 3.11+):

```python
# In agent.py

async def _execute_tools(self, actions):
    results = [ToolResult(...)] * len(actions)

    async def _run_one(idx):
        action = actions[idx]
        tool = self.tools.get(action.tool_name)
        # ... hooks, execution, error handling ...
        results[idx] = result

    async with asyncio.TaskGroup() as tg:
        for i in range(len(actions)):
            tg.create_task(_run_one(i))

    return results
```

Benefits of `asyncio.TaskGroup`:

- **Structured concurrency** -- If any task raises an exception, all other tasks are cancelled and the group raises an `ExceptionGroup`. No orphaned tasks.
- **Automatic cleanup** -- The `async with` block ensures all tasks complete or are cancelled before proceeding.
- **Zero thread overhead** for async tools -- Only sync tools (wrapped via `to_thread`) use threads.

### Example: 3 Tool Calls in Parallel

```python
@tool
async def search(query: str) -> str: ...

@tool
async def fetch_price(ticker: str) -> str: ...

@tool
def calculate_tax(amount: float) -> str: ...  # sync, auto-wrapped

# If the LLM requests all 3 in one response:
# search("AI stocks"), fetch_price("NVDA"), calculate_tax(1000.0)
#
# They execute concurrently:
# - search and fetch_price run as native async coroutines
# - calculate_tax runs in a thread via asyncio.to_thread()
# - All 3 complete before the results are sent back to the LLM
```

## Streaming

`run.stream()` is an async generator that yields `StreamEvent` objects:

```python
async def _stream(agent, input, ...) -> AsyncIterator[StreamEvent]:
    ...
    for _step in range(steps):
        async for chunk in provider.stream(msg_list, tools=...):
            if chunk.delta:
                yield TextEvent(text=chunk.delta, agent_name=agent.name)
            # accumulate tool call deltas...

        if tool_calls:
            # execute tools, feed results back, loop
            ...
        else:
            return  # done
```

Key patterns:
- The outer loop handles tool iterations (LLM -> tools -> LLM -> ...)
- The inner loop streams individual LLM responses
- `TextEvent` is yielded immediately for each text delta
- `ToolCallEvent` is yielded after the full tool call is assembled from deltas
- Tool results are appended to the message list and the LLM is re-streamed

## Event Bus and Hooks

Both `EventBus` and `HookManager` are async:

```python
# EventBus -- handlers are called sequentially, exceptions suppressed
async def emit(self, event: str, **data) -> None:
    for handler in self._handlers[event]:
        await handler(**data)

# HookManager -- hooks are called sequentially, exceptions propagate
async def run(self, point: HookPoint, **data) -> None:
    for hook in self._hooks[point]:
        await hook(**data)
```

The distinction:
- **EventBus** is for decoupled communication (observers). Handler failures should not abort the run.
- **HookManager** is for lifecycle interception. A failing hook (e.g., a validation hook on `PRE_TOOL_CALL`) should abort execution.

## Common Pitfalls

### Do not nest asyncio.run()

```python
# BAD -- will raise "asyncio.run() cannot be called from a running event loop"
async def my_handler():
    result = run.sync(agent, "Hello!")  # WRONG

# GOOD
async def my_handler():
    result = await run(agent, "Hello!")
```

### Do not block the event loop

```python
# BAD -- blocks the event loop, starving other coroutines
@tool
async def slow_tool(query: str) -> str:
    time.sleep(10)  # WRONG -- use asyncio.sleep or to_thread

# GOOD -- non-blocking
@tool
async def slow_tool(query: str) -> str:
    await asyncio.sleep(10)

# ALSO GOOD -- sync function is auto-wrapped in to_thread
@tool
def slow_tool(query: str) -> str:
    time.sleep(10)  # OK -- runs in thread pool
```
