---
name: exo:harness
description: "Use when building orchestration harnesses in Exo — Harness ABC, execute() async generator, HarnessContext (run_agent, stream_agent, emit, checkpoint, check_cancelled), HarnessEvent, SessionState, Middleware (TimeoutMiddleware, CostTrackingMiddleware), CheckpointAdapter, HarnessNode for Swarm composition, event interception, agent routing, cross-cutting middleware, session state persistence. Triggers on: harness, Harness, orchestration, HarnessContext, run_agent, stream_agent, HarnessEvent, SessionState, middleware, TimeoutMiddleware, CostTrackingMiddleware, CheckpointAdapter, HarnessNode, event interception, agent routing, harness checkpoint, harness cancel, harness middleware, execute method, harness stream."
---

> **Branch:** These skills are written for the `rename/orbiter-to-exo` branch. The Exo APIs referenced here may differ on other branches.

# Exo Harness — Composable Agent Orchestration

## When To Use This Skill

Use this skill when the developer needs to:
- Build custom orchestration logic that decides which agent runs, when, and with what input
- Intercept agent events in real-time to make routing decisions
- Control what message history each agent sees (fresh, shared, or forked)
- Maintain session state across multiple agent runs
- Apply cross-cutting middleware (timeout, cost tracking, custom)
- Checkpoint and resume harness execution after process restart
- Compose harnesses with other harnesses or Swarms
- Emit custom events alongside standard agent StreamEvent instances
- Cancel a running harness from outside

## Decision Guide

1. **Need to run agents in a custom order with logic between them?** → Subclass `Harness`, implement `execute()`
2. **Need to watch what an agent does and react?** → Iterate `ctx.stream_agent()` inside `execute()`, inspect events, yield them through
3. **Need to run an agent and just get the result?** → Use `ctx.run_agent()` (non-streaming, returns `RunResult`)
4. **Need to control what history an agent sees?** → Pass `messages=None` (fresh), `messages=ctx.messages` (shared), or `messages=list(ctx.messages)` (forked) to `ctx.run_agent()` / `ctx.stream_agent()`
5. **Need to emit a custom event from the harness?** → `yield ctx.emit("my_kind", key="value")`
6. **Need a timeout?** → Add `TimeoutMiddleware(seconds)` to the middleware list
7. **Need to track total token cost?** → Add `CostTrackingMiddleware()` — writes to `ctx.state["_cost"]`
8. **Need to write custom middleware?** → Subclass `Middleware`, implement `async def wrap(stream, ctx) -> AsyncIterator[StreamEvent]`
9. **Need to persist state between runs?** → Use `ctx.state["key"] = value` (it's a `SessionState` dict wrapper with dirty tracking)
10. **Need to save/restore checkpoints?** → Pass a `MemoryStore` as `checkpoint_store`, call `ctx.checkpoint()` inside `execute()`
11. **Need to cancel a running harness?** → Call `harness.cancel()` from outside; inside `execute()`, call `ctx.check_cancelled()` to raise `HarnessError`
12. **Need to use a harness as a node in a Swarm?** → Wrap it with `HarnessNode(harness=my_harness)`
13. **Need to nest harnesses?** → Pass an inner harness to `ctx.run_agent()` or `ctx.stream_agent()` — it works automatically because the runner detects `is_harness`
14. **Want the harness to work with `run()` and `run.stream()`?** → It already does. `run(harness, input)` and `run.stream(harness, input)` dispatch automatically

## Reference

### Package Location

```
packages/exo-harness/src/exo/harness/
    __init__.py      # Public API exports
    base.py          # Harness ABC, HarnessContext, HarnessNode, HarnessError
    types.py         # HarnessEvent, SessionState, HarnessCheckpoint
    middleware.py    # Middleware ABC, TimeoutMiddleware, CostTrackingMiddleware
    checkpoint.py    # CheckpointAdapter — serializes to MemoryStore
```

### Imports

```python
from exo.harness import (
    Harness,            # ABC — subclass and implement execute()
    HarnessContext,     # Runtime context passed to execute()
    HarnessError,       # Exception for harness-level errors
    HarnessEvent,       # Custom event type (type="harness")
    HarnessNode,        # Wrapper for using a Harness in a Swarm
    SessionState,       # Mutable dict wrapper with dirty tracking
    HarnessCheckpoint,  # Immutable snapshot for persistence
    Middleware,          # ABC for stream middleware
    TimeoutMiddleware,   # Built-in: wall-clock timeout
    CostTrackingMiddleware,  # Built-in: token usage accumulation
    CheckpointAdapter,  # Serializes checkpoints to MemoryStore
)
```

### Harness ABC

The core abstraction. One abstract method: `execute()`.

```python
from exo.harness import Harness, HarnessContext

class Harness(ABC):
    is_harness: bool = True  # Marker for runner.py detection

    def __init__(
        self,
        *,
        name: str,                                      # Required: unique name
        agents: dict[str, Any] | list[Any] | None = None,  # Agents indexed by name
        state: SessionState | dict[str, Any] | None = None, # Initial session state
        checkpoint_store: Any = None,                    # MemoryStore for checkpoints
        middleware: list[Middleware] | None = None,       # Stream middleware chain
    ) -> None: ...

    # The ONE abstract method
    @abstractmethod
    async def execute(self, ctx: HarnessContext) -> AsyncIterator[StreamEvent]: ...

    # Public API — same shape as Agent and Swarm
    async def run(self, input, *, messages=None, provider=None, max_retries=3) -> RunResult
    async def stream(self, input, *, messages=None, provider=None, detailed=False,
                     max_steps=None, event_types=None) -> AsyncIterator[StreamEvent]

    # Cancellation
    def cancel(self) -> None        # Signal cancellation
    @property
    def cancelled(self) -> bool     # Check if cancelled
    def reset(self) -> None         # Clear cancellation signal

    # Checkpointing
    async def save_checkpoint(self, *, pending_agent=None) -> None
    async def restore_checkpoint(self) -> HarnessCheckpoint | None
```

**Key attributes:**
- `self.name: str` — harness name
- `self.agents: dict[str, Any]` — agents indexed by name
- `self.session: SessionState` — mutable session state

**How `run()` works:** Drains the `execute()` generator. Accumulates text from `TextEvent` instances and usage from `UsageEvent` instances. Returns a `RunResult` with the aggregated output.

**How `stream()` works:** Iterates the `execute()` generator, applies `event_types` filter, yields events to the consumer. Same interface as `Agent.stream()` and `Swarm.stream()`.

**Middleware chain:** Both `run()` and `stream()` wrap `execute()` through the middleware chain via `_execute_with_middleware()`. Middleware is applied in reverse order (outermost first). Exceptions from `execute()` are caught, an `ErrorEvent` is yielded, then the exception re-raises.

### HarnessContext

The runtime handle passed to `execute()`. Provides utilities for running agents, managing state, and emitting events.

```python
class HarnessContext:
    # Read-only attributes
    input: MessageContent           # The user's input
    messages: list[Message]         # The message history (mutable list)
    state: SessionState             # The mutable session state
    provider: Any                   # LLM provider
    detailed: bool                  # Whether to emit rich event types
    max_steps: int | None           # Max LLM-tool round-trips per agent

    # Run an agent (non-streaming) — returns RunResult
    async def run_agent(
        self,
        agent: Any,                     # Agent, Swarm, or Harness
        input: MessageContent,
        *,
        messages: Sequence[Message] | None = None,  # History visibility control
        provider: Any = None,           # Override harness provider
    ) -> RunResult

    # Stream an agent — yields StreamEvent
    async def stream_agent(
        self,
        agent: Any,
        input: MessageContent,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        detailed: bool | None = None,
        max_steps: int | None = None,
        event_types: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]

    # Cancellation
    @property
    def cancelled(self) -> bool
    def check_cancelled(self) -> None   # Raises HarnessError if cancelled

    # Custom event emission
    def emit(self, kind: str, **data: Any) -> HarnessEvent

    # Checkpointing
    async def checkpoint(self, *, pending_agent: str | None = None) -> None
```

**History visibility control** — the `messages` parameter on `run_agent()` and `stream_agent()`:
- `messages=None` — agent starts with a fresh conversation (no history)
- `messages=ctx.messages` — agent shares the harness's message list (shared reference, mutations visible to both)
- `messages=list(ctx.messages)` — agent gets a forked copy (isolated from harness)

### execute() — The Abstract Method

`execute()` is an async generator. It yields `StreamEvent` instances. The generator IS the stream — `run()` drains it, `stream()` yields from it.

```python
async def execute(self, ctx: HarnessContext) -> AsyncIterator[StreamEvent]:
    # Use ctx to run/stream agents
    # Use normal Python control flow for orchestration
    # Yield events to pass them through to consumers
    # Yield ctx.emit(...) for custom events
    yield  # must yield at least once (it's a generator)
```

**Why an async generator?**
- Unifies streaming and non-streaming in one code path
- Event interception is a for-loop with conditionals — no callbacks
- Custom events emitted by yielding `HarnessEvent`
- Full Python control flow for orchestration decisions

### HarnessEvent

Custom event type emitted by harness logic. NOT part of the `StreamEvent` union (to avoid coupling exo-core to exo-harness), but follows the same frozen BaseModel pattern with `type` and `agent_name` fields.

```python
class HarnessEvent(BaseModel):
    model_config = {"frozen": True}

    type: Literal["harness"] = "harness"   # Discriminator
    kind: str                               # Developer-defined sub-kind
    agent_name: str = ""                    # Harness name (auto-set by ctx.emit)
    data: dict[str, Any] = Field(default_factory=dict)  # Arbitrary payload
```

**Creating events:** Use `ctx.emit(kind, **data)` — it sets `agent_name` automatically.

```python
yield ctx.emit("classified", category="billing")
# Creates: HarnessEvent(type="harness", kind="classified", agent_name="my_harness", data={"category": "billing"})
```

**Filtering:** Works with `event_types={"harness"}` to receive only harness events. The runner's string-based `event.type` filter passes `"harness"` through.

### SessionState

Mutable dict wrapper with dirty tracking for checkpoint optimization.

```python
@dataclass
class SessionState:
    data: dict[str, Any]    # The underlying dict (serialization format)
    _dirty: bool            # Whether state has been modified since last checkpoint

    def __getitem__(self, key: str) -> Any       # state["key"]
    def __setitem__(self, key: str, value: Any)  # state["key"] = value (sets dirty)
    def __contains__(self, key: str) -> bool      # "key" in state
    def get(self, key: str, default=None) -> Any  # state.get("key", default)
    def mark_clean(self) -> None                  # Reset dirty flag
    @property
    def dirty(self) -> bool                       # Check if modified
```

**Usage in execute():**
```python
ctx.state["step"] = 1              # Sets dirty=True
ctx.state["route"] = "billing"
if "category" in ctx.state: ...
val = ctx.state.get("fallback", "default")
```

**Initialization:** Pass a dict or SessionState to the Harness constructor:
```python
harness = MyHarness(name="h", state={"initial_key": "value"})
harness = MyHarness(name="h", state=SessionState(data={"key": "val"}))
```

### HarnessCheckpoint

Immutable snapshot for persistence and resumption.

```python
@dataclass(frozen=True)
class HarnessCheckpoint:
    harness_name: str                    # Which harness created this
    session_state: dict[str, Any]        # Serialized SessionState.data
    completed_agents: list[str]          # Agents that have finished
    pending_agent: str | None = None     # Agent about to execute
    messages: list[dict[str, Any]] = ... # Serialized message history
    timestamp: float = ...               # Unix timestamp
    metadata: dict[str, Any] = ...       # Arbitrary metadata
```

### CheckpointAdapter

Serializes `HarnessCheckpoint` to any `MemoryStore` backend (SQLite, Postgres, etc.).

```python
from exo.harness import CheckpointAdapter
from exo.memory.backends.sqlite import SQLiteMemoryStore

async with SQLiteMemoryStore() as store:
    adapter = CheckpointAdapter(store, "my_harness")
    await adapter.save(checkpoint)       # Persist as MemoryItem
    restored = await adapter.load_latest()  # Load most recent
```

**Storage format:** JSON in `MemoryItem.content`, `memory_type="harness_checkpoint"`, scoped by `agent_id=harness_name`.

### Middleware

Middleware wraps the event stream. Each middleware is an async generator transformer — it receives an upstream `AsyncIterator[StreamEvent]` and yields events downstream.

```python
from exo.harness import Middleware

class Middleware(ABC):
    @abstractmethod
    async def wrap(
        self,
        stream: AsyncIterator[StreamEvent],
        ctx: HarnessContext,
    ) -> AsyncIterator[StreamEvent]:
        async for event in stream:
            yield event
```

**Middleware chain:** Applied in reverse order of the `middleware` list. The first middleware in the list is the outermost wrapper.

```python
harness = MyHarness(
    name="h",
    middleware=[TimeoutMiddleware(30), CostTrackingMiddleware()],
    # Chain: CostTracking wraps execute(), Timeout wraps CostTracking
)
```

### TimeoutMiddleware

Emits an `ErrorEvent` and stops iteration after a wall-clock deadline.

```python
from exo.harness import TimeoutMiddleware

middleware = TimeoutMiddleware(timeout_seconds=30.0)
```

The timeout is checked between yielded events. If the deadline is exceeded, an `ErrorEvent(error_type="TimeoutError", recoverable=False)` is yielded and the stream stops.

### CostTrackingMiddleware

Accumulates token usage from `UsageEvent` instances into `ctx.state["_cost"]`.

```python
from exo.harness import CostTrackingMiddleware

middleware = CostTrackingMiddleware()
# After run: harness.session["_cost"] == {"input_tokens": N, "output_tokens": N, "total_tokens": N}
```

All events pass through unmodified — this middleware only writes to session state. Note: `UsageEvent` is only emitted when `detailed=True` is passed to `stream_agent()`.

### HarnessNode

Wraps a `Harness` for use as a node in a Swarm's flow DSL. Follows the same pattern as `SwarmNode` and `RalphNode`.

```python
from exo.harness import HarnessNode
from exo.swarm import Swarm

inner_harness = MyHarness(name="classifier_harness", agents=[classifier])
node = HarnessNode(harness=inner_harness, name="classify")

swarm = Swarm(
    agents=[agent_a, node, agent_b],
    flow="agent_a >> classify >> agent_b",
)
```

**Context isolation:** `HarnessNode` does NOT forward outer messages to the inner harness. Each execution creates a fresh context.

**Marker:** `is_group = True` makes the Swarm's duck-typing check route to `.stream()` for streaming and `.run()` for non-streaming.

### Runner Integration

Harness works with the standard `run()` and `run.stream()` entry points:

```python
from exo import run

# Non-streaming — returns RunResult
result = await run(harness, "Hello!", provider=provider)

# Streaming — yields StreamEvent
async for event in run.stream(harness, "Hello!", provider=provider, detailed=True):
    print(event)

# Blocking — for scripts/notebooks
result = run.sync(harness, "Hello!", provider=provider)
```

**Detection:** The runner checks `hasattr(agent, "is_harness")` before the Swarm check (`hasattr(agent, "flow_order")`). If detected, it delegates to `harness.run()` or `harness.stream()`.

**Provider resolution:** `_resolve_provider()` checks `harness.agents` (same dict pattern as Swarm), resolves from the first agent's model string.

## Patterns

### Basic Passthrough Harness

Streams one agent and passes all events through unchanged. The simplest possible harness.

```python
from exo.harness import Harness, HarnessContext

class PassthroughHarness(Harness):
    async def execute(self, ctx):
        agent = self.agents["bot"]
        async for event in ctx.stream_agent(agent, ctx.input):
            yield event

harness = PassthroughHarness(name="passthrough", agents=[bot])
result = await run(harness, "Hello!")
```

### Router Harness

Classifies input, then routes to a specialist agent based on the classification.

```python
class RouterHarness(Harness):
    async def execute(self, ctx):
        # Phase 1: Classify the input
        result = await ctx.run_agent(self.agents["classifier"], ctx.input)
        category = result.output.strip()
        ctx.state["category"] = category
        yield ctx.emit("classified", category=category)

        # Phase 2: Route to specialist
        route_map = {"billing": "billing_agent", "tech": "tech_agent"}
        agent_name = route_map.get(category, "general_agent")
        agent = self.agents[agent_name]

        # Phase 3: Stream the specialist (events pass through to consumer)
        async for event in ctx.stream_agent(agent, ctx.input):
            yield event

harness = RouterHarness(
    name="support_router",
    agents=[classifier, billing_agent, tech_agent, general_agent],
)
```

### Event Interception

Watch what an agent does and react based on observed behavior.

```python
from exo.types import ToolCallEvent, TextEvent

class WatchdogHarness(Harness):
    async def execute(self, ctx):
        agent = self.agents["worker"]
        async for event in ctx.stream_agent(agent, ctx.input, detailed=True):
            # React to specific tool calls
            if isinstance(event, ToolCallEvent) and event.tool_name == "delete_file":
                yield ctx.emit("alert", message="Agent tried to delete a file!")
                # Could cancel, switch agents, inject a message, etc.

            yield event  # Always yield to pass through

harness = WatchdogHarness(name="watchdog", agents=[worker])
async for event in run.stream(harness, "Clean up old files"):
    if isinstance(event, HarnessEvent) and event.kind == "alert":
        print(f"ALERT: {event.data['message']}")
```

### Sequential Pipeline with State

Run agents in sequence, piping output from one to the next.

```python
class PipelineHarness(Harness):
    async def execute(self, ctx):
        current_input = ctx.input
        for agent_name in ["researcher", "writer", "editor"]:
            agent = self.agents[agent_name]
            result = await ctx.run_agent(agent, current_input)
            current_input = result.output
            ctx.state[agent_name] = result.output
            yield ctx.emit("step_complete", agent=agent_name)

        # Final output — yield as TextEvent so run() captures it
        from exo.types import TextEvent
        yield TextEvent(text=current_input, agent_name=self.name)

harness = PipelineHarness(
    name="content_pipeline",
    agents=[researcher, writer, editor],
)
result = await run(harness, "Write about quantum computing")
print(result.output)  # Editor's refined output
```

### Forked History

Give an agent a copy of the conversation history (isolated from the harness).

```python
class ForkedHarness(Harness):
    async def execute(self, ctx):
        # Agent A sees shared history
        result_a = await ctx.run_agent(self.agents["a"], ctx.input, messages=ctx.messages)

        # Agent B sees a forked copy (won't see A's mutations)
        result_b = await ctx.run_agent(self.agents["b"], ctx.input, messages=list(ctx.messages))

        # Agent C sees no history (fresh conversation)
        result_c = await ctx.run_agent(self.agents["c"], ctx.input, messages=None)

        yield ctx.emit("done", a=result_a.output, b=result_b.output, c=result_c.output)
```

### Cancellation

Cancel a running harness from outside.

```python
import asyncio

class LongRunningHarness(Harness):
    async def execute(self, ctx):
        for i in range(10):
            ctx.check_cancelled()  # Raises HarnessError if cancelled
            agent = self.agents["worker"]
            result = await ctx.run_agent(agent, f"Step {i}: {ctx.input}")
            ctx.state[f"step_{i}"] = result.output
            yield ctx.emit("progress", step=i)

harness = LongRunningHarness(name="long", agents=[worker])

async def cancel_after(h, delay):
    await asyncio.sleep(delay)
    h.cancel()

# Run with a cancellation timer
task = asyncio.create_task(cancel_after(harness, 5.0))
try:
    result = await run(harness, "Process all items")
except HarnessError:
    print(f"Cancelled after step {harness.session.get('step_9', '?')}")
```

### Checkpoint and Resume

Persist state for resumption after process restart.

```python
from exo.memory.backends.sqlite import SQLiteMemoryStore

class ResumableHarness(Harness):
    async def execute(self, ctx):
        # Try to restore from checkpoint
        checkpoint = await self.restore_checkpoint()
        start_step = 0
        if checkpoint:
            ctx.state.data.update(checkpoint.session_state)
            start_step = len(checkpoint.completed_agents)

        agents_order = ["step_a", "step_b", "step_c"]
        for i, agent_name in enumerate(agents_order):
            if i < start_step:
                continue  # Skip completed steps
            agent = self.agents[agent_name]
            result = await ctx.run_agent(agent, ctx.input)
            ctx.state[agent_name] = result.output
            ctx.state.setdefault("_completed_agents", []).append(agent_name)
            await ctx.checkpoint(pending_agent=agents_order[i + 1] if i + 1 < len(agents_order) else None)
            yield ctx.emit("step_done", agent=agent_name)

async with SQLiteMemoryStore() as store:
    harness = ResumableHarness(
        name="resumable",
        agents=[step_a, step_b, step_c],
        checkpoint_store=store,
    )
    result = await run(harness, "Process")
```

### Custom Middleware

Write middleware that filters, transforms, or augments the event stream.

```python
from exo.harness import Middleware
from exo.types import TextEvent

class CensorMiddleware(Middleware):
    def __init__(self, banned_words: list[str]):
        self._banned = banned_words

    async def wrap(self, stream, ctx):
        async for event in stream:
            if isinstance(event, TextEvent):
                text = event.text
                for word in self._banned:
                    text = text.replace(word, "***")
                yield TextEvent(text=text, agent_name=event.agent_name)
            else:
                yield event

harness = MyHarness(
    name="safe",
    agents=[bot],
    middleware=[CensorMiddleware(["password", "secret"]), TimeoutMiddleware(60)],
)
```

### Nested Harness Composition

A harness can run another harness as a sub-unit.

```python
class InnerHarness(Harness):
    async def execute(self, ctx):
        result = await ctx.run_agent(self.agents["worker"], ctx.input)
        yield ctx.emit("inner_done", output=result.output)

class OuterHarness(Harness):
    async def execute(self, ctx):
        # Run inner harness — detected automatically via is_harness
        inner = self.agents["inner"]
        async for event in ctx.stream_agent(inner, ctx.input):
            yield event
        yield ctx.emit("outer_done")

inner = InnerHarness(name="inner", agents=[worker])
outer = OuterHarness(name="outer", agents={"inner": inner})
result = await run(outer, "Hello!")
```

## Gotchas

- **`execute()` must be an async generator** — it must contain at least one `yield` statement, even if just `yield ctx.emit(...)`. A regular `async def` that never yields will not work.
- **`run()` only captures `TextEvent` text** — if your `execute()` only yields `HarnessEvent` or other event types, `run().output` will be an empty string. Yield `TextEvent` for content you want in the `RunResult.output`.
- **`UsageEvent` requires `detailed=True`** — token usage is only emitted when `ctx.stream_agent(agent, input, detailed=True)`. Without it, `CostTrackingMiddleware` and `run()` usage aggregation get nothing.
- **`stream_agent()` needs a provider with `stream()` method** — the runner calls `provider.stream()`. A mock provider with only `complete()` will fail. Use `_make_stream_provider` in tests.
- **`run_agent()` needs a provider with `complete()` method** — the runner calls `provider.complete()` for non-streaming. Use `_make_provider` in tests.
- **History visibility is your responsibility** — `messages=None` (fresh), `ctx.messages` (shared reference), `list(ctx.messages)` (forked copy). The harness does not choose for you.
- **`HarnessEvent` is NOT in the `StreamEvent` union** — consumers that type-narrow with `isinstance(event, TextEvent | ToolCallEvent | ...)` will not match `HarnessEvent`. Check for it separately: `isinstance(event, HarnessEvent)`.
- **`event_types={"harness"}` works** — the runner's string-based filter checks `event.type`, so `"harness"` passes through even though `HarnessEvent` is not part of `StreamEvent`.
- **Middleware order matters** — middleware is applied in reverse list order. `[A, B]` means B wraps execute(), then A wraps B. The first middleware in the list is the outermost.
- **`cancel()` is cooperative** — it sets a flag. The `execute()` method must call `ctx.check_cancelled()` at yield points for cancellation to take effect. In-flight `run_agent()` / `stream_agent()` calls are not interrupted.
- **`save_checkpoint()` is a no-op without `checkpoint_store`** — you must pass a `MemoryStore` backend to the Harness constructor for checkpointing to persist.
- **`SessionState._dirty` tracks writes, not reads** — `state["key"]` (read) does not set dirty. Only `state["key"] = value` (write) sets dirty.
- **Duplicate agent names raise `HarnessError`** — when passing a list to `agents=`, all agents must have unique `.name` attributes. Use a dict for explicit naming.
- **`HarnessNode` provides context isolation** — outer Swarm messages are NOT forwarded to the inner harness. Each execution starts fresh.
- **Exceptions from `execute()` yield `ErrorEvent` then re-raise** — `_execute_with_middleware()` catches exceptions, yields an `ErrorEvent`, then re-raises. `HarnessError` is re-raised directly without yielding `ErrorEvent`.
- **`reset()` clears the cancellation signal** — call it before reusing a harness instance after cancellation.
