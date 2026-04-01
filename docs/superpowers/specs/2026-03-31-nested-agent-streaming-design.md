# Nested Agent Streaming Design

**Date:** 2026-03-31
**Status:** Draft
**Scope:** Event forwarding from inner agents through tools, RalphRunner streaming, Swarm node pluggability

## Problem

When an agent (X) has a tool that internally runs another agent (Y), Y's stream events are invisible to X's consumer. The tool is opaque — the parent stream only sees a `ToolResultEvent` with the final return value. This blocks two key use cases:

1. **Ralph loops** — `RalphRunner` calls `execute_fn` each iteration (typically `agent.run()`). No events stream out during execution. Consumers can't observe progress across iterations.
2. **Iterative tool loops** — A tool on Agent X runs Agent Y (possibly in a loop). Y's `TextEvent`, `ToolCallEvent`, etc. are lost.

Additionally:
- `SwarmNode` has `.run()` but no `.stream()`, so nested swarms lose all inner events in the streaming path.
- `RalphRunner` has no way to be plugged into a Swarm as a node.

## Design: Event Queue on Agent (Approach A)

### Core Mechanism

Each `Agent` gets an `_event_queue: asyncio.Queue[StreamEvent]` initialized in `__init__`. Tools push events onto this queue via an injected `ToolContext`. `run.stream()` drains the queue after each tool batch completes — the same spot where MCP progress queues are already drained.

### ToolContext

A lightweight object injected into tools that declare it as a parameter. Uses the existing `injected_tool_args` mechanism.

```python
class ToolContext:
    """Injected into tools that declare a ToolContext parameter."""

    def __init__(self, agent_name: str, queue: asyncio.Queue[StreamEvent]):
        self._queue = queue
        self.agent_name = agent_name

    def emit(self, event: StreamEvent) -> None:
        """Push an event to the parent agent's stream."""
        self._queue.put_nowait(event)
```

Tool authors opt in by adding `ctx: ToolContext` to their signature:

```python
@tool
async def research(query: str, ctx: ToolContext) -> str:
    parts = []
    async for event in inner_agent.stream(query):
        ctx.emit(event)
        if isinstance(event, TextEvent):
            parts.append(event.text)
    return "".join(parts)
```

Tools that don't declare `ToolContext` are completely unaffected. When using `run()` (non-streaming), the queue exists but is never drained — events are silently discarded on GC. Zero cost when not streaming.

### Event drain in run.stream()

Added after `_execute_tools()` returns, before `ToolResultEvent` emission. Extends the existing MCP progress drain pattern:

```python
# runner.py — after _execute_tools(), before ToolResultEvent yield

# Drain inner agent events pushed by tools via ToolContext.emit()
while not agent._event_queue.empty():
    try:
        inner_event = agent._event_queue.get_nowait()
        if _passes_filter(inner_event):
            yield inner_event
    except Exception:
        break

# Existing MCP progress drain follows (unchanged)
```

Events from inner agents arrive with their original `agent_name` (e.g., `agent_name="Y"`), so consumers can distinguish them from outer agent events. No wrapper event type needed.

### ToolContext injection

Detection uses the same pattern as existing `injected_tool_args`. If a tool's signature has a parameter typed as `ToolContext`, it gets auto-injected during `_execute_tools`. The `ToolContext` is constructed with the agent's `_event_queue`.

## RalphRunner.stream()

### New event types

Two new event types added to `StreamEvent` union:

```python
class RalphIterationEvent(BaseModel):
    """Emitted at the start/end of each Ralph loop iteration."""
    model_config = {"frozen": True}
    type: Literal["ralph_iteration"] = "ralph_iteration"
    iteration: int
    status: Literal["started", "completed", "failed"]
    scores: dict[str, float] = {}
    agent_name: str = ""

class RalphStopEvent(BaseModel):
    """Emitted when the Ralph loop terminates."""
    model_config = {"frozen": True}
    type: Literal["ralph_stop"] = "ralph_stop"
    stop_type: str
    reason: str
    iterations: int
    final_scores: dict[str, float] = {}
    agent_name: str = ""
```

### Dual execute_fn

`RalphRunner` accepts two function signatures:

- `execute_fn: async (str) -> str` — used by `.run()` (existing, unchanged)
- `stream_execute_fn: async (str) -> AsyncIterator[StreamEvent]` — used by `.stream()`. If not provided, `.stream()` raises `ValueError("stream_execute_fn required for streaming")`.

A convenience classmethod wires both from an Agent:

```python
@classmethod
def from_agent(cls, agent, scorers, **kwargs) -> RalphRunner:
    async def _execute(input: str) -> str:
        result = await run(agent, input)
        return result.output

    async def _stream(input: str) -> AsyncIterator[StreamEvent]:
        async for event in run.stream(agent, input):
            yield event

    return cls(execute_fn=_execute, stream_execute_fn=_stream, scorers=scorers, **kwargs)
```

### stream() method

```python
async def stream(self, input: str, *, name: str = "ralph") -> AsyncIterator[StreamEvent]:
    state = LoopState()
    current_input = input

    while True:
        state.iteration += 1
        yield RalphIterationEvent(iteration=state.iteration, status="started", agent_name=name)

        # Phase 1: Run — stream the execute_fn, forward all inner events
        output_parts = []
        success = True
        try:
            async for event in self._stream_execute_fn(current_input):
                yield event
                if isinstance(event, TextEvent):
                    output_parts.append(event.text)
            output = "".join(output_parts)
            state.record_success()
        except Exception as exc:
            output = str(exc)
            state.record_failure()
            success = False

        # Phase 2: Analyze
        scores = await self._analyze(output, current_input, state) if success else {}

        yield RalphIterationEvent(
            iteration=state.iteration,
            status="completed" if success else "failed",
            scores=scores,
            agent_name=name,
        )

        # Phase 3: Learn
        reflection = await self._learn(current_input, output, scores, success, state)

        # Phase 4: Plan
        current_input = self._plan(input, reflection)

        # Phase 5: Halt
        decision = await self._halt(state)
        if decision.should_stop:
            yield RalphStopEvent(
                stop_type=decision.stop_type.value,
                reason=decision.reason,
                iterations=state.iteration,
                final_scores=scores,
                agent_name=name,
            )
            return
```

## Swarm Pluggability

### SwarmNode.stream() — fix missing method

`SwarmNode` currently has `.run()` but no `.stream()`. The Swarm code already checks `hasattr(agent, "stream")` and delegates to it. Adding the method:

```python
class SwarmNode:
    # existing __init__, run(), describe() unchanged

    async def stream(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        detailed: bool = False,
        max_steps: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream inner swarm events with context isolation."""
        async for event in self._swarm.stream(
            input, provider=provider, detailed=detailed, max_steps=max_steps,
        ):
            yield event
```

### RalphNode — new wrapper

Lives in `exo/_internal/nested.py` alongside `SwarmNode`. Wraps `RalphRunner` so it can be placed in a Swarm's agent list.

```python
class RalphNode:
    """Wraps a RalphRunner so it can be used as a node in a Swarm."""

    def __init__(self, *, runner: RalphRunner, name: str = "ralph"):
        self._runner = runner
        self.name = name
        self.is_group = True  # triggers Swarm's .stream() delegation path

    async def run(self, input: str, *, messages=None, provider=None, **kw) -> RunResult:
        result = await self._runner.run(input)
        return RunResult(output=result.output)

    async def stream(
        self, input: str, *, messages=None, provider=None,
        detailed=False, max_steps=None,
    ) -> AsyncIterator[StreamEvent]:
        async for event in self._runner.stream(input, name=self.name):
            yield event
```

The `is_group = True` marker makes the Swarm's existing duck-typing check (`getattr(agent, "is_group", False)`) route to `.stream()`. No changes to `Swarm` itself.

### Usage

```python
researcher = Agent(name="researcher", ...)
ralph = RalphNode(
    runner=RalphRunner.from_agent(researcher, scorers=[quality_scorer]),
    name="research_loop",
)
summarizer = Agent(name="summarizer", ...)

swarm = Swarm(
    agents=[ralph, summarizer],
    flow="research_loop >> summarizer",
)

async for event in swarm.stream("analyze market trends"):
    print(event)
```

## Files Changed

| File | Change |
|---|---|
| `packages/exo-core/src/exo/types.py` | Add `RalphIterationEvent`, `RalphStopEvent`; update `StreamEvent` union |
| `packages/exo-core/src/exo/tool_context.py` | New file: `ToolContext` class (depends on `asyncio.Queue`, kept out of types.py) |
| `packages/exo-core/src/exo/agent.py` | Add `_event_queue = asyncio.Queue()` in `__init__`; inject `ToolContext` in `_execute_tools` |
| `packages/exo-core/src/exo/runner.py` | Drain `_event_queue` after tool execution (~8 lines) |
| `packages/exo-core/src/exo/_internal/nested.py` | Add `SwarmNode.stream()`; add `RalphNode` class |
| `packages/exo-eval/src/exo/eval/ralph/runner.py` | Add `.stream()`, `stream_execute_fn` param, `from_agent()` classmethod |
| `packages/exo-eval/src/exo/eval/ralph/__init__.py` | Re-export new types |
| `packages/exo-core/src/exo/__init__.py` | Export `ToolContext` (from `tool_context`), new event types |

No changes to: `exo/eval/ralph/config.py`, `exo/eval/ralph/detectors.py`, `exo/swarm.py`.

## Testing Strategy

- **ToolContext + event drain:** Mock tool that emits events via `ctx.emit()`, verify they appear in `run.stream()` output in correct order.
- **RalphRunner.stream():** Mock `stream_execute_fn` that yields known events, verify `RalphIterationEvent`/`RalphStopEvent` interleave correctly with inner events.
- **SwarmNode.stream():** Nest a swarm inside another, stream outer, verify inner events flow through.
- **RalphNode in Swarm:** Place `RalphNode` in a workflow swarm, verify iteration events + inner agent events + downstream agent events all stream.
- **Backwards compatibility:** Existing tools without `ToolContext` param still work unchanged. Existing `run()` (non-streaming) unaffected.
