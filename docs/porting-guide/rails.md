# Agent Rails — agent-core to Exo Mapping

**Epic:** 6 — Typed Agent Rails
**Date:** 2026-03-10

This document maps agent-core's (openJiuwen) rail system to Exo's typed
rails, helping contributors familiar with either framework navigate both.

---

## 1. Agent-Core Overview

Agent-core's rail system lives in `openjiuwen/core/single_agent/rail/` and
provides typed lifecycle guards for agent execution.

### Key Components

**`AgentCallbackEvent`** — An enum of 10 lifecycle events that rails can
intercept:

| Event | When it fires |
|-------|--------------|
| `ON_AGENT_START` | Agent invocation begins |
| `ON_AGENT_END` | Agent invocation completes |
| `ON_AGENT_ERROR` | Agent encounters an unrecoverable error |
| `ON_MODEL_CALL` | Before an LLM call is made |
| `ON_MODEL_RESPONSE` | After an LLM response is received |
| `ON_MODEL_EXCEPTION` | LLM call raises an exception |
| `ON_TOOL_CALL` | Before a tool is executed |
| `ON_TOOL_RESPONSE` | After a tool returns a result |
| `ON_TOOL_EXCEPTION` | Tool execution raises an exception |
| `ON_HANDOFF` | Agent transfers control to another agent |

**`AgentRail` ABC** — The base class for all rails. Subclasses implement
`handle(event, context)` where `context` is an untyped dict containing the
agent reference, event-specific data, and a shared `extra` dict for cross-rail
state.

```python
# agent-core pattern
class AgentRail(ABC):
    name: str
    priority: int = 50  # Lower = runs first

    @abstractmethod
    async def handle(self, event: AgentCallbackEvent, context: dict) -> RailResult:
        ...
```

**`@rail` decorator** — A convenience decorator that wraps a plain async
function as an `AgentRail` instance:

```python
# agent-core pattern
@rail(name="block_rm", events=[AgentCallbackEvent.ON_TOOL_CALL], priority=10)
async def block_rm(event, context):
    if context["tool_name"] == "rm_rf":
        return RailResult.ABORT
    return RailResult.CONTINUE
```

**Typed inputs** — Agent-core passes event-specific data as a flat dict
(`context`). The caller is responsible for knowing which keys are available
at each event.

**Priority ordering** — Rails execute in ascending priority order (lower value
= higher priority). Default is 50. Security rails use 10–20; observability
rails use 80–90.

**Cross-rail state** — `context["extra"]` is a mutable dict shared across all
rails in a single invocation, enabling upstream rails to pass data to
downstream ones.

**`RailResult`** — An enum of actions: `CONTINUE`, `SKIP`, `RETRY`, `ABORT`.
`RETRY` is accompanied by retry parameters (delay, max attempts).

---

## 2. Exo Equivalent

Exo's rail system is in `packages/exo-core/src/exo/` across two
modules: `rail.py` and `rail_types.py`. It extends (not replaces) the existing
`HookManager` system.

### Concept Mapping

| Agent-Core Concept | Exo Counterpart | Notes |
|-------------------|---------------------|-------|
| `AgentCallbackEvent` (10 events) | `HookPoint` enum (7 events) | Exo consolidates exception events into `ERROR`; no `ON_HANDOFF` yet |
| `AgentRail` ABC | `Rail` ABC | Same pattern: `name`, `priority`, abstract `handle()` |
| `@rail` decorator | *(no decorator)* | Exo uses class-based rails only; subclass `Rail` directly |
| `RailResult` enum | `RailAction` StrEnum | Same 4 values: `CONTINUE`, `SKIP`, `RETRY`, `ABORT` |
| `context` dict (untyped) | `RailContext` Pydantic model | Typed `inputs` field with event-specific model |
| Event-specific data (flat dict keys) | Typed input models | `InvokeInputs`, `ModelCallInputs`, `ToolCallInputs` |
| `context["extra"]` dict | `RailContext.extra` dict | Same pattern — mutable dict shared across rails in one invocation |
| Rail manager (internal) | `RailManager` class | Explicit class with `add()`, `remove()`, `run()`, `hook_for()` |
| Priority ordering | Priority ordering | Identical: ascending sort, lower = first, default 50 |
| Retry parameters | `RetryRequest` dataclass | Frozen dataclass with `delay`, `max_retries`, `reason` |
| Rail abort exception | `RailAbortError` | Inherits from `ExoError`; includes `rail_name` and `reason` |

### Lifecycle Event Mapping

| Agent-Core Event | Exo HookPoint | Input Model |
|-----------------|-------------------|-------------|
| `ON_AGENT_START` | `HookPoint.START` | `InvokeInputs` |
| `ON_AGENT_END` | `HookPoint.FINISHED` | `InvokeInputs` |
| `ON_AGENT_ERROR` | `HookPoint.ERROR` | `InvokeInputs` |
| `ON_MODEL_CALL` | `HookPoint.PRE_LLM_CALL` | `ModelCallInputs` |
| `ON_MODEL_RESPONSE` | `HookPoint.POST_LLM_CALL` | `ModelCallInputs` |
| `ON_MODEL_EXCEPTION` | `HookPoint.ERROR` | `InvokeInputs` |
| `ON_TOOL_CALL` | `HookPoint.PRE_TOOL_CALL` | `ToolCallInputs` |
| `ON_TOOL_RESPONSE` | `HookPoint.POST_TOOL_CALL` | `ToolCallInputs` |
| `ON_TOOL_EXCEPTION` | `HookPoint.ERROR` | `InvokeInputs` |
| `ON_HANDOFF` | *(not yet ported)* | — |

### Integration with HookManager

Unlike agent-core where rails are a standalone system, Exo rails **extend
the existing hook system**:

1. `RailManager` registers itself as a hook on `HookManager` via `hook_for()`.
2. Plain hooks (`hook_manager.add(HookPoint.X, my_func)`) continue to work.
3. Both hooks and rails coexist on the same `HookManager`.
4. Agents without rails behave identically to before — `rail_manager` is `None`.

---

## 3. Side-by-Side Code Examples

### Example: Block a dangerous tool

**Agent-core:**

```python
# agent-core — uses @rail decorator and untyped context dict
@rail(name="block_rm", events=[AgentCallbackEvent.ON_TOOL_CALL], priority=10)
async def block_rm(event, context):
    if context["tool_name"] == "rm_rf":
        return RailResult.ABORT
    return RailResult.CONTINUE

agent = Agent(name="assistant", rails=[block_rm])
```

**Exo:**

```python
from exo.hooks import HookPoint
from exo.rail import Rail, RailAction
from exo.rail_types import RailContext, ToolCallInputs


class BlockDangerousTool(Rail):
    """Block execution of a specific tool by name."""

    def __init__(self, blocked_tool: str) -> None:
        super().__init__("block_tool", priority=10)
        self.blocked_tool = blocked_tool

    async def handle(self, ctx: RailContext) -> RailAction | None:
        if (
            ctx.event == HookPoint.PRE_TOOL_CALL
            and isinstance(ctx.inputs, ToolCallInputs)
            and ctx.inputs.tool_name == self.blocked_tool
        ):
            return RailAction.ABORT
        return RailAction.CONTINUE
```

### Example: Cross-rail state sharing

**Agent-core:**

```python
# agent-core — upstream rail writes to context["extra"]
@rail(name="rate_counter", priority=10)
async def rate_counter(event, context):
    context["extra"]["calls_remaining"] = 5
    return RailResult.CONTINUE

@rail(name="rate_logger", priority=80)
async def rate_logger(event, context):
    remaining = context["extra"].get("calls_remaining", "unknown")
    print(f"Rate limit remaining: {remaining}")
    return RailResult.CONTINUE
```

**Exo:**

```python
from exo.rail import Rail, RailAction
from exo.rail_types import RailContext


class RateCounterRail(Rail):
    """Write rate-limit state for downstream rails."""

    def __init__(self) -> None:
        super().__init__("rate_counter", priority=10)

    async def handle(self, ctx: RailContext) -> RailAction | None:
        ctx.extra["calls_remaining"] = 5
        return RailAction.CONTINUE


class RateLoggerRail(Rail):
    """Read rate-limit state from upstream rails."""

    def __init__(self) -> None:
        super().__init__("rate_logger", priority=80)

    async def handle(self, ctx: RailContext) -> RailAction | None:
        remaining = ctx.extra.get("calls_remaining", "unknown")
        print(f"Rate limit remaining: {remaining}")
        return RailAction.CONTINUE
```

### Example: Registering rails on an agent

**Agent-core:**

```python
# agent-core
agent = Agent(name="my_agent", rails=[block_rm, rate_counter, rate_logger])
```

**Exo:**

```python
from exo.agent import Agent
from exo.rail import Rail

# Rails are passed as a list — RailManager is created automatically
agent = Agent(
    name="my_agent",
    model="openai:gpt-4o",
    rails=[
        BlockDangerousTool("rm_rf"),
        RateCounterRail(),
        RateLoggerRail(),
    ],
)

# Equivalent manual setup (if you need fine-grained control):
from exo.hooks import HookPoint
from exo.rail import RailManager

manager = RailManager()
manager.add(BlockDangerousTool("rm_rf"))

agent2 = Agent(name="my_agent2", model="openai:gpt-4o")
for point in HookPoint:
    agent2.hook_manager.add(point, manager.hook_for(point))
```

---

## 4. Migration Table

| Agent-Core Path | Exo Import |
|----------------|---------------|
| `rail.base.AgentCallbackEvent` | `exo.hooks.HookPoint` |
| `rail.base.AgentRail` | `exo.rail.Rail` |
| `rail.base.RailResult` | `exo.rail.RailAction` |
| `rail.base.RailResult.CONTINUE` | `exo.rail.RailAction.CONTINUE` |
| `rail.base.RailResult.SKIP` | `exo.rail.RailAction.SKIP` |
| `rail.base.RailResult.RETRY` | `exo.rail.RailAction.RETRY` |
| `rail.base.RailResult.ABORT` | `exo.rail.RailAction.ABORT` |
| `rail.base.@rail` decorator | *(removed — subclass `Rail` directly)* |
| `rail.base.RetryConfig` | `exo.rail.RetryRequest` |
| `rail.base.RailAbortException` | `exo.rail.RailAbortError` |
| `rail.manager.RailManager` | `exo.rail.RailManager` |
| Event context dict | `exo.rail_types.RailContext` |
| `context["extra"]` | `RailContext.extra` |
| *(untyped event data)* | `exo.rail_types.InvokeInputs` |
| *(untyped event data)* | `exo.rail_types.ModelCallInputs` |
| *(untyped event data)* | `exo.rail_types.ToolCallInputs` |
