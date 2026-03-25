# Rails System Design — Typed Agent Lifecycle Guards

**Status:** Proposed
**Epic:** 6 — Typed Agent Rails
**Date:** 2026-03-10

---

## 1. Motivation

Exo's `HookManager` provides lifecycle interception via async callables
registered at `HookPoint` enum values. While functional, it lacks:

- **Typed event inputs** — hooks receive untyped `**kwargs`, making it easy
  to mishandle data or miss available fields.
- **Priority ordering** — hooks execute in registration order only.
- **Cross-hook state** — no mechanism for hooks in the same invocation to
  share intermediate state.
- **Retry mechanism** — a failing hook aborts the run; there is no way for
  a guard to request a retry with a delay.

Agent-core's rail system (`openjiuwen/core/single_agent/rail/`) addresses all
four gaps with 10 lifecycle events, typed input models, `RetryRequest`,
priority-based execution, and a shared `extra` dict.

This document proposes porting the rail concept into Exo as an **extension
of the existing hook system**, not a replacement.

---

## 2. Key Decision: Rails Extend Hooks

### Option A — Rails as a parallel system (rejected)

A separate `RailManager` with its own lifecycle, independent of `HookManager`.
This duplicates the dispatch logic and forces consumers to choose between
hooks and rails.

### Option B — Rails extend hooks (chosen)

Rails are a **new optional typed interface** layered on top of `HookManager`:

1. `Rail` is an abstract class with a `handle(ctx: RailContext)` method.
2. `RailManager` collects rails, sorts by priority, and exposes an async
   callable with the `Hook` signature.
3. `RailManager` registers itself as a single hook on the agent's
   `HookManager` for each relevant `HookPoint`.
4. Existing `hook_manager.add(HookPoint.X, my_func)` calls continue to work
   unchanged — rails are just another hook in the list.

**Why Option B:**

- Zero breaking changes to the existing hook API.
- Rails and plain hooks coexist on the same `HookManager`.
- `HookManager` remains the single source of truth for lifecycle dispatch.
- Rails are opt-in — agents without rails behave identically to today.

---

## 3. Typed Event Input Models

Three Pydantic models capture the data available at each lifecycle point.
Models are **not frozen** because hooks/rails may need to mutate inputs
(e.g., redacting messages before an LLM call).

```python
class InvokeInputs(BaseModel):
    """Data for START / FINISHED events."""
    input: str
    messages: list[Any] | None = None
    result: Any | None = None

class ModelCallInputs(BaseModel):
    """Data for PRE_LLM_CALL / POST_LLM_CALL events."""
    messages: list[Any]
    tools: list[dict] | None = None
    response: Any | None = None
    usage: Any | None = None

class ToolCallInputs(BaseModel):
    """Data for PRE_TOOL_CALL / POST_TOOL_CALL events."""
    tool_name: str
    arguments: dict[str, Any]
    result: Any | None = None
    metadata: Any | None = None
```

A `RailContext` model bundles the agent reference, event type, typed inputs,
and the shared `extra` dict:

```python
class RailContext(BaseModel):
    """Context passed to each rail's handle() method."""
    model_config = {"arbitrary_types_allowed": True}

    agent: Any                  # Agent instance (Any to avoid circular imports)
    event: HookPoint
    inputs: InvokeInputs | ModelCallInputs | ToolCallInputs
    extra: dict[str, Any]       # Shared cross-rail state
```

### Mapping HookPoints to Input Models

| HookPoint | Input Model | Key fields populated |
|-----------|-------------|---------------------|
| `START` | `InvokeInputs` | `input`, `messages` |
| `FINISHED` | `InvokeInputs` | `input`, `result` |
| `ERROR` | `InvokeInputs` | `input`, `result` (exception) |
| `PRE_LLM_CALL` | `ModelCallInputs` | `messages`, `tools` |
| `POST_LLM_CALL` | `ModelCallInputs` | `messages`, `response`, `usage` |
| `PRE_TOOL_CALL` | `ToolCallInputs` | `tool_name`, `arguments` |
| `POST_TOOL_CALL` | `ToolCallInputs` | `tool_name`, `result` |

---

## 4. Rail ABC

```python
class RailAction(StrEnum):
    CONTINUE = "continue"   # Proceed normally
    SKIP = "skip"           # Skip this step (e.g., skip a tool call)
    RETRY = "retry"         # Retry the operation (with RetryRequest)
    ABORT = "abort"         # Abort the agent run

@dataclass
class RetryRequest:
    delay: float = 0.0
    max_retries: int = 1
    reason: str = ""

class RailAbortError(ExoError):
    """Raised when a rail returns ABORT."""

class Rail(ABC):
    name: str
    priority: int = 50          # Lower = runs first

    @abstractmethod
    async def handle(self, ctx: RailContext) -> RailAction | None:
        """Process the event. Return None or CONTINUE to proceed."""
        ...
```

### Priority Ordering

Rails are sorted by `priority` ascending (lower value = higher priority).
Default priority is 50. Security rails (guardrails) should use priority
10-20; logging/observability rails should use 80-90.

### Action Semantics

| Action | Behavior |
|--------|----------|
| `None` / `CONTINUE` | Proceed to next rail, then normal execution |
| `SKIP` | Stop remaining rails, skip the operation |
| `RETRY` | Attach a `RetryRequest` to context, re-execute the operation |
| `ABORT` | Raise `RailAbortError` immediately |

---

## 5. RailManager

```python
class RailManager:
    def __init__(self) -> None:
        self._rails: list[Rail] = []

    def add(self, rail: Rail) -> None: ...
    def remove(self, rail: Rail) -> None: ...
    def clear(self) -> None: ...

    async def run(self, event: HookPoint, **data: Any) -> RailAction:
        """Build RailContext, run rails in priority order, return first
        non-CONTINUE action (or CONTINUE if all pass)."""
        ...
```

### Cross-Rail State

Each `run()` invocation creates a fresh `extra: dict[str, Any]` on the
`RailContext`. All rails in that invocation share the same dict, allowing
upstream rails to pass data to downstream ones.

Example: a rate-limit rail sets `extra["rate_limit_remaining"] = 5` and a
logging rail reads it.

### HookManager Compatibility

`RailManager` can be registered as a plain hook on `HookManager`:

```python
agent.hook_manager.add(HookPoint.PRE_LLM_CALL, rail_manager.run)
```

Because `RailManager.run` matches the `Hook` signature
(`async (**data) -> None`), it integrates seamlessly. The `RailManager`
internally handles priority ordering, typed inputs, and action dispatch.

When a `RailManager` is registered this way:
- It runs alongside any other hooks at that `HookPoint`.
- Registration order determines when the `RailManager` runs relative to
  other hooks (typically registered first).
- If a rail returns `ABORT`, the `RailAbortError` propagates through
  `HookManager` as any exception would.

---

## 6. Agent Integration

```python
class Agent:
    def __init__(
        self,
        *,
        # ... existing params ...
        rails: list[Rail] | None = None,  # NEW
    ) -> None:
        # ... existing init ...

        # Rails (optional typed lifecycle guards)
        if rails:
            self._rail_manager = RailManager()
            for rail in rails:
                self._rail_manager.add(rail)
            # Register rail_manager.run as a hook for all HookPoints
            for point in HookPoint:
                self.hook_manager.add(point, self._rail_manager.run)
```

### Backward Compatibility Guarantees

1. **No rails = no change.** If `rails` is `None` (default), no
   `RailManager` is created, no hooks are added. Behavior is identical to
   the current implementation.

2. **Existing hooks preserved.** Rails register via `hook_manager.add()`,
   so they append to the hook list. Previously registered hooks continue
   to run in their original order.

3. **Existing tests pass unchanged.** The `Agent` constructor signature
   is additive-only (new optional keyword arg). No existing call sites
   break.

4. **Serialization.** Agents with rails cannot be serialized (same as
   agents with hooks today). This is consistent with existing behavior.

---

## 7. Event Flow Diagram

```
Agent.run(input)
  │
  ├─ hook_manager.run(START, ...)
  │    ├─ [plain hooks]
  │    └─ [RailManager.run → sorted rails → action]
  │
  ├─ Agent._call_llm()
  │    ├─ hook_manager.run(PRE_LLM_CALL, ...)
  │    │    ├─ [plain hooks]
  │    │    └─ [RailManager.run → sorted rails → action]
  │    │         ├─ CONTINUE → proceed to LLM call
  │    │         ├─ SKIP → skip this LLM call
  │    │         ├─ RETRY → re-attempt with delay
  │    │         └─ ABORT → raise RailAbortError
  │    │
  │    ├─ provider.complete(...)
  │    │
  │    └─ hook_manager.run(POST_LLM_CALL, ...)
  │         ├─ [plain hooks]
  │         └─ [RailManager.run → sorted rails]
  │
  ├─ Agent._execute_tools()
  │    ├─ hook_manager.run(PRE_TOOL_CALL, ...)
  │    │    └─ [RailManager.run → sorted rails → action]
  │    │         ├─ ABORT → raise RailAbortError
  │    │         └─ SKIP → skip tool execution
  │    │
  │    ├─ tool.execute(...)
  │    │
  │    └─ hook_manager.run(POST_TOOL_CALL, ...)
  │         └─ [RailManager.run → sorted rails]
  │
  └─ hook_manager.run(FINISHED, ...)
       ├─ [plain hooks]
       └─ [RailManager.run → sorted rails]
```

---

## 8. File Layout

All new files live in `packages/exo-core/src/exo/`:

| File | Contents |
|------|----------|
| `rail_types.py` | `InvokeInputs`, `ModelCallInputs`, `ToolCallInputs`, `RailContext` |
| `rail.py` | `Rail`, `RailAction`, `RetryRequest`, `RailAbortError`, `RailManager` |

Tests in `packages/exo-core/tests/`:

| File | Contents |
|------|----------|
| `test_rail_types.py` | Model creation and validation |
| `test_rail.py` | Rail actions, priority ordering, cross-rail state, abort propagation |

---

## 9. Open Questions

1. **RETRY semantics in _call_llm.** The agent already has `max_retries`
   on `_call_llm`. Should rail-requested retries decrement the same counter
   or have their own? **Recommendation:** Separate counter via
   `RetryRequest.max_retries`.

2. **SKIP semantics for LLM calls.** Skipping an LLM call means no
   response is generated. Should this return a sentinel output or raise?
   **Recommendation:** Return `AgentOutput(text="", tool_calls=[])` — the
   loop will terminate since there are no tool calls.

3. **Exception events.** Agent-core has `ON_MODEL_EXCEPTION` and
   `ON_TOOL_EXCEPTION`. Exo has `ERROR` but it is not currently fired.
   **Recommendation:** Defer exception-specific events to a follow-up story;
   wire up `ERROR` as a general exception hook first.

---

## 10. Summary

- Rails **extend** the existing hook system — they do not replace it.
- `RailManager` registers as a single hook on `HookManager`.
- Typed inputs provide structured data; priority ordering ensures
  deterministic execution; `extra` dict enables cross-rail coordination.
- Zero breaking changes to existing APIs, tests, or behavior.
- Implementation spans 2 new files (~400 lines total) + tests.
