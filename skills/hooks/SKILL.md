---
name: exo:hooks
description: "Use when intercepting Exo agent lifecycle events — HookPoint, HookManager, pre/post LLM/tool hooks, error hooks, runtime mutation (add_tool, remove_tool, add_handoff, add_mcp_server, inject_message). Triggers on: hook, HookPoint, lifecycle, pre_llm_call, post_tool_call, agent hook, runtime mutation, add_tool, inject_message, dynamic tool."
---

> **Branch:** These skills are written for the `rename/orbiter-to-exo` branch. The Exo APIs referenced here may differ on other branches.

# Exo Hooks — Lifecycle Hooks & Runtime Mutation

## When To Use This Skill

Use this skill when the developer needs to:
- Intercept agent execution at specific lifecycle points (start, pre/post LLM, pre/post tool, finish, error)
- Add logging, metrics, or observability to agent execution
- Dynamically add/remove tools, handoffs, or MCP servers during a run
- Inject messages into a running agent
- Understand the hook execution model (sequential, exceptions propagate)

## Decision Guide

1. **Need to observe/log lifecycle events?** → Register hooks at relevant `HookPoint`s
2. **Need to block execution on a condition?** → Raise an exception from a hook (it propagates immediately)
3. **Need security guardrails?** → Use Rails instead (see `exo:guardrails` skill) — they offer CONTINUE/SKIP/RETRY/ABORT actions
4. **Need fire-and-forget observation?** → Wrap hook body in try/except (hooks abort the run on exception by default)
5. **Need to add tools during execution?** → Use `agent.add_tool()` (async-safe)
6. **Need to steer a running agent?** → Use `agent.inject_message(content)`

## Reference

### Hook Points

| HookPoint | Fires When | Data Kwargs |
|-----------|------------|-------------|
| `START` | Agent execution begins | `agent_name: str`, `input: str` |
| `PRE_LLM_CALL` | Before each LLM invocation | `agent_name: str`, `messages: list`, `tools: list` |
| `POST_LLM_CALL` | After LLM returns | `agent_name: str`, `response: Any` |
| `PRE_TOOL_CALL` | Before tool execution | `agent_name: str`, `tool_name: str`, `arguments: dict` |
| `POST_TOOL_CALL` | After tool returns | `agent_name: str`, `tool_name: str`, `result: Any` |
| `FINISHED` | Execution completed successfully | `agent_name: str`, `output: Any` |
| `ERROR` | Exception occurred | `agent_name: str`, `error: Exception` |
| `CONTEXT_WINDOW` | Context windowing runs | `agent: Agent`, `messages: list`, `info: ContextWindowInfo`, `provider: Any`, `actions: list` |

### Registering Hooks on Agent

```python
from exo import Agent, HookPoint
from typing import Any

async def log_start(agent_name: str, input: str, **_: Any) -> None:
    print(f"[{agent_name}] Starting with: {input[:50]}")

async def log_tool(agent_name: str, tool_name: str, result: Any, **_: Any) -> None:
    print(f"[{agent_name}] Tool {tool_name} returned: {str(result)[:100]}")

async def log_error(agent_name: str, error: Exception, **_: Any) -> None:
    print(f"[{agent_name}] ERROR: {error}")

agent = Agent(
    name="bot",
    hooks=[
        (HookPoint.START, log_start),
        (HookPoint.POST_TOOL_CALL, log_tool),
        (HookPoint.ERROR, log_error),
    ],
)
```

**Hook function signature:** `async def my_hook(**data: Any) -> None`
- Must be async
- Receives kwargs matching the hook point's data (see table above)
- Use `**_` or `**data` to absorb extra kwargs for forward compatibility

### HookManager API

Exposed as `agent.hook_manager`:

```python
from exo.hooks import HookManager, HookPoint

manager = agent.hook_manager

# Add a hook
manager.add(HookPoint.PRE_LLM_CALL, my_hook)

# Remove a hook (silently no-ops if not found)
manager.remove(HookPoint.PRE_LLM_CALL, my_hook)

# Check if any hooks registered
if manager.has_hooks(HookPoint.ERROR):
    print("Error hooks are registered")

# Remove all hooks at all points
manager.clear()

# Fire hooks manually (called by runtime — rarely needed directly)
await manager.run(HookPoint.START, agent_name="bot", input="Hello")
```

### Execution Semantics

- **Sequential:** Hooks run in registration order at each point
- **Exceptions propagate:** A failing hook aborts the agent run immediately
- **No suppression:** Unlike `EventBus`, hook exceptions are never swallowed
- **All hooks are async:** Must be `async def`

### Runtime Mutation API

Dynamically modify an agent during execution:

```python
# Add a tool (async-safe via _tools_lock)
await agent.add_tool(new_tool)

# Remove a tool
agent.remove_tool("old_tool_name")

# Add a handoff target
await agent.add_handoff(target_agent)

# Add an MCP server (connects, lists tools, registers them)
from exo.mcp import MCPServerConfig
config = MCPServerConfig(name="server", command="npx", args=["..."])
await agent.add_mcp_server(config)
```

**Thread safety:** `add_tool()` and `add_mcp_server()` use `agent._tools_lock` (asyncio.Lock) for concurrent safety. `remove_tool()` is safe without the lock (dict.pop is atomic in CPython's single-threaded event loop).

### inject_message()

Push a user message into a running agent's context, picked up before the next LLM call:

```python
# From another coroutine:
agent.inject_message("New priority: focus on security aspects first.")
```

**Behavior:**
- Message added to `agent._injected_messages` queue (asyncio.Queue)
- Drained before each LLM call in the agent's run loop
- Each injected message becomes a `UserMessage` in the conversation
- Emits `MessageInjectedEvent` in streaming mode
- Raises `ValueError` if content is empty

**Use case:** Live steering of a running agent from external code (e.g., a web socket handler, a monitoring system, or a parent agent).

**Snapshot interaction:** Injected messages become part of the context snapshot at end-of-run (when `enable_snapshots=True`). They will persist across runs without needing re-injection.

## Patterns

### Execution Timer

```python
import time

_start_times: dict[str, float] = {}

async def on_start(agent_name: str, **_):
    _start_times[agent_name] = time.monotonic()

async def on_finish(agent_name: str, **_):
    elapsed = time.monotonic() - _start_times.pop(agent_name, 0)
    print(f"[{agent_name}] Completed in {elapsed:.2f}s")

agent = Agent(
    name="bot",
    hooks=[
        (HookPoint.START, on_start),
        (HookPoint.FINISHED, on_finish),
    ],
)
```

### Token Usage Logger

```python
async def log_usage(agent_name: str, response: Any, **_):
    usage = getattr(response, "usage", None)
    if usage:
        print(f"[{agent_name}] Tokens: {usage.input_tokens}in/{usage.output_tokens}out")

agent = Agent(name="bot", hooks=[(HookPoint.POST_LLM_CALL, log_usage)])
```

### Tool Call Auditor

```python
audit_log: list[dict] = []

async def audit_tool(agent_name: str, tool_name: str, arguments: dict, **_):
    audit_log.append({
        "agent": agent_name,
        "tool": tool_name,
        "args": arguments,
        "timestamp": time.time(),
    })

agent = Agent(name="bot", hooks=[(HookPoint.PRE_TOOL_CALL, audit_tool)])
```

### Conditional Tool Blocking (Simple)

```python
BLOCKED_TOOLS = {"delete_file", "drop_table"}

async def block_dangerous(agent_name: str, tool_name: str, **_):
    if tool_name in BLOCKED_TOOLS:
        raise RuntimeError(f"Tool '{tool_name}' is blocked by policy")

agent = Agent(name="bot", hooks=[(HookPoint.PRE_TOOL_CALL, block_dangerous)])
```

For more sophisticated blocking with SKIP/RETRY/ABORT actions, use the Rail system (see `exo:guardrails`).

### Dynamic Tool Loading

```python
async def maybe_load_tools(agent_name: str, input: str, **_):
    """Load specialized tools based on the user's query."""
    if "code" in input.lower():
        await agent.add_tool(python_executor)
    if "search" in input.lower():
        await agent.add_tool(web_search)

agent = Agent(name="bot", hooks=[(HookPoint.START, maybe_load_tools)])
```

### Snapshot-Aware Context Injection (Idempotent)

```python
from exo.memory.snapshot import has_message_content

MARKER = "[INJECTED_CONTEXT]"

async def inject_context(agent, messages, **_):
    """Inject persistent context — safe with snapshots enabled."""
    if not has_message_content(messages, MARKER):
        messages.insert(1, UserMessage(content=f"{MARKER}\n{load_my_data()}"))

agent = Agent(
    name="bot",
    hooks=[(HookPoint.PRE_LLM_CALL, inject_context)],
    context_mode="navigator",  # snapshots enabled
)
```

When `enable_snapshots=True`, hook-injected messages are saved in the snapshot and loaded on the next run. The idempotency check prevents duplicate injection.

### Custom Context Management (overflow="hook")

```python
from exo import Agent, HookPoint
from exo.context.info import ContextWindowInfo
from exo.types import SystemMessage

async def token_aware_window(*, messages, info: ContextWindowInfo, **_):
    """Keep messages until we're under 60% of context window."""
    if info.fill_ratio < 0.6:
        return  # plenty of room
    system = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    # Estimate tokens per message from last call
    avg_tokens = info.input_tokens / max(len(non_system), 1)
    target_count = int((info.context_window * 0.5) / avg_tokens) if info.context_window else info.limit
    messages.clear()
    messages.extend(system + non_system[-target_count:])

agent = Agent(
    name="bot",
    overflow="hook",
    hooks=[(HookPoint.CONTEXT_WINDOW, token_aware_window)],
)
```

### Context Window Hook with ContextWindowHook ABC

```python
from exo.context.hook import ContextWindowHook
from exo.context.info import ContextWindowInfo

class TrajectoryAwareCompressor(ContextWindowHook):
    """Compress aggressively if token growth predicts overflow."""

    async def window(self, *, messages, info: ContextWindowInfo, **_):
        if len(info.trajectory) < 2 or not info.context_window:
            return
        growth = info.trajectory[-1].prompt_tokens - info.trajectory[-2].prompt_tokens
        steps_left = info.max_steps - info.step
        projected = info.input_tokens + (growth * min(steps_left, 3))
        if projected > info.context_window * 0.9:
            system = [m for m in messages if isinstance(m, SystemMessage)]
            non_system = [m for m in messages if not isinstance(m, SystemMessage)]
            messages.clear()
            messages.extend(system + non_system[-info.keep_recent:])

agent = Agent(
    name="bot",
    overflow="hook",
    hooks=[(HookPoint.CONTEXT_WINDOW, TrajectoryAwareCompressor())],
)
```

### Post-Windowing Augmentation (pin messages after summarization)

```python
async def pin_tool_results(*, messages, info, **_):
    """After built-in summarization, ensure important messages survive."""
    # This fires AFTER overflow="summarize" runs
    # info.total_messages reflects post-windowing state
    pass  # ... re-inject pinned messages from external store ...

agent = Agent(
    name="bot",
    overflow="summarize",  # built-in runs first
    hooks=[(HookPoint.CONTEXT_WINDOW, pin_tool_results)],
)
```

### Fire-and-Forget Hook (Exception-Safe)

```python
async def safe_metric(agent_name: str, **data):
    """Send metrics without risking agent abort."""
    try:
        await metrics_client.record(agent_name, data)
    except Exception:
        pass  # Swallow — don't abort the agent run

agent = Agent(name="bot", hooks=[(HookPoint.FINISHED, safe_metric)])
```

## Gotchas

- **Hooks abort on exception** — a failing hook stops the agent run. Wrap in try/except for fire-and-forget behavior.
- **`_has_user_hooks`** tracks whether the user explicitly provided hooks (vs auto-attached memory/rail hooks). Useful for debugging.
- **Memory persistence hooks are auto-attached** — when `memory` is provided, `POST_LLM_CALL` and `POST_TOOL_CALL` hooks are auto-registered for memory persistence. You don't need to wire this manually.
- **Rails also register as hooks** — `Agent(rails=[...])` creates hooks at ALL hook points. Rails hooks run after user hooks.
- **Tool schemas are re-enumerated each step** — so dynamically added/removed tools take effect on the next LLM call, not retroactively.
- **inject_message is async-safe** — uses `asyncio.Queue.put_nowait()`, can be called from any coroutine.
- **Hook data kwargs may evolve** — always use `**_` or `**data` to absorb unknown kwargs for forward compatibility.
- **`CONTEXT_WINDOW` hook has two modes**: With `overflow="hook"`, the hook replaces built-in windowing entirely. With `overflow="summarize"` or `"truncate"`, the hook fires after built-in windowing as an augmentation pass.
- **`CONTEXT_WINDOW` hook receives `ContextWindowInfo`**: A frozen dataclass with step number, message counts, token pressure (fill_ratio, context_window, input/output tokens), cumulative trajectory, config values, and agent identity. Import from `exo.context.info`.
- **`CONTEXT_WINDOW` hook mutates `messages` in place**: The `messages` kwarg is a mutable list. Modify it directly (`.clear()`, `.extend()`, etc.). Return value is ignored (hooks return None).
- **`CONTEXT_WINDOW` hook can report actions**: Append `_ContextAction` instances to the `actions` list to emit `ContextEvent`s in streaming mode.
- **PRE_LLM_CALL mutations persist in snapshots** — when `enable_snapshots=True`, messages injected by PRE_LLM_CALL hooks become part of the snapshot. On the next run, the hook fires again on the snapshot-loaded messages. Hooks that inject messages must be idempotent (check before injecting) to avoid duplicates. Use `has_message_content()` from `exo.memory.snapshot`.
- **inject_message persists in snapshots** — injected messages are part of the final `msg_list` and get saved in the snapshot. They survive across runs without re-injection.
