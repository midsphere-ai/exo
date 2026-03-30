---
name: exo:spawn
description: "Use when an Exo agent needs to dynamically spawn copies of itself for parallel sub-tasks — allow_self_spawn, spawn_self() tool, max_spawn_depth, depth guards, memory isolation, context forking. Triggers on: spawn_self, allow_self_spawn, self spawn, max_spawn_depth, parallel sub-task, agent clone, dynamic sub-agent."
---

# Exo Spawn — Self-Spawning Agents

## When To Use This Skill

Use this skill when the developer needs to:
- Let an agent dynamically spawn copies of itself at runtime
- Delegate parallel sub-tasks from within an agent's execution
- Configure spawn depth limits
- Understand memory isolation between parent and spawned children
- Understand context forking for spawned agents

## Decision Guide

1. **Does the agent need to delegate parallel work to itself?** → Set `allow_self_spawn=True`
2. **How deep should recursive spawning go?** → Set `max_spawn_depth` (default 3; root agent at depth 0)
3. **Does the child need to share long-term knowledge?** → Automatic: shared long-term memory, fresh short-term
4. **Should children be able to spawn further?** → Default: No. Children get `allow_self_spawn=False`. For multi-level, parent must have `max_spawn_depth > 2`.

## Reference

### Enabling Self-Spawn

```python
from exo import Agent

agent = Agent(
    name="coordinator",
    model="openai:gpt-4o",
    instructions=(
        "You are a research coordinator. When asked about multiple topics, "
        "use spawn_self to delegate each topic as a separate sub-task. "
        "After all spawn_self calls return, synthesize the results."
    ),
    tools=[search, summarize],
    allow_self_spawn=True,       # Registers spawn_self(task) tool automatically
    max_spawn_depth=2,           # Root (depth 0) + 1 level of children (depth 1)
)
```

**Agent parameters:**
- `allow_self_spawn: bool = False` — When `True`, registers the `spawn_self(task: str) -> str` tool
- `max_spawn_depth: int = 3` — Maximum recursive depth. Default allows root + 2 levels.

### spawn_self(task: str) -> str

Auto-registered tool that the LLM calls to spawn a child agent:

**What happens when the LLM calls `spawn_self(task)`:**

1. **Depth check:** If `parent._spawn_depth >= max_spawn_depth`, returns error string (no exception)
2. **Provider check:** Verifies the parent's `_current_provider` is available
3. **Child agent creation:**
   - Same `model`, `instructions`, `temperature`, `max_tokens`, `max_steps`
   - Parent's tools **minus** `spawn_self` and context tools
   - Fresh `ShortTermMemory` (conversation isolation)
   - Shared `LongTermMemory` instance (knowledge accumulates across spawns)
   - Forked context via `parent.context.fork(child_name)` (reads inherited, writes isolated)
   - `allow_self_spawn=False` — children cannot spawn further
   - `_spawn_depth = parent._spawn_depth + 1`
4. **Execution:** Runs child agent on `task` with same provider
5. **Returns:** `result.text` (string)

### Child Agent Properties

| Property | Inherited From Parent? | Details |
|----------|----------------------|---------|
| `model` | Yes | Same model string |
| `instructions` | Yes | Same system prompt |
| `tools` | Yes, minus exclusions | Excludes `spawn_self` and `_is_context_tool` tools |
| `temperature` | Yes | Same value |
| `max_tokens` | Yes | Same value |
| `max_steps` | Yes | Same value |
| `allow_self_spawn` | No | Always `False` |
| `_spawn_depth` | Incremented | `parent._spawn_depth + 1` |
| Short-term memory | No | Fresh `ShortTermMemory()` instance |
| Long-term memory | Yes | Shared instance (same reference) |
| Context | Forked | `parent.context.fork(child_name)` |

### Child Naming

Children are named: `{parent.name}_spawn_{uuid_hex[:8]}`

Example: `coordinator_spawn_a3f1b2c4`

### Memory Isolation

```
Parent Agent
├── ShortTermMemory: [msg1, msg2, msg3, ...]  ← parent's conversation
├── LongTermMemory: shared_store               ← shared across all
│
├── Child A (spawn_self("Research topic A"))
│   ├── ShortTermMemory: []                    ← fresh, empty
│   └── LongTermMemory: shared_store           ← same reference
│
└── Child B (spawn_self("Research topic B"))
    ├── ShortTermMemory: []                    ← fresh, empty
    └── LongTermMemory: shared_store           ← same reference
```

- **Short-term:** Each child gets a brand new `ShortTermMemory` — no access to parent's or sibling's conversation
- **Long-term:** All share the same `LongTermMemory` instance — writes by child A are visible to child B via `search()`

### Error Handling

All errors return strings (no exceptions raised to the LLM):

```
"[spawn_self error] Maximum spawn depth (2) reached. Cannot spawn further sub-agents."
"[spawn_self error] No provider available for spawned agent."
```

The LLM sees these as tool results and can react accordingly.

### Provider Lifecycle

```python
# Inside Agent.run():
self._current_provider = provider  # Set before execution
try:
    return await self._run_inner(...)
finally:
    self._current_provider = None  # Cleaned up after
```

The `spawn_self` tool closure captures `parent` and accesses `parent._current_provider`. This is safe because spawn_self only runs during the parent's `run()` execution.

### Depth Guard

```
max_spawn_depth=3 (default)

Depth 0: Root agent        → can spawn (0 < 3)
Depth 1: Child of root     → can spawn (1 < 3) IF allow_self_spawn=True on child
Depth 2: Grandchild        → can spawn (2 < 3) IF allow_self_spawn=True
Depth 3: Great-grandchild  → blocked (3 >= 3), returns error string
```

**Important:** By default, children get `allow_self_spawn=False`, so only the root can spawn. For multi-level spawning, you would need to create children with `allow_self_spawn=True` manually (the auto-spawned children do not get this).

## Patterns

### Parallel Research Coordinator

```python
agent = Agent(
    name="researcher",
    model="openai:gpt-4o",
    instructions=(
        "You research topics by delegating to sub-agents. "
        "For each distinct sub-topic, call spawn_self with a focused task. "
        "After all calls return, synthesize findings into a report."
    ),
    tools=[web_search, fetch_page],
    allow_self_spawn=True,
    max_spawn_depth=2,
    memory=None,    # Disable memory for stateless research
    context=None,   # Disable context for simplicity
)

result = await run(
    agent,
    "Compare AI chip architectures: NVIDIA H100, Google TPU v5, AMD MI300X",
    provider=provider,
)
# LLM calls spawn_self 3 times (once per chip)
# Each child researches one chip independently
# Parent synthesizes all results
```

### Data Processing Fan-Out

```python
agent = Agent(
    name="processor",
    instructions=(
        "Process each data item by spawning a sub-agent. "
        "Use spawn_self(task='Process item: <item>') for each."
    ),
    tools=[validate, transform, store],
    allow_self_spawn=True,
    max_spawn_depth=2,
)
```

### Spawn with Explicit Memory Control

```python
from exo.memory.base import AgentMemory
from exo.memory.short_term import ShortTermMemory
from exo.memory.backends.vector import ChromaVectorMemoryStore, OpenAIEmbeddingProvider

# Shared long-term store for knowledge accumulation
shared_lt = ChromaVectorMemoryStore(OpenAIEmbeddingProvider())

agent = Agent(
    name="coordinator",
    memory=AgentMemory(
        short_term=ShortTermMemory(),
        long_term=shared_lt,
    ),
    allow_self_spawn=True,
)
# Children will share shared_lt but get fresh ShortTermMemory
```

## Gotchas

- **Children get `allow_self_spawn=False`** — only the parent can spawn. Children cannot spawn further by default.
- **Spawned agents run sequentially** from the parent's perspective — each `spawn_self(task)` call awaits completion before returning. The LLM can issue multiple spawn_self calls in one step, but they execute via the normal parallel tool execution path.
- **Context tools are excluded** from children's tool sets — children get fresh context tool bindings via their own `__init__`
- **`retrieve_artifact` is also excluded** — auto-registered by child's `__init__` if any `large_output=True` tool exists
- **Guide the LLM via instructions** — the LLM decides when to spawn. Be explicit: "use spawn_self for each country separately"
- **Memory=None propagates** — if parent has `memory=None`, children also get `memory=None`
- **Context fork fallback** — if `parent.context.fork()` fails, child shares parent's context (fallback, not isolation)
- **Provider is shared** — parent and all children use the same provider instance. Rate limits and quotas are shared.
