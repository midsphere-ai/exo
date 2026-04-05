---
name: exo:context
description: "Use when configuring Exo context management — context_limit, overflow strategy (summarize/truncate/none), cache, ContextConfig, neurons, fork/merge, budget awareness, token tracking. Triggers on: context_limit, overflow, cache, context mode, ContextConfig, history_rounds, summarization, offload, neuron, context fork, budget_awareness, token budget."
---

> **Branch:** These skills are written for the `rename/orbiter-to-exo` branch. The Exo APIs referenced here may differ on other branches.

# Exo Context — Context Management

## When To Use This Skill

Use this skill when the developer needs to:
- Configure how an agent manages conversation history length
- Choose an overflow strategy (summarize, truncate, or none)
- Enable caching of processed context between runs
- Use neurons for modular prompt composition
- Fork/merge context for hierarchical task decomposition
- Enable budget awareness for token tracking
- Disable context management entirely

## Decision Guide

1. **Want sensible defaults?** → Just use `Agent(name="bot")` (default: limit=20, overflow="summarize")
2. **Need a different limit?** → `Agent(name="bot", context_limit=50)`
3. **Want cheap, no-LLM context management?** → `Agent(name="bot", overflow="truncate")`
4. **Want no context management at all?** → `Agent(name="bot", overflow="none")`
5. **Want to persist context between runs?** → `Agent(name="bot", cache=True)`
6. **Need full custom config?** → Use `ContextConfig(limit=..., overflow=..., ...)`
7. **Need modular prompt snippets?** → Use Neurons (composable prompt fragments with priority ordering)
8. **Need hierarchical sub-tasks?** → Use `context.fork(child_name)` and `context.merge(child)`
9. **Want the LLM to see its token usage?** → Set `budget_awareness="per-message"` on Agent
10. **Want to disable context entirely?** → `Agent(context=None)`

## Reference

### Setting Context on Agent

```python
from exo import Agent

# Simple: set a limit (default overflow is "summarize")
agent = Agent(name="bot", context_limit=30)

# With explicit strategy
agent = Agent(name="bot", context_limit=20, overflow="summarize")

# Cheaper: drop old messages instead of summarizing
agent = Agent(name="bot", context_limit=20, overflow="truncate")

# No management: grows until model token limit
agent = Agent(name="bot", overflow="none")

# Persist processed context between runs (avoids re-summarizing)
agent = Agent(name="bot", context_limit=20, cache=True)

# Disable context entirely
agent = Agent(name="bot", context=None)

# Default (if exo-context is installed): limit=20, overflow="summarize"
agent = Agent(name="bot")
```

### Overflow Strategies

| Strategy | What happens at `context_limit` | LLM cost | Good for |
|---|---|---|---|
| `"summarize"` | Oldest messages compressed into a summary, recent kept verbatim | 1 extra LLM call | Agents that need long-term context |
| `"truncate"` | Oldest messages dropped, recent kept | None | Stateless / cost-sensitive agents |
| `"none"` | Nothing -- grows until model token limit | None | Short conversations, manual control |

### ContextConfig (Advanced)

For full control, use `ContextConfig` directly:

```python
from exo.context.config import ContextConfig
from exo.context.context import Context

# New simplified API
config = ContextConfig(
    limit=20,             # Max non-system messages to keep
    overflow="summarize", # Strategy: "summarize", "truncate", or "none"
    keep_recent=5,        # Messages kept verbatim after summarization
    token_pressure=0.8,   # Auto-trigger overflow when token fill exceeds this
    cache=True,           # Persist processed messages between runs
)
ctx = Context(task_id="task-123", config=config)
agent = Agent(name="bot", context=ctx)
```

**Legacy API** (still fully supported):

```python
from exo.context.config import make_config

# Factory with mode presets + overrides
config = make_config(
    "copilot",                    # Base mode (pilot/copilot/navigator)
    history_rounds=30,            # Override: keep 30 rounds
    summary_threshold=15,         # Override: summarize at 15 messages
)
ctx = Context(task_id="task-123", config=config)
agent = Agent(name="bot", context=ctx)

# Mode shorthand
agent = Agent(name="bot", context_mode="navigator")
```

Legacy mode presets:

| Mode | `history_rounds` | `summary_threshold` | `offload_threshold` | `enable_retrieval` | `enable_snapshots` |
|------|-------------------|---------------------|---------------------|--------------------|--------------------
| `pilot` | 100 | 100 | 100 | False | False |
| `copilot` | 20 | 10 | 50 | False | False |
| `navigator` | 10 | 5 | 20 | True | True |

### How Overflow Works

**overflow="summarize"** (default):
1. At `limit`, uses LLM to compress older messages into a `[Conversation Summary]` system message
2. Keeps `keep_recent` most recent messages verbatim
3. Emergency fallback: if messages exceed 2.5x limit, aggressively truncates first
4. **Requires exo-memory installed** -- falls back gracefully if not

**overflow="truncate"**:
- At `limit`, drops oldest non-system messages, keeps the most recent

**overflow="none"**:
- No windowing at all -- conversation grows unbounded

**Token pressure**: When input tokens exceed `token_pressure` ratio (default 0.8) of the model's context window, overflow fires early regardless of message count. This is automatic -- no configuration needed for most users.

### Context Caching (Snapshots)

When `cache=True`, the processed `msg_list` (after summarization, truncation, hook mutations) is persisted at the end of each agent run. On the next run, the cached context is loaded instead of rebuilding from raw history.

```python
# Simple
agent = Agent(name="bot", context_limit=20, cache=True)

# Or via ContextConfig
config = ContextConfig(limit=20, cache=True)
```

**How it works:**
1. **Save**: At end of run, processed messages are serialized and stored
2. **Load**: On next run, if the cache is fresh, it's loaded directly (skipping windowing)
3. **Invalidation**: Cache is invalidated if: new raw items exist, context config changed, or external `messages` parameter is passed

**Restore**: To force rebuild from raw history, call `await agent.clear_snapshot()`.

**Key rules:**
- Instruction SystemMessages excluded (regenerated fresh each run)
- `[Conversation Summary]` SystemMessages ARE included
- `messages` parameter invalidates cache
- `branch()` does not copy cached context
- `spawn_self()` children never load/save cached context
- Requires memory to be configured

### Context Object

```python
from exo.context.context import Context
from exo.context.config import ContextConfig

ctx = Context(
    task_id="task-123",          # Required: unique task identifier
    config=ContextConfig(limit=20),  # Optional: defaults to ContextConfig()
    parent=None,                  # Optional: parent context for fork/merge
    state=None,                   # Optional: initial ContextState
)
```

**Properties:**
- `ctx.config` -- The immutable ContextConfig
- `ctx.state` -- Hierarchical ContextState (key-value store with parent chain)
- `ctx.task_id` -- Task identifier

### Fork and Merge

For hierarchical task decomposition -- child contexts inherit parent state but write in isolation:

```python
# Create a child context
child_ctx = parent_ctx.fork("subtask-research")
# child_ctx.state inherits from parent_ctx.state
# Reads: child sees parent's state entries
# Writes: child's writes are isolated

# ... child executes ...

# Merge child state back into parent
parent_ctx.merge(child_ctx)
# Consolidates child state with net token calculation
```

**Used internally by `spawn_self()`** -- spawned agents get a forked context automatically.

### Neurons

Composable prompt fragments that produce content from context:

```python
from exo.context.neuron import Neuron, neuron_registry
from exo.context.context import Context

class CustomNeuron(Neuron):
    def __init__(self):
        super().__init__("custom", priority=25)

    async def format(self, ctx: Context, **kwargs) -> str:
        # Return empty string to contribute nothing
        return "<custom>\nRelevant context here.\n</custom>"

# Register globally
neuron_registry.register("custom", CustomNeuron())
```

**Built-in neurons (by priority):**

| Priority | Neuron | Content |
|----------|--------|---------|
| 1 | `TaskNeuron` | Task ID, input, plan |
| 2 | `TodoNeuron` | Todo/checklist items |
| 10 | `HistoryNeuron` | Conversation history with windowing |
| 20 | `KnowledgeNeuron` | Knowledge base snippets (`<knowledge>` blocks) |
| 30 | `WorkspaceNeuron` | Workspace artifact summaries |
| 40 | `SkillNeuron` | Available skill descriptions |
| 50 | `FactNeuron` | Extracted facts |
| 60 | `EntityNeuron` | Named entities |
| 100 | `SystemNeuron` | Date, time, platform info |

Lower priority = earlier in the assembled prompt.

### Budget Awareness

Set on Agent (not Context) -- injects token usage info:

```python
# Per-message: LLM sees token usage in system message each call
agent = Agent(
    name="bot",
    budget_awareness="per-message",
)
# System message includes: [Context: 3500/128000 tokens (3% full)]

# Limit: trigger early summarization at a percentage
agent = Agent(
    name="bot",
    budget_awareness="limit:80",
)
# When input tokens exceed 80% of context window, forces summarization
```

**Valid values:** `"per-message"` or `"limit:<0-100>"`.

### Context Tools

When context is set (not None), 7 tools are auto-loaded:
- `add_todo` -- Add a todo/checklist item
- `search_knowledge` -- Search the knowledge base
- `read_file` -- Read a workspace file
- And others

These tools have `_is_context_tool=True` and are:
- Auto-loaded per agent (fresh instances, not shared)
- Excluded from `spawn_self` child agents (children get their own fresh bindings)
- Skipped if a user-registered tool has the same name

## Patterns

### Long-Running Conversation Agent

```python
agent = Agent(
    name="assistant",
    context_limit=50,         # Keep more history
    overflow="summarize",     # Compress old messages
    cache=True,               # Don't re-summarize between runs
)
```

### Aggressive Context for Tool-Heavy Agent

```python
agent = Agent(
    name="researcher",
    context_limit=10,              # Tight window
    overflow="summarize",          # Summarize aggressively
    budget_awareness="limit:70",   # Force summarize at 70% token fill
    tools=[search, fetch, analyze],
)
```

### Cost-Sensitive Agent

```python
agent = Agent(
    name="helper",
    context_limit=20,
    overflow="truncate",     # No LLM calls for context management
)
```

### Disable Context for Stateless Tool

```python
agent = Agent(
    name="calculator",
    context=None,                # No context management
    memory=None,                 # No memory either
    tools=[calculate],
    max_steps=1,                 # Single-shot
)
```

## Gotchas

- **`context_limit`/`overflow`/`cache` cannot be combined with `context=` or `context_mode=`** -- use one approach or the other
- **Summarization requires exo-memory** -- without it installed, the summarize stage is silently skipped
- **Token pressure default is 0.8** (80%) -- when input tokens exceed this ratio of context window, forced summarization fires
- **Context tools have `_is_context_tool=True`** -- they are excluded from `spawn_self` child agents and serialization
- **`context_limit`/`overflow`/`cache` on Swarm propagates to ALL agents** -- overrides their individual context settings
- **Default auto-creation:** If exo-context is installed and no context params are passed, the agent gets limit=20, overflow="summarize" automatically
- **`context=None` vs not passing context:** `None` explicitly disables; not passing triggers auto-creation
- **Cache requires memory persistence** -- `cache=True` has no effect if `memory` is not configured
- **Cache-aware hooks must be idempotent** -- PRE_LLM_CALL hooks that inject messages should check before injecting (use `has_message_content(messages, "MARKER")` from `exo.memory.snapshot`)
- **Cache excludes instruction SystemMessages** -- they are regenerated fresh each run. `[Conversation Summary]` SystemMessages are preserved.
