---
name: exo:context
description: "Use when configuring Exo context management — automation modes (pilot/copilot/navigator), ContextConfig, history windowing, summarization, offloading, neurons, fork/merge, budget awareness, token tracking. Triggers on: context mode, ContextConfig, history_rounds, summarization, offload, neuron, context fork, budget_awareness, token budget."
---

> **Branch:** These skills are written for the `rename/orbiter-to-exo` branch. The Exo APIs referenced here may differ on other branches.

# Exo Context — Context Management

## When To Use This Skill

Use this skill when the developer needs to:
- Configure how an agent manages conversation history length
- Choose an automation mode (pilot, copilot, navigator)
- Tune summarization and offloading thresholds
- Use neurons for modular prompt composition
- Fork/merge context for hierarchical task decomposition
- Enable budget awareness for token tracking
- Disable context management entirely

## Decision Guide

1. **Want minimal automation, manual control?** → `context_mode="pilot"` (keeps 100 rounds, no auto-summarization)
2. **Want sensible defaults?** → `context_mode="copilot"` (default — 20 rounds, summarize at 10, offload at 50)
3. **Want aggressive context management?** → `context_mode="navigator"` (10 rounds, summarize at 5, offload at 20, retrieval enabled)
4. **Need custom thresholds?** → Use `make_config(mode, history_rounds=..., ...)` and pass explicit `Context`
5. **Need modular prompt snippets?** → Use Neurons (composable prompt fragments with priority ordering)
6. **Need hierarchical sub-tasks?** → Use `context.fork(child_name)` and `context.merge(child)`
7. **Want the LLM to see its token usage?** → Set `budget_awareness="per-message"` on Agent
8. **Want to disable context entirely?** → `Agent(context=None)`

## Reference

### Automation Modes

| Mode | `history_rounds` | `summary_threshold` | `offload_threshold` | `enable_retrieval` | Use Case |
|------|-------------------|---------------------|---------------------|--------------------|----------|
| `pilot` | 100 | 100 | 100 | False | Long conversations, manual control |
| `copilot` | 20 | 10 | 50 | False | General-purpose (default) |
| `navigator` | 10 | 5 | 20 | True | Agentic workflows, heavy tool use |

### Setting Context on Agent

```python
from exo import Agent

# Option 1: Mode string shorthand (creates Context internally)
agent = Agent(name="bot", context_mode="navigator")

# Option 2: Explicit Context object
from exo.context.config import make_config
from exo.context.context import Context

config = make_config("copilot", history_rounds=30, summary_threshold=15)
ctx = Context(task_id="task-123", config=config)
agent = Agent(name="bot", context=ctx)

# Option 3: Disable context entirely
agent = Agent(name="bot", context=None)

# Default (if exo-context is installed): auto-creates copilot mode
agent = Agent(name="bot")  # context = Context(mode="copilot")
```

**Precedence:** Explicit `context=` takes precedence over `context_mode=`. Both unset triggers auto-creation.

### ContextConfig

```python
from exo.context.config import ContextConfig, make_config

# Factory with mode presets + overrides
config = make_config(
    "copilot",                    # Base mode
    history_rounds=30,            # Override: keep 30 rounds
    summary_threshold=15,         # Override: summarize at 15 messages
    offload_threshold=60,         # Override: offload at 60 messages
    enable_retrieval=True,        # Override: enable RAG retrieval
    neuron_names=("system", "task", "knowledge"),  # Select neurons
    token_budget_trigger=0.75,    # Override: trigger at 75% context fill
)

# Or build directly
config = ContextConfig(
    mode="copilot",
    history_rounds=20,
    summary_threshold=10,
    offload_threshold=50,
    enable_retrieval=False,
    neuron_names=(),
    token_budget_trigger=0.8,
    extra={},                     # Extensible metadata for custom processors
)
```

**Validation:** `summary_threshold` must be <= `offload_threshold`.

### Three-Stage Windowing Pipeline

Applied before each LLM call, in order:

**Stage 1 — Offload** (when messages > `offload_threshold`):
- Aggressively trims to `summary_threshold` messages
- Keeps the most recent messages
- Emits `ContextEvent(action="offload")`

**Stage 2 — Summarize** (when messages >= `summary_threshold`):
- Uses LLM-powered summarization via exo-memory
- Compresses older messages into a `[Conversation Summary]` system message
- Keeps `keep_recent` most recent messages intact
- Emits `ContextEvent(action="summarize")`
- **Requires exo-memory installed** — falls back gracefully if not

**Stage 3 — Window** (always applied):
- Keeps last `history_rounds` messages
- Emits `ContextEvent(action="window")`

### Context Object

```python
from exo.context.context import Context
from exo.context.config import make_config

ctx = Context(
    task_id="task-123",          # Required: unique task identifier
    config=make_config("copilot"),  # Optional: defaults to ContextConfig()
    parent=None,                  # Optional: parent context for fork/merge
    state=None,                   # Optional: initial ContextState
)
```

**Properties:**
- `ctx.config` — The immutable ContextConfig
- `ctx.state` — Hierarchical ContextState (key-value store with parent chain)
- `ctx.task_id` — Task identifier

### Fork and Merge

For hierarchical task decomposition — child contexts inherit parent state but write in isolation:

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

**Used internally by `spawn_self()`** — spawned agents get a forked context automatically.

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

**Selecting neurons:** Use `ContextConfig(neuron_names=("system", "task", "knowledge"))` to pick which neurons contribute.

### Budget Awareness

Set on Agent (not Context) — injects token usage info:

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
- `add_todo` — Add a todo/checklist item
- `search_knowledge` — Search the knowledge base
- `read_file` — Read a workspace file
- And others

These tools have `_is_context_tool=True` and are:
- Auto-loaded per agent (fresh instances, not shared)
- Excluded from `spawn_self` child agents (children get their own fresh bindings)
- Skipped if a user-registered tool has the same name

## Patterns

### Long-Running Conversation Agent

```python
config = make_config(
    "copilot",
    history_rounds=50,           # Keep more history
    summary_threshold=30,        # Summarize later
    offload_threshold=100,       # Offload much later
)
ctx = Context(task_id="chat-session", config=config)
agent = Agent(name="assistant", context=ctx)
```

### Aggressive Context for Tool-Heavy Agent

```python
agent = Agent(
    name="researcher",
    context_mode="navigator",    # Aggressive: 10 rounds, summarize at 5
    budget_awareness="limit:70", # Force summarize at 70% token fill
    tools=[search, fetch, analyze],
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

- **`summary_threshold` must be <= `offload_threshold`** — validated at creation, raises `ValueError`
- **Summarization requires exo-memory** — without it installed, the summarize stage is silently skipped
- **`token_budget_trigger` default is 0.8** (80%) — when input tokens exceed this ratio of context window, forced summarization fires
- **Context tools have `_is_context_tool=True`** — they are excluded from `spawn_self` child agents and serialization
- **`context_mode` on Swarm propagates to ALL agents** — overrides their individual context settings
- **Default auto-creation:** If exo-context is installed and no `context`/`context_mode` is passed, the agent gets `copilot` mode automatically
- **`context=None` vs not passing context:** `None` explicitly disables; not passing triggers auto-creation
