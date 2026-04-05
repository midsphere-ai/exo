# exo-context

Context engine for the [Exo](../../README.md) multi-agent framework. Provides hierarchical state management, composable prompt building via neurons, event-driven processors, and workspace with artifact versioning.

## Installation

```bash
pip install exo-context
```

Requires Python 3.11+ and `exo-core`.

## Quick Start

```python
from exo import Agent

# Set a conversation limit with automatic summarization
agent = Agent(name="bot", context_limit=30, overflow="summarize")

# Cheaper: just drop old messages
agent = Agent(name="bot", context_limit=20, overflow="truncate")

# Persist context between runs (avoids re-summarizing)
agent = Agent(name="bot", context_limit=20, cache=True)

# No context management
agent = Agent(name="bot", overflow="none")
```

### Overflow Strategies

| Strategy | What happens at limit | LLM cost |
|---|---|---|
| `"summarize"` | Oldest messages compressed into summary, recent kept verbatim | 1 extra call |
| `"truncate"` | Oldest messages dropped, recent kept | None |
| `"none"` | No management -- grows until model token limit | None |

### Advanced: ContextConfig

```python
from exo.context import ContextConfig, Context

config = ContextConfig(
    limit=20,             # Max non-system messages
    overflow="summarize", # "summarize", "truncate", or "none"
    keep_recent=5,        # Messages kept verbatim after summarization
    token_pressure=0.8,   # Auto-trigger overflow at this token fill ratio
    cache=True,           # Persist processed messages between runs
)
ctx = Context(task_id="task-123", config=config)
agent = Agent(name="bot", context=ctx)
```

## What's Included

- **Context** -- core context object with hierarchical state, fork/merge, and lifecycle management.
- **ContextConfig** -- configuration with `limit`, `overflow`, `cache`, and advanced tuning.
- **OverflowStrategy** -- enum: `summarize`, `truncate`, `none`.
- **ContextState** -- hierarchical key-value store with parent inheritance.
- **PromptBuilder** -- composable prompt construction from prioritized neurons.
- **Neurons** -- 9 built-in neurons: System, Task, History, Todo, Knowledge, Workspace, Skill, Fact, Entity.
- **Processors** -- event-driven pipeline with built-in Summarize and ToolResultOffloader processors.
- **Workspace** -- versioned artifact storage with filesystem persistence and observer pattern.
- **TokenTracker** -- per-agent per-step token usage tracking.
- **Checkpoint** -- save/restore state for long-running tasks.
- **Context Tools** -- planning, knowledge, and file tools for agents.

## Public API

```python
from exo.context import (
    Context,             # Core context object
    ContextConfig,       # Configuration (limit, overflow, cache, ...)
    OverflowStrategy,    # Enum: SUMMARIZE, TRUNCATE, NONE
    ContextState,        # Hierarchical key-value state
    PromptBuilder,       # Composable prompt builder
    Neuron,              # Base class for prompt neurons
    ContextProcessor,    # Base class for processors
    ProcessorPipeline,   # Chain of processors
    Workspace,           # Artifact storage with versioning
    TokenTracker,        # Token usage tracking
    Checkpoint,          # State save/restore
)
```

## Documentation

- [Context Engine Guide](../../docs/guides/context/)
- [API Reference](../../docs/reference/context/)
