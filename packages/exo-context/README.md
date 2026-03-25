# exo-context

Context engine for the [Exo](../../README.md) multi-agent framework. Provides hierarchical state management, composable prompt building via neurons, event-driven processors, and workspace with artifact versioning.

## Installation

```bash
pip install exo-context
```

Requires Python 3.11+ and `exo-core`.

## What's Included

- **Context** -- core context object with hierarchical state, fork/merge, and lifecycle management.
- **ContextState** -- hierarchical key-value store with parent inheritance.
- **PromptBuilder** -- composable prompt construction from prioritized neurons.
- **Neurons** -- 9 built-in neurons: System, Task, History, Todo, Knowledge, Workspace, Skill, Fact, Entity.
- **Processors** -- event-driven pipeline with built-in Summarize and ToolResultOffloader processors.
- **Workspace** -- versioned artifact storage with filesystem persistence and observer pattern.
- **TokenTracker** -- per-agent per-step token usage tracking.
- **Checkpoint** -- save/restore state for long-running tasks.
- **Context Tools** -- planning, knowledge, and file tools for agents.
- **DynamicVariables** -- template variable resolution for prompt templates.

## Quick Example

```python
from exo.context import Context, ContextConfig

config = ContextConfig(automation_mode="copilot")
ctx = Context(config=config)

# Set state values
ctx.state.set("user_name", "Alice")
ctx.state.set("task", "Research quantum computing")

# Build prompts from neurons
messages = ctx.build_prompt(agent_name="researcher")

# Store artifacts in workspace
ctx.workspace.store("notes.md", "# Research Notes\n...")
```

## Automation Modes

| Mode | Description |
|------|-------------|
| `pilot` | Fully autonomous -- agent decides everything |
| `copilot` | Agent proposes, human approves critical steps |
| `navigator` | Human guides, agent executes |

## Public API

```python
from exo.context import (
    Context,             # Core context object
    ContextConfig,       # Configuration with automation mode
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
