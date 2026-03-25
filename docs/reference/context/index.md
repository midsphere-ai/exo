# exo-context

Hierarchical state, prompt building, processors, and workspace for the Exo framework.

## Module Path

```
exo.context
```

## Installation

```bash
pip install exo-context
```

## Overview

The `exo-context` package provides the context engine that manages per-task state, prompt composition, context processing, artifact storage, and checkpointing. It implements a hierarchical state model where child contexts inherit from parents, a composable neuron-based prompt builder, an event-driven processor pipeline, and a versioned workspace for artifact management.

## Exports

| Export | Type | Description |
|---|---|---|
| `Context` | Class | Per-task context with fork/merge lifecycle |
| `ContextError` | Exception | Context lifecycle errors |
| `ContextConfig` | Pydantic model | Immutable context configuration |
| `AutomationMode` | StrEnum | Automation level (pilot/copilot/navigator) |
| `make_config` | Function | Factory for preset ContextConfig |
| `ContextState` | Class | Hierarchical key-value state |
| `Neuron` | ABC | Composable prompt fragment producer |
| `neuron_registry` | `Registry[Neuron]` | Global neuron instance registry |
| `PromptBuilder` | Class | Composes neurons into system prompts |
| `ContextProcessor` | ABC | Event-driven context processor |
| `ProcessorPipeline` | Class | Registers and fires processors by event |
| `SummarizeProcessor` | Class | Marks context for summarization |
| `ToolResultOffloader` | Class | Offloads large tool results |
| `Workspace` | Class | Persistent artifact storage with versioning |
| `ArtifactType` | StrEnum | Artifact classification |
| `Checkpoint` | Dataclass | Immutable state snapshot |
| `CheckpointStore` | Class | Per-session checkpoint management |
| `TokenTracker` | Class | Per-agent token usage tracking |
| `get_context_tools` | Function | All context tools (planning + knowledge + file) |
| `get_planning_tools` | Function | Todo/checklist tools |
| `get_knowledge_tools` | Function | Knowledge retrieval tools |
| `get_file_tools` | Function | File reading tools |

## Import Patterns

```python
# Import from package root
from exo.context import (
    Context, ContextConfig, ContextState,
    PromptBuilder, Neuron, neuron_registry,
    Workspace, ArtifactType,
    Checkpoint, CheckpointStore,
    TokenTracker,
)

# Import specific modules
from exo.context.context import Context, ContextError
from exo.context.config import ContextConfig, AutomationMode, make_config
from exo.context.state import ContextState
from exo.context.neuron import Neuron, neuron_registry
from exo.context.prompt_builder import PromptBuilder
from exo.context.processor import ContextProcessor, ProcessorPipeline
from exo.context.workspace import Workspace, ArtifactType
from exo.context.tools import get_context_tools
```

## Quick Example

```python
import asyncio
from exo.context import (
    Context, ContextConfig, AutomationMode,
    PromptBuilder, Workspace,
)

async def main():
    # Create context with navigator automation
    ctx = Context("task-1", config=ContextConfig(mode=AutomationMode.NAVIGATOR))

    # Set task state
    ctx.state.set("task_input", "Write a Python function to sort a list")

    # Build prompt from neurons
    builder = PromptBuilder(ctx)
    builder.add("task").add("system")
    prompt = await builder.build()
    print(prompt)

    # Fork for sub-task
    child = ctx.fork("subtask-1")
    child.state.set("progress", "step 1 done")

    # Merge back
    ctx.merge(child)

    # Workspace for artifacts
    ws = Workspace("ws-1")
    await ws.write("solution.py", "def sort_list(lst): return sorted(lst)")
    print(ws.read("solution.py"))

asyncio.run(main())
```

## Architecture

```
Context
  |-- ContextConfig (immutable settings)
  |-- ContextState (hierarchical key-value, parent chain)
  |-- CheckpointStore (save/restore snapshots)
  |
  |-- fork() --> child Context (inherited state)
  |-- merge() <-- child state consolidated back

PromptBuilder
  |-- add("neuron_name") --> resolves from neuron_registry
  |-- build() --> sorted by priority, formatted, joined

ProcessorPipeline
  |-- register(SummarizeProcessor())
  |-- fire("pre_llm_call", ctx, payload)

Workspace
  |-- write/read/delete artifacts
  |-- version history with revert
  |-- observer callbacks (on_create, on_update, on_delete)
  |-- KnowledgeStore integration (auto-indexing)
```

## See Also

- [context](context.md) -- Context class and ContextError
- [config](config.md) -- ContextConfig and AutomationMode
- [state](state.md) -- ContextState hierarchical state
- [neuron](neuron.md) -- Neuron ABC and built-in neurons
- [prompt-builder](prompt-builder.md) -- PromptBuilder
- [processor](processor.md) -- ContextProcessor and ProcessorPipeline
- [workspace](workspace.md) -- Workspace and ArtifactType
- [checkpoint](checkpoint.md) -- Checkpoint and CheckpointStore
- [token-tracker](token-tracker.md) -- TokenTracker
- [tools](tools.md) -- Context tools for agents
- [variables](variables.md) -- DynamicVariableRegistry
- [knowledge](knowledge.md) -- KnowledgeStore internal module
