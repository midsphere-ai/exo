# Context Engine

The Context Engine (`exo-context`) gives agents structured, hierarchical memory that persists across LLM calls. It manages state, prompt assembly, token budgets, workspace artifacts, knowledge retrieval, checkpointing, and context-aware tools -- all coordinated through a single `Context` object.

## Why Use Context?

Without the context engine, each agent turn starts from scratch. With it you get:

- **Hierarchical state** -- parent/child contexts with inherited key-value data.
- **Composable prompts** -- neurons assemble prompt sections by priority.
- **Event-driven processors** -- run logic before/after LLM calls and tool invocations.
- **Workspace artifacts** -- versioned file storage with observer notifications.
- **Knowledge retrieval** -- TF-IDF search over ingested documents.
- **Token tracking** -- per-agent, per-step usage accounting.
- **Checkpointing** -- save and restore context snapshots.
- **Context tools** -- planning, knowledge, and file tools the agent can call.

## Quick Start

```python
from exo.context import Context, ContextConfig, make_config, AutomationMode

# Create a context with sensible defaults
config = make_config(AutomationMode.COPILOT)
ctx = Context(task_id="task-1", config=config)

# Store and retrieve state
ctx.state.set("user_name", "Alice")
print(ctx.state.get("user_name"))  # "Alice"

# Fork a child context for a sub-task
child = ctx.fork("subtask-1")
child.state.set("step", 3)

# Child inherits parent state
print(child.state.get("user_name"))  # "Alice" (inherited)

# Merge child results back
ctx.merge(child)
```

## Architecture Overview

```
Context
  |-- config: ContextConfig     (immutable settings)
  |-- state: ContextState       (hierarchical key-value store)
  |-- children: list[Context]   (forked child contexts)
  |-- token_usage: TokenTracker (per-step token accounting)
  |
  +-- PromptBuilder             (assembles prompt from neurons)
  +-- ProcessorPipeline         (event-driven pre/post processing)
  +-- Workspace                 (versioned artifact storage)
  +-- KnowledgeStore            (text search + TF-IDF scoring)
  +-- CheckpointStore           (save/restore snapshots)
  +-- Context Tools             (planning, knowledge, file tools)
```

## Automation Modes

The `AutomationMode` enum controls how much autonomy the agent has:

| Mode | Description | History Rounds | Summary Threshold |
|------|-------------|---------------|-------------------|
| `PILOT` | Full autonomy, minimal history | 3 | 5 |
| `COPILOT` | Balanced autonomy and context | 10 | 15 |
| `NAVIGATOR` | Maximum context, human-guided | 20 | 30 |

```python
from exo.context import make_config, AutomationMode

pilot_cfg = make_config(AutomationMode.PILOT)
copilot_cfg = make_config(AutomationMode.COPILOT)
navigator_cfg = make_config(AutomationMode.NAVIGATOR)
```

## Guides

| Guide | What It Covers |
|-------|---------------|
| [State Management](state.md) | Hierarchical key-value state, parent inheritance, fork/merge |
| [Prompt Building](prompt-building.md) | Neurons, priority ordering, variable substitution |
| [Processors](processors.md) | Event-driven pipeline (pre_llm_call, post_tool_call, etc.) |
| [Workspace](workspace.md) | Versioned artifact storage, observers, filesystem persistence |
| [Knowledge](knowledge.md) | Text chunking, TF-IDF search, knowledge store |
| [Checkpoints](checkpoints.md) | Snapshot save/restore, version history |
| [Token Tracking](token-tracking.md) | Per-agent per-step usage, trajectories, summaries |
| [Context Tools](context-tools.md) | Planning, knowledge, and file tools for agents |

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `Context` | `exo.context` | Central context object with state, children, and token tracking |
| `ContextConfig` | `exo.context` | Frozen configuration (mode, history_rounds, thresholds) |
| `ContextState` | `exo.context` | Hierarchical key-value store with parent inheritance |
| `PromptBuilder` | `exo.context` | Assembles prompts from neurons with priority ordering |
| `Neuron` | `exo.context` | ABC for named prompt sections with priority |
| `neuron_registry` | `exo.context` | Registry of built-in and custom neurons |
| `ContextProcessor` | `exo.context` | ABC for event-driven context processing |
| `ProcessorPipeline` | `exo.context` | Manages and fires processors by event |
| `SummarizeProcessor` | `exo.context` | Built-in processor for pre_llm_call summarization |
| `ToolResultOffloader` | `exo.context` | Built-in processor for post_tool_call result offloading |
| `Workspace` | `exo.context` | Versioned artifact storage with observer pattern |
| `ArtifactType` | `exo.context` | Enum: CODE, CSV, IMAGE, JSON, MARKDOWN, TEXT |
| `TokenTracker` | `exo.context` | Per-agent per-step token usage tracking |
| `Checkpoint` | `exo.context` | Frozen snapshot of context state |
| `CheckpointStore` | `exo.context` | Save, list, and restore checkpoints |
| `make_config` | `exo.context` | Factory for preset configurations by automation mode |
| `AutomationMode` | `exo.context` | Enum: PILOT, COPILOT, NAVIGATOR |
| `get_context_tools` | `exo.context` | Returns all context tools (planning + knowledge + file) |
| `get_planning_tools` | `exo.context` | Returns planning tools (add_todo, complete_todo, get_todo) |
| `get_knowledge_tools` | `exo.context` | Returns knowledge tools (get, grep, search) |
| `get_file_tools` | `exo.context` | Returns file tools (read_file) |
