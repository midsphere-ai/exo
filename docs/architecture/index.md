# Architecture Overview

Exo is a modern multi-agent framework built as a ground-up rewrite of AWorld (96,500 LOC). It is organized as a **UV workspace monorepo** with 13 focused packages under `packages/`, each with a single responsibility and minimal dependencies.

## Design Philosophy

Exo's design draws from the best ideas across existing frameworks while avoiding their pitfalls:

- **Simplicity from AWorld's lessons** -- AWorld grew to 96k lines with deep inheritance, stringly-typed routing, and duplicated sync/async classes. Exo starts clean with a ~200 line-per-file cap and flat composition.
- **Async-first** -- All internal functions are `async def`. A single `run.sync()` bridge covers synchronous callers. No sync/async class duplication.
- **Type-safe** -- Full pyright strict-mode compliance. Typed message classes replace stringly-typed routing. Pydantic v2 models for all configuration and data.
- **Composable** -- One `Agent` class, one `Swarm` orchestrator, one `Registry[T]` pattern. Behavior changes through composition (tools, hooks, modes), not inheritance hierarchies.

## Package Overview

```
exo (workspace root)
|
+-- packages/
    |
    +-- exo-core       Core types, Agent, Tool, Swarm, Runner, Hooks, Events
    +-- exo-models     LLM provider abstractions (OpenAI, Anthropic)
    +-- exo-context    Context engine: state, prompt building, neurons, processors
    +-- exo-memory     Short-term and long-term memory backends
    +-- exo-mcp        Model Context Protocol client + server decorator
    +-- exo-sandbox    Local and Kubernetes sandboxed execution
    +-- exo-trace      OpenTelemetry-based tracing and instrumentation
    +-- exo-eval       Evaluation framework: scorers, reflection, evaluator
    +-- exo-a2a        Agent-to-Agent protocol for remote delegation
    +-- exo-cli        Command-line interface
    +-- exo-server     HTTP server for serving agents
    +-- exo-train      Training: trajectory collection, dataset, trainers
    +-- exo            Meta-package that re-exports everything
```

## Dependency DAG

The dependency structure is strictly layered. `exo-core` sits at the bottom with zero heavy dependencies (only `pydantic`). Every other package depends on `exo-core`, but packages at the same level do not depend on each other.

```
                        exo (meta)
                            |
     +----------+-----------+-----------+----------+
     |          |           |           |          |
  exo-   exo-   exo-   exo-   exo-
  cli        server     train      a2a        eval
     |          |           |           |          |
     +----------+-----------+-----------+----------+
                            |
     +----------+-----------+-----------+----------+
     |          |           |           |          |
  exo-   exo-   exo-   exo-   exo-
  models     context    memory     mcp        sandbox
     |          |           |           |          |
     +----------+-----------+-----------+----------+
                            |
                      exo-core
                      (pydantic only)
```

## Core Concepts

| Concept | Module | Description |
|---------|--------|-------------|
| **Agent** | `exo.agent` | Autonomous LLM-powered unit with tools, handoffs, and hooks |
| **Tool** | `exo.tool` | Function or class that an agent can call via LLM tool-use |
| **Swarm** | `exo.swarm` | Multi-agent orchestration: workflow, handoff, or team mode |
| **Runner** | `exo.runner` | Entry points: `run()`, `run.sync()`, `run.stream()` |
| **Hook** | `exo.hooks` | Lifecycle interception at PRE/POST_LLM_CALL, PRE/POST_TOOL_CALL |
| **EventBus** | `exo.events` | Decoupled async pub/sub for framework-level events |
| **Registry** | `exo.registry` | Generic `Registry[T]` for agents, tools, models, neurons |
| **Context** | `exo.context` | Hierarchical state, prompt building, RAG, workspace |

## What Users Import

Most applications need only the top-level imports:

```python
from exo import Agent, Swarm, Tool, tool, run
```

For advanced usage:

```python
from exo.types import UserMessage, AssistantMessage, RunResult, StreamEvent
from exo.config import AgentConfig, ModelConfig
from exo.hooks import Hook, HookPoint
from exo.models import ModelProvider, get_provider
from exo.context import Context, ContextConfig, PromptBuilder
```

## Further Reading

- [Execution Flow](./execution-flow.md) -- What happens when you call `run()`
- [Dependency Graph](./dependency-graph.md) -- Package dependencies and internal modules
- [Design Decisions](./design-decisions.md) -- Why Exo is structured the way it is
- [Agent Runtime Control Contracts](./agent-runtime-control-contracts.md) -- Decision memo for planning, HITL, injected args, sub-agents, and progress controls
- [Context Management and Infinite Context Stages](./context-management-infinite-context-stages.md) -- Decision memo for persisted summaries, checkpoints, retrieval-aware compaction, and branch-scoped context
- [Temporal Parity Gaps](./temporal-parity-gaps.md) -- Decision memo for bringing durable execution to the same observable contract as local execution
- [Async Patterns](./async-patterns.md) -- Async-first design and sync bridge
- [Error Handling](./error-handling.md) -- Exception hierarchy and propagation
