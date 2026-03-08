# Architecture Overview

Orbiter is a modern multi-agent framework built as a ground-up rewrite of AWorld (96,500 LOC). It is organized as a **UV workspace monorepo** with 13 focused packages under `packages/`, each with a single responsibility and minimal dependencies.

## Design Philosophy

Orbiter's design draws from the best ideas across existing frameworks while avoiding their pitfalls:

- **Simplicity from AWorld's lessons** -- AWorld grew to 96k lines with deep inheritance, stringly-typed routing, and duplicated sync/async classes. Orbiter starts clean with a ~200 line-per-file cap and flat composition.
- **Async-first** -- All internal functions are `async def`. A single `run.sync()` bridge covers synchronous callers. No sync/async class duplication.
- **Type-safe** -- Full pyright strict-mode compliance. Typed message classes replace stringly-typed routing. Pydantic v2 models for all configuration and data.
- **Composable** -- One `Agent` class, one `Swarm` orchestrator, one `Registry[T]` pattern. Behavior changes through composition (tools, hooks, modes), not inheritance hierarchies.

## Package Overview

```
orbiter (workspace root)
|
+-- packages/
    |
    +-- orbiter-core       Core types, Agent, Tool, Swarm, Runner, Hooks, Events
    +-- orbiter-models     LLM provider abstractions (OpenAI, Anthropic)
    +-- orbiter-context    Context engine: state, prompt building, neurons, processors
    +-- orbiter-memory     Short-term and long-term memory backends
    +-- orbiter-mcp        Model Context Protocol client + server decorator
    +-- orbiter-sandbox    Local and Kubernetes sandboxed execution
    +-- orbiter-trace      OpenTelemetry-based tracing and instrumentation
    +-- orbiter-eval       Evaluation framework: scorers, reflection, evaluator
    +-- orbiter-a2a        Agent-to-Agent protocol for remote delegation
    +-- orbiter-cli        Command-line interface
    +-- orbiter-server     HTTP server for serving agents
    +-- orbiter-train      Training: trajectory collection, dataset, trainers
    +-- orbiter            Meta-package that re-exports everything
```

## Dependency DAG

The dependency structure is strictly layered. `orbiter-core` sits at the bottom with zero heavy dependencies (only `pydantic`). Every other package depends on `orbiter-core`, but packages at the same level do not depend on each other.

```
                        orbiter (meta)
                            |
     +----------+-----------+-----------+----------+
     |          |           |           |          |
  orbiter-   orbiter-   orbiter-   orbiter-   orbiter-
  cli        server     train      a2a        eval
     |          |           |           |          |
     +----------+-----------+-----------+----------+
                            |
     +----------+-----------+-----------+----------+
     |          |           |           |          |
  orbiter-   orbiter-   orbiter-   orbiter-   orbiter-
  models     context    memory     mcp        sandbox
     |          |           |           |          |
     +----------+-----------+-----------+----------+
                            |
                      orbiter-core
                      (pydantic only)
```

## Core Concepts

| Concept | Module | Description |
|---------|--------|-------------|
| **Agent** | `orbiter.agent` | Autonomous LLM-powered unit with tools, handoffs, and hooks |
| **Tool** | `orbiter.tool` | Function or class that an agent can call via LLM tool-use |
| **Swarm** | `orbiter.swarm` | Multi-agent orchestration: workflow, handoff, or team mode |
| **Runner** | `orbiter.runner` | Entry points: `run()`, `run.sync()`, `run.stream()` |
| **Hook** | `orbiter.hooks` | Lifecycle interception at PRE/POST_LLM_CALL, PRE/POST_TOOL_CALL |
| **EventBus** | `orbiter.events` | Decoupled async pub/sub for framework-level events |
| **Registry** | `orbiter.registry` | Generic `Registry[T]` for agents, tools, models, neurons |
| **Context** | `orbiter.context` | Hierarchical state, prompt building, RAG, workspace |

## What Users Import

Most applications need only the top-level imports:

```python
from orbiter import Agent, Swarm, Tool, tool, run
```

For advanced usage:

```python
from orbiter.types import UserMessage, AssistantMessage, RunResult, StreamEvent
from orbiter.config import AgentConfig, ModelConfig
from orbiter.hooks import Hook, HookPoint
from orbiter.models import ModelProvider, get_provider
from orbiter.context import Context, ContextConfig, PromptBuilder
```

## Further Reading

- [Execution Flow](./execution-flow.md) -- What happens when you call `run()`
- [Dependency Graph](./dependency-graph.md) -- Package dependencies and internal modules
- [Design Decisions](./design-decisions.md) -- Why Orbiter is structured the way it is
- [Agent Runtime Control Contracts](./agent-runtime-control-contracts.md) -- Decision memo for planning, HITL, injected args, sub-agents, and progress controls
- [Async Patterns](./async-patterns.md) -- Async-first design and sync bridge
- [Error Handling](./error-handling.md) -- Exception hierarchy and propagation
