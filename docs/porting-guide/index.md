# Porting Guide: agent-core (openJiuwen) → Orbiter

## 1. Introduction

**agent-core** is the core SDK of [openJiuwen](https://github.com/OpenJiuwen), a Chinese-origin open-source AI Agent framework (Apache 2.0, Python 3.11+). It provides a comprehensive, monolithic toolkit for building self-optimizing agents with features spanning LLM abstraction, multi-agent orchestration, Pregel-based graph execution, enterprise-grade RAG pipelines, context engineering, multi-dimensional memory, and security guardrails.

**Orbiter** is a modular Python monorepo that reimplements and extends agent-core's most valuable features as independent, composable packages. Rather than porting agent-core wholesale, Orbiter selectively adapted the strongest patterns — typed rails, context processors, memory evolution, retrieval pipelines — while replacing others with simpler alternatives (e.g., Swarm transfers instead of Pregel channels).

This guide documents every architectural decision made during the port, providing migration paths for developers familiar with either framework.

For a detailed feature-by-feature comparison, see [agent-core-vs-orbiter-analysis.md](../../agent-core-vs-orbiter-analysis.md).

## 2. Architecture Comparison

### agent-core: Single-Package Monolith

agent-core ships as a single `openjiuwen` package (~540+ files) where all features are bundled together:

```
openjiuwen/
├── llm/                # LLM abstraction, prompts, KV stores
├── tool/               # Tool system, vector store tools
├── agent/              # ReActAgent, ControllerAgent
├── agent_group/        # Multi-agent coordination
├── workflow/           # Pregel-based DAG execution
├── memory/             # 5-type taxonomy, encryption, deduplication
├── context_engine/     # Offloading, compression, windowing
├── retrieval/          # Embeddings, vector stores, hybrid search, reranking
├── session/            # Session lifecycle, checkpointing, streaming
├── runner/             # Global resource registry, callbacks
├── security/           # Guardrails, risk assessment
├── agent_evolution/    # LLM-driven optimization of prompts/tools/memory
├── deep_agent/         # Autonomous task execution (file/shell/code/todo)
└── extensions/         # Redis, Pulsar, context evolution algorithms
```

### Orbiter: Modular Monorepo

Orbiter splits these concerns across 18 independently installable packages:

```
packages/
├── orbiter-core/         # Agent, Swarm, Tool, HookManager, Runner
├── orbiter-models/       # ModelProvider ABC, 11 provider adapters
├── orbiter-memory/       # Typed memory hierarchy, stores, search
├── orbiter-context/      # ProcessorPipeline, token budgeting
├── orbiter-guardrail/    # RiskAssessment, guardrail backends
├── orbiter-retrieval/    # Embeddings, vector stores, retrievers
├── orbiter-train/        # Operator, Optimizer, textual gradient tuning
├── orbiter-sandbox/      # Isolated code execution
├── orbiter-eval/         # Evaluation framework
├── orbiter-mcp/          # Model Context Protocol integration
├── orbiter-a2a/          # Agent-to-Agent protocol
├── orbiter-server/       # FastAPI server runtime
├── orbiter-web/          # Astro frontend + FastAPI backend
├── orbiter-cli/          # Command-line interface
├── orbiter-distributed/  # Redis task queue, workers
├── orbiter-observability/ # OpenTelemetry logging, tracing, metrics
├── orbiter-perplexica/   # Search-focused agent example
└── orbiter/              # Meta-package (convenience re-exports)
```

### Package Mapping Diagram

```
agent-core (openjiuwen/)            Orbiter (packages/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

llm/, tool/, agent/          ──→    orbiter-core + orbiter-models
    runner/                  ──→    orbiter-core (Runner)
workflow/                    ──→    orbiter-core (Swarm, BranchNode, LoopNode)
memory/                      ──→    orbiter-memory
context_engine/              ──→    orbiter-context
security/                    ──→    orbiter-guardrail
retrieval/                   ──→    orbiter-retrieval
agent_evolution/             ──→    orbiter-train
deep_agent/                  ──→    orbiter-sandbox + orbiter-core (Tools)
session/                     ──→    orbiter-core (state) + orbiter-observability
agent_group/                 ──→    orbiter-core (Swarm handoffs)
extensions/redis             ──→    orbiter-distributed
extensions/context_evolver   ──→    orbiter-memory (evolution strategies)
```

## 3. Design Philosophy Differences

| Dimension | agent-core | Orbiter |
|-----------|-----------|---------|
| **Package structure** | Single package, all-or-nothing | Modular monorepo, install what you need |
| **Resource management** | Global `Runner` singleton with registry | Dependency injection through package hierarchy |
| **Agent composition** | Inheritance-heavy (`ReActAgent` → `ControllerAgent`) | Composition via Swarm transfers + HookManager |
| **Workflow execution** | Pregel graph with super-steps, channels, barriers | Swarm DSL (`"a >> b >> c"`) + topological sort |
| **Context lifecycle** | Stateful `ContextManager` orchestrates all processing | Event-driven `ProcessorPipeline` — processors are independent |
| **Agent lifecycle guards** | `AgentRail` callbacks with untyped context dict | Typed `Rail` ABC with Pydantic event models |
| **Memory model** | Flat `MemoryItem` with 5 string types | Typed hierarchy (`SystemMemory`, `HumanMemory`, `AIMemory`, `ToolMemory`) |
| **Security** | Pluggable guardrail backends on callback framework | `BaseGuardrail` integrates via `HookManager` |
| **Evolution/optimization** | LLM-driven Operators (prompt, tool, memory tuning) | Textual gradient optimization + genetic algorithms |
| **Observability** | Session tracing with spans | Full OpenTelemetry (logging, tracing, metrics, cost tracking) |
| **Provider support** | 4 providers (OpenAI-compatible, DashScope, SiliconFlow) | 11 providers (OpenAI, Anthropic, Gemini, Vertex, Ollama, etc.) |

## 4. Package Mapping Table

| agent-core Directory | Purpose | Orbiter Package(s) |
|---------------------|---------|-------------------|
| `openjiuwen/llm/` | LLM abstraction, prompt templates | `orbiter-models` (ModelProvider ABC, adapters) |
| `openjiuwen/tool/` | Tool definitions, execution | `orbiter-core` (Tool, ToolResult) |
| `openjiuwen/agent/` | ReActAgent, ControllerAgent | `orbiter-core` (Agent, Swarm) |
| `openjiuwen/agent_group/` | Multi-agent coordination | `orbiter-core` (Swarm handoffs) |
| `openjiuwen/workflow/` | Pregel-based DAG execution | `orbiter-core` (Swarm, BranchNode, LoopNode, GroupNode) |
| `openjiuwen/memory/` | 5-type memory, encryption, dedup | `orbiter-memory` (MemoryCategory, typed stores, SearchManager) |
| `openjiuwen/context_engine/` | Offloading, compression, windowing | `orbiter-context` (ProcessorPipeline, processors) |
| `openjiuwen/retrieval/` | Embeddings, vector stores, search | `orbiter-retrieval` (EmbeddingProvider, VectorStore, Retriever) |
| `openjiuwen/security/` | Guardrails, risk assessment | `orbiter-guardrail` (BaseGuardrail, RiskAssessment) |
| `openjiuwen/session/` | Session lifecycle, checkpointing | `orbiter-core` (WorkflowState, checkpointing) |
| `openjiuwen/runner/` | Global resource registry, callbacks | `orbiter-core` (Runner, HookManager) |
| `openjiuwen/agent_evolution/` | LLM-driven prompt/tool optimization | `orbiter-train` (Operator, BaseOptimizer, OperatorTrainer) |
| `openjiuwen/deep_agent/` | Autonomous file/shell/code/todo tools | `orbiter-sandbox` + `orbiter-core` (Tool definitions) |
| `openjiuwen/extensions/redis/` | Redis checkpointer, distributed | `orbiter-distributed` (Redis task queue, workers) |
| `openjiuwen/extensions/context_evolver/` | ACE, ReasoningBank, ReMe algorithms | `orbiter-memory` (MemoryEvolutionStrategy subclasses) |

## 5. Per-Epic Porting Guides

| Epic | Guide | Summary |
|------|-------|---------|
| 1 — Security Guardrails | [guardrails.md](guardrails.md) | Event-driven content moderation with RiskAssessment models and multiple detection backends |
| 2 — Advanced Workflow Engine | [workflow-engine.md](workflow-engine.md) | Pregel graph engine replaced by Swarm orchestrator with flow DSL, branch/loop/group nodes |
| 3 — RAG/Retrieval Pipeline | [rag-retrieval.md](rag-retrieval.md) | Complete RAG pipeline mapping: embeddings, vector stores, 5 retriever types, reranking |
| 4 — Context Engine | [context-engine.md](context-engine.md) | Monolithic ContextManager decomposed into composable ProcessorPipeline with pluggable processors |
| 5 — Enhanced Memory System | [memory-system.md](memory-system.md) | 5-type memory taxonomy with AES-256 encryption mapped to typed hierarchy with pluggable stores |
| 6 — Typed Agent Rails | [rails.md](rails.md) | Rail lifecycle guards with 10 callback events mapped to typed Rails with Pydantic models |
| 7 — Task Management | [task-management.md](task-management.md) | TaskManager and TaskScheduler mapped to Pydantic-based TaskManager with mid-execution steering |
| 8 — Deep Agent Toolkit | [deep-agent-toolkit.md](deep-agent-toolkit.md) | Autonomous toolkit (file I/O, shell, code, todo) distributed across Orbiter's tool system and sandbox |
| 9 — Context Evolution | [context-evolver.md](context-evolver.md) | Three memory evolution strategies (ACE, ReasoningBank, ReMe) mapped to composable strategy subclasses |
| 10 — Operator & Self-Optimization | [operator-optimization.md](operator-optimization.md) | Iterative optimization system mapped to orbiter-train with Operator ABC and textual gradient tuning |

## 6. Acknowledgments

This porting effort builds extensively on the work of the [openJiuwen](https://github.com/OpenJiuwen) project and its contributors. The agent-core SDK represents a thoughtful and comprehensive approach to AI agent development, and many of Orbiter's features — particularly the typed rail system, context engine processors, memory evolution strategies, and retrieval pipeline architecture — were directly inspired by or adapted from openJiuwen's designs.

We are grateful to the openJiuwen team for releasing their work under the Apache 2.0 license, making this kind of cross-project knowledge transfer possible.
