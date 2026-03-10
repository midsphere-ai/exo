# Agent-Core (openJiuwen) vs Orbiter: Deep Feature Analysis

> **Date:** 2026-03-10
> **Scope:** Complete codebase exploration of [agent-core](https://github.com/openjiuwen/agent-core) (openJiuwen Core v0.1.9) compared against Orbiter AI framework
> **Method:** 12 parallel exploration agents covering every module, doc, and example

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Framework Overview Comparison](#framework-overview-comparison)
3. [Important Features in Agent-Core](#important-features-in-agent-core)
4. [Features Missing in Orbiter](#features-missing-in-orbiter)
5. [Feature-by-Feature Comparison](#feature-by-feature-comparison)
6. [Architectural Differences](#architectural-differences)
7. [Recommendations](#recommendations)

---

## Executive Summary

**Agent-Core (openJiuwen)** is a Chinese-origin open-source AI Agent SDK (Apache 2.0, Python 3.11+) with a single-package architecture. It emphasizes **self-optimizing agents**, **Pregel-based graph execution**, **enterprise RAG pipelines**, and **context engineering** as its core differentiators.

**Orbiter** is a modular UV workspace monorepo (15 packages) with clear separation of concerns. It emphasizes **production-grade multi-agent orchestration**, **comprehensive observability**, **distributed execution**, and a **full web platform**.

### Key Takeaways

| Dimension | Agent-Core Advantage | Orbiter Advantage |
|-----------|---------------------|-------------------|
| **RAG/Retrieval** | Full pipeline (embeddings, vector stores, graph retrieval, agentic retrieval) | None built-in |
| **Self-Optimization** | 3-dimension agent evolution (prompt, tool, memory) with trajectory analysis | Has orbiter-train (synthesis + evolution + VeRL) but different approach |
| **Workflow Engine** | Pregel graph with super-steps, barriers, stream actors, loops, sub-workflows | Swarm DSL (simpler but less powerful for complex workflows) |
| **Context Engineering** | Sophisticated context engine with processors, offloading, reload, compression | Has orbiter-context with processors + checkpointing |
| **Security** | Pluggable guardrail framework with risk assessment | No guardrail system |
| **Memory** | 5 memory types + encryption + LLM deduplication + migration system | Short/long-term memory with vector search |
| **Provider Support** | 4 providers (OpenAI-compatible focus, DashScope, SiliconFlow) | 11 providers (OpenAI, Anthropic, Gemini, Vertex, Ollama, etc.) |
| **Distributed** | Message queue (Pulsar) + remote agents | Redis task queue + workers + optional Temporal |
| **Observability** | Session tracing with spans | Full OpenTelemetry (logging, tracing, metrics, cost, SLO) |
| **Web Platform** | Studio mentioned (separate product, not in repo) | Full web platform (30+ API routes, React canvas) |
| **Modularity** | Single package | 15-package monorepo with clear dependency graph |
| **Protocol Support** | No A2A | A2A protocol support |

---

## Framework Overview Comparison

### Architecture

| Aspect | Agent-Core | Orbiter |
|--------|-----------|---------|
| **Structure** | Single package (`openjiuwen/`) with sub-modules | UV workspace monorepo (15 packages) |
| **Core Dep** | Pydantic, aiohttp, openai, SQLAlchemy, tiktoken | Pydantic (core only depends on pydantic) |
| **Python** | 3.11-3.13 | 3.11+ |
| **Async** | Async-first (all APIs) | Async-first (all APIs) |
| **License** | Apache 2.0 | (workspace monorepo) |
| **Version** | 0.1.9 | Multi-package versioning |
| **Tests** | Unit + system tests | ~2,900 tests across packages |

### Agent Types

| Agent Type | Agent-Core | Orbiter |
|-----------|-----------|---------|
| **ReAct Agent** | `ReActAgent` (Reason-Act-Observe loop) | `Agent` (built-in tool loop) |
| **Workflow Agent** | `WorkflowAgent` (predefined flows) | Swarm workflow mode |
| **Controller Agent** | `ControllerAgent` (event-driven task management) | No equivalent |
| **Deep Agent** | `DeepAgent` (autonomous task execution with todo/file/shell/code tools) | No equivalent |
| **Self-Evolving Agent** | `ReActAgentEvolve` (operator-based optimization) | Via orbiter-train |
| **Custom Agent** | Extend `BaseAgent` | Extend `Agent` |

---

## Important Features in Agent-Core

### 1. Pregel-Based Graph Execution Engine
**Location:** `openjiuwen/core/graph/`

A LangGraph-style graph execution engine implementing the Pregel model:

- **Super-step execution**: All ready nodes execute in parallel per step, results buffered then flushed
- **Channel system**: `TriggerChannel` (activation signals) and `BarrierChannel` (fan-in synchronization)
- **Router types**: `StaticRouter` (fixed targets), `ConditionalRouter` (dynamic), `BarrierRouter` (fan-in)
- **Stream actors**: Per-node streaming I/O with `StreamActor`, `StreamProcessor`, `StreamConsumer`
- **4 I/O patterns**: INVOKE (batch-batch), STREAM (batch-stream), COLLECT (stream-batch), TRANSFORM (stream-stream)
- **Graph state checkpointing**: Save/restore on error with `GraphStore`
- **Recursion limits**: Prevents infinite loops
- **Mermaid visualization**: Auto-generated flowcharts (PNG/SVG/Mermaid)

**Orbiter equivalent:** Swarm with DSL (`"a >> b >> c"`) and topological sort. Much simpler, no Pregel model, no barriers, no stream actors.

---

### 2. Full RAG/Retrieval Pipeline
**Location:** `openjiuwen/core/retrieval/`

Enterprise-grade retrieval system:

- **Embedding providers**: OpenAI, vLLM (multimodal), generic API (any HTTP endpoint)
- **Vector stores**: Milvus (distributed), ChromaDB (local), PostgreSQL/pgvector
- **Hybrid search**: Dense vector + sparse BM25 with Reciprocal Rank Fusion (RRF)
- **5 retriever types**:
  - `VectorRetriever` — pure dense search
  - `SparseRetriever` — pure BM25
  - `HybridRetriever` — RRF fusion of vector + sparse
  - `GraphRetriever` — knowledge graph expansion via triple beam search
  - `AgenticRetriever` — multi-round LLM-driven query rewriting with iterative retrieval
- **Reranking**: StandardReranker (vLLM-compatible), AliyunReranker (vendor-specific)
- **Query rewriting**: LLM-based with history compression
- **Document processing**: PDF, Word, Excel, JSON, Markdown, web pages, WeChat articles, images with captions
- **Chunking strategies**: Character, hybrid, tokenizer, text-based with configurable overlap
- **Knowledge graph**: Triple extraction (subject-predicate-object), beam search expansion

**Orbiter equivalent:** None. Orbiter has no built-in RAG system. Context tools exist in orbiter-context but no document ingestion, vector stores, or retrieval pipeline.

---

### 3. Agent Self-Optimization (agent_evolving)
**Location:** `openjiuwen/agent_evolving/`

Three-dimensional agent evolution via LLM-driven optimization:

- **Dimension 1 — Prompt Optimization** (`InstructionOptimizer`):
  - Uses LLM to analyze failure cases and generate "textual gradients"
  - Generates improved system/user prompts from gradients
  - Preserves placeholder variables (e.g., `{{query}}`) across optimization
  - Handles joint system+user prompt optimization

- **Dimension 2 — Tool Description Optimization** (`ToolOptimizer`):
  - Beam search across tool description candidates
  - Multi-stage refinement (example generation, description optimization, review)
  - Cross-checking + translation for quality assurance

- **Dimension 3 — Memory Configuration** (`MemoryOptimizer`):
  - Optimizes enabled/disabled state and retry counts
  - Memory tunable parameters exposed via operator interface

- **Training loop**: Evaluate -> backward (compute textual gradients) -> step (generate updates) -> apply -> validate
- **Trajectory analysis**: Complete execution DAG extraction from session traces
- **Checkpoint/resume**: Full training interruption/resumption with FileCheckpointStore
- **Metrics**: ExactMatch + LLM-as-Judge evaluation
- **Operator pattern**: `LLMCallOperator`, `ToolCallOperator`, `MemoryCallOperator` as atomic tunable units

**Orbiter equivalent:** orbiter-train has `EvolutionPipeline` (genetic algorithm style), `SynthesisPipeline` (data generation), `VeRLTrainer` (RL reward training). Different approach — Orbiter focuses on data synthesis + RL, Agent-Core focuses on LLM-driven prompt/tool/memory optimization from trajectories.

---

### 4. Context Engine with Processors
**Location:** `openjiuwen/core/context_engine/`

Sophisticated context window management:

- **Message offloading**: Messages too large are replaced with `[[OFFLOAD: handle=<id>]]` markers; model can request reload via tool
- **Dialogue compression**: LLM-based compression of tool call chains into summaries
- **Round-level windowing**: Understands dialogue structure (user → assistant without tool_calls = 1 round)
- **Token budgeting**: TiktokenCounter (cl100k_base for GPT-3.5/4, o200k_base for GPT-4o)
- **Processor pipeline**: Pluggable `ContextProcessor` chain with `register_processor()` decorator
- **Built-in processors**: `MessageOffloader`, `DialogueCompressor`, `CurrentRoundCompressor`, `RoundLevelCompressor`
- **State persistence**: Save/load context state to session storage
- **KV cache management**: Optional GPU memory optimization for vLLM

**Orbiter equivalent:** orbiter-context has `ContextProcessor` pipeline, `SummarizeProcessor`, `ToolResultOffloader`, `TokenTracker`, `Checkpoint`. Similar concept but Agent-Core's offload-and-reload pattern and dialogue round semantics are more sophisticated.

---

### 5. Security Guardrail Framework
**Location:** `openjiuwen/core/security/guardrail/`

Pluggable security detection:

- **Risk assessment model**: `RiskAssessment` with levels (SAFE, LOW, MEDIUM, HIGH, CRITICAL)
- **Backend-agnostic**: `GuardrailBackend` abstract interface with `analyze(data) -> RiskAssessment`
- **Event-driven**: Registers with callback framework, monitors user_input, llm_input, llm_output, tool_call events
- **Built-in**: `UserInputGuardrail` for injection/jailbreak detection
- **Integration**: Hooks into `AsyncCallbackFramework` with priority=100, ERROR hook re-throws

**Orbiter equivalent:** None. Orbiter has no guardrail/security detection framework. Only safety prompts in the web UI.

---

### 6. Comprehensive Memory Engine
**Location:** `openjiuwen/core/memory/`

Multi-type memory with enterprise features:

- **5 memory types**: USER_PROFILE, SEMANTIC_MEMORY, EPISODIC_MEMORY, VARIABLE, SUMMARY
- **AES-256 encryption**: Optional encryption at rest (nonce + tag + ciphertext)
- **Intelligent deduplication**: LLM-based `MemUpdateChecker` decides ADD/DELETE/MERGE for new memories (top-5 similarity check at 0.75 threshold)
- **Migration system**: Pluggable registries for SQL, vector store, and KV store schema migrations with version tracking
- **Specialized managers**: FragmentMemoryManager (profile/semantic/episodic), SummaryManager, VariableManager, MessageManager
- **Unified search**: `SearchManager` traverses all memory types with threshold filtering

**Orbiter equivalent:** orbiter-memory has `ShortTermMemory` and `LongTermMemory` with vector search and extraction tasks. Simpler — no memory types taxonomy, no encryption, no deduplication, no migration system.

---

### 7. Controller Architecture (Event-Driven Task Management)
**Location:** `openjiuwen/core/controller/`

Full task management system:

- **TaskManager**: CRUD with priority indexing, hierarchical parent-child relationships, async locking
- **TaskScheduler**: Concurrent task execution with configurable limits, pause/cancel support
- **EventQueue**: Pub/sub via MessageQueueInMemory with per-session topic routing
- **IntentRecognizer**: LLM-based intent detection for routing (create_task, pause_task, resume_task, cancel_task)
- **Task states**: SUBMITTED, WORKING, PAUSED, INPUT_REQUIRED, COMPLETED, CANCELED, FAILED, WAITING
- **TaskExecutor registry**: Pluggable executors per task type

**Orbiter equivalent:** No direct equivalent. Orbiter's Swarm handles multi-agent coordination but not task lifecycle management with priorities, pause/resume, and intent-driven routing.

---

### 8. DeepAgents (Autonomous Task Execution)
**Location:** `openjiuwen/deepagents/`

Extended agent framework for autonomous operation:

- **Built-in tools**: File I/O (read, write, edit, glob, list, grep), shell execution, code execution, todo management
- **Todo system**: Full CRUD with status tracking (PENDING, IN_PROGRESS, COMPLETED), single-active constraint
- **Task loop architecture**: Multi-iteration outer loop with priority event queue (ABORT > STEER > FOLLOWUP)
- **Rail-based extensibility**: `TaskPlanningRail`, `ContextEngineeringRail` (skeleton)
- **Session-scoped state**: Per-invoke mutable runtime state with persistence
- **Selective callback routing**: Rails routed to inner ReActAgent or outer DeepAgent based on event type

**Orbiter equivalent:** No direct equivalent. Orbiter agents can use tools but don't have the built-in autonomous execution toolkit (file/shell/code/todo).

---

### 9. Context Evolver (Memory Evolution Algorithms)
**Location:** `openjiuwen/extensions/context_evolver/`

Three research-grade memory evolution algorithms:

- **ACE** (Adaptive Context Engine): Playbook-based with helpful/harmful/neutral counters, reflection + curation
- **ReasoningBank**: Title/description/content memories with query embeddings, recall + summarize
- **ReMe**: When-to-use based memories, success/failure extraction, deduplication
- **Operation composition**: `>>` (sequential) and `|` (parallel) operators for composable pipelines
- **Service context**: Singleton pattern for shared LLM/embedding services
- **File-based persistence**: JSON connector for memory storage

**Orbiter equivalent:** None. Orbiter-memory does basic extraction tasks but has no algorithm-specific memory evolution strategies.

---

### 10. Workflow Components Library
**Location:** `openjiuwen/core/workflow/components/`

Rich component set:

| Component | Purpose | Orbiter Equivalent |
|-----------|---------|-------------------|
| `LLMComponent` | LLM call with formatting, history, schema validation | Agent node in Swarm |
| `BranchComponent` | Conditional routing with safe expression evaluator | Swarm conditional handoff |
| `LoopComponent` | Array/number/expression loops with break control | No equivalent |
| `SubWorkflowComponent` | Nested workflow execution | Nested Swarm (SwarmNode) |
| `IntentDetectionComponent` | LLM-based intent classification | No equivalent |
| `QuestionerComponent` | Human-in-the-loop interactive prompts | Tool approval workflow |
| `KnowledgeRetrievalComponent` | RAG integration in workflow | No equivalent |
| `Start/End` | Workflow boundaries | Implicit in Swarm |

**Expression evaluator security**: AST-safe parsing, blocks dunder access, limits nesting depth, prevents large exponentiation.

---

### 11. Async Callback Framework
**Location:** `openjiuwen/core/runner/callback/`

Production-grade event system:

- **Priority-based execution**: Callbacks ordered by priority
- **Callback chains with rollback**: Atomic multi-callback sequences
- **Circuit breakers**: Automatic failure isolation
- **Rate limiting**: Prevents callback storms
- **Lifecycle hooks**: BEFORE, AFTER, AROUND, ERROR
- **Decorator system**: `@create_on_decorator()`, `@create_before_decorator()`, `@create_emit_around_decorator()`, etc.
- **Performance metrics**: Built-in callback performance tracking

**Orbiter equivalent:** `HookManager` with 8 lifecycle points. Simpler — no circuit breakers, rate limiting, callback chains, or rollback.

---

### 12. Session & Checkpointing System
**Location:** `openjiuwen/core/session/`

Comprehensive session lifecycle:

- **Session hierarchy**: BaseSession -> ProxySession, AgentSession, WorkflowSession (with parent relationships)
- **State management**: StateCollection (agent), CommitState (workflow) with staged updates, commit/rollback
- **Checkpointers**: InMemoryCheckpointer (dev), PersistenceCheckpointer (SQLite/Shelve), RedisCheckpointer (distributed)
- **Interruption/resumption**: `InteractiveInput` with per-node user inputs, `GraphInterrupt` for workflow pause
- **Streaming**: Queue-based `StreamEmitter`, typed `StreamWriter`, `StreamWriterManager` with timeout handling
- **Tracing**: `Tracer` with agent + workflow span managers, parent-child span relationships, 15+ trace events

**Orbiter equivalent:** orbiter-context has `Checkpoint` + `CheckpointStore`. orbiter-web has session management. But no transactional state (commit/rollback), no hierarchical session types, no built-in checkpointer implementations.

---

### 13. Runner & Resource Management
**Location:** `openjiuwen/core/runner/`

Central execution coordination:

- **ResourceMgr**: Registry for agents, groups, workflows, models, prompts, tools, operations
- **Tagging system**: `Tag` objects with `TagMatchStrategy` (ALL/ANY) for resource discovery
- **Distributed runner**: Message queue-based with `DMessage` protocol, `RemoteAgent` proxying
- **Metaclass instrumentation**: `_BaseModelClientMeta` and `_ToolMeta` auto-wrap invoke/stream with callback events

**Orbiter equivalent:** orbiter-core has a registry system. orbiter-distributed has Redis-based task queue + workers. Different approach but similar capability.

---

### 14. Operator Pattern (Tunable Execution Units)
**Location:** `openjiuwen/core/operator/`

Atomic executable units designed for self-evolution:

- **LLMCallOperator**: Wraps LLM calls, exposes system_prompt and user_prompt as tunables
- **ToolCallOperator**: Wraps tool invocation, exposes tool descriptions as tunables
- **MemoryCallOperator**: Wraps memory operations, exposes enabled/max_retries as tunables
- **TunableSpec**: Declares optimizable parameters with kind (prompt, continuous, discrete, text)
- **State management**: `get_state()` / `load_state()` for checkpoint/restore

**Orbiter equivalent:** No direct equivalent. Orbiter's training system works at a higher level (evolution pipelines) rather than exposing individual operator tunables.

---

### 15. Rail System (Agent Lifecycle Guards)
**Location:** `openjiuwen/core/single_agent/rail/`

Class-based input/output guards:

- **10 lifecycle events**: BEFORE/AFTER_INVOKE, BEFORE/AFTER_TASK_ITERATION, BEFORE/AFTER_MODEL_CALL, ON_MODEL_EXCEPTION, BEFORE/AFTER_TOOL_CALL, ON_TOOL_EXCEPTION
- **Typed event inputs**: `InvokeInputs`, `ModelCallInputs`, `ToolCallInputs`, `TaskIterationInputs`
- **RetryRequest**: Guards can request retry with configurable delay
- **Cross-rail communication**: `extra` dict shared across rails for the same invocation
- **Priority ordering**: Rails executed in priority order

**Orbiter equivalent:** `HookManager` provides lifecycle hooks but without typed inputs, retry requests, or cross-hook communication.

---

## Features Missing in Orbiter

### Critical Gaps

| # | Feature | Impact | Effort to Add |
|---|---------|--------|---------------|
| 1 | **RAG/Retrieval Pipeline** | Cannot build knowledge-grounded agents without external tools | High — need embedding, vector store, indexing, retrieval, reranking subsystems |
| 2 | **Security Guardrails** | No protection against prompt injection, jailbreak, or unsafe tool use | Medium — need guardrail framework + detection backends |
| 3 | **Advanced Workflow Engine** | Cannot express complex flows (loops, barriers, conditional branches, sub-workflows) | High — Swarm DSL is too simple for enterprise workflows |
| 4 | **Context Offloading & Reload** | Long conversations degrade — no way to offload and reload context on demand | Medium — extend orbiter-context processors |
| 5 | **Memory Type Taxonomy** | No distinction between user profile, semantic, episodic memory | Medium — extend orbiter-memory |

### Notable Gaps

| # | Feature | Impact | Effort to Add |
|---|---------|--------|---------------|
| 6 | **Task Management Controller** | No built-in task lifecycle (create, pause, resume, cancel) with priority | Medium |
| 7 | **Agent Rails (Typed Guards)** | Hooks exist but lack typed inputs, retry requests, cross-hook state | Low-Medium |
| 8 | **DeepAgent Toolkit** | No built-in file/shell/code/todo tools for autonomous execution | Medium |
| 9 | **Memory Encryption** | No encryption at rest for sensitive conversation data | Low |
| 10 | **Memory Deduplication** | No LLM-based semantic dedup to prevent memory bloat | Medium |
| 11 | **Workflow Visualization** | No auto-generated Mermaid/SVG diagrams from agent flows | Low |
| 12 | **Intent Recognition** | No built-in LLM-based intent detection for task routing | Low-Medium |
| 13 | **Query Rewriting** | No LLM-based query reformulation for better retrieval | Low (with RAG) |
| 14 | **Knowledge Graph Retrieval** | No triple extraction or graph-based expansion for RAG | High |
| 15 | **Context Evolver Algorithms** | No ACE/ReasoningBank/ReMe memory evolution strategies | Medium |
| 16 | **Operator Tunables** | No atomic tunable units for fine-grained self-optimization | Medium |

### Where Orbiter is Ahead

| # | Feature | Orbiter | Agent-Core |
|---|---------|---------|-----------|
| 1 | **Provider breadth** | 11 providers (incl. Anthropic, Gemini, Vertex, Ollama) | 4 providers (OpenAI-compatible focus) |
| 2 | **Full web platform** | 30+ API routes, React canvas, Astro frontend | Studio is separate product (not in repo) |
| 3 | **A2A protocol** | Full agent-to-agent with task store + streaming | Not supported |
| 4 | **OpenTelemetry observability** | Logging + tracing + metrics + cost + SLO + alerts | Session tracing only |
| 5 | **Distributed execution** | Redis queue + workers + optional Temporal | Pulsar MQ (single impl) |
| 6 | **Kubernetes sandbox** | KubernetesSandbox built-in | Sandbox gateway (HTTP) |
| 7 | **Package modularity** | 15 packages, install only what you need | Single package (all-or-nothing) |
| 8 | **VeRL/RL training** | VeRL integration for reward-based training | LLM-driven optimization only |
| 9 | **MCP server publishing** | `mcp_server()` decorator to publish agents as MCP servers | MCP client only |
| 10 | **Multi-agent DSL** | Swarm DSL (`"a >> b >> c"`) with handoff/team/workflow modes | Controller-based (more verbose) |
| 11 | **Cost tracking** | `CostTracker` with model pricing lookup | No cost tracking |
| 12 | **Health monitoring** | `HealthRegistry` + `AlertManager` with severity levels | No health monitoring |

---

## Feature-by-Feature Comparison

### LLM Integration

| Feature | Agent-Core | Orbiter |
|---------|-----------|---------|
| Provider abstraction | `Model` + `BaseModelClient` + registry | `ModelProvider` + `model_registry` |
| Supported providers | OpenAI, DashScope, SiliconFlow, OpenRouter | OpenAI, Azure, Anthropic, Gemini, Vertex, Ollama, LM Studio, LiteLLM |
| Streaming | AsyncIterator with chunk merging | StreamEvent with delta tokens |
| Tool call parsing | OpenAI-format conversion + fragment merging | Per-provider parsing |
| Image/video generation | `generate_image()`, `generate_speech()`, `generate_video()` | DALL-E, Imagen, Veo tools |
| Multimodal input | MultimodalDocument (text, image, audio, video) | Content blocks (text, image, audio, video, document) |
| Metaclass callbacks | Auto-wraps invoke/stream with events | Via HookManager |
| vLLM affinity | `InferenceAffinityModel` with session caching | Not supported |
| Output parsers | JSON, Markdown parsers (streaming-aware) | Not built-in (in-agent parsing) |

### Tool System

| Feature | Agent-Core | Orbiter |
|---------|-----------|---------|
| Decorator | `@tool` with flexible overloads | `@tool` decorator |
| Schema extraction | `CallableSchemaExtractor` with type handlers | Auto from docstrings |
| MCP integration | `MCPTool` (client only) | Full MCP client + server |
| REST API tools | `RestfulApi` class | Not built-in |
| Tool registry | `AbilityManager` (cards, not instances) | Tool registry |
| Parallel execution | `AbilityManager.execute()` runs tools in parallel | Parallel tool calls in run loop |

### Memory

| Feature | Agent-Core | Orbiter |
|---------|-----------|---------|
| Types | 5 types (USER_PROFILE, SEMANTIC, EPISODIC, VARIABLE, SUMMARY) | 2 types (ShortTermMemory, LongTermMemory) |
| Encryption | AES-256 optional | None |
| Deduplication | LLM-based semantic dedup (top-5 similarity) | None |
| Persistence | SQL (SQLAlchemy), KV store, Vector store | SQLite, PostgreSQL, Vector |
| Migration | Pluggable migration registries with version tracking | None |
| Search | Unified `SearchManager` across all types | Vector similarity search |

### Workflow/Orchestration

| Feature | Agent-Core | Orbiter |
|---------|-----------|---------|
| Engine | Pregel graph (super-steps, channels, barriers) | Swarm (topological sort, serial/parallel groups) |
| DSL | Fluent API (`add_comp()`, `add_connection()`) | String DSL (`"a >> b >> c"`) |
| Components | 10+ built-in (LLM, Branch, Loop, SubWorkflow, Intent, Questioner, KnowledgeRetrieval) | Agent nodes with handoff functions |
| Loops | Array, Number, Expression, AlwaysTrue with break control | No loop construct |
| Branching | Safe expression evaluator (AST-parsed, security-hardened) | Conditional handoff functions |
| Sub-workflows | Native `SubWorkflowComponent` | Nested SwarmNode |
| HITL | `QuestionerComponent` + GraphInterrupt | Tool approval workflow |
| Streaming | 4 I/O patterns (INVOKE, STREAM, COLLECT, TRANSFORM) | Stream mode |
| Visualization | Mermaid/PNG/SVG auto-generation | React canvas in web UI |
| State persistence | GraphStore with checkpoint/resume on error | No graph-level checkpointing |

### Multi-Agent

| Feature | Agent-Core | Orbiter |
|---------|-----------|---------|
| Group types | BaseGroup, ControllerGroup, HierarchicalGroup | Swarm (workflow, handoff, team modes) |
| Coordination | Message queue-based pub/sub, subscription routing | DSL-based flow + handoff functions |
| Controller | EventQueue + TaskManager + TaskScheduler | Swarm executor |
| Communication | Point-to-point + broadcast via EventDrivenGroupCard | Agent-to-agent via A2A protocol |

### Session & State

| Feature | Agent-Core | Orbiter |
|---------|-----------|---------|
| Session types | AgentSession, WorkflowSession, SubWorkflowSession | Basic session in web |
| State model | StateCollection (agent), CommitState (workflow) with commit/rollback | Context state |
| Checkpointers | InMemory, Persistence (SQLite/Shelve), Redis | Checkpoint + CheckpointStore in context |
| Interruption | InteractiveInput + GraphInterrupt + resume | Interrupt not built-in to core |
| Streaming | StreamEmitter + StreamWriter + StreamWriterManager | StreamEvent |

### Security

| Feature | Agent-Core | Orbiter |
|---------|-----------|---------|
| Guardrail framework | `BaseGuardrail` + `GuardrailBackend` + risk levels | None |
| Input validation | `UserInputGuardrail` (injection/jailbreak) | Safety prompts only |
| Risk assessment | 5 levels (SAFE, LOW, MEDIUM, HIGH, CRITICAL) | None |
| Expression safety | AST-safe evaluator (blocks dunder, limits depth) | N/A |
| Shell allowlist | Configurable command whitelist | N/A (sandbox isolation) |

### Observability

| Feature | Agent-Core | Orbiter |
|---------|-----------|---------|
| Tracing | Tracer with agent/workflow span managers | OpenTelemetry spans + decorators |
| Logging | 4 logger types (general, llm, prompt, sys_operation) | Structured logging with formatters |
| Metrics | None built-in | OpenTelemetry metrics (run duration, token usage, etc.) |
| Cost tracking | None | CostTracker with model pricing |
| SLO monitoring | None | SLOTracker |
| Health checks | None | HealthRegistry + AlertManager |

---

## Architectural Differences

### 1. Monolith vs Monorepo

**Agent-Core** ships as a single `openjiuwen` package. All features are always available. Simpler dependency management but larger install footprint.

**Orbiter** uses 15 separate packages with explicit dependency graph. Users install only what they need. Better for production deployments but more complex workspace management.

### 2. Card-Based vs Config-Based

**Agent-Core** uses a "Card" pattern (`AgentCard`, `ToolCard`, `WorkflowCard`, `GroupCard`) as immutable identity metadata, separate from mutable runtime `Config` objects. Cards serve as "digital business cards" for resource discovery.

**Orbiter** uses `AgentConfig` and model configuration directly. Tools are registered by instance, not by card metadata.

### 3. Operator Pattern vs Direct Execution

**Agent-Core** wraps LLM calls, tool calls, and memory operations in `Operator` objects that expose tunable parameters. This enables the self-evolution system to optimize individual operator behavior.

**Orbiter** calls providers and tools directly. Optimization happens at a higher level (evolution pipelines operating on full agent behavior).

### 4. Runner Singleton vs Modular Injection

**Agent-Core** uses a `Runner` singleton that holds the global `ResourceMgr`, `MessageQueue`, and `CallbackFramework`. All agents access resources through this singleton.

**Orbiter** injects dependencies through the package hierarchy. Each package is self-contained with explicit imports.

### 5. Chinese-First vs English-First

**Agent-Core** has Chinese-language prompts built into the codebase (todo tools, deep agent system prompt, evaluation templates). English docs exist but Chinese docs are primary.

**Orbiter** is English-first throughout.

---

## Recommendations

### High-Priority Features to Consider from Agent-Core

1. **RAG/Retrieval Pipeline** — The most significant gap. Agent-Core's modular retrieval system (embedding -> indexing -> retrieval -> reranking) with hybrid search and graph expansion is production-ready. Consider building `orbiter-retrieval` with:
   - Embedding abstraction (OpenAI, Vertex, custom API)
   - Vector store abstraction (at minimum: pgvector since orbiter-web already uses SQL)
   - Hybrid search with RRF fusion
   - Document processing pipeline (chunking, parsing)

2. **Security Guardrails** — Given audit finding B-1 (sandbox escape), a guardrail framework should be a priority. Consider:
   - `GuardrailBackend` abstraction for pluggable detection
   - Integration with existing `HookManager` lifecycle
   - Built-in input sanitization for prompt injection

3. **Context Offloading** — Agent-Core's offload-and-reload pattern is elegant. When context grows too large, messages are replaced with markers; the model can request reload via a tool. This could be added to orbiter-context as a new processor.

4. **Workflow Loops & Branches** — Swarm DSL is clean for simple flows but can't express loops, conditional branches, or sub-workflows. Consider:
   - Adding `LoopNode` and `BranchNode` to Swarm
   - Supporting conditional edge functions
   - Or building a separate workflow engine alongside Swarm

### Medium-Priority Features

5. **Memory Type Taxonomy** — Distinguishing user_profile, semantic, episodic, and summary memory types enables more targeted retrieval and better context assembly.

6. **Agent Rails** — Upgrading HookManager to support typed event inputs, retry requests, and cross-hook state would enable more sophisticated lifecycle guards.

7. **Memory Encryption** — AES-256 encryption at rest for sensitive conversation data. Relevant for enterprise deployments.

8. **Workflow Visualization** — Auto-generated Mermaid diagrams from Swarm flows would improve developer experience.

### Architecture Observations

9. **Operator Pattern** — Agent-Core's tunable operator abstraction is interesting but adds complexity. Orbiter's higher-level evolution pipeline approach may be more practical for most use cases.

10. **Single Package vs Monorepo** — Orbiter's modularity is a significant advantage for production. Agent-Core's single-package approach creates tight coupling and forces all-or-nothing dependency installation.

---

## Appendix: File Count & Scope

| Module | Agent-Core Files | Key Abstractions |
|--------|-----------------|------------------|
| Foundation (LLM, prompt, store, tool) | ~87 | Model, PromptTemplate, BaseKVStore, BaseVectorStore, Tool |
| Single Agent + Controller | ~55 | ReActAgent, ControllerAgent, Controller, TaskManager |
| Multi-Agent | ~15 | BaseGroup, ControllerGroup, EventDrivenGroupCard |
| Workflow + Graph | ~40+ | Workflow, PregelGraph, PregelLoop, Vertex, StreamActor |
| Memory | ~25 | FragmentMemoryManager, MemUpdateChecker, SearchManager |
| Context Engine | ~20 | ContextEngine, SessionModelContext, ContextProcessor |
| Retrieval | ~80+ | VectorRetriever, HybridRetriever, GraphRetriever, AgenticRetriever |
| Session | ~30 | Session, Checkpointer, StreamEmitter, Tracer |
| Runner | ~25 | Runner, ResourceMgr, AsyncCallbackFramework |
| Security | ~10 | BaseGuardrail, GuardrailBackend, RiskAssessment |
| Agent Evolving | ~46 | Trainer, InstructionOptimizer, ToolOptimizer, TracerTrajectoryExtractor |
| DeepAgents | ~20 | DeepAgent, TodoTools, CodeTool, BashTool |
| Extensions | ~56 | RedisCheckpointer, ContextEvolver (ACE/RB/ReMe), PulsarMQ |
| Dev Tools | ~34 | PromptBuilder, SkillCreator, Tune framework |
| **Total** | **~540+** | |

---

*Report generated by 12 parallel exploration agents analyzing every file in the agent-core codebase.*
