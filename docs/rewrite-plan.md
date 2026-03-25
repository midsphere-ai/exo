# Exo: Granular Rewrite Plan

## Context

AWorld (96,500 LOC) is being rewritten as **Exo** — a modern, minimal, multi-package agent framework. This plan breaks the work into **unit-level sessions** (~100-150 lines of new code each) for maximum accuracy. The `old/` directory is read-only reference — understand the logic, rewrite clean. No copy-paste-modify.

**Decisions:**
- Port ALL examples (including environment-specific ones)
- Keep A2A protocol from experimental; keep AMNI context engine (rewritten clean as exo-context); drop CAST, PTC, continuous
- Training: port core + VeRL integration only (drop Swift, AREAL, TRL)
- ~100-150 lines new code per session

---

## Package Architecture

```
exo/                          # Monorepo root
├── pyproject.toml                # UV workspace + shared tool config
├── .python-version               # 3.11
├── .pre-commit-config.yaml
├── CLAUDE.md
├── packages/
│   ├── exo-core/             # Agent, Tool, Runner, Config, Events, Hooks, Swarm
│   ├── exo-models/           # LLM provider abstractions
│   ├── exo-context/          # Context engine (prompt building, state, RAG, checkpoints)
│   ├── exo-memory/           # Memory backends
│   ├── exo-sandbox/          # Sandboxed execution
│   ├── exo-trace/            # Observability
│   ├── exo-eval/             # Evaluation framework
│   ├── exo-mcp/              # MCP client/tools
│   ├── exo-a2a/              # Agent-to-Agent protocol
│   ├── exo-cli/              # CLI agent runner
│   ├── exo-server/           # Web UI + API server
│   ├── exo-train/            # Training framework (core + VeRL)
│   └── exo/                  # Meta-package (installs all above)
├── examples/
├── docs/
└── old/                          # Read-only reference (deleted at end)
```

---

## Session Plan

### Phase 0: Planning & Documentation
> Goal: Establish CLAUDE.md, planning docs, and session guide. No code.

**Session 0.1** — Write CLAUDE.md for the Exo project with build instructions, architecture overview, and session references. Write this plan document.

---

### Phase 1: Monorepo Infrastructure [COMPLETE]

**Session 1.1 — Root pyproject.toml + tooling** [DONE]
- Root `pyproject.toml` (UV workspace, ruff, pyright, pytest config)
- `.python-version` (3.11)
- `.pre-commit-config.yaml` (ruff hooks)

**Session 1.2 — Package pyproject.toml files (batch 1)** [DONE]
- `packages/exo-core/pyproject.toml`
- `packages/exo-models/pyproject.toml`
- `packages/exo-memory/pyproject.toml`
- `packages/exo-mcp/pyproject.toml`
- Minimal `__init__.py` stubs so packages are importable

**Session 1.3 — Package pyproject.toml files (batch 2)** [DONE]
- `packages/exo-sandbox/pyproject.toml`
- `packages/exo-trace/pyproject.toml`
- `packages/exo-eval/pyproject.toml`
- `packages/exo-a2a/pyproject.toml`
- `packages/exo-cli/pyproject.toml`
- `packages/exo-server/pyproject.toml`
- `packages/exo-train/pyproject.toml`
- `packages/exo/pyproject.toml` (meta-package)
- Minimal `__init__.py` stubs
- **Verified:** `uv sync` succeeds, all 12 packages importable, ruff clean

---

### Phase 2: Core Types & Config [COMPLETE]

**Reference files:**
- `old/aworld/core/event/base.py` (305 LOC, 11 classes — message types, routing)
- `old/aworld/config/conf.py` (415 LOC, 13 classes — Pydantic configs)
- `old/aworld/core/common.py` (ActionModel, Observation types)
- `old/aworld/core/factory.py` (89 LOC — registry pattern)

**Session 2.1 — Core types: Messages** [DONE]
- `packages/exo-core/src/exo/types.py`
- Message types: `UserMessage`, `AssistantMessage`, `SystemMessage`, `ToolCall`, `ToolResult`
- Base `Message` union type
- ~100 lines
- **Verify:** pytest on message creation, serialization, type narrowing

**Session 2.2 — Core types: Agent I/O & Actions** [DONE]
- Add to `exo/types.py`: `AgentInput`, `AgentOutput`, `ActionModel`, `RunResult`
- Streaming event types: `TextEvent`, `ToolCallEvent`, `StreamEvent`
- ~100 lines
- **Verify:** pytest on all new types

**Session 2.3 — Config: AgentConfig & ModelConfig** [DONE]
- `packages/exo-core/src/exo/config.py`
- `AgentConfig` (name, model, instructions, tools, temperature, max_tokens, etc.)
- `ModelConfig` (provider, model, api_key, base_url, etc.)
- Pure Pydantic v2, sensible defaults
- ~120 lines
- **Verify:** pytest on config creation, validation, defaults, serialization

**Session 2.4 — Config: TaskConfig & RunConfig** [DONE]
- Add to `exo/config.py`: `TaskConfig`, `RunConfig`, `MemoryConfig`
- ~100 lines
- **Verify:** pytest on all config models

**Session 2.5 — Registry** [DONE]
- `packages/exo-core/src/exo/registry.py`
- Single generic `Registry[T]` class replacing 8+ factories
- `register()`, `get()`, `__contains__()`, decorator form
- Global `agent_registry` and `tool_registry` instances
- ~80 lines
- **Verify:** pytest on register/get/contains/decorator

**Session 2.6 — Events** [DONE]
- `packages/exo-core/src/exo/events.py`
- Simple async in-memory event bus: `EventBus` with `emit()`, `on()`, `off()`
- Typed event names (enum or literal)
- ~80 lines
- **Verify:** pytest on emit/subscribe/unsubscribe, async event handling

**Session 2.7 — Hooks** [DONE]
- `packages/exo-core/src/exo/hooks.py`
- `HookPoint` enum (START, FINISHED, ERROR, PRE_LLM_CALL, POST_LLM_CALL, PRE_TOOL_CALL, POST_TOOL_CALL)
- `Hook` protocol, `HookManager` class
- `run_hooks()` async function
- ~100 lines
- **Verify:** pytest on hook registration, execution order, async hooks

---

### Phase 3: Tool System [COMPLETE]

**Reference files:**
- `old/aworld/core/tool/base.py` (835 LOC, 6 classes — dual sync/async, 6-level call chain)
- `old/aworld/tools/function_tools.py` (495 LOC — function wrapping, schema generation)
- `old/aworld/tools/function_tools_executor.py` (144 LOC)

**Session 3.1 — Tool base class + @tool decorator** [DONE]
- `packages/exo-core/src/exo/tool.py`
- `Tool` base class with async `execute()` method
- `@tool` decorator that wraps a Python function into a `FunctionTool`
- ~120 lines
- **Verify:** pytest — decorate sync/async functions, verify Tool interface

**Session 3.2 — Schema generation + tool execution** [DONE]
- Add to `exo/tool.py`: JSON Schema generation from function signatures
- `FunctionTool.execute()` — call the wrapped function, handle sync/async
- Type mapping (Python types → JSON Schema)
- ~120 lines
- **Verify:** pytest — schema generation for various signatures, execution of sync/async tools

**Session 3.3 — Tool tests + tool registry integration** [DONE]
- `packages/exo-core/tests/test_tool.py`
- Wire `FunctionTool` into `tool_registry`
- Test edge cases: optional params, default values, complex types, docstring extraction
- ~100 lines tests
- **Verify:** full pytest pass

---

### Phase 4: LLM Models Package [COMPLETE]

**Reference files:**
- `old/aworld/models/llm.py` (752 LOC — LLMModel, provider detection, sync/async/stream)
- `old/aworld/models/openai_provider.py` (693 LOC)
- `old/aworld/models/anthropic_provider.py` (333 LOC)
- `old/aworld/models/model_response.py` (715 LOC)
- `old/aworld/models/utils.py` (445 LOC — token counting)

**Session 4.1 — Model types** [DONE]
- `packages/exo-models/src/exo/models/types.py`
- `ModelResponse`, `Usage`, `ToolCallRequest`, `StreamChunk`
- Provider-agnostic response types
- ~100 lines
- **Verify:** pytest on type creation/serialization

**Session 4.2 — Provider base** [DONE]
- `packages/exo-models/src/exo/models/provider.py`
- `ModelProvider` abstract base: `async complete()`, `async stream()`
- `get_provider()` factory function (provider detection from model name/config)
- ~100 lines
- **Verify:** pytest on provider interface

**Session 4.3 — OpenAI provider: completion** [DONE]
- `packages/exo-models/src/exo/models/openai.py`
- `OpenAIProvider` — `async complete()` using openai SDK
- Message formatting, tool call parsing
- ~130 lines
- **Verify:** pytest with mocked openai client

**Session 4.4 — OpenAI provider: streaming** [DONE]
- Add streaming to `OpenAIProvider` — `async stream()` yielding `StreamChunk`
- ~100 lines
- **Verify:** pytest with mocked streaming responses

**Session 4.5 — Anthropic provider: completion** [DONE]
- `packages/exo-models/src/exo/models/anthropic.py`
- `AnthropicProvider` — `async complete()` using anthropic SDK
- Message/tool format conversion (Anthropic uses different format)
- ~130 lines
- **Verify:** pytest with mocked anthropic client

**Session 4.6 — Anthropic provider: streaming** [DONE]
- Add streaming to `AnthropicProvider`
- ~100 lines
- **Verify:** pytest with mocked streaming responses

**Session 4.7 — Models __init__ + integration tests** [DONE]
- `packages/exo-models/src/exo/models/__init__.py` — public API (10 exports)
- Integration tests: public API validation, auto-registration, get_provider() end-to-end, cross-provider consistency
- ~85 lines (13 tests)
- **Verify:** full pytest pass for exo-models (127 tests), full suite (288 tests)

---

### Phase 5: Agent Core

**Reference files:**
- `old/aworld/agents/llm_agent.py` (1,156 LOC — god class with 33 methods)
  - Key methods: `async_policy()` (lines 623-739), `async_messages_transform()` (lines 362-500)
  - Retry logic: `invoke_model()` (lines 868-983)
- `old/aworld/core/agent/base.py` (451 LOC — lifecycle, status, trajectory)
- `old/aworld/core/agent/agent_desc.py` (98 LOC — agent description/capability metadata)
- `old/aworld/tools/human/human.py` (115 LOC — HITL tool)
- `old/aworld/core/model_output_parser/` (~150 LOC — output parsing framework)

**Session 5.1 — Message builder**
- `packages/exo-core/src/exo/_internal/message_builder.py`
- Build LLM message list from: system instructions, conversation history, tool results
- Handle message ordering (user → assistant → tool cycles)
- ~120 lines
- **Verify:** pytest — various conversation histories produce correct message lists

**Session 5.2 — Output parser**
- `packages/exo-core/src/exo/_internal/output_parser.py`
- Parse LLM response into: text output, tool calls, or both
- Extract tool call names, arguments, IDs
- Handle structured output (Pydantic model validation)
- ~100 lines
- **Verify:** pytest — parse text-only, tool-call-only, mixed responses

**Session 5.3 — Agent class: init + configuration**
- `packages/exo-core/src/exo/agent.py`
- `Agent` class with `__init__`: name, model, instructions, tools, hooks, memory, handoffs, output_type
- Tool registration, config validation
- Agent description/metadata: `describe()` method for capability advertisement
- ~100 lines
- **Verify:** pytest — agent creation with various configs

**Session 5.4 — Agent class: run method with retry**
- Add `async run()` to Agent: build messages → call LLM → parse → return
- Single-turn execution (no tool loop yet)
- Wire in hooks (PRE_LLM_CALL, POST_LLM_CALL)
- **LLM retry logic:** configurable `max_retries` with exponential backoff; context-length errors fail immediately, transient errors retry
- ~130 lines
- **Verify:** pytest with mocked model provider — agent returns text output, retry on transient error

**Session 5.5 — Agent class: tool execution loop**
- Add tool dispatch to `Agent.run()`: if LLM returns tool calls → execute → feed back → re-call LLM
- Max iterations guard (`max_steps`)
- PRE_TOOL_CALL / POST_TOOL_CALL hooks
- Parallel tool execution via `asyncio.TaskGroup` when multiple tool calls returned
- ~120 lines
- **Verify:** pytest — agent calls tool, gets result, produces final answer

**Session 5.6 — Agent tests + edge cases**
- `packages/exo-core/tests/test_agent.py`
- Multi-tool calls, parallel tool calls, tool errors, max iterations, retry behavior
- ~120 lines tests
- **Verify:** full pytest pass

**Session 5.7 — Human-in-the-loop tool**
- `packages/exo-core/src/exo/human.py`
- `HumanInputTool` — async tool that blocks for user confirmation/input
- `HumanInputHandler` protocol — pluggable handler for different UIs (CLI, web, etc.)
- Default console handler for interactive use
- ~120 lines
- **Verify:** pytest — tool schema, handler invocation with mocked input

---

### Phase 6: Runner & Execution

**Reference files:**
- `old/aworld/runners/call_driven_runner.py` (851 LOC — WorkflowRunner, HandoffRunner, LoopWorkflowRunner)
- `old/aworld/runners/state_manager.py` (841 LOC — RunNode, RuntimeStateManager, NodeGroup)
- `old/aworld/runner.py` (320 LOC)
- `old/aworld/runners/event_runner.py` (522 LOC — event-driven runner variant)
- `old/aworld/runners/handler/` (~2,400 LOC — 10 handler types: agent, tool, group, memory, human, output, context, background_task, task)
- `old/aworld/runners/runtime_engine.py` (283 LOC — distributed execution backends)

**Session 6.1 — Run state tracking**
- `packages/exo-core/src/exo/_internal/state.py`
- `RunState` — tracks messages, tool calls, iterations, status
- `RunNodeStatus` enum (INIT, RUNNING, SUCCESS, FAILED, TIMEOUT)
- `RunNode` — per-step state with agent_id, group tracking
- ~120 lines
- **Verify:** pytest on state transitions, node lifecycle

**Session 6.2 — Call runner: core loop**
- `packages/exo-core/src/exo/_internal/call_runner.py`
- `async call_runner(agent, input, state)` — the LLM→tool→LLM loop
- Endless loop detection with configurable threshold
- ~120 lines
- **Verify:** pytest — single-turn and multi-turn execution with mocked agent

**Session 6.3 — Public run() entry point**
- `packages/exo-core/src/exo/runner.py`
- `async run(agent_or_swarm, input)` → `RunResult`
- `run.sync()` — sync wrapper using `asyncio.run()`
- ~100 lines
- **Verify:** pytest — `run()` and `run.sync()` both work

**Session 6.4 — Streaming run**
- `run.stream()` — async generator yielding `StreamEvent`
- ~120 lines
- **Verify:** pytest — streaming produces text/tool events in order

**Session 6.5 — Handler system: base + agent handler**
- `packages/exo-core/src/exo/_internal/handlers.py`
- `Handler[IN, OUT]` ABC with `async handle()` → `AsyncGenerator`
- `AgentHandler` — routes between agents in swarm, handles handoff dispatch
- Swarm topology-aware stop checks (workflow/handoff/team modes)
- ~130 lines
- **Verify:** pytest — handler dispatch, agent routing

**Session 6.6 — Handler system: tool + group handlers**
- `ToolHandler` — dynamic tool loading, execution, result aggregation
- `GroupHandler` — parallel agent/tool group execution with dependency resolution
- ~130 lines
- **Verify:** pytest — parallel execution, dependency ordering

**Session 6.7 — Background task handler**
- `packages/exo-core/src/exo/_internal/background.py`
- `BackgroundTaskHandler` — hot-merge (running task) and wake-up-merge (checkpoint restore) patterns
- Pending message queue for background results
- Integration with checkpoint system
- ~120 lines
- **Verify:** pytest — hot-merge, wake-up-merge, pending message handling

**Session 6.8 — Runner integration tests**
- End-to-end tests: `Agent` + `@tool` + `run()` with mocked LLM
- Handler pipeline tests, background task scenarios
- ~100 lines tests
- **Verify:** full pytest pass for exo-core

---

### Phase 7: Swarm / Multi-Agent

**Reference files:**
- `old/aworld/core/agent/swarm.py` (1,211 LOC — 8 classes, 3 topology types)
- `old/aworld/agents/loop_llm_agent.py` (65 LOC — self-loop capability)
- `old/aworld/agents/parallel_llm_agent.py` (79 LOC — concurrent agent execution)
- `old/aworld/agents/serial_llm_agent.py` (72 LOC — sequential agent chaining)
- `old/aworld/agents/task_llm_agent.py` (62 LOC — nested swarm-as-agent)
- `old/aworld/agents/swarm_composer_agent.py` (1,042 LOC — LLM-based YAML planning)

**Session 7.1 — Graph utilities**
- `packages/exo-core/src/exo/_internal/graph.py`
- Simple adjacency list graph
- `topological_sort()` (Kahn's algorithm), cycle detection
- `parse_flow_dsl()` — parse `"a >> b >> c"` into edges
- ~100 lines
- **Verify:** pytest — topo sort, cycle detection, DSL parsing

**Session 7.2 — Swarm class: workflow mode**
- `packages/exo-core/src/exo/swarm.py`
- `Swarm` class with `mode="workflow"`
- Takes agents list + `flow="a >> b >> c"` DSL
- Sequential execution: run agents in topological order, pass output as input
- ~120 lines
- **Verify:** pytest — workflow executes agents in order

**Session 7.3 — Swarm: handoff mode**
- Add `mode="handoff"` to Swarm
- Agents can return a handoff to another agent
- Endless loop detection with configurable threshold
- ~120 lines
- **Verify:** pytest — agent A hands off to agent B

**Session 7.4 — Swarm: team mode**
- Add `mode="team"` to Swarm
- Lead agent coordinates, can delegate to team members
- ~120 lines
- **Verify:** pytest — lead delegates, members respond, lead synthesizes

**Session 7.5 — Agent grouping: parallel + serial**
- `packages/exo-core/src/exo/_internal/agent_group.py`
- `ParallelGroup` — concurrent agent execution via `asyncio.TaskGroup`, custom result aggregation
- `SerialGroup` — dependency-based sequential execution with output→input chaining
- Both integrate as nodes in Swarm flow DSL (e.g., `"(a | b) >> c"` for parallel a,b then serial c)
- ~120 lines
- **Verify:** pytest — parallel execution, serial chaining, mixed topologies

**Session 7.6 — Nested swarms (TaskAgent pattern)**
- `packages/exo-core/src/exo/_internal/nested.py`
- Allow `Swarm` to be used as an agent within another `Swarm` (hierarchical nesting)
- Recursive execution with context isolation
- ~100 lines
- **Verify:** pytest — nested swarm execution, 2-level hierarchy

**Session 7.7 — Swarm integration + runner wiring**
- Wire `Swarm` into `run()` — detect if input is Agent or Swarm
- `packages/exo-core/src/exo/__init__.py` — public API exports
- Integration tests for all swarm modes including parallel/serial groups and nesting
- ~100 lines
- **Verify:** full end-to-end test

---

### Phase 8: Context Engine

**Reference files:**
- `old/aworld/core/context/base.py` (913 LOC — Context class, token trajectories, forking/merging)
- `old/aworld/core/context/context_state.py` (232 LOC — hierarchical state with parent inheritance)
- `old/aworld/core/context/amni/config.py` (~200 LOC — AgentContextConfig, automation levels)
- `old/aworld/core/context/amni/contexts.py` (~500 LOC — ContextManager, checkpoint, memory integration)
- `old/aworld/core/context/amni/state/` (~400 LOC — TaskInput, TaskWorkingState, AgentState, SubTask)
- `old/aworld/core/context/amni/prompt/neurons/` (~600 LOC — 12+ neuron implementations)
- `old/aworld/core/context/amni/processor/op/` (~400 LOC — SystemPromptAugment, ToolResultOffload)
- `old/aworld/core/context/amni/services/` (~300 LOC — Memory, Prompt, Skill, TaskState services)
- `old/aworld/core/context/amni/retrieval/` (~800 LOC — RAG pipeline: chunking, embeddings, vector, reranker)
- `old/aworld/core/context/amni/worksapces.py` (~200 LOC — workspace artifact management)
- `old/aworld/core/context/amni/tool/` (~400 LOC — context-specific tools)
- `old/aworld/core/context/prompts/` (~500 LOC — prompt templates, dynamic variables)

**Session 8.1 — ContextConfig + ContextState**
- `packages/exo-context/src/exo/context/config.py`
- `ContextConfig` — Pydantic v2 model (mode, history_rounds, summary_threshold, offload_threshold, enable_retrieval, neuron_names)
- `packages/exo-context/src/exo/context/state.py`
- `ContextState` — hierarchical key-value state with parent inheritance (get/set/local_dict/to_dict)
- ~120 lines
- **Verify:** pytest — config creation/validation, state inheritance, local vs parent, merge

**Session 8.2 — Context class: core lifecycle**
- `packages/exo-context/src/exo/context/context.py`
- `Context` class: task_id, config, state, parent reference
- `fork()` — create child context with state inheritance
- `merge()` — consolidate child state back into parent (net token calculation)
- ~130 lines
- **Verify:** pytest — context creation, fork, merge, hierarchical state

**Session 8.3 — TokenTracker**
- `packages/exo-context/src/exo/context/token_tracker.py`
- `TokenTracker` — per-agent, per-step token tracking
- `TokenStep` — prompt_tokens, output_tokens per step
- `add_step()`, `get_trajectory()`, `total_usage()`
- ~100 lines
- **Verify:** pytest — step tracking, usage aggregation, multi-agent

**Session 8.4 — Neuron base + core built-in neurons**
- `packages/exo-context/src/exo/context/neuron.py`
- `Neuron` ABC with `async format(ctx) -> str` and `priority: int`
- `neuron_registry` — `Registry[Neuron]` for neuron discovery
- Built-in core: `SystemNeuron` (date/time/platform), `TaskNeuron` (task info), `HistoryNeuron` (conversation windowing)
- ~130 lines
- **Verify:** pytest — neuron formatting, priority ordering, registry

**Session 8.4b — Extended neurons**
- Additional neurons: `SkillNeuron` (active skills/capabilities), `WorkspaceNeuron` (artifact listing), `TodoNeuron` (task plan/checklist), `FactNeuron` (long-term facts), `EntityNeuron` (extracted entities), `KnowledgeNeuron` (RAG results)
- Dynamic variable system: `DynamicVariableRegistry` with nested path resolution, processor/formatter pipelines
- ~130 lines
- **Verify:** pytest — each neuron type produces correct prompt fragment

**Session 8.5 — PromptBuilder**
- `packages/exo-context/src/exo/context/prompt_builder.py`
- `PromptBuilder` — composable prompt construction via `add(neuron_name, **kwargs)`
- `async build()` — resolve all neurons in priority order, compose final prompt
- Template variable resolution with hierarchical context traversal
- ~120 lines
- **Verify:** pytest — add neurons, build prompt, variable resolution

**Session 8.6 — ContextProcessor pipeline**
- `packages/exo-context/src/exo/context/processor.py`
- `ContextProcessor` ABC with `async process(ctx, payload) -> None`
- `ProcessorPipeline` — event-driven execution (pre_llm_call, post_tool_call, etc.)
- Built-in: `SummarizeProcessor`, `ToolResultOffloader`
- ~130 lines
- **Verify:** pytest — processor registration, event filtering, pipeline execution

**Session 8.7 — Workspace + artifact system**
- `packages/exo-context/src/exo/context/workspace.py`
- `Workspace` — artifact storage (write, read, list, delete)
- `ArtifactType` enum (TEXT, CODE, MARKDOWN, JSON, CSV, IMAGE, etc.)
- Artifact versioning: `version_history`, `revert_to_version()`
- Observer pattern: `on_create`, `on_update`, `on_delete` callbacks
- Support local filesystem backend (OSS backend as future extension)
- ~130 lines
- **Verify:** pytest — write/read/list artifacts, versioning, observer notifications

**Session 8.7b — Workspace-retriever integration**
- Wire workspace into RAG pipeline: artifacts added to workspace are auto-indexed in KnowledgeStore
- Chunk range queries for large artifacts
- ~100 lines
- **Verify:** pytest — artifact → chunk → search round-trip

**Session 8.8 — Checkpoint**
- `packages/exo-context/src/exo/context/checkpoint.py`
- `Checkpoint` — serialized context snapshot (values, metadata, version)
- `Context.snapshot()` / `Context.restore()` — save/restore execution state
- ~100 lines
- **Verify:** pytest — snapshot, restore, version incrementing

**Session 8.9 — Knowledge store + RAG basics**
- `packages/exo-context/src/exo/context/_internal/knowledge.py`
- `KnowledgeStore` — add artifacts, semantic search, range queries
- Basic chunking + in-memory vector store for testing
- ~130 lines
- **Verify:** pytest — add/search/get artifacts

**Session 8.10 — Context tools**
- `packages/exo-context/src/exo/context/tools.py`
- `planning_tool` — add_todo, get_todo (task planning checklist)
- `knowledge_tool` — get_knowledge, grep_knowledge (artifact search)
- `file_tool` — read_file from working directory
- ~120 lines
- **Verify:** pytest — tool execution, context mutation

**Session 8.11 — Context __init__ + integration tests**
- `packages/exo-context/src/exo/context/__init__.py` — public API exports
- Integration tests: Context + PromptBuilder + Processor + Workspace end-to-end
- Wire context into Agent (agent.py context parameter)
- ~100 lines
- **Verify:** full pytest pass for exo-context

---

### Phase 9: Memory Package

**Reference files:**
- `old/aworld/memory/main.py` (928 LOC — AworldMemory orchestrator, summary logic, vector integration)
- `old/aworld/memory/models.py` (592 LOC — MemoryItem hierarchy, UserProfile, Fact, AgentExperience)
- `old/aworld/memory/db/sqlite.py` (426 LOC)
- `old/aworld/memory/db/postgres.py` (375 LOC)
- `old/aworld/memory/longterm/` (~260 LOC — MemoryOrchestrator, MemoryGungnir extraction)
- `old/aworld/memory/embeddings/` (~100 LOC — embedding providers)
- `old/aworld/memory/vector/dbs/` (~200 LOC — Chroma, Qdrant backends)

**Session 9.1 — Memory interface + types**
- `packages/exo-memory/src/exo/memory/base.py`
- `MemoryStore` protocol: `add()`, `get()`, `search()`, `clear()`
- `MemoryItem` model with subtypes: `SystemMemory`, `HumanMemory`, `AIMemory`, `ToolMemory`
- `MemoryMetadata` with user_id, session_id, task_id, agent_id scoping
- Status lifecycle: DRAFT → ACCEPTED → DISCARD
- ~120 lines
- **Verify:** pytest on types, status transitions

**Session 9.2 — Short-term memory**
- `packages/exo-memory/src/exo/memory/short_term.py`
- `ShortTermMemory` — conversation context management
- Scope-based filtering: user, session, task
- Incomplete message pair filtering (tool call/response integrity for LLM API compatibility)
- ~130 lines
- **Verify:** pytest — add messages, truncation, windowing, scope filtering

**Session 9.3 — Summary + compression**
- `packages/exo-memory/src/exo/memory/summary.py`
- Summary trigger logic: message count threshold, token count threshold, incomplete pair check
- Multi-template summary generation with typed summaries (conversation, facts, profiles)
- `SummaryConfig` — prompts, thresholds, compression rules
- ~130 lines
- **Verify:** pytest — trigger detection, summary generation with mocked LLM

**Session 9.4 — Long-term memory: orchestrator + extraction**
- `packages/exo-memory/src/exo/memory/long_term.py`
- `LongTermMemory` — persistent memory across sessions
- `MemoryOrchestrator` — async processing tasks for extracting UserProfile, AgentExperience, Facts from conversations
- Processing task queue with status tracking (initial/processing/completed/failed)
- ~130 lines
- **Verify:** pytest with in-memory backend, mocked LLM extraction

**Session 9.5 — SQLite backend**
- `packages/exo-memory/src/exo/memory/backends/sqlite.py`
- `SQLiteMemoryStore` implementing `MemoryStore`
- JSON indexes for metadata fields, soft deletes, version field
- ~120 lines
- **Verify:** pytest with temp SQLite database

**Session 9.6 — Postgres backend**
- `packages/exo-memory/src/exo/memory/backends/postgres.py`
- `PostgresMemoryStore`
- ~120 lines
- **Verify:** pytest (may need mock or skip if no postgres)

**Session 9.7 — Embeddings + vector search**
- `packages/exo-memory/src/exo/memory/backends/vector.py`
- `VectorMemoryStore` — wraps embedding + vector DB
- `Embeddings` ABC with sync + async variants
- OpenAI-compatible embedding provider with dimension support
- ~130 lines
- **Verify:** pytest with mocked embeddings

**Session 9.8 — Memory __init__ + integration**
- Public API, wire memory into Agent
- Memory event integration (emit memory events for async processing)
- ~80 lines
- **Verify:** full pytest pass for exo-memory

---

### Phase 10: MCP Integration

**Reference files:**
- `old/aworld/mcp_client/decorator.py` (233 LOC — @mcp_server decorator, method→tool conversion)
- `old/aworld/mcp_client/utils.py` (1,460 LOC — tool descriptor transforms, server lifecycle, retry logic)
- `old/aworld/tools/mcp_tool/async_mcp_tool.py` (148 LOC)
- `old/aworld/tools/mcp_tool/executor.py` (281 LOC)

**Session 10.1 — MCP client: server connection** (~130 lines)
- Multiple transport types: SSE, stdio, streamable-http
- Server instance caching/reuse with session isolation
**Session 10.2 — MCP tools: loading + conversion** (~120 lines)
- Tool schema extraction, `mcp__` namespace mapping
- Tool black/white-list filtering by skill config
**Session 10.3 — MCP server decorator** (~120 lines)
- `@mcp_server()` class decorator converting Python methods to MCP tools
- `MCPServerRegistry` for singleton server instances
**Session 10.4 — MCP execution + tests** (~100 lines)
- Retry logic with configurable timeout, sandbox integration fallback
- Environment variable substitution in mcp.json config

---

### Phase 10.5: Config-Driven Loading & Skill Registry (NEW)

> Goal: YAML-based agent/task composition and multi-source skill management. Enables non-programmatic agent creation and reusable skill libraries.

**Reference files:**
- `old/aworld/config/agent_loader.py` (196 LOC — YAML agent/swarm definitions)
- `old/aworld/config/task_loader.py` (537 LOC — skill loading, MCP config merging)
- `old/aworld/utils/skill_loader.py` (822 LOC — multi-source skill registry, GitHub cloning)

**Session 10.5a — YAML agent loader**
- `packages/exo-core/src/exo/loader.py`
- Load agent/swarm definitions from YAML with `${ENV_VAR}` and `${vars.KEY}` substitution
- Swarm topology patterns: workflow, handoff, team
- Agent factory dispatch (builtin vs. custom classes)
- ~130 lines
- **Verify:** pytest — YAML parsing, variable substitution, swarm creation

**Session 10.5b — Skill registry**
- `packages/exo-core/src/exo/skills.py`
- `SkillRegistry` — multi-source skill management (local paths, GitHub URLs)
- GitHub URL parsing & shallow clone with branch support, cached at `~/.exo/skills/`
- YAML front-matter extraction (name, desc, tool_list, type, active)
- Conflict resolution strategies (keep_first, keep_last, raise)
- Search + filtering capabilities
- ~130 lines
- **Verify:** pytest — local skill loading, registry operations, search

---

### Phase 11: Sandbox Package

**Reference files:**
- `old/aworld/sandbox/base.py` (628 LOC — Sandbox ABC with 15+ abstract methods)
- `old/aworld/sandbox/builder/sandbox_builder.py` (324 LOC — fluent builder API)
- `old/aworld/sandbox/builtin/filesystem.py` (~250 LOC — sandboxed filesystem tool)
- `old/aworld/sandbox/builtin/terminal.py` (~150 LOC — sandboxed terminal tool)
- `old/aworld/sandbox/implementations/` (local, kubernetes, super sandbox)

**Session 11.1 — Sandbox interface + local sandbox** (~130 lines)
- `SandboxStatus` enum (INIT, RUNNING, IDLE, ERROR, CLOSED)
- Sandbox ABC with workspace, MCP integration, agent configuration
**Session 11.2 — Built-in sandbox tools** (~130 lines)
- `FilesystemTool` with allowed_directories sandboxing, safe path validation
- `TerminalTool` with dangerous command blacklist, platform detection, timeout
**Session 11.3 — Sandbox builder** (~120 lines)
- Fluent API for sandbox construction with method chaining
- Lazy evaluation: auto-build on first API call
**Session 11.4 — Kubernetes sandbox** (~120 lines)

---

### Phase 12: Trace Package

**Reference files:**
- `old/aworld/trace/function_trace.py` (181 LOC — decorator-based tracing)
- `old/aworld/trace/instrumentation/semconv.py` (49 LOC — gen_ai.* semantic conventions)
- `old/aworld/trace/instrumentation/agent/__init__.py` (~100 LOC — agent metrics)
- `old/aworld/trace/instrumentation/tool/__init__.py` (~80 LOC — tool metrics)
- `old/aworld/trace/asyncio_monitor/` (~476 LOC — real-time async monitoring)
- `old/aworld/trace/baggage/` (~195 LOC — W3C + SofaTracer propagation)
- `old/aworld/trace/stack_info.py` (91 LOC — user-code frame extraction)
- `old/aworld/logs/prompt_log.py` (873 LOC — structured LLM execution logging)
- `old/aworld/metrics/metric.py` (288 LOC — metric provider ABC)

**Session 12.1 — Trace config + base** (~120 lines)
- Semantic conventions for gen_ai.*, agent.*, tool.* attributes
- `TraceConfig` with backend selection, sampling, export settings
**Session 12.2 — Span decorator + context manager** (~120 lines)
- `@traced` decorator supporting sync, async, generators, async generators
- Function metadata extraction (qualname, module, line number, parameters)
- Stack frame analysis with user-code filtering (skip library/framework frames)
**Session 12.3 — Agent/tool instrumentation** (~120 lines)
- Agent metrics: `agent_run_duration` (histogram), `agent_run_counter`, `agent_token_usage`
- Tool metrics: `tool_step_duration`, `tool_step_counter`
**Session 12.4 — Trace context propagation** (~100 lines)
- W3C Baggage standard (RFC 9110) for cross-service correlation
- Span consumer plugin system with `@register_span_consumer` decorator
**Session 12.5 — Prompt execution logger** (~130 lines)
- Structured LLM execution logging: token breakdown by role, context window usage analysis
- Multi-modal content logging (text, images, tool_use)

---

### Phase 13: Evaluation Package

**Reference files:**
- `old/aworld/evaluations/base.py` (~482 LOC — Evaluator, EvalTarget, EvalCriteria)
- `old/aworld/evaluations/scorers/` (~1,200 LOC — 11 scorer types)
- `old/aworld/evaluations/reflect/` (~474 LOC — reflection framework)
- `old/aworld/evaluations/types.py` (~120 LOC — 29+ metric names)

**Session 13.1 — Eval types + base evaluator** (~100 lines)
- `EvalTarget` ABC, `EvalCriteria` with threshold-based pass/fail
- `Evaluator` with parallel execution, repeat_times, pass@k metrics
- `EvalResult`, `EvalCaseResult`, `ScorerResult` dataclasses
**Session 13.2 — Rule-based scorers** (~130 lines)
- `FormatValidationScorer` — JSON, XML, YAML, Markdown, CSV format checking
- `SchemaValidationScorer` — JSON schema compliance
- `OutputCorrectnessScorer` — ground truth matching, keyword checking, number extraction
- `OutputLengthScorer`, `OutputRelevanceScorer`, `OutputCompletenessScorer`
**Session 13.3 — LLM-as-Judge + quality scorers** (~130 lines)
- `LLMAsJudgeScorer` — configurable judge prompts
- `OutputQualityScorer` — weighted 5-dimensional: correctness (40%), relevance (20%), completeness (20%), clarity (10%), professionalism (10%)
- `LogicConsistencyScorer`, `ReasoningValidityScorer`, `ConstraintSatisfactionScorer`
**Session 13.4 — Trajectory + time scorers** (~100 lines)
- `TrajectoryValidators` — trajectory step validation
- `TimeCostScorer`, `AnswerAccuracyLLMScorer`, `LabelDistributionScorer`
- Scorer registry with `@scorer_register()` decorator
**Session 13.5 — Reflection framework** (~130 lines)
- `Reflector` ABC with three-step template: `analyze()`, `insight()`, `suggest()`
- `GeneralReflector` using LLM to extract: summary, key_findings, root_causes, insights, suggestions
- `ReflectionType` enum (SUCCESS, FAILURE, OPTIMIZATION, PATTERN, INSIGHT)
- `ReflectionLevel` enum (SHALLOW, MEDIUM, DEEP, META)
- `ReflectionHistory` tracking with summarization

---

### Phase 13.5: Ralph Loop — Iterative Refinement (NEW)

> Goal: Run-Analyze-Learn-Plan-Halt execution pattern for production-quality agent refinement. This is a core differentiator — enables agents to self-improve across iterations.

**Reference files:**
- `old/aworld/ralph_loop/ralph_runner.py` (425 LOC — 5-phase RALPH loop)
- `old/aworld/ralph_loop/config.py` (119 LOC — RalphConfig with validation/reflection/stop configs)
- `old/aworld/ralph_loop/detect/` (~200 LOC — stop condition detectors)
- `old/aworld/ralph_loop/state/` (~100 LOC — LoopContext, LoopState)

**Session 13.5a — Ralph loop: state + config**
- `packages/exo-eval/src/exo/eval/ralph/config.py`
- `RalphConfig` unifying: `ValidationConfig` (scorers, min_score_threshold), `ReflectionConfig` (reflectors, level), `StopConditionConfig` (max_iterations, timeout, max_cost, max_consecutive_failures)
- `LoopState` — iteration tracking, score history, reflection history
- ~120 lines
- **Verify:** pytest — config creation, state transitions

**Session 13.5b — Ralph loop: stop detectors**
- `packages/exo-eval/src/exo/eval/ralph/detectors.py`
- `StopDetector` ABC with pluggable implementations
- Built-in: `MaxIterationDetector`, `TimeoutDetector`, `CostLimitDetector`, `ConsecutiveFailureDetector`, `ScoreThresholdDetector`
- ~100 lines
- **Verify:** pytest — each detector type, composite detection

**Session 13.5c — Ralph loop: runner**
- `packages/exo-eval/src/exo/eval/ralph/runner.py`
- `RalphRunner` implementing 5-phase loop: Run → Analyze (score) → Learn (reflect) → Plan (re-prompt) → Halt (detect stop)
- Integration with Evaluator scorers and Reflection framework
- ~130 lines
- **Verify:** pytest — full loop execution with mocked agent, early stopping, score improvement

---

### Phase 14: A2A Protocol

**Reference files:**
- `old/aworld/experimental/a2a/agent_executor.py` (115 LOC)
- `old/aworld/experimental/a2a/agent_server.py` (220 LOC)
- `old/aworld/experimental/a2a/client_manager.py` (209 LOC)
- `old/aworld/experimental/a2a/client_proxy.py` (217 LOC)
- `old/aworld/experimental/a2a/remote_agent.py` (129 LOC)
- `old/aworld/experimental/a2a/config.py` (55 LOC)

**Session 14.1 — A2A types + agent card** (~130 lines)
- `AgentCard` with skills, transport modes, streaming capabilities
- `ServingConfig`, `ClientConfig` Pydantic models
- Task event types: `TaskArtifactUpdateEvent`, `TaskStatusUpdateEvent`
**Session 14.2 — A2A server** (~130 lines)
- FastAPI-based server with agent card endpoint (`.well-known/agent-card`)
- Agent executor wrapping for A2A protocol, streaming support
- `TaskStore` abstraction (in-memory default)
**Session 14.3 — A2A client + remote agent** (~130 lines)
- Thread-safe client manager with per-thread instances, cleanup on thread death
- Agent card resolution from URL/file
- `RemoteAgent` — BaseAgent subclass for calling remote A2A agents
- Task→A2A message conversion, streaming event handling

---

### Phase 15: CLI Package

**Session 15.1 — CLI entry point + config** (~130 lines)
**Session 15.2 — Agent discovery + loading** (~130 lines)
**Session 15.3 — Interactive console** (~130 lines)
**Session 15.4 — Local executor** (~120 lines)
**Session 15.5 — Plugin system** (~100 lines)
**Session 15.6 — Batch execution** (~120 lines)

---

### Phase 16: Server Package

**Session 16.1 — FastAPI app + chat route** (~130 lines)
**Session 16.2 — Session management route** (~120 lines)
**Session 16.3 — Agent management + workspace routes** (~120 lines)
**Session 16.4 — Streaming + WebSocket support** (~100 lines)

---

### Phase 17: Training Package

**Reference files:**
- `old/aworld/dataset/trajectory_dataset.py` (502 LOC — trajectory capture, strategy pattern)
- `old/aworld/dataset/trajectory_strategy.py` (~100 LOC — trajectory generation strategies)
- `old/aworld/dataset/types.py`, `dataset.py`, `dataloader.py`, `sampler.py`

**Session 17.1 — Trajectory dataset** (~130 lines)
- `TrajectoryItem` model, `TrajectoryDataset` with strategy pattern
- `append_trajectory()`, `from_messages()`, `save_task_trajectory()`
- Export to JSON/CSV with storage integration
**Session 17.2 — Base trainer** (~120 lines)
**Session 17.3 — Data synthesis: core** (~130 lines)
**Session 17.4 — Agent evolution** (~120 lines)
**Session 17.5 — VeRL integration** (~130 lines)

---

### Phase 18: Examples (Quick-Start)

**Session 18.1 — Basic examples: define_agent, use_llm** (~80 lines)
**Session 18.2 — Tool examples: local_tool, mcp_tool** (~80 lines)
**Session 18.3 — Swarm examples: workflow, handoff, hybrid_swarm** (~80 lines)
**Session 18.4 — Memory + trace examples** (~80 lines)
**Session 18.5 — Advanced quickstart: parallel_task, HITL, multi_root_agent, serving** (~100 lines)
**Session 18.6 — Config-driven + CLI examples** (~80 lines)

---

### Phase 19: Examples (Multi-Agent Patterns)

**Session 19.1 — Collaborative examples: debate, travel** (~100 lines)
**Session 19.2 — Coordination examples: custom_agent, deepresearch, master_worker** (~100 lines)
**Session 19.3 — Workflow examples: search patterns** (~60 lines)

---

### Phase 20: Examples (Benchmarks)

**Session 20.1 — GAIA benchmark** (~150 lines)
**Session 20.2 — IMO benchmark** (~150 lines)
**Session 20.3 — OSWorld benchmark** (~150 lines)
**Session 20.4 — VisualWebArena benchmark** (~150 lines)
**Session 20.5 — XBench benchmark** (~150 lines)
**Session 20.6 — BFCL + other benchmarks** (~120 lines)

---

### Phase 21: Examples (Advanced)

**Session 21.1 — Skill agent + web deployment examples** (~120 lines)
**Session 21.2 — Common tools library** (~150 lines)
**Session 21.3 — Training examples** (~120 lines)

---

### Phase 22: Final Cleanup & CI

**Session 22.1 — Public API finalization** (~50 lines)
**Session 22.2 — GitHub Actions CI** (~80 lines)
**Session 22.3 — README + migration guide** (~150 lines)
**Session 22.4 — Delete old/ + final verification**

---

## Session Verification Checklist (Every Session)

```bash
uv run ruff check .                 # zero lint errors
uv run ruff format --check .        # all formatted
uv run pyright                      # no critical type errors
uv run pytest                       # all tests pass
```

---

## Total Session Count

| Phase | Sessions | Description | Status |
|-------|----------|-------------|--------|
| 0 | 1 | Planning & CLAUDE.md | DONE |
| 1 | 3 | Monorepo infrastructure | DONE |
| 2 | 7 | Core types & config | DONE |
| 3 | 3 | Tool system | DONE |
| 4 | 7 | LLM models | DONE |
| 5 | 7 | Agent core (+HITL, +retry) | TODO |
| 6 | 8 | Runner & execution (+handlers, +background tasks) | TODO |
| 7 | 7 | Swarm / multi-agent (+parallel/serial groups, +nesting) | TODO |
| 8 | 13 | Context engine (+extended neurons, +artifact versioning) | TODO |
| 9 | 8 | Memory package (+summary, +orchestrator) | TODO |
| 10 | 4 | MCP integration (+server decorator) | TODO |
| 10.5 | 2 | Config-driven loading & skill registry (NEW) | TODO |
| 11 | 4 | Sandbox (+builder, +builtin tools) | TODO |
| 12 | 5 | Trace (+propagation, +prompt logger) | TODO |
| 13 | 5 | Evaluation (+LLM-as-Judge, +quality scorers) | TODO |
| 13.5 | 3 | Ralph Loop — iterative refinement (NEW) | TODO |
| 14 | 3 | A2A protocol (+remote agent, +task store) | TODO |
| 15 | 6 | CLI | TODO |
| 16 | 4 | Server | TODO |
| 17 | 5 | Training (+trajectory dataset) | TODO |
| 18 | 6 | Examples (quickstart) | TODO |
| 19 | 3 | Examples (multi-agent) | TODO |
| 20 | 6 | Examples (benchmarks) | TODO |
| 21 | 3 | Examples (advanced) | TODO |
| 22 | 4 | Cleanup & CI | TODO |
| **Total** | **~127** | | |

---

## Key Rules

1. **No copy-paste-modify.** Read old code for logic understanding, rewrite clean from scratch.
2. **~100-150 lines new code per session.** One small thing, done right.
3. **Every session ends with passing tests.** No broken state between sessions.
4. **Async-first.** Sync wrappers only at `run.sync()` entry point.
5. **Pydantic v2 only.** No custom config classes.
6. **One registry.** Not 8+ factories.
7. **Clean English.** All docstrings, comments, error messages.

---

## Reference: Source Files by Session

| New Module | Reference Source (old/) | Complexity |
|---|---|---|
| `exo/types.py` | `core/event/base.py`, `core/common.py` | Medium |
| `exo/config.py` | `config/conf.py` | Medium |
| `exo/registry.py` | `core/factory.py` | Low |
| `exo/events.py` | `events/inmemory.py` | Low |
| `exo/hooks.py` | `runners/hook/` | Medium |
| `exo/tool.py` | `core/tool/base.py`, `tools/function_tools.py` | High |
| `exo/human.py` | `tools/human/human.py`, `runners/handler/human.py` | Medium |
| `exo/agent.py` | `agents/llm_agent.py`, `core/agent/base.py`, `core/agent/agent_desc.py` | **Very High** |
| `exo/runner.py` | `runner.py`, `runners/call_driven_runner.py` | **Very High** |
| `exo/_internal/handlers.py` | `runners/handler/` (10 handlers, ~2,400 LOC) | **Very High** |
| `exo/_internal/background.py` | `runners/handler/background_task.py` | High |
| `exo/_internal/state.py` | `runners/state_manager.py` | High |
| `exo/swarm.py` | `core/agent/swarm.py` | **Very High** |
| `exo/_internal/agent_group.py` | `agents/parallel_llm_agent.py`, `agents/serial_llm_agent.py` | High |
| `exo/_internal/nested.py` | `agents/task_llm_agent.py` | Medium |
| `exo/loader.py` | `config/agent_loader.py`, `config/task_loader.py` | High |
| `exo/skills.py` | `utils/skill_loader.py` | High |
| `exo/models/` | `models/` | High |
| `exo/context/` | `core/context/amni/`, `core/context/prompts/` | **Very High** |
| `exo/memory/` | `memory/` | High |
| `exo/memory/summary.py` | `memory/main.py` (summary logic) | High |
| `exo/memory/long_term.py` | `memory/longterm/` | High |
| `exo/mcp/` | `mcp_client/`, `tools/mcp_tool/` | High |
| `exo/sandbox/` | `sandbox/` | Medium-High |
| `exo/trace/` | `trace/`, `logs/`, `metrics/` | High |
| `exo/eval/` | `evaluations/` | High |
| `exo/eval/ralph/` | `ralph_loop/` | High |
| `exo/a2a/` | `experimental/a2a/` | Medium-High |
| `exo-cli` | `aworld-cli/` | **Very High** |
| `exo-server` | `cmd/web/` | Medium |
| `exo-train` | `train/`, `dataset/` | High |

---

## Implementation Notes

### Namespace Packages
The `exo` namespace is shared across multiple packages using `pkgutil.extend_path()` in `exo-core`'s `__init__.py`. This allows `exo.models`, `exo.memory`, etc. to be provided by separate installable packages while sharing the `exo` top-level namespace.

### Meta-Package
The `exo` meta-package (in `packages/exo/`) installs all sub-packages. It uses a dummy `_exo_meta` package for hatchling build compatibility since it has no source code of its own.
