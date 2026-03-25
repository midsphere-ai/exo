# Changelog

All notable changes to Exo are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Live Message Injection** -- `Agent.inject_message(content)` pushes a `UserMessage` into a running agent's context, picked up before the next LLM call. Enables mid-run steering without cancelling. (`exo-core`)
- **`MessageInjectedEvent`** -- New streaming event type emitted by `run.stream()` when an injected message is drained. Always emitted (not gated by `detailed`). (`exo-core`)
- **`POST /inject` endpoint** -- HTTP endpoint on `exo-server` for injecting messages into a running agent via the `InjectRequest` model. (`exo-server`)

## [0.1.0] - 2025-02-16

Initial release of Exo, a ground-up rewrite of the AWorld multi-agent framework.

### Core Framework (`exo-core`)

- **Agent** -- Single `Agent` class replacing AWorld's 5 agent types (`LLMAgent`, `TaskLLMAgent`, `LoopLLMAgent`, `ParallelLLMAgent`, `SerialLLMAgent`). Supports tools, handoffs, lifecycle hooks, structured output, and configurable max steps.
- **Tool** -- `@tool` decorator for turning functions into LLM-callable tools with auto-generated JSON schemas from type hints and docstrings. `Tool` ABC for complex tools. Sync functions auto-wrapped via `asyncio.to_thread()`.
- **Swarm** -- Multi-agent orchestration with three modes: `"workflow"` (sequential pipeline via flow DSL), `"handoff"` (agent-driven delegation), and `"team"` (lead-worker pattern with auto-generated delegate tools).
- **Runner** -- Three entry points: `run()` (async), `run.sync()` (blocking), `run.stream()` (async generator). State tracking, loop detection, and retry logic built in.
- **Types** -- Typed message classes (`UserMessage`, `AssistantMessage`, `SystemMessage`, `ToolResult`) as frozen Pydantic models. `RunResult`, `StreamEvent`, `Usage`, `AgentOutput`.
- **Config** -- Pydantic v2 models: `AgentConfig`, `ModelConfig`, `TaskConfig`, `RunConfig`. `parse_model_string()` for `"provider:model_name"` format.
- **Registry** -- Generic `Registry[T]` with fail-fast duplicate detection. Pre-built `agent_registry` and `tool_registry`.
- **Events** -- `EventBus` for decoupled async pub/sub communication.
- **Hooks** -- `HookManager` with `HookPoint` enum (`START`, `FINISHED`, `ERROR`, `PRE_LLM_CALL`, `POST_LLM_CALL`, `PRE_TOOL_CALL`, `POST_TOOL_CALL`).
- **Internal** -- `message_builder` (correct-by-construction message ordering), `output_parser` (LLM response to AgentOutput bridging, structured output validation), `call_runner` (execution loop with state tracking and loop detection), `state` (RunState/RunNode lifecycle), `graph` (flow DSL parser, topological sort).

### LLM Models (`exo-models`)

- **ModelProvider** -- Abstract base class with `complete()` and `stream()` methods.
- **OpenAIProvider** -- Full implementation for OpenAI's chat completions API.
- **AnthropicProvider** -- Full implementation for Anthropic's messages API.
- **Model Registry** -- `get_provider("openai:gpt-4o")` factory for building providers from model strings.
- **Types** -- `ModelResponse`, `StreamChunk`, `ToolCallDelta`, `FinishReason`, `ModelError`.

### Package Structure

- UV workspace monorepo with 13 packages under `packages/`
- Namespace packages via `pkgutil.extend_path()` for unified `exo.*` import path
- Zero heavy dependencies in `exo-core` (only `pydantic`)
- Provider SDKs isolated in `exo-models`

### Design Highlights

- Async-first with single sync bridge (`run.sync()`)
- Parallel tool execution via `asyncio.TaskGroup`
- Typed message union (`Message = UserMessage | AssistantMessage | SystemMessage | ToolResult`)
- Frozen Pydantic models for all data types
- Flow DSL for declarative multi-agent pipelines (`"a >> b >> c"`, `"a >> (b | c) >> d"`)
- Structured exception hierarchy rooted at `ExoError`
- Tool errors captured as `ToolResult(error=...)`, not propagated
- Exponential backoff retry for transient LLM errors

### Planned Packages (stubs created)

- `exo-context` -- Context engine: hierarchical state, prompt neurons, processors, workspace, RAG
- `exo-memory` -- Short-term and long-term memory backends
- `exo-mcp` -- Model Context Protocol client and server
- `exo-sandbox` -- Local and Kubernetes sandboxed execution
- `exo-trace` -- OpenTelemetry-based tracing
- `exo-eval` -- Evaluation framework with scorers and reflection
- `exo-a2a` -- Agent-to-Agent protocol
- `exo-cli` -- Command-line interface
- `exo-server` -- HTTP server for agent serving
- `exo-train` -- Training: trajectory collection and trainers

[0.1.0]: https://github.com/anthropics/exo/releases/tag/v0.1.0
