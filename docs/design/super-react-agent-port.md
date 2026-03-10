# SuperReActAgent Porting Strategy

**Status:** Active
**Epic:** 11 — DeepAgent Example Port
**Date:** 2026-03-11

---

## 1. Motivation

`agent/super_react_agent.py` is the largest file in the deepagent example (~1,223 lines). It contains the core ReAct loop, MCP tool registration, sub-agent management, plan tracking, and context overflow handling. This document maps every openjiuwen import to its Orbiter replacement, identifies which methods are already covered by Orbiter's `Agent` class, which must remain as custom logic, proposes a class hierarchy, and defines the splitting strategy for US-097 through US-099.

---

## 2. All 9 openjiuwen Imports and Their Replacements

| # | openjiuwen Import | Used For | Orbiter Replacement |
|---|-------------------|----------|---------------------|
| 1 | `BaseAgent` from `core.agent.agent` | Parent class for SuperReActAgent | `orbiter.agent.Agent` (compose around, not extend — see §5) |
| 2 | `Runtime` from `core.runtime.runtime` | Tool dispatch, agent config retrieval | Remove — Orbiter Agent handles tool dispatch via `_execute_tools()`. Tool filtering via `Agent.tools` dict. |
| 3 | `Workflow` from `core.runtime.runtime` | Workflow definition | Remove — not used in the ported example (workflows replaced by Orbiter Swarm) |
| 4 | `Tool` from `core.utils.tool.base` | Type hint for tool lists | `orbiter.tool.Tool` |
| 5 | `logger` from `core.common.logging` | Logging throughout | `logging.getLogger(__name__)` (Python stdlib) |
| 6 | `AIMessage` from `core.utils.llm.messages` | LLM response type | `orbiter.types.AgentOutput` (returned by `Agent.run()`) |
| 7 | `Param` from `core.utils.tool.param` | MCP tool parameter schema | Remove — Orbiter Tool uses `parameters` dict (JSON Schema format) |
| 8 | `LocalFunction` from `core.utils.tool.function.function` | Wrapping MCP tools as callable functions | `orbiter.tool.FunctionTool` or custom `Tool` subclass |
| 9 | `ToolServerConfig`, `Runner`, `resource_mgr` from MCP/runner modules | MCP server registration and tool invocation | Custom MCP integration using `mcp` SDK directly + Orbiter `Tool` wrappers |

---

## 3. Methods That Map to Built-in Orbiter Agent Behavior

These SuperReActAgent methods are **already provided** by `orbiter.agent.Agent` and need no custom reimplementation:

| SuperReActAgent Method | Orbiter Agent Equivalent | Notes |
|------------------------|--------------------------|-------|
| `call_model()` — LLM invocation | `Agent._call_llm()` | Orbiter handles message building, tool schema injection, retry logic, and hook firing. |
| `_execute_tool_call()` — single tool dispatch | `Agent._execute_tools()` | Orbiter dispatches tools by name from `self.tools` dict, handles errors, fires PRE/POST_TOOL_CALL hooks. |
| Tool registration (via `add_tools()`) | `Agent._register_tool()` | Orbiter stores tools in `self.tools: dict[str, Tool]` with O(1) lookup. |
| Tool schema generation | `Agent.get_tool_schemas()` | Returns OpenAI-format schemas via `Tool.to_schema()`. |
| ReAct loop (LLM → tool calls → results → repeat) | `Agent.run()` | Built-in tool loop with `max_steps`, history management, and `TaskLoopQueue` support. |
| Context-length error detection | `Agent._call_llm()` → `_is_context_length_error()` | Detects context overflow and raises `AgentError`. |
| Retry on transient errors | `Agent._call_llm()` with `max_retries` | Exponential backoff retry built in. |

---

## 4. Methods That Must Remain as Custom Logic

These are **not provided** by Orbiter's Agent and must be ported as custom classes or hooks:

### 4.1 PlanTracker (~350 lines)
- Extracts step-by-step plans from LLM output via regex and structured blocks (`#PLAN#`, `<TODO_PLAN>`, `<TODO_STATUS>`)
- Writes `todo.md` and injects plan summaries into the context
- LLM-based plan update interpretation
- **Port strategy:** Extract as standalone class `PlanTracker` — no framework dependency, only needs a context-append callback and optional LLM callable.

### 4.2 PlanStepState dataclass (~15 lines)
- Simple dataclass for plan step representation
- **Port strategy:** Keep as-is (already framework-agnostic).

### 4.3 ContextManager integration
- `_context_manager.add_user_message()`, `add_assistant_message()`, `add_tool_message()`, `get_history()`, `generate_summary()`, `upsert_system_message()`
- The custom `ContextManager` (already ported in `context_manager.py`) manages a separate message history with summarization
- **Port strategy:** Wire as a PRE_LLM_CALL hook that injects context, or compose around `Agent.run()` with a wrapper.

### 4.4 QAHandler integration
- Question hint extraction (`extract_hints()`)
- Answer type detection (`get_answer_type()`)
- Final answer extraction (`extract_final_answer()`)
- **Port strategy:** Keep `QAHandler` as standalone (already ported). Call from the orchestrator before/after the agent loop.

### 4.5 MCP server registration (`_register_mcp_server_as_local_tools`, ~60 lines)
- Registers MCP servers via openjiuwen's `Runner`/`resource_mgr`
- Converts MCP tools to `LocalFunction` with parameter schemas
- **Port strategy:** Rewrite to use `mcp` SDK directly + wrap each tool as Orbiter `Tool` subclass. The `_make_mcp_call_coroutine` / `_normalize_mcp_server_config` helpers are framework-agnostic and mostly portable.

### 4.6 Sub-agent management (`register_sub_agent`, ~20 lines)
- Registers sub-agents as callable tools on the main agent
- **Port strategy:** Already handled by Orbiter's `Swarm(mode="team")` which auto-generates `delegate_to_<worker>` tools. The `SubAgentTool` in `tool_call_handler.py` (already ported) covers the direct-call case.

### 4.7 Tool whitelist filtering (~40 lines in `call_model`)
- Filters runtime tools by agent config's tool name list
- **Port strategy:** Not needed — Orbiter Agent only exposes tools in `self.tools` dict, which are explicitly registered at construction time.

### 4.8 Context-limit retry and summarization (~30 lines)
- Checks context window capacity, triggers summarization
- **Port strategy:** Implement as a PRE_LLM_CALL hook or catch `AgentError` (context_length) in the orchestrator and retry with a summarized history.

### 4.9 `invoke()` orchestration (~240 lines)
- The main ReAct loop with:
  - Input processing (GAIA format)
  - Question hints injection
  - Iteration loop with tool calls
  - Context limit checking
  - Summary generation
  - Final answer extraction
  - Plan tracker lifecycle
- **Port strategy:** Rewrite as a top-level `run_super_agent()` async function that composes Orbiter primitives.

---

## 5. Proposed Class Hierarchy: Compose Around Agent

**Decision: Compose around `orbiter.agent.Agent`, do NOT extend it.**

Rationale:
- `Agent` is designed as a self-contained unit with a clean `run()` method
- SuperReActAgent's custom logic (plan tracking, context management, QA extraction) is orchestration-level — it wraps around the agent loop rather than modifying its internals
- The already-ported `super_factory.py` constructs `Agent` instances, confirming this pattern
- Extending `Agent` would require overriding `run()` and `_call_llm()`, duplicating significant built-in behavior

### Proposed Structure

```
agent/
  super_react_agent.py    →  REWRITE as orchestrator (compose Agent, not extend)
  plan_tracker.py         →  NEW — extracted PlanTracker + PlanStepState
  super_config.py         →  ALREADY PORTED
  super_factory.py        →  ALREADY PORTED (builds Agent instances)
  context_manager.py      →  ALREADY PORTED
  qa_handler.py           →  ALREADY PORTED
  tool_call_handler.py    →  ALREADY PORTED
  mcp_tools.py            →  NEW — MCP server registration as Orbiter Tools
```

### Key Classes

```python
# plan_tracker.py — standalone, no Orbiter imports
class PlanStepState:
    """Dataclass for plan step state."""

class PlanTracker:
    """Extracts plans from LLM output, writes todo.md, injects context summaries."""

# mcp_tools.py — thin MCP-to-Orbiter bridge
class McpTool(Tool):
    """Orbiter Tool wrapping an MCP server tool."""

async def register_mcp_server(server_name, client_type, params) -> list[Tool]:
    """Register an MCP server and return Orbiter Tool instances."""

# super_react_agent.py — orchestrator
async def run_super_agent(
    agent: Agent,
    query: str,
    *,
    config: SuperAgentConfig,
    context_manager: ContextManager,
    qa_handler: QAHandler | None = None,
    plan_tracker: PlanTracker | None = None,
) -> dict[str, Any]:
    """Run the full SuperReAct loop using Orbiter's Agent.run()."""
```

---

## 6. Splitting Strategy for US-097 through US-099

### US-097: Port PlanTracker as standalone module

**Scope:** ~380 lines (PlanStepState + PlanTracker)
- Extract `PlanStepState` dataclass and `PlanTracker` class into `agent/plan_tracker.py`
- Replace `openjiuwen.core.common.logging.logger` → `logging.getLogger(__name__)`
- Replace `AIMessage` type hint → generic protocol (content: str, tool_calls: list)
- Replace `ContextManager.upsert_system_message()` call → callback interface
- Replace `OpenRouterLLM._ainvoke()` call → generic async LLM callable
- Keep `_make_mcp_call_coroutine` and `_normalize_mcp_server_config` helpers as-is (move to `mcp_tools.py` or keep inline)
- Unit tests for plan extraction, step updates, and markdown rendering
- Typecheck passes

### US-098: Port MCP tool registration to Orbiter

**Scope:** ~80 lines
- Create `agent/mcp_tools.py` with:
  - `McpTool(Tool)` — Orbiter Tool subclass wrapping an MCP call
  - `register_mcp_server()` → returns `list[Tool]`
  - `_normalize_mcp_server_config()` helper (moved from super_react_agent.py)
- Replace `Runner.run_tool()`, `Runner.list_tools()`, `ToolServerConfig`, `Param`, `LocalFunction` with direct `mcp` SDK usage + Orbiter `Tool`
- Tests for tool schema generation and mock MCP execution
- Typecheck passes

### US-099: Port SuperReActAgent core loop as orchestrator

**Scope:** ~300 lines
- Rewrite `super_react_agent.py` as a composition-based orchestrator:
  - `run_super_agent()` async function (replaces `invoke()`)
  - Uses `orbiter.runner.run()` or `Agent.run()` for the inner loop
  - Wires `PlanTracker`, `ContextManager`, `QAHandler` around the agent
  - Handles context-limit retry (catch `AgentError`, summarize, retry)
  - Input processing (GAIA format via `process_input` + `get_task_instruction_prompt`)
  - Summary generation and final answer extraction post-loop
- Remove all openjiuwen imports — zero remaining references
- Remove `BaseAgent` inheritance entirely
- Integration with `Swarm` for multi-agent via `create_agent_system_with_sub_agents()` (already in super_factory.py)
- Typecheck passes

---

## 7. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| PlanTracker depends on ContextManager's `upsert_system_message` | Use callback/protocol — PlanTracker accepts `on_context_update: Callable[[str, str], None]` |
| PlanTracker's LLM-based update uses OpenRouterLLM directly | Accept generic `Callable[[list[dict], list], Awaitable[Any]]` for LLM calls |
| MCP registration uses openjiuwen's Runner which manages connections | Use `mcp` SDK's `ClientSession` directly — the deepagent already has `mcp` as a dependency |
| Context overflow retry requires intercepting Agent.run() mid-loop | Catch `AgentError` (context_length) at the orchestrator level, summarize history, and re-invoke |
| invoke() has GAIA-specific input processing | Keep in orchestrator — it's example-specific, not framework logic |

---

## 8. Summary

- **9 openjiuwen imports** → all have clear Orbiter or stdlib replacements
- **7 methods** map directly to built-in Orbiter Agent behavior (tool loop, LLM calls, retry, tool dispatch)
- **9 custom concerns** must be preserved (PlanTracker, context management, QA handler, MCP, sub-agents, tool filtering, context-limit retry, input processing, orchestration)
- **Class hierarchy:** Compose around `Agent` (not extend) — orchestrator pattern
- **3 stories:** US-097 (PlanTracker), US-098 (MCP tools), US-099 (orchestrator + final integration)
- **Estimated total:** ~760 lines of new/rewritten code across 3 new files
