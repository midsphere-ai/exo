# DeepAgent Example — Porting Plan

**Status:** Complete
**Epic:** 11 — DeepAgent Example Port
**Date:** 2026-03-11

---

## 1. Motivation

The `examples/advanced/deepagent/` directory is a multi-agent ReAct system originally built on the **openjiuwen** framework. This plan inventories every openjiuwen dependency, maps each to its Orbiter equivalent, and defines the porting order for subsequent stories.

---

## 2. File-by-File Inventory

### 2.1 Files WITH openjiuwen Imports (Must Port)

| File | openjiuwen Imports | Orbiter Equivalent |
|------|-------------------|-------------------|
| `agent/utils.py` | `logger, LogManager` from `core.common.logging`; `log_config` from `extensions.common.configs`; `DefaultLogger, SafeRotatingFileHandler, ThreadContextFilter` from `extensions.common.log` | Python `logging` module (standard library) |
| `agent/context_manager.py` | `logger` from `core.common.logging` | Python `logging` module |
| `agent/qa_handler.py` | `logger` from `core.common.logging` | Python `logging` module |
| `agent/super_config.py` | `ReActAgentConfig, ConstrainConfig` from `agent.config.react_config`; `PluginSchema, WorkflowSchema` from `agent.common.schema`; `ModelConfig` from `core.component.common.configs.model_config` | Pydantic `BaseModel` config classes (see deepsearch `config.py` pattern) |
| `agent/super_factory.py` | `Tool` from `core.utils.tool.base`; `Workflow` from `core.workflow.base`; `ModelConfig` from `core.component.common.configs.model_config`; `BaseModelInfo` from `core.utils.llm.base`; `PluginSchema, WorkflowSchema` from `agent.common.schema` | `orbiter.agent.Agent`, `orbiter.tools.Tool`, Pydantic config models |
| `agent/super_react_agent.py` | `BaseAgent` from `core.agent.agent`; `Runtime, Workflow` from `core.runtime.runtime`; `Tool` from `core.utils.tool.base`; `logger` from `core.common.logging`; `AIMessage` from `core.utils.llm.messages`; `Param` from `core.utils.tool.param`; `LocalFunction` from `core.utils.tool.function.function`; `ToolServerConfig` from `core.utils.tool.mcp.base`; `Runner, resource_mgr` from `core.runner.runner` | `orbiter.agent.Agent` + `orbiter.runner` + `orbiter.swarm.Swarm` + `orbiter.tools.Tool` |
| `agent/tool_call_handler.py` | `Runtime` from `core.runtime.runtime`; `logger` from `core.common.logging`; `LocalFunction` from `core.utils.tool.function.function`; `Param` from `core.utils.tool.param` | `orbiter.tools.Tool`, Python `logging` |
| `llm/__init__.py` | `ModelConfig` from `core.component.common.configs.model_config` | Pydantic config model or `orbiter-models` |
| `test/super_react_agent_test_run.py` | Multiple agent config, model setup, tool definition imports | Orbiter Agent, Swarm, Tool equivalents |

### 2.2 Files WITHOUT openjiuwen Imports (No Changes Needed)

| File/Directory | Notes |
|---------------|-------|
| `__init__.py` (root) | Empty |
| `agent/__init__.py` | Empty or local-only imports |
| `agent/prompt_templates.py` | Pure string templates, no framework imports |
| `tool/__init__.py` | Empty |
| `tool/logger.py` | Standalone logging utility |
| `tool/browser/` (12 files) | Uses browser_use, playwright, pydantic — no openjiuwen |
| `tool/browser/utils/` (9 files) | Standalone utilities — no openjiuwen |
| `tool/browser/utils/utils/` (9 files) | Duplicate of parent utils — no openjiuwen |
| `tool/mcp_servers/` (13 files) | Uses fastmcp, langchain, anthropic SDK — no openjiuwen |
| `tool/mcp_servers/utils/` (7 files) | Search/API utilities — no openjiuwen |
| `llm/openrouter_llm.py` | Uses OpenAI SDK directly |
| `llm/openrouter_function_call_example.py` | Standalone example |

---

## 3. Dependency Graph

```
agent/utils.py                 ← openjiuwen.logging (leaf — no internal deps)
agent/context_manager.py       ← openjiuwen.logging (leaf)
agent/qa_handler.py            ← openjiuwen.logging (leaf)
agent/prompt_templates.py      ← (no openjiuwen — standalone)

agent/super_config.py          ← openjiuwen.agent.config, openjiuwen.agent.schema, openjiuwen.model_config
                                  (depends on: nothing internal)

agent/tool_call_handler.py     ← openjiuwen.runtime, openjiuwen.logging, openjiuwen.tool
                                  (depends on: agent/utils.py for logging)

agent/super_factory.py         ← openjiuwen.tool, openjiuwen.workflow, openjiuwen.model_config, openjiuwen.schema
                                  (depends on: super_config.py, tool_call_handler.py)

agent/super_react_agent.py     ← openjiuwen.agent, openjiuwen.runtime, openjiuwen.tool, openjiuwen.logging,
                                  openjiuwen.messages, openjiuwen.runner
                                  (depends on: super_config.py, super_factory.py, tool_call_handler.py,
                                   context_manager.py, qa_handler.py, utils.py)

llm/__init__.py                ← openjiuwen.model_config (leaf)

test/super_react_agent_test_run.py ← depends on everything above
```

---

## 4. Porting Order (Leaves First)

| Phase | File(s) | Story | Rationale |
|-------|---------|-------|-----------|
| 0 | `pyproject.toml`, `__init__.py` | US-088 | Switch deps from openjiuwen to orbiter packages |
| 1 | `agent/utils.py` | US-089 | Leaf — only imports openjiuwen logging |
| 2 | `llm/__init__.py` | US-090 | Leaf — only imports ModelConfig |
| 3 | `agent/super_config.py` | US-091 | Config classes — depends on nothing internal |
| 4 | `agent/context_manager.py` | US-089 (logging only) | Leaf — swap logger import |
| 5 | `agent/qa_handler.py` | US-089 (logging only) | Leaf — swap logger import |
| 6 | `agent/tool_call_handler.py` | US-092 | Depends on utils.py (already ported) |
| 7 | `agent/super_factory.py` | US-093 | Depends on super_config, tool_call_handler |
| 8 | `agent/super_react_agent.py` | US-094–US-095 | Core agent — depends on everything above |
| 9 | `test/super_react_agent_test_run.py` | US-096 | Integration test — must be last |
| 10 | Final cleanup | US-097 | Verify, document, close |

---

## 5. Files That Need No Changes

The following directories contain **zero openjiuwen imports** and work independently:

- **`tool/browser/`** — Browser automation using browser_use + playwright
- **`tool/browser/utils/`** — Image/token/function utilities
- **`tool/browser/utils/utils/`** — Duplicate of above (candidate for cleanup)
- **`tool/mcp_servers/`** — All MCP servers use fastmcp, langchain, etc.
- **`tool/mcp_servers/utils/`** — Search engine wrappers
- **`agent/prompt_templates.py`** — Pure string templates
- **`tool/logger.py`** — Standalone logging utility
- **`llm/openrouter_llm.py`** — Uses OpenAI SDK directly
- **`llm/openrouter_function_call_example.py`** — Standalone example

---

## 6. Action Per File: Delete vs. Rewrite vs. Light Edit

| File | Action | Effort |
|------|--------|--------|
| `agent/utils.py` | **Light edit** — replace 4 openjiuwen logging imports with stdlib `logging` | Low |
| `agent/context_manager.py` | **Light edit** — swap 1 logger import | Low |
| `agent/qa_handler.py` | **Light edit** — swap 1 logger import | Low |
| `agent/super_config.py` | **Rewrite** — replace openjiuwen config classes with Pydantic BaseModel | Medium |
| `agent/tool_call_handler.py` | **Rewrite** — replace openjiuwen Runtime/Tool with orbiter equivalents | Medium |
| `agent/super_factory.py` | **Rewrite** — replace factory pattern with Orbiter Agent/Swarm assembly | Medium |
| `agent/super_react_agent.py` | **Heavy rewrite** — core agent, 9 openjiuwen imports, maps to Orbiter Agent + Swarm | High |
| `llm/__init__.py` | **Light edit** — replace ModelConfig import | Low |
| `test/super_react_agent_test_run.py` | **Rewrite** — update all imports and config to use Orbiter | Medium |
| `tool/browser/utils/utils/` | **Delete candidate** — duplicates parent `utils/` directory | Low |
| `pyproject.toml` | **Light edit** — swap openjiuwen dep for orbiter packages | Low |

---

## 7. Risk Assessment

| File | Risk | Notes |
|------|------|-------|
| `agent/super_react_agent.py` | **HIGH** | Core agent with 9 openjiuwen imports. ReAct loop, MCP integration, sub-agent management, and tool execution all tightly coupled to openjiuwen's BaseAgent/Runtime/Runner. Needs careful decomposition into Orbiter Agent + Swarm patterns. |
| `agent/super_factory.py` | **MEDIUM** | Factory creates agents with openjiuwen's Tool/Workflow/ModelConfig. Must map to Orbiter's composition model. Logic is mostly wiring, not complex. |
| `agent/super_config.py` | **MEDIUM** | Inherits from openjiuwen's `ReActAgentConfig`. Must be rewritten as standalone Pydantic models. Risk of missing config fields that downstream code depends on. |
| `agent/tool_call_handler.py` | **MEDIUM** | Uses openjiuwen's Runtime for tool dispatch. Must map to Orbiter's tool execution. Has type conversion logic that should port cleanly. |
| `test/super_react_agent_test_run.py` | **MEDIUM** | End-to-end test depends on all ported components. Must be updated last. |
| `agent/utils.py` | **LOW** | Only logging imports — straightforward stdlib swap. |
| `agent/context_manager.py` | **LOW** | Only 1 logger import to swap. Core logic is framework-agnostic. |
| `agent/qa_handler.py` | **LOW** | Only 1 logger import to swap. Uses OpenAI SDK directly for model calls. |
| `llm/__init__.py` | **LOW** | Only 1 ModelConfig import to replace. |
| All `tool/` files | **NONE** | No openjiuwen imports. No changes needed. |

---

## 8. Reference: DeepSearch Example Structure

The completed `examples/advanced/deepsearch/` port demonstrates the target patterns:

- **Config:** Pydantic `BaseModel` with `from_env()` classmethod
- **Agent assembly:** `build_deep_agent(config) → Swarm` function
- **Tools:** Subclass `Tool` ABC with `name`, `description`, `parameters`, `async execute()`
- **Prompts:** Pure functions returning formatted strings
- **Entry:** `__main__.py` with argparse CLI → async main
- **Memory:** Custom classes with JSON persistence

---

## 9. Summary

- **64 total Python files** in deepagent
- **9 files** require porting (have openjiuwen imports)
- **~45 files** need no changes (browser tools, MCP servers, utilities)
- **~10 files** are candidates for cleanup (duplicate utils directory)
- **Highest risk:** `super_react_agent.py` (9 imports, core agent logic)
- **Lowest risk:** Logging swaps in utils.py, context_manager.py, qa_handler.py
- **Porting order:** leaves (logging) → config → handlers → factory → core agent → test

---

## 10. Completion Summary

**Date completed:** 2026-03-11

All 9 files with openjiuwen imports have been successfully ported to use Orbiter equivalents:

1. **agent/utils.py** — Replaced openjiuwen logging with stdlib `logging`
2. **agent/context_manager.py** — Swapped logger import to stdlib
3. **agent/qa_handler.py** — Swapped logger import to stdlib
4. **agent/super_config.py** — Rewritten as standalone Pydantic BaseModel classes
5. **agent/tool_call_handler.py** — Replaced openjiuwen Runtime/Tool with Orbiter equivalents
6. **agent/super_factory.py** — Rewired to use `orbiter.agent.Agent` and `orbiter.swarm.Swarm`
7. **agent/super_react_agent.py** — Core agent rewritten using Orbiter Agent + MCP client + Swarm
8. **llm/__init__.py** — Replaced ModelConfig import
9. **test/super_react_agent_test_run.py** — Updated all imports and config to Orbiter

**Validation:**
- `grep -r 'openjiuwen' examples/advanced/deepagent/` returns zero matches
- `pyproject.toml` depends only on `orbiter-core`, `orbiter-models`, `orbiter-mcp`
- Stale `uv.lock` removed (will be regenerated when orbiter packages are published)
- README updated to reference Orbiter instead of openjiuwen
