# Migration Guide

This section covers migrating from AWorld to Exo. Exo is a ground-up rewrite of AWorld that simplifies the API surface while preserving the framework's capabilities.

## Why Migrate?

AWorld grew to 96,500 lines of code with accumulated complexity:

- **5 agent types** that resist composition (`LLMAgent`, `TaskLLMAgent`, `LoopLLMAgent`, `ParallelLLMAgent`, `SerialLLMAgent`)
- **3 config systems** (`ConfigDict`, `BaseConfig`, Pydantic models) mixed throughout the codebase
- **Sync/async duplication** (`BaseTool` and `AsyncBaseTool`, `run()` and `sync_run()`)
- **Stringly-typed message routing** (`category="tool"`, `topic="GROUP_RESULTS"`)
- **Deep factory chains** (`Factory[T]` + `AgentManager` + `ToolsManager`)

Exo addresses all of these with:

- **1 Agent class** with composable behavior via Swarm modes
- **1 config system** (Pydantic v2 models)
- **Async-first** with automatic sync wrapping
- **Typed message classes** (`UserMessage`, `AssistantMessage`, `ToolResult`)
- **1 Registry** pattern (`Registry[T]`)

## Migration Scope

The migration is broken into these areas:

| Area | AWorld Package | Exo Package | Difficulty |
|------|---------------|-----------------|------------|
| Agent definition | `aworld.agents` | `exo.agent` | Low |
| Tool registration | `aworld.core.tool`, `aworld.tools` | `exo.tool` | Low |
| Running agents | `aworld.runner`, `aworld.runners` | `exo.runner` | Low |
| Configuration | `aworld.config.conf` | `exo.config` | Medium |
| Multi-agent | `aworld.agents.swarm_composer_agent` | `exo.swarm` | Medium |
| Context engine | `aworld.core.context.amni` | `exo.context` | High |
| Memory | `aworld.memory` | `exo.memory` | Medium |
| Models/LLM | `aworld.models` | `exo.models` | Low |
| Tracing | `aworld.trace` | `exo.trace` | Medium |
| Evaluation | `aworld.evaluations` | `exo.eval` | Medium |
| MCP | `aworld.mcp_client` | `exo.mcp` | Low |
| Sandbox | `aworld.sandbox` | `exo.sandbox` | Medium |

## Getting Started

1. Start with the [detailed migration guide](./from-aworld.md) for step-by-step instructions
2. Migrate the simplest agents first (single agent, no context, no memory)
3. Add tools, then multi-agent, then context/memory

## Package Installation

AWorld was a monolith. Exo is modular -- install only what you need:

```bash
# Minimal -- core + models
pip install exo-core exo-models

# Or the meta-package for everything
pip install exo

# With specific extras
pip install exo-memory[qdrant]
pip install exo-sandbox[kubernetes]
```
