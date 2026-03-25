# Exo Framework Guides

Comprehensive guides covering every feature of the Exo multi-agent framework.

## Core Concepts

| Guide | Description |
|-------|-------------|
| [Agents](agents.md) | Creating and configuring agents -- the core autonomous unit |
| [Tools](tools.md) | Defining tools with the `@tool` decorator, `FunctionTool`, or the `Tool` ABC |
| [Running Agents](running.md) | Executing agents with `run()`, `run.sync()`, and `run.stream()` |
| [Streaming](streaming.md) | Real-time streaming with `TextEvent` and `ToolCallEvent` |
| [Structured Output](structured-output.md) | Constraining agent output to Pydantic models |

## Multi-Agent Systems

| Guide | Description |
|-------|-------------|
| [Multi-Agent Swarms](multi-agent.md) | Orchestrating agents with workflow, handoff, and team modes |
| [Agent Groups](agent-groups.md) | Parallel and serial execution groups, nested swarms |

## Lifecycle and Observability

| Guide | Description |
|-------|-------------|
| [Hooks](hooks.md) | Intercepting agent lifecycle events (pre/post LLM call, tool call) |
| [Events](events.md) | Decoupled async event bus for cross-cutting concerns |

## Human Oversight

| Guide | Description |
|-------|-------------|
| [Human-in-the-Loop](human-in-the-loop.md) | Pausing agents for human input, confirmation, or review |

## Context Engine

| Guide | Description |
|-------|-------------|
| [Context Overview](context/index.md) | Architecture, automation modes, and full context engine guide index |
| [State Management](context/state.md) | Hierarchical key-value state, parent inheritance, fork/merge |
| [Prompt Building](context/prompt-building.md) | Neurons, priority ordering, variable substitution |
| [Processors](context/processors.md) | Event-driven pipeline (pre_llm_call, post_tool_call, etc.) |
| [Workspace](context/workspace.md) | Versioned artifact storage, observers, filesystem persistence |
| [Knowledge](context/knowledge.md) | Text chunking, TF-IDF search, knowledge store |
| [Checkpoints](context/checkpoints.md) | Snapshot save/restore, version history |
| [Token Tracking](context/token-tracking.md) | Per-agent per-step usage, trajectories, summaries |
| [Context Tools](context/context-tools.md) | Planning, knowledge, and file tools for agents |

## Memory

| Guide | Description |
|-------|-------------|
| [Memory](memory.md) | Typed memory hierarchy, short/long-term memory, orchestration, summarization |
| [Memory Backends](memory-backends.md) | SQLite, Postgres, and Vector storage backends |

## Integration

| Guide | Description |
|-------|-------------|
| [MCP](mcp.md) | Model Context Protocol client/server, tool loading, multi-server management |
| [Sandbox](sandbox.md) | Isolated execution environments (local, Kubernetes), filesystem and terminal tools |
| [Agent-to-Agent (A2A)](a2a.md) | Agent discovery, HTTP-based agent communication, remote agent wrappers |

## Observability and Evaluation

| Guide | Description |
|-------|-------------|
| [Tracing](tracing.md) | OpenTelemetry spans, @traced decorator, prompt logging, propagation |
| [Evaluation](evaluation.md) | Rule-based and LLM-as-Judge scorers, parallel evaluation, pass@k |
| [Ralph Loop](ralph-loop.md) | Iterative refinement: Run -> Analyze -> Learn -> Plan -> Halt |

## Training

| Guide | Description |
|-------|-------------|
| [Training](training.md) | Trajectories, data synthesis, evolution, VeRL/RLHF integration |

## Deployment

| Guide | Description |
|-------|-------------|
| [Server](server.md) | FastAPI-based HTTP server with chat, SSE, WebSocket streaming |
| [CLI](cli.md) | Command-line runner, interactive console, batch processing, plugins |

## Configuration and Extensibility

| Guide | Description |
|-------|-------------|
| [Config-Driven Agents](config-driven.md) | Loading agents and swarms from YAML files |
| [Skills](skills.md) | Skill registry, markdown skill format, local and GitHub sources |
| [Model Providers](models.md) | Using OpenAI, Anthropic, and custom LLM providers |

## Quick Reference

All framework types live in `exo.types`:

- **Messages:** `UserMessage`, `SystemMessage`, `AssistantMessage`, `ToolResult`
- **Execution:** `AgentOutput`, `RunResult`, `ActionModel`
- **Streaming:** `TextEvent`, `ToolCallEvent`, `StreamEvent`
- **Stats:** `Usage`

Configuration types live in `exo.config`:

- `ModelConfig`, `AgentConfig`, `TaskConfig`, `RunConfig`
- `parse_model_string()` -- split `"provider:model"` strings

For full API details, see the [API Reference](../reference/).
