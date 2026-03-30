# Exo Claude Code Skills — Design Spec

**Date:** 2026-03-30
**Status:** Approved (design reviewed in conversation)

## Goal

Create 9 Claude Code skills that guide developers through Exo's advanced features. Each skill uses a hybrid approach: a decision/guidance section (picks the right pattern) + a reference section (exact syntax and gotchas).

## Skills

| # | Skill ID | Domain | Triggers On |
|---|----------|--------|-------------|
| 1 | `exo:tools` | Tool creation | `@tool`, `FunctionTool`, `Tool` ABC, `injected_tool_args`, `large_output`, structured output, MCP |
| 2 | `exo:swarms` | Multi-agent orchestration | `Swarm`, flow DSL, workflow/handoff/team modes, groups |
| 3 | `exo:context` | Context management | `ContextConfig`, automation modes, neurons, fork/merge, budget awareness |
| 4 | `exo:spawn` | Self-spawning agents | `allow_self_spawn`, `spawn_self()`, depth guards, memory isolation |
| 5 | `exo:hooks` | Lifecycle hooks + runtime mutation | `HookPoint`, `HookManager`, `add_tool()`, `inject_message()` |
| 6 | `exo:guardrails` | Security guardrails | `BaseGuardrail`, `Rail`, `PatternBackend`, `LLMGuardrailBackend` |
| 7 | `exo:streaming` | Streaming execution | `run.stream()`, event types, filtering, `detailed=True` |
| 8 | `exo:memory` | Memory system | `AgentMemory`, `ShortTermMemory`, backends, persistence, `conversation_id` |
| 9 | `exo:testing` | Test patterns | `MockProvider`, async test config, tool/swarm/streaming test patterns |

## Skill Structure (each file)

```
---
name: exo:<topic>
description: <when to trigger — includes keyword patterns>
---

# <Title>

## When To Use This Skill
<trigger conditions>

## Decision Guide
<questions that pick the right pattern>

## Reference
<exact API signatures, constructor params, code snippets>

## Patterns
<common usage patterns with complete examples>

## Gotchas
<things that trip people up>
```

## Installation

Skills will be written to a new `exo-skills` plugin directory, each skill as a `SKILL.md` file in its own subdirectory.

## Source Files Referenced

- `packages/exo-core/src/exo/agent.py` — Agent class, spawn_self, runtime mutation
- `packages/exo-core/src/exo/tool.py` — Tool ABC, FunctionTool, @tool decorator
- `packages/exo-core/src/exo/swarm.py` — Swarm, flow DSL, modes
- `packages/exo-core/src/exo/hooks.py` — HookPoint, HookManager
- `packages/exo-core/src/exo/rail.py` — Rail ABC, RailManager, RailAction
- `packages/exo-core/src/exo/runner.py` — run(), run.stream()
- `packages/exo-core/src/exo/config.py` — AgentConfig, validation
- `packages/exo-core/src/exo/types.py` — StreamEvent hierarchy, message types
- `packages/exo-context/src/exo/context/` — Context, ContextConfig, Neuron
- `packages/exo-memory/src/exo/memory/` — Memory stores, persistence, summary
- `packages/exo-guardrail/src/exo/guardrail/` — BaseGuardrail, backends
