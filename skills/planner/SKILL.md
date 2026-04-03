---
name: exo:planner
description: "Use when configuring Exo agent planning — planning_enabled, planning_model, planning_instructions, two-phase plan-then-execute pattern, planner pre-pass, plan injection. Triggers on: planning_enabled, planning_model, planning_instructions, plan agent, planner, plan-then-execute, two-phase, planner pre-pass."
---

> **Branch:** These skills are written for the `rename/orbiter-to-exo` branch. The Exo APIs referenced here may differ on other branches.

# Exo Planner — Plan-Then-Execute Agents

## When To Use This Skill

Use this skill when the developer needs to:
- Enable a two-phase plan-then-execute pattern on an agent
- Configure a separate planning model (e.g., cheaper/faster model for planning)
- Provide custom planning instructions distinct from the executor instructions
- Understand how the planner pre-pass works and how the plan is injected

## Decision Guide

1. **Should the agent plan before executing?** → Set `planning_enabled=True`
2. **Should planning use a different model?** → Set `planning_model` (e.g., `"openai:gpt-4o-mini"` for cheaper planning)
3. **Should the planner have different instructions?** → Set `planning_instructions` (otherwise it reuses the agent's own instructions)

## Reference

### Enabling Planning

```python
from exo import Agent, run

agent = Agent(
    name="researcher",
    model="openai:gpt-4o",
    instructions="You are a thorough research assistant.",
    tools=[search, summarize],
    planning_enabled=True,  # Enables the planner pre-pass
)

result = await run(agent, "Compare React vs Svelte", provider=provider)
```

### Agent Parameters

- `planning_enabled: bool = False` — When `True`, runs an ephemeral planner agent before the main execution
- `planning_model: str | None = None` — Model for the planner phase. When `None`, uses the agent's own `model`
- `planning_instructions: str = ""` — Instructions for the planner agent. When empty, uses the agent's own `instructions`

### How It Works

The planner is a **two-phase execution pattern** that runs automatically when `planning_enabled=True`. It is called from both `agent.run()` (via `call_runner.py`) and `run.stream()` (via `runner.py`).

```
User Input
    │
    ▼
┌─────────────────────────────┐
│  Phase 1: Planner Pre-Pass  │
│                             │
│  Ephemeral planner agent    │
│  runs the full LLM-tool     │
│  loop on the original input │
│  → produces plan_text       │
│                             │
│  Transcript is discarded    │
└──────────────┬──────────────┘
               │ plan_text
               ▼
┌─────────────────────────────┐
│  Plan Injection             │
│                             │
│  String input:              │
│    "Original task:\n{input} │
│     Planner output:\n{plan} │
│     Use the planner output  │
│     while completing the    │
│     task."                  │
│                             │
│  Content-block input:       │
│    Appends SystemMessage    │
│    with plan to messages    │
└──────────────┬──────────────┘
               │ augmented input
               ▼
┌─────────────────────────────┐
│  Phase 2: Main Execution    │
│                             │
│  Agent runs its normal      │
│  LLM-tool loop with the     │
│  plan injected as context   │
└─────────────────────────────┘
```

### Planner Agent Properties

The ephemeral planner agent is created with:

| Property | Value |
|----------|-------|
| `name` | `{agent.name}_planner` |
| `model` | `planning_model` or agent's `model` |
| `instructions` | `planning_instructions` or agent's `instructions` |
| `tools` | Agent's tools **minus** `spawn_self`, `retrieve_artifact`, context tools |
| `memory` | `None` (stateless — no conversation persisted) |
| `context` | Shared with parent agent |
| `max_steps` | Same as agent |
| `temperature` | Same as agent |
| `max_tokens` | Same as agent |

The planner is a real agent — it can call tools, reason over multiple steps, etc. Its full conversation transcript is discarded after the plan text is extracted.

### Provider Resolution

When `planning_model` differs from the agent's model, the runtime resolves a provider in this order:

1. **Same model** → reuse the executor's provider directly
2. **Same backend, different model** (e.g., both `openai:*`) → clone the provider with the new model name
3. **Different backend** → resolve a fresh provider via `get_provider(planning_model)`

If all three fail, an `AgentError` is raised.

### Plan Injection Format

**String input:**
```
Original task:
Compare React vs Svelte

Planner output:
1. Research React — strengths, weaknesses, ecosystem
2. Research Svelte — strengths, weaknesses, ecosystem
3. Compare on performance, DX, ecosystem, learning curve
4. Synthesize findings

Use the planner output while completing the task.
```

**Content-block input** (e.g., images, multi-part):
```
Appended SystemMessage:
"Planner output:\n{plan}\n\nUse the planner output while responding to the next user task."
```

### Empty Plan Handling

If the planner produces empty text, the plan injection is skipped entirely — the executor receives the original input unchanged.

## Patterns

### Basic Planning

```python
agent = Agent(
    name="analyst",
    model="openai:gpt-4o",
    instructions="You are a data analyst. Break down complex queries methodically.",
    tools=[query_db, chart],
    planning_enabled=True,
)
```

### Cheap Planner, Expensive Executor

```python
agent = Agent(
    name="coder",
    model="anthropic:claude-sonnet-4-6",       # powerful executor
    instructions="You are an expert software engineer.",
    tools=[read_file, write_file, run_tests],
    planning_enabled=True,
    planning_model="openai:gpt-4o-mini",         # cheap planner
    planning_instructions=(
        "Analyze the task and create a step-by-step implementation plan. "
        "List files to modify, functions to change, and tests to add. "
        "Do NOT execute — only plan."
    ),
)
```

### Planner with Streaming

Planning works with `run.stream()` — the planner runs first, then streaming begins for the executor phase:

```python
async for event in run.stream(agent, "Build a REST API for users", provider=provider):
    if isinstance(event, TextEvent):
        print(event.text, end="")  # Only executor output is streamed
```

### Planning + Self-Spawn

Planning and self-spawn are compatible. The planner agent inherits `allow_self_spawn` and `max_spawn_depth` but gets `spawn_self` excluded from its tools (since planning should plan, not execute):

```python
agent = Agent(
    name="coordinator",
    model="openai:gpt-4o",
    instructions="Plan research, then delegate via spawn_self.",
    tools=[search],
    planning_enabled=True,
    planning_instructions="Create a research plan listing topics to investigate.",
    allow_self_spawn=True,
    max_spawn_children=4,
)
```

## Gotchas

- **Planner transcript is discarded** — only the final `result.text` survives as the plan. Intermediate tool calls, reasoning, etc. are thrown away.
- **Planner is a full agent** — it can call tools and run multiple LLM steps. If you want a lightweight plan, set low `max_steps` or use instructions that say "do NOT call tools, only plan."
- **Planner has `memory=None`** — it's stateless. It doesn't remember previous conversations or planning sessions.
- **Double LLM cost** — planning adds an extra agent run. Use a cheaper `planning_model` to mitigate.
- **Empty plan = no-op** — if the planner returns empty text, the executor gets the original input unchanged.
- **`spawn_self` excluded from planner tools** — the planner can see other tools (for schema awareness) but cannot call `spawn_self`.
- **Provider must be resolvable** — if `planning_model` is set and no provider can be resolved for it, `AgentError` is raised at runtime, not at init.
- **Works with both `agent.run()` and `run.stream()`** — the planner pre-pass runs in both entry points.

## Key Files

- Implementation: `packages/exo-core/src/exo/_internal/planner.py`
- Called from: `packages/exo-core/src/exo/_internal/call_runner.py` (line 77) and `packages/exo-core/src/exo/runner.py` (line 256)
- Config: `packages/exo-core/src/exo/config.py` (`validate_planning_model`)
- Agent params: `packages/exo-core/src/exo/agent.py` (lines 593-595)
