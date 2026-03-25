# Migrating from AWorld to Exo

This guide maps AWorld concepts and imports to their Exo equivalents.

## Package Mapping

| AWorld | Exo | Notes |
|--------|---------|-------|
| `aworld` (monolith) | `exo` (meta-package) | Split into 13 focused packages |
| `aworld.agents` | `exo.agent` | Single `Agent` class replaces `LLMAgent`, `TaskLLMAgent`, etc. |
| `aworld.core.tool` | `exo.tool` | `@tool` decorator, `FunctionTool`, `Tool` ABC |
| `aworld.runner` | `exo.runner` | `run()`, `run.sync()`, `run.stream()` |
| `aworld.models` | `exo.models` | `get_provider("openai:gpt-4o")` factory |
| `aworld.core.context.amni` | `exo.context` | Clean rewrite — neurons, processors, workspace |
| `aworld.memory` | `exo.memory` | Short/long-term, SQLite/Postgres backends |
| `aworld.mcp_client` | `exo.mcp` | MCP client + `@mcp_server` decorator |
| `aworld.sandbox` | `exo.sandbox` | Local + Kubernetes sandboxes |
| `aworld.trace` | `exo.trace` | OpenTelemetry-based tracing |
| `aworld.evaluations` | `exo.eval` | Scorers, reflection, evaluator |
| `aworld.ralph_loop` | `exo.ralph` | Ralph loop — state, detectors, runner |
| `aworld.experimental.a2a` | `exo.a2a` | Agent-to-Agent protocol |

## Agent Definition

**AWorld:**
```python
from aworld.agents import LLMAgent
from aworld.config.conf import AgentConfig

config = AgentConfig(name="my-agent", llm_provider="openai", llm_model_id="gpt-4o")
agent = LLMAgent(agent_config=config, task_config=task_config)
```

**Exo:**
```python
from exo import Agent

agent = Agent(name="my-agent", model="openai:gpt-4o", instructions="...", tools=[...])
```

## Running Agents

**AWorld:**
```python
from aworld.runner import create_runner
runner = create_runner(agent_config=config, task_config=task_config)
result = await runner.run(task)
```

**Exo:**
```python
from exo import run

result = await run(agent, "What is 2+2?")      # async
result = run.sync(agent, "What is 2+2?")        # sync
async for event in run.stream(agent, "prompt"):  # streaming
    print(event)
```

## Tools

**AWorld:**
```python
from aworld.tools.function_tools import FunctionTool
tool = FunctionTool(name="search", func=search_fn, description="...")
```

**Exo:**
```python
from exo import tool

@tool
async def search(query: str) -> str:
    """Search the web."""
    return "results"
```

## Multi-Agent (Swarm)

**AWorld:**
```python
from aworld.agents import SwarmComposerAgent
swarm = SwarmComposerAgent(agents=[a, b], swarm_config=config)
```

**Exo:**
```python
from exo import Swarm
swarm = Swarm(agents=[a, b], mode="workflow")  # or "handoff", "team"
```

## Context

**AWorld:**
```python
from aworld.core.context.amni.contexts import AmniContext
from aworld.core.context.amni.config import AmniConfig
```

**Exo:**
```python
from exo.context import Context, ContextConfig
ctx = Context(config=ContextConfig(automation_level="copilot"))
```

## Key Differences

1. **Single Agent class** — AWorld had `LLMAgent`, `TaskLLMAgent`, `LoopLLMAgent`, `ParallelLLMAgent`, `SerialLLMAgent`. Exo has one `Agent` with composable behavior via Swarm modes and agent groups.

2. **Model strings** — use `"provider:model"` format (e.g., `"openai:gpt-4o"`, `"anthropic:claude-sonnet-4-20250514"`) instead of separate provider/model config fields.

3. **Async-first** — all agent execution is async. Use `run.sync()` for synchronous contexts.

4. **Modular packages** — install only what you need. `exo-core` has zero heavy dependencies.

5. **Type safety** — full pyright strict-mode compliance across all packages.
