# exo-core

Core agent framework for the [Exo](../../README.md) multi-agent platform.

## Installation

```bash
pip install exo-core
```

Requires Python 3.11+. Dependencies: `pydantic>=2.0`, `pyyaml>=6.0`.

## What's Included

- **Agent** -- the core autonomous unit. Wraps an LLM model, tools, handoffs, hooks, and optional structured output. Supports context snapshot persistence across runs (`clear_snapshot()` to restore from raw history).
- **Tool** -- `@tool` decorator for turning functions into LLM-callable tools with auto-generated JSON schemas. `Tool` ABC for custom tools.
- **Runner** -- `run()` (async), `run.sync()` (blocking), `run.stream()` (streaming). State tracking, loop detection, retry logic. All three modes fire the same lifecycle hooks (`PRE_LLM_CALL`, `POST_LLM_CALL`, etc.).
- **Swarm** -- multi-agent orchestration with three modes: `workflow`, `handoff`, `team`. Flow DSL: `"a >> b >> c"`.
- **Agent Groups** -- `ParallelGroup` and `SerialGroup` for concurrent/sequential sub-pipelines.
- **Config** -- Pydantic v2 models: `AgentConfig`, `ModelConfig`, `TaskConfig`, `RunConfig`.
- **Registry** -- generic `Registry[T]` with fail-fast duplicate detection.
- **Events** -- async `EventBus` for decoupled pub/sub communication.
- **Hooks** -- lifecycle hooks: `PRE_LLM_CALL`, `POST_LLM_CALL`, `PRE_TOOL_CALL`, `POST_TOOL_CALL`, `START`, `FINISHED`, `ERROR`.
- **Human-in-the-Loop** -- `HumanInputTool` and `HumanInputHandler` ABC for pausing agents for human input.
- **Loader** -- YAML-based agent and swarm loading with variable substitution.
- **Skills** -- multi-source skill registry for loading skills from local paths and GitHub.

## Quick Example

```python
from exo import Agent, run, tool


@tool
async def search(query: str) -> str:
    """Search the web.

    Args:
        query: The search query.
    """
    return f"Results for: {query}"


agent = Agent(
    name="assistant",
    model="openai:gpt-4o-mini",
    instructions="You are a helpful assistant.",
    tools=[search],
)

result = run.sync(agent, "Search for Python tutorials")
print(result.output)
```

## Public API

```python
from exo import (
    Agent,              # Core agent class
    Tool,               # Tool abstract base class
    FunctionTool,       # Function-based tool wrapper
    tool,               # @tool decorator
    run,                # Runner (run, run.sync, run.stream)
    Swarm,              # Multi-agent orchestrator
    SwarmNode,          # Nested swarm wrapper
    ParallelGroup,      # Concurrent agent group
    SerialGroup,        # Sequential agent group
)
```

## Documentation

- [Getting Started](../../docs/getting-started/)
- [Agent Guide](../../docs/guides/agents.md)
- [Tool Guide](../../docs/guides/tools.md)
- [API Reference](../../docs/reference/core/)
