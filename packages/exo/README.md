# exo

Meta-package for the [Exo](../../README.md) multi-agent framework. Installs `exo-core` plus all standard extras in a single command.

## Installation

```bash
pip install exo
```

This installs:

- `exo-core` -- Agent, Tool, Runner, Swarm, Config, Events, Hooks
- `exo-models` -- LLM providers (OpenAI, Anthropic, Gemini)
- `exo-context` -- Context engine, neurons, prompt builder
- `exo-memory` -- Short/long-term memory, vector search
- `exo-mcp` -- Model Context Protocol client/server
- `exo-sandbox` -- Sandboxed execution environments
- `exo-observability` -- Logging, tracing, metrics, and health checks
- `exo-eval` -- Evaluation and scoring
- `exo-a2a` -- Agent-to-Agent protocol

CLI (`exo-cli`), server (`exo-server`), and training (`exo-train`) are installed separately.

## Quick Start

```python
from exo import Agent, run, tool


@tool
async def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


agent = Agent(
    name="greeter",
    model="openai:gpt-4o-mini",
    instructions="You are a friendly greeter.",
    tools=[greet],
)

result = run.sync(agent, "Say hi to Alice")
print(result.output)
```

## Documentation

See the full [Exo documentation](../../docs/).
