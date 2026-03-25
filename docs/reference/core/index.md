# exo-core

The `exo-core` package is the foundation of the Exo framework. It provides all the building blocks for creating and running LLM-powered agents: the type system, configuration, tool abstractions, the agent class, execution runners, multi-agent swarms, lifecycle hooks, events, and more.

## Installation

```bash
pip install exo-core
```

## Import Patterns

The top-level `exo` namespace re-exports the most commonly used classes:

```python
from exo import Agent, Tool, FunctionTool, tool, run, Swarm
from exo import ParallelGroup, SerialGroup, SwarmNode
```

For more specific imports, use the individual modules:

```python
from exo.types import UserMessage, AssistantMessage, RunResult, StreamEvent
from exo.config import ModelConfig, AgentConfig, RunConfig, parse_model_string
from exo.registry import Registry, RegistryError, agent_registry, tool_registry
from exo.events import EventBus, EventHandler
from exo.hooks import HookPoint, HookManager, Hook, run_hooks
from exo.tool import Tool, FunctionTool, tool, ToolError
from exo.agent import Agent, AgentError
from exo.runner import run
from exo.swarm import Swarm, SwarmError
from exo.human import HumanInputTool, HumanInputHandler, ConsoleHandler
from exo.loader import load_agents, load_swarm, register_agent_class, LoaderError
from exo.skills import Skill, SkillRegistry, ConflictStrategy, SkillError
```

## Namespace Package Architecture

`exo-core` uses `pkgutil.extend_path()` to create a namespace package. This allows other Exo packages (such as `exo-models`, `exo-context`) to add modules to the `exo` namespace without conflicts.

## Modules

| Module | Description |
|--------|-------------|
| [types](types.md) | Core message types, agent I/O, run results, streaming events |
| [config](config.md) | Configuration dataclasses for models, agents, tasks, and runs |
| [registry](registry.md) | Generic named registry with duplicate detection |
| [events](events.md) | Async event bus for decoupled communication |
| [hooks](hooks.md) | Lifecycle hook system for agent execution interception |
| [tool](tool.md) | Tool ABC, FunctionTool wrapper, @tool decorator |
| [agent](agent.md) | The Agent class -- the core autonomous unit |
| [runner](runner.md) | `run()`, `run.sync()`, `run.stream()` execution API |
| [swarm](swarm.md) | Multi-agent orchestration with workflow, handoff, and team modes |
| [human](human.md) | Human-in-the-loop tool and input handlers |
| [loader](loader.md) | YAML-based agent and swarm loading |
| [skills](skills.md) | Multi-source skill registry |

## Quick Example

```python
import asyncio
from exo import Agent, tool, run

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is sunny in {city}."

agent = Agent(
    name="weather_bot",
    model="openai:gpt-4o",
    instructions="You are a helpful weather assistant.",
    tools=[get_weather],
)

result = run.sync(agent, "What is the weather in Tokyo?")
print(result.output)
```
