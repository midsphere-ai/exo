# Agents

Agents are the core building block in Orbiter. Each agent wraps an LLM model, a set of tools, optional handoff targets, and lifecycle hooks. When executed, an agent runs an LLM-tool loop: it calls the LLM, executes any requested tool calls in parallel, feeds the results back, and repeats until the LLM produces a text-only response or the step limit is reached.

## Basic Usage

```python
from orbiter.agent import Agent
from orbiter.tool import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 72F in {city}"

agent = Agent(
    name="weather_bot",
    model="openai:gpt-4o",
    instructions="You are a helpful weather assistant.",
    tools=[get_weather],
)
```

To execute an agent, use `run()` (see [Running Agents](running.md)):

```python
from orbiter.runner import run

result = await run(agent, "What's the weather in Tokyo?")
print(result.output)
```

## Constructor Parameters

All parameters are keyword-only. Only `name` is required.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Unique identifier for this agent |
| `model` | `str` | `"openai:gpt-4o"` | Model string in `"provider:model_name"` format |
| `instructions` | `str \| Callable[..., str]` | `""` | System prompt, or a callable that returns one |
| `tools` | `list[Tool] \| None` | `None` | Tools available to this agent |
| `handoffs` | `list[Agent] \| None` | `None` | Other agents this agent can delegate to |
| `hooks` | `list[tuple[HookPoint, Hook]] \| None` | `None` | Lifecycle hooks as `(HookPoint, Hook)` tuples |
| `output_type` | `type[BaseModel] \| None` | `None` | Pydantic model for [structured output](structured-output.md) validation |
| `max_steps` | `int` | `10` | Maximum LLM-tool round-trips before stopping |
| `temperature` | `float` | `1.0` | LLM sampling temperature |
| `max_tokens` | `int \| None` | `None` | Maximum output tokens per LLM call |
| `memory` | `Any` | `None` | Optional memory store for persistent memory |
| `context` | `Any` | `None` | Optional context engine for hierarchical state |

## Dynamic Instructions

The `instructions` parameter can be a callable. This is useful when the system prompt needs to change based on runtime context:

```python
def make_instructions(agent_name: str) -> str:
    return f"You are {agent_name}. Today is Monday. Be concise."

agent = Agent(
    name="dynamic_bot",
    instructions=make_instructions,
)
```

When `instructions` is callable, it is invoked with the agent's name as the argument at the start of each `run()` call.

## Agent.describe()

The `describe()` method returns a summary dict of the agent's capabilities. This is useful for debugging, logging, and capability advertisement in multi-agent systems.

```python
agent = Agent(
    name="helper",
    model="anthropic:claude-sonnet-4-20250514",
    tools=[get_weather],
    max_steps=5,
)

print(agent.describe())
# {
#     "name": "helper",
#     "model": "anthropic:claude-sonnet-4-20250514",
#     "tools": ["get_weather"],
#     "handoffs": [],
#     "max_steps": 5,
#     "output_type": None,
# }
```

## Tool Registration

Tools are indexed by name for O(1) lookup. Duplicate tool names raise `AgentError`:

```python
from orbiter.agent import Agent, AgentError
from orbiter.tool import tool

@tool
def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"

@tool(name="greet")  # duplicate!
def greet_v2(name: str) -> str:
    """Say hello differently."""
    return f"Hi there, {name}!"

try:
    agent = Agent(name="bot", tools=[greet, greet_v2])
except AgentError as e:
    print(e)  # "Duplicate tool name 'greet' on agent 'bot'"
```

## Handoffs

Handoffs allow an agent to delegate to another agent. The target agents are registered by name:

```python
researcher = Agent(name="researcher", instructions="You research topics deeply.")
writer = Agent(name="writer", instructions="You write clearly and concisely.")

coordinator = Agent(
    name="coordinator",
    instructions="Delegate research to 'researcher' and writing to 'writer'.",
    handoffs=[researcher, writer],
)
```

Handoffs are primarily used within [Swarm](multi-agent.md) orchestration in handoff mode.

## The Agent.run() Method

The `run()` method executes the agent's LLM-tool loop directly:

```python
output = await agent.run(
    "What's the weather?",
    provider=my_provider,   # required: object with async complete()
    messages=history,       # optional: prior conversation
    max_retries=3,          # default: 3 retry attempts
)
# output is an AgentOutput with .text, .tool_calls, .usage
```

In most cases, you should use the top-level `run()` function from `orbiter.runner` instead, which adds state tracking, loop detection, and auto-resolved providers. See [Running Agents](running.md).

## AgentConfig

For serializable configuration, use `AgentConfig` from `orbiter.config`:

```python
from orbiter.config import AgentConfig

config = AgentConfig(
    name="my_agent",
    model="openai:gpt-4o",
    instructions="Be helpful.",
    temperature=0.7,
    max_steps=5,
    planning_enabled=True,
    planning_model="openai:gpt-4o-mini",
    budget_awareness="limit:70",
    hitl_tools=["deploy_service"],
    injected_tool_args={"ui_request_id": "Opaque UI correlation id"},
)
```

| Field | Type | Default |
|-------|------|---------|
| `name` | `str` | *required* |
| `model` | `str` | `"openai:gpt-4o"` |
| `instructions` | `str` | `""` |
| `temperature` | `float` | `1.0` (range: 0.0-2.0) |
| `max_tokens` | `int \| None` | `None` |
| `max_steps` | `int` | `10` (min: 1) |
| `planning_enabled` | `bool` | `False` |
| `planning_model` | `str \| None` | `None` |
| `planning_instructions` | `str` | `""` |
| `budget_awareness` | `str \| None` | `None` (`per-message` or `limit:<0-100>`) |
| `hitl_tools` | `list[str]` | `[]` |
| `emit_mcp_progress` | `bool` | `True` |
| `injected_tool_args` | `dict[str, str]` | `{}` |
| `allow_parallel_subagents` | `bool` | `False` |
| `max_parallel_subagents` | `int` | `3` (range: 1-7) |

## Live Message Injection

The `inject_message()` method pushes a user message into a running agent's context. The message is picked up before the next LLM call, letting external code steer or augment the agent mid-run without cancelling.

```python
import asyncio
from orbiter import Agent, run, tool

@tool
def slow_search(query: str) -> str:
    """Search that takes time."""
    return f"Results for {query}"

agent = Agent(name="researcher", tools=[slow_search])

async def main():
    task = asyncio.create_task(run(agent, "Search for Python tutorials"))

    # Inject additional context while the agent is running
    await asyncio.sleep(0.1)
    agent.inject_message("Also check the official Python docs")

    result = await task
    print(result.output)
```

Key points:
- Messages are drained in FIFO order before each LLM call
- Injected messages never break provider message compliance (they appear after tool results, before the next LLM call)
- Raises `ValueError` if content is empty
- Safe to call from any coroutine in the same event loop
- In streaming mode (`run.stream()`), each injection emits a `MessageInjectedEvent`

For full details and streaming examples, see [Streaming Events > Live Message Injection](streaming-events.md#live-message-injection).

## Error Handling

Agent errors raise `AgentError`, a subclass of `OrbiterError`:

- **Duplicate tools:** Registering two tools with the same name
- **Duplicate handoffs:** Registering two handoff targets with the same name
- **No provider:** Calling `run()` without a provider
- **Retries exhausted:** All retry attempts failed for an LLM call
- **Context length exceeded:** Input exceeds the model's context window

```python
from orbiter.agent import Agent, AgentError

try:
    result = await agent.run("Hello", provider=None)
except AgentError as e:
    print(f"Agent error: {e}")
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `Agent` | `orbiter.agent` | Core agent class |
| `AgentError` | `orbiter.agent` | Agent-level error |
| `AgentConfig` | `orbiter.config` | Serializable agent configuration |
| `AgentOutput` | `orbiter.types` | Output from a single LLM call |
| `RunResult` | `orbiter.types` | Final result of `run()` |
