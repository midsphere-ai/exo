# Core Concepts

This page describes the fundamental building blocks of Exo. Every agent system you build combines these primitives.

## Agent

An `Agent` is the core autonomous unit. It wraps an LLM model, a set of tools, optional handoff targets, and lifecycle hooks. Agents are created with keyword-only arguments:

```python
from exo import Agent

agent = Agent(
    name="my-agent",                       # required -- unique identifier
    model="openai:gpt-4o",                 # default: "openai:gpt-4o"
    instructions="You are helpful.",        # system prompt (str or callable)
    tools=[...],                            # list of Tool instances
    handoffs=[other_agent],                 # agents this one can delegate to
    hooks=[(HookPoint.START, my_hook)],     # lifecycle hooks
    output_type=MyModel,                    # Pydantic model for structured output
    max_steps=10,                           # max LLM-tool round-trips (default: 10)
    temperature=1.0,                        # LLM sampling temperature (default: 1.0)
    max_tokens=None,                        # max output tokens per call (default: None)
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *(required)* | Unique identifier for the agent |
| `model` | `str` | `"openai:gpt-4o"` | Model string in `"provider:model_name"` format |
| `instructions` | `str \| Callable[..., str]` | `""` | System prompt. Can be a string or an async callable that receives a context dict |
| `tools` | `list[Tool] \| None` | `None` | Tools available to the agent |
| `handoffs` | `list[Agent] \| None` | `None` | Other agents this agent can delegate to |
| `hooks` | `list[tuple[HookPoint, Hook]] \| None` | `None` | Lifecycle hooks as `(HookPoint, Hook)` tuples |
| `output_type` | `type[BaseModel] \| None` | `None` | Pydantic model for structured output validation |
| `max_steps` | `int` | `10` | Maximum LLM-tool round-trips before stopping |
| `temperature` | `float` | `1.0` | LLM sampling temperature |
| `max_tokens` | `int \| None` | `None` | Maximum output tokens per LLM call |
| `memory` | `Any` | `None` | Optional memory store for persistent memory |
| `context` | `Any` | `None` | Optional context engine for hierarchical state |

### Agent Methods

- **`agent.describe()`** -- Returns a dict summary of the agent's name, model, tools, handoffs, max_steps, and output_type.
- **`agent.get_tool_schemas()`** -- Returns OpenAI-format JSON schemas for all registered tools.
- **`await agent.run(input, messages=None, provider=None, max_retries=3)`** -- Executes the LLM-tool loop directly. Prefer using `run()` or `run.sync()` instead, which add state tracking and loop detection.

### Dynamic Instructions

Instructions can be a callable for runtime customization:

```python
def dynamic_instructions(agent_name: str) -> str:
    return f"You are {agent_name}. Today is Monday."

agent = Agent(
    name="assistant",
    instructions=dynamic_instructions,
)
```

---

## Tool

A `Tool` is a typed function the agent can call. The `@tool` decorator is the simplest way to create one:

```python
from exo import tool

@tool
async def search(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query string.
    """
    return f"Results for: {query}"
```

### How Schema Generation Works

The `@tool` decorator inspects the function's:

1. **Name** -- becomes the tool name (override with `@tool(name="custom_name")`)
2. **Docstring** -- first line becomes the description; `Args:` section provides parameter descriptions
3. **Type hints** -- converted to JSON Schema types (`str` to `"string"`, `int` to `"integer"`, `float` to `"number"`, `bool` to `"boolean"`, `list[X]` to `"array"`, `dict` to `"object"`)
4. **Default values** -- parameters without defaults are marked as `required`

### Decorator Forms

```python
# Bare decorator
@tool
async def my_tool(x: str) -> str: ...

# With parentheses
@tool()
async def my_tool(x: str) -> str: ...

# With overrides
@tool(name="custom_name", description="Custom description")
async def my_tool(x: str) -> str: ...
```

### Sync and Async

Both sync and async functions are supported. Sync functions are automatically wrapped via `asyncio.to_thread()` so they do not block the event loop:

```python
@tool
def read_file(path: str) -> str:
    """Read a file from disk."""
    return open(path).read()

@tool
async def fetch_url(url: str) -> str:
    """Fetch a URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()
```

### Tool ABC

For more control, subclass `Tool` directly:

```python
from exo.tool import Tool

class DatabaseTool(Tool):
    def __init__(self, connection_string: str):
        self.name = "query_db"
        self.description = "Run a SQL query."
        self.parameters = {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL query to execute"}
            },
            "required": ["sql"],
        }
        self.conn = connect(connection_string)

    async def execute(self, **kwargs) -> str:
        result = self.conn.execute(kwargs["sql"])
        return str(result.fetchall())
```

### ToolError

Raise `ToolError` in your tool to send an error message back to the LLM (instead of crashing the agent):

```python
from exo.tool import ToolError

@tool
async def divide(a: float, b: float) -> str:
    """Divide a by b."""
    if b == 0:
        raise ToolError("Cannot divide by zero")
    return str(a / b)
```

---

## Runner

The runner is the primary API for executing agents. It provides three entry points:

### `await run(agent, input, ...)` -- Async

```python
from exo import run

result = await run(agent, "Hello!")
```

### `run.sync(agent, input, ...)` -- Synchronous

A blocking wrapper that calls `asyncio.run()` internally. Use this in scripts, notebooks, and any context without an existing event loop:

```python
result = run.sync(agent, "Hello!")
```

### `run.stream(agent, input, ...)` -- Async Streaming

An async generator that yields `StreamEvent` objects in real time:

```python
async for event in run.stream(agent, "Hello!"):
    if event.type == "text":
        print(event.text, end="")
    elif event.type == "tool_call":
        print(f"\n[Tool: {event.tool_name}]")
```

### Run Parameters

All three entry points accept the same core parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `Agent` or `Swarm` | *(required)* | The agent or swarm to execute |
| `input` | `str` | *(required)* | User query string |
| `messages` | `Sequence[Message] \| None` | `None` | Prior conversation history |
| `provider` | `Any` | `None` | LLM provider (auto-resolved from agent's model if None) |

`run()` and `run.sync()` also accept:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_retries` | `int` | `3` | Retry attempts for transient LLM errors |
| `loop_threshold` | `int` | `3` | Consecutive identical tool-call patterns before raising a loop error |

`run.stream()` also accepts:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_steps` | `int \| None` | `None` | Maximum LLM-tool round-trips (defaults to `agent.max_steps`) |

### Provider Auto-Resolution

When `provider` is `None`, the runner automatically resolves a provider from the agent's model string. If `exo-models` is installed, it uses the model registry to look up `"openai"`, `"anthropic"`, etc. If auto-resolution fails, the agent's own `run()` method raises an `AgentError`.

---

## Swarm

A `Swarm` groups multiple agents and defines their execution topology:

```python
from exo import Agent, Swarm

researcher = Agent(name="researcher", model="openai:gpt-4o", ...)
writer = Agent(name="writer", model="openai:gpt-4o", ...)
reviewer = Agent(name="reviewer", model="openai:gpt-4o", ...)

swarm = Swarm(
    agents=[researcher, writer, reviewer],
    flow="researcher >> writer >> reviewer",
    mode="workflow",
)

result = await run(swarm, "Write an article about quantum computing")
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `list[Agent]` | *(required)* | Agents to include in the swarm |
| `flow` | `str \| None` | `None` | Flow DSL defining execution order (e.g., `"a >> b >> c"`) |
| `mode` | `str` | `"workflow"` | Execution mode: `"workflow"`, `"handoff"`, or `"team"` |
| `max_handoffs` | `int` | `10` | Maximum handoff transitions (handoff mode only) |

### Execution Modes

**Workflow** (`mode="workflow"`) -- Agents execute sequentially in the order defined by `flow`. Each agent's output becomes the next agent's input. The final agent's output is the swarm's output.

```python
swarm = Swarm(
    agents=[researcher, writer, reviewer],
    flow="researcher >> writer >> reviewer",
    mode="workflow",
)
```

**Handoff** (`mode="handoff"`) -- The first agent runs and can dynamically delegate to other agents via its `handoffs` list. Control transfers when an agent's output matches a handoff target name.

```python
triage = Agent(name="triage", handoffs=[billing, support, sales], ...)
billing = Agent(name="billing", ...)
support = Agent(name="support", ...)
sales = Agent(name="sales", ...)

swarm = Swarm(
    agents=[triage, billing, support, sales],
    mode="handoff",
)
```

**Team** (`mode="team"`) -- The first agent is the lead; others are workers. The lead receives auto-generated `delegate_to_{name}` tools to invoke workers. Workers run independently and their output is returned as tool results to the lead.

```python
swarm = Swarm(
    agents=[lead, analyst, coder],
    mode="team",
)
# The lead agent gets tools: delegate_to_analyst, delegate_to_coder
```

### Flow DSL

When `flow` is not provided, agents execute in the order they appear in the `agents` list. The DSL uses `>>` to define sequential ordering:

```python
flow="a >> b >> c"  # a runs first, then b, then c
```

### Agent Groups

For concurrent execution within a flow, use `ParallelGroup` and `SerialGroup`:

```python
from exo import ParallelGroup, SerialGroup

parallel = ParallelGroup(
    name="research_team",
    agents=[agent_a, agent_b],
    separator="\n\n",           # how to join outputs (default: "\n\n")
)

serial = SerialGroup(
    name="pipeline",
    agents=[agent_c, agent_d],  # c's output becomes d's input
)
```

Groups behave like agents in a Swarm's flow DSL and can be placed in the `agents` list.

### Nested Swarms

Use `SwarmNode` to nest one swarm inside another:

```python
from exo import Swarm, SwarmNode

inner = Swarm(agents=[a, b], flow="a >> b")
node = SwarmNode(swarm=inner, name="inner_pipeline")
outer = Swarm(agents=[c, node, d], flow="c >> inner_pipeline >> d")
```

Each nested swarm runs with context isolation -- it creates its own message history and does not share mutable state with the outer swarm.

---

## Message Types

Exo uses a typed message system. All messages are frozen Pydantic models:

### UserMessage

A message from the user:

```python
from exo.types import UserMessage

msg = UserMessage(content="What's the weather?")
# msg.role == "user"
```

### SystemMessage

A system instruction:

```python
from exo.types import SystemMessage

msg = SystemMessage(content="You are a helpful assistant.")
# msg.role == "system"
```

### AssistantMessage

A response from the LLM. May contain text, tool calls, or both:

```python
from exo.types import AssistantMessage, ToolCall

msg = AssistantMessage(
    content="Let me check the weather.",
    tool_calls=[
        ToolCall(id="call_1", name="get_weather", arguments='{"city": "Tokyo"}')
    ],
)
# msg.role == "assistant"
```

### ToolResult

The result of executing a tool call:

```python
from exo.types import ToolResult

msg = ToolResult(
    tool_call_id="call_1",
    tool_name="get_weather",
    content="Sunny, 22 C in Tokyo.",
)
# msg.role == "tool"
# msg.error is None for successful calls
```

### ToolCall

A request from the LLM to invoke a tool (embedded in `AssistantMessage.tool_calls`):

```python
from exo.types import ToolCall

tc = ToolCall(
    id="call_abc123",
    name="get_weather",
    arguments='{"city": "Tokyo"}',  # JSON-encoded string
)
```

### Message Union

The `Message` type is a union of all message types:

```python
Message = UserMessage | AssistantMessage | SystemMessage | ToolResult
```

---

## RunResult

The return type of `run()` and `run.sync()`:

```python
from exo.types import RunResult
```

| Field | Type | Description |
|-------|------|-------------|
| `output` | `str` | Final text output from the agent (default: `""`) |
| `messages` | `list[Message]` | Full message history of the run |
| `usage` | `Usage` | Aggregated token usage across all steps |
| `steps` | `int` | Number of LLM call steps taken (default: `0`) |

### Usage

Token usage statistics:

```python
from exo.types import Usage

usage = Usage(input_tokens=150, output_tokens=50, total_tokens=200)
```

| Field | Type | Default |
|-------|------|---------|
| `input_tokens` | `int` | `0` |
| `output_tokens` | `int` | `0` |
| `total_tokens` | `int` | `0` |

---

## Stream Events

`run.stream()` yields `StreamEvent` objects, which is a union of two types:

### TextEvent

A chunk of text from the LLM:

```python
from exo.types import TextEvent

# event.type == "text"
# event.text -- the text chunk
# event.agent_name -- which agent produced this
```

### ToolCallEvent

Notification that a tool is being invoked:

```python
from exo.types import ToolCallEvent

# event.type == "tool_call"
# event.tool_name -- name of the tool
# event.tool_call_id -- unique ID for this invocation
# event.agent_name -- which agent produced this
```

The `StreamEvent` union:

```python
StreamEvent = TextEvent | ToolCallEvent
```

---

## Hooks

Hooks let you intercept the agent lifecycle at specific points. They are async functions registered as `(HookPoint, Hook)` tuples:

```python
from exo.hooks import HookPoint

async def log_llm_call(**data):
    print(f"LLM call on agent: {data['agent'].name}")

agent = Agent(
    name="my-agent",
    hooks=[(HookPoint.PRE_LLM_CALL, log_llm_call)],
)
```

### Hook Points

| HookPoint | When It Fires | Data Passed |
|-----------|---------------|-------------|
| `START` | Run begins | `agent` |
| `FINISHED` | Run completes | `agent` |
| `ERROR` | Run encounters an error | `agent` |
| `PRE_LLM_CALL` | Before each LLM call | `agent`, `messages` |
| `POST_LLM_CALL` | After each LLM call | `agent`, `response` |
| `PRE_TOOL_CALL` | Before each tool execution | `agent`, `tool_name`, `arguments` |
| `POST_TOOL_CALL` | After each tool execution | `agent`, `tool_name`, `result` |

Hooks are called sequentially in registration order. Unlike the EventBus, hook exceptions are **not** suppressed -- a failing hook aborts the run.

---

## Events

The `EventBus` provides decoupled async pub-sub communication:

```python
from exo.events import EventBus

bus = EventBus()

async def on_step(**data):
    print(f"Step completed: {data}")

bus.on("step_complete", on_step)
await bus.emit("step_complete", agent_name="bot", step=1)
```

Events are distinct from hooks: hooks are tightly coupled to the agent lifecycle, while events are for decoupled, application-level communication.

---

## Next Steps

- **[Your First Agent](first-agent.md)** -- Put these concepts into practice with a step-by-step tutorial
- **[Quickstart](quickstart.md)** -- Run a working example in 5 minutes
