# exo.agent

The Agent class -- the core autonomous LLM-powered unit in Exo.

**Module:** `exo.agent`

```python
from exo.agent import Agent, AgentError
```

---

## AgentError

```python
class AgentError(ExoError)
```

Raised for agent-level errors (duplicate tools, invalid config, missing provider, context length exceeded, retries exhausted, etc.). Inherits from `ExoError`.

---

## Agent

```python
class Agent
```

An autonomous LLM-powered agent with tools and lifecycle hooks. Agents are the core building block in Exo. Each agent wraps an LLM model, a set of tools, optional handoff targets, and lifecycle hooks.

All parameters are keyword-only; only `name` is required.

### Constructor

```python
def __init__(
    self,
    *,
    name: str,
    model: str = "openai:gpt-4o",
    instructions: str | Callable[..., str] = "",
    tools: list[Tool] | None = None,
    handoffs: list[Agent] | None = None,
    hooks: list[tuple[HookPoint, Hook]] | None = None,
    output_type: type[BaseModel] | None = None,
    max_steps: int = 10,
    temperature: float = 1.0,
    max_tokens: int | None = None,
    memory: Any = None,
    context: Any = None,
) -> None
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | *(required)* | Unique identifier for this agent. |
| `model` | `str` | `"openai:gpt-4o"` | Model string in `"provider:model_name"` format. |
| `instructions` | `str \| Callable[..., str]` | `""` | System prompt. Can be a string or a callable that receives a context dict and returns a string. |
| `tools` | `list[Tool] \| None` | `None` | Tools available to this agent. |
| `handoffs` | `list[Agent] \| None` | `None` | Other agents this agent can delegate to via handoff. |
| `hooks` | `list[tuple[HookPoint, Hook]] \| None` | `None` | Lifecycle hooks as `(HookPoint, Hook)` tuples. |
| `output_type` | `type[BaseModel] \| None` | `None` | Pydantic model class for structured output validation. |
| `max_steps` | `int` | `10` | Maximum LLM-tool round-trips before stopping. |
| `temperature` | `float` | `1.0` | LLM sampling temperature. |
| `max_tokens` | `int \| None` | `None` | Maximum output tokens per LLM call. |
| `memory` | `Any` | `None` | Optional memory store for persistent memory across sessions. |
| `context` | `Any` | `None` | Optional context engine for hierarchical state and prompt building. |

### Instance Attributes

| Name | Type | Description |
|------|------|-------------|
| `name` | `str` | Agent identifier. |
| `model` | `str` | Full model string (e.g. `"openai:gpt-4o"`). |
| `provider_name` | `str` | Parsed provider name (e.g. `"openai"`). |
| `model_name` | `str` | Parsed model name (e.g. `"gpt-4o"`). |
| `instructions` | `str \| Callable` | System prompt or callable. |
| `output_type` | `type[BaseModel] \| None` | Structured output type. |
| `max_steps` | `int` | Max LLM-tool round-trips. |
| `temperature` | `float` | Sampling temperature. |
| `max_tokens` | `int \| None` | Max output tokens. |
| `memory` | `Any` | Memory store. |
| `context` | `Any` | Context engine. |
| `tools` | `dict[str, Tool]` | Registered tools indexed by name. |
| `handoffs` | `dict[str, Agent]` | Handoff targets indexed by name. |
| `hook_manager` | `HookManager` | Lifecycle hook manager. |

### Methods

#### run()

```python
async def run(
    self,
    input: str,
    *,
    messages: Sequence[Message] | None = None,
    provider: Any = None,
    max_retries: int = 3,
) -> AgentOutput
```

Execute the agent's LLM-tool loop with retry logic. Builds the message list, calls the LLM, and if tool calls are returned, executes them in parallel, feeds results back, and re-calls the LLM. The loop continues until a text-only response is produced or `max_steps` is reached.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input` | `str` | *(required)* | User query string for this turn. |
| `messages` | `Sequence[Message] \| None` | `None` | Prior conversation history. |
| `provider` | `Any` | `None` | An object with an `async complete()` method (e.g. a `ModelProvider` instance). |
| `max_retries` | `int` | `3` | Maximum retry attempts for transient errors. |

**Returns:** Parsed `AgentOutput` from the final LLM response.

**Raises:**
- `AgentError` -- if no provider is supplied.
- `AgentError` -- if all retries are exhausted.
- `AgentError` -- if context length is exceeded.

#### inject_message()

```python
def inject_message(self, content: str) -> None
```

Push a user message into the running agent's context. The message is picked up before the next LLM call. Safe to call from any coroutine in the same event loop.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `content` | `str` | *(required)* | The message text to inject. Must be non-empty. |

**Raises:** `ValueError` -- if `content` is empty.

Injected messages are drained in FIFO order at the start of each step, after tool results from the previous step and before the LLM call. This preserves provider message compliance.

In the streaming path (`run.stream()`), each drained message emits a `MessageInjectedEvent`.

#### get_tool_schemas()

```python
def get_tool_schemas(self) -> list[dict[str, Any]]
```

Return OpenAI-format tool schemas for all registered tools.

**Returns:** A list of tool schema dicts suitable for LLM function calling.

#### describe()

```python
def describe(self) -> dict[str, Any]
```

Return a summary of the agent's capabilities. Useful for debugging, logging, and capability advertisement in multi-agent systems.

**Returns:** A dict with keys:
- `name` (str)
- `model` (str)
- `tools` (list[str])
- `handoffs` (list[str])
- `max_steps` (int)
- `output_type` (str | None)

#### \_\_repr\_\_()

```python
def __repr__(self) -> str
```

Return a readable string representation, e.g. `Agent(name='bot', model='openai:gpt-4o', tools=['get_weather'])`.

### Example

```python
import asyncio
from exo.agent import Agent
from exo.tool import tool
from exo.hooks import HookPoint

@tool
def search(query: str) -> str:
    """Search the web.

    Args:
        query: The search query.
    """
    return f"Results for: {query}"

async def log_llm(**data):
    print("LLM call starting...")

agent = Agent(
    name="researcher",
    model="openai:gpt-4o",
    instructions="You are a research assistant.",
    tools=[search],
    hooks=[(HookPoint.PRE_LLM_CALL, log_llm)],
    max_steps=5,
    temperature=0.7,
)

# Inspect
print(agent.describe())
# {
#     "name": "researcher",
#     "model": "openai:gpt-4o",
#     "tools": ["search"],
#     "handoffs": [],
#     "max_steps": 5,
#     "output_type": None,
# }

# Run (requires a provider)
# result = asyncio.run(agent.run("Find papers on AI safety", provider=my_provider))
```

### Dynamic Instructions

Instructions can be a callable that receives the agent name:

```python
def dynamic_instructions(agent_name: str) -> str:
    return f"You are {agent_name}. Today is Monday."

agent = Agent(
    name="assistant",
    instructions=dynamic_instructions,
)
```

### Handoffs

Agents can hand off to other agents in a multi-agent system:

```python
researcher = Agent(name="researcher", model="openai:gpt-4o")
writer = Agent(name="writer", model="openai:gpt-4o")

lead = Agent(
    name="lead",
    model="openai:gpt-4o",
    handoffs=[researcher, writer],
)
```
