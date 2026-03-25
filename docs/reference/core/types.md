# exo.types

Core message types, agent I/O models, run results, and streaming events for the Exo framework.

**Module:** `exo.types`

```python
from exo.types import (
    ExoError,
    UserMessage,
    SystemMessage,
    AssistantMessage,
    ToolCall,
    ToolResult,
    Message,
    Usage,
    AgentInput,
    AgentOutput,
    ActionModel,
    RunResult,
    TextEvent,
    ToolCallEvent,
    StreamEvent,
)
```

---

## ExoError

```python
class ExoError(Exception)
```

Base exception for all Exo errors. All framework-specific exceptions inherit from this class.

```python
from exo.types import ExoError

try:
    raise ExoError("something went wrong")
except ExoError as e:
    print(e)
```

---

## UserMessage

```python
class UserMessage(BaseModel)
```

A message from the user. Immutable (frozen Pydantic model).

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `role` | `Literal["user"]` | `"user"` | Message role discriminator. |
| `content` | `str` | *(required)* | The user's message text. |

### Example

```python
from exo.types import UserMessage

msg = UserMessage(content="What is the weather today?")
print(msg.role)     # "user"
print(msg.content)  # "What is the weather today?"
```

---

## SystemMessage

```python
class SystemMessage(BaseModel)
```

A system instruction message. Immutable (frozen Pydantic model).

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `role` | `Literal["system"]` | `"system"` | Message role discriminator. |
| `content` | `str` | *(required)* | The system instruction text. |

### Example

```python
from exo.types import SystemMessage

msg = SystemMessage(content="You are a helpful assistant.")
print(msg.role)     # "system"
print(msg.content)  # "You are a helpful assistant."
```

---

## ToolCall

```python
class ToolCall(BaseModel)
```

A request from the LLM to invoke a tool. Immutable (frozen Pydantic model).

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `id` | `str` | *(required)* | Unique identifier for this tool call. |
| `name` | `str` | *(required)* | Name of the tool to invoke. |
| `arguments` | `str` | `""` | JSON-encoded string of the tool arguments. |

### Example

```python
from exo.types import ToolCall

tc = ToolCall(id="call_1", name="get_weather", arguments='{"city": "Tokyo"}')
print(tc.name)       # "get_weather"
print(tc.arguments)  # '{"city": "Tokyo"}'
```

---

## AssistantMessage

```python
class AssistantMessage(BaseModel)
```

A response from the LLM assistant. May contain text content, tool calls, or both. Immutable (frozen Pydantic model).

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `role` | `Literal["assistant"]` | `"assistant"` | Message role discriminator. |
| `content` | `str` | `""` | Text content of the response. |
| `tool_calls` | `list[ToolCall]` | `[]` | Tool invocations requested by the assistant. |

### Example

```python
from exo.types import AssistantMessage, ToolCall

# Text-only response
msg = AssistantMessage(content="The weather in Tokyo is sunny.")

# Response with tool calls
msg = AssistantMessage(
    content="Let me check that for you.",
    tool_calls=[ToolCall(id="call_1", name="get_weather", arguments='{"city": "Tokyo"}')],
)
```

---

## ToolResult

```python
class ToolResult(BaseModel)
```

The result of executing a tool call. Immutable (frozen Pydantic model).

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `role` | `Literal["tool"]` | `"tool"` | Message role discriminator. |
| `tool_call_id` | `str` | *(required)* | The ID of the ToolCall this responds to. |
| `tool_name` | `str` | *(required)* | Name of the tool that was executed. |
| `content` | `str` | `""` | The string result from the tool. |
| `error` | `str \| None` | `None` | Error message if the tool failed. |

### Example

```python
from exo.types import ToolResult

# Successful result
result = ToolResult(
    tool_call_id="call_1",
    tool_name="get_weather",
    content="Sunny, 25C",
)

# Error result
result = ToolResult(
    tool_call_id="call_1",
    tool_name="get_weather",
    error="API rate limit exceeded",
)
```

---

## Message

```python
Message = UserMessage | AssistantMessage | SystemMessage | ToolResult
```

Union type alias for all message types. Used as the element type in conversation history lists.

---

## Usage

```python
class Usage(BaseModel)
```

Token usage statistics from an LLM call. Immutable (frozen Pydantic model).

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input_tokens` | `int` | `0` | Number of tokens in the prompt. |
| `output_tokens` | `int` | `0` | Number of tokens in the completion. |
| `total_tokens` | `int` | `0` | Total tokens consumed. |

### Example

```python
from exo.types import Usage

usage = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
print(usage.total_tokens)  # 150
```

---

## AgentInput

```python
class AgentInput(BaseModel)
```

Normalized input for an agent run. Immutable (frozen Pydantic model).

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `query` | `str` | *(required)* | The user's query string. |
| `messages` | `list[Message]` | `[]` | Prior conversation messages for context. |

### Example

```python
from exo.types import AgentInput, UserMessage

inp = AgentInput(
    query="What is the weather?",
    messages=[UserMessage(content="Hello")],
)
```

---

## AgentOutput

```python
class AgentOutput(BaseModel)
```

Output from a single LLM call within a run. Immutable (frozen Pydantic model).

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `str` | `""` | Text content of the LLM response. |
| `tool_calls` | `list[ToolCall]` | `[]` | Tool invocations requested by the LLM. |
| `usage` | `Usage` | `Usage()` | Token usage for this call. |

### Example

```python
from exo.types import AgentOutput, Usage

output = AgentOutput(
    text="Here is your answer.",
    usage=Usage(input_tokens=50, output_tokens=20, total_tokens=70),
)
```

---

## ActionModel

```python
class ActionModel(BaseModel)
```

A parsed tool action ready for execution. Unlike `ToolCall` where `arguments` is a JSON string, here `arguments` is already parsed into a dict. Immutable (frozen Pydantic model).

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tool_call_id` | `str` | *(required)* | Identifier linking back to the originating ToolCall. |
| `tool_name` | `str` | *(required)* | Name of the tool to execute. |
| `arguments` | `dict[str, Any]` | `{}` | Parsed keyword arguments for the tool. |

### Example

```python
from exo.types import ActionModel

action = ActionModel(
    tool_call_id="call_1",
    tool_name="get_weather",
    arguments={"city": "Tokyo"},
)
```

---

## RunResult

```python
class RunResult(BaseModel)
```

Return type of `run()` -- the final result of an agent execution. Immutable (frozen Pydantic model).

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `output` | `str` | `""` | Final text output from the agent. |
| `messages` | `list[Message]` | `[]` | Full message history of the run. |
| `usage` | `Usage` | `Usage()` | Aggregated token usage across all steps. |
| `steps` | `int` | `0` | Number of LLM call steps taken. Must be >= 0. |

### Example

```python
from exo.types import RunResult, Usage

result = RunResult(
    output="The weather in Tokyo is sunny.",
    usage=Usage(input_tokens=200, output_tokens=50, total_tokens=250),
    steps=2,
)
print(result.output)  # "The weather in Tokyo is sunny."
print(result.steps)   # 2
```

---

## TextEvent

```python
class TextEvent(BaseModel)
```

Streaming event for a text delta. Immutable (frozen Pydantic model).

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `type` | `Literal["text"]` | `"text"` | Discriminator literal. |
| `text` | `str` | *(required)* | The text chunk. |
| `agent_name` | `str` | `""` | Name of the agent producing this event. |

### Example

```python
from exo.types import TextEvent

event = TextEvent(text="Hello", agent_name="assistant")
print(event.type)  # "text"
print(event.text)  # "Hello"
```

---

## ToolCallEvent

```python
class ToolCallEvent(BaseModel)
```

Streaming event for a tool call notification. Immutable (frozen Pydantic model).

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `type` | `Literal["tool_call"]` | `"tool_call"` | Discriminator literal. |
| `tool_name` | `str` | *(required)* | Name of the tool being called. |
| `tool_call_id` | `str` | *(required)* | Identifier for this tool call. |
| `agent_name` | `str` | `""` | Name of the agent producing this event. |

### Example

```python
from exo.types import ToolCallEvent

event = ToolCallEvent(
    tool_name="get_weather",
    tool_call_id="call_1",
    agent_name="weather_bot",
)
```

---

## MessageInjectedEvent

```python
class MessageInjectedEvent(BaseModel)
```

Streaming event emitted when a message is injected into a running agent via `agent.inject_message()`. Immutable (frozen Pydantic model).

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `type` | `Literal["message_injected"]` | `"message_injected"` | Discriminator literal. |
| `content` | `str` | *(required)* | The injected message text. |
| `agent_name` | `str` | `""` | Name of the agent that received the injection. |

### Example

```python
from exo.types import MessageInjectedEvent

event = MessageInjectedEvent(content="Also check the docs", agent_name="researcher")
print(event.type)     # "message_injected"
print(event.content)  # "Also check the docs"
```

---

## StreamEvent

```python
StreamEvent = (
    TextEvent | ToolCallEvent | StepEvent | ToolResultEvent
    | ReasoningEvent | ErrorEvent | StatusEvent | UsageEvent
    | MCPProgressEvent | ContextEvent | MessageInjectedEvent
)
```

Union type alias for all streaming event types. Yielded by `run.stream()`.
