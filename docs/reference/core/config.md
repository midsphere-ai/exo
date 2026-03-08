# orbiter.config

Configuration types for models, agents, tasks, and runs in the Orbiter framework.

**Module:** `orbiter.config`

```python
from orbiter.config import (
    parse_model_string,
    ModelConfig,
    AgentConfig,
    TaskConfig,
    RunConfig,
)
```

---

## parse_model_string()

```python
def parse_model_string(model: str) -> tuple[str, str]
```

Split a model string into provider and model name.

Parses the `"provider:model_name"` format. If no colon is present, defaults the provider to `"openai"`.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `str` | *(required)* | Model string, e.g. `"openai:gpt-4o"` or `"gpt-4o"`. |

### Returns

A `(provider, model_name)` tuple of strings.

### Example

```python
from orbiter.config import parse_model_string

provider, model_name = parse_model_string("openai:gpt-4o")
# provider = "openai", model_name = "gpt-4o"

provider, model_name = parse_model_string("gpt-4o")
# provider = "openai", model_name = "gpt-4o"

provider, model_name = parse_model_string("anthropic:claude-3-opus")
# provider = "anthropic", model_name = "claude-3-opus"
```

---

## ModelConfig

```python
class ModelConfig(BaseModel)
```

Configuration for an LLM provider connection. Immutable (frozen Pydantic model).

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `provider` | `str` | `"openai"` | Provider name, e.g. `"openai"` or `"anthropic"`. |
| `model_name` | `str` | `"gpt-4o"` | Model identifier within the provider. |
| `api_key` | `str \| None` | `None` | API key for authentication. |
| `base_url` | `str \| None` | `None` | Custom API base URL. |
| `max_retries` | `int` | `3` | Maximum number of retries on transient failures. Must be >= 0. |
| `timeout` | `float` | `30.0` | Request timeout in seconds. Must be > 0. |

### Example

```python
from orbiter.config import ModelConfig

config = ModelConfig(
    provider="openai",
    model_name="gpt-4o",
    api_key="sk-...",
    timeout=60.0,
)
```

---

## AgentConfig

```python
class AgentConfig(BaseModel)
```

Configuration for an Agent. Immutable (frozen Pydantic model).

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | *(required)* | Unique identifier for the agent. |
| `model` | `str` | `"openai:gpt-4o"` | Model string in `"provider:model_name"` format. |
| `instructions` | `str` | `""` | System prompt for the agent. |
| `temperature` | `float` | `1.0` | LLM sampling temperature. Must be between 0.0 and 2.0. |
| `max_tokens` | `int \| None` | `None` | Maximum output tokens per LLM call. |
| `max_steps` | `int` | `10` | Maximum LLM-tool round-trips. Must be >= 1. |
| `planning_enabled` | `bool` | `False` | Enables the planner phase contract. |
| `planning_model` | `str \| None` | `None` | Optional planner model override. |
| `planning_instructions` | `str` | `""` | Optional planner-only instructions. |
| `budget_awareness` | `str \| None` | `None` | Either `per-message` or `limit:<0-100>`. |
| `hitl_tools` | `list[str]` | `[]` | Tool names that require approval before execution. |
| `emit_mcp_progress` | `bool` | `True` | Controls MCP progress-event emission. |
| `injected_tool_args` | `dict[str, str]` | `{}` | Schema-only tool arguments exposed to the model. |
| `allow_parallel_subagents` | `bool` | `False` | Enables the parallel sub-agent contract. |
| `max_parallel_subagents` | `int` | `3` | Maximum child jobs per parallel call. Must be `1..7`. |

### Example

```python
from orbiter.config import AgentConfig

config = AgentConfig(
    name="researcher",
    model="anthropic:claude-3-opus",
    instructions="You are a research assistant.",
    temperature=0.7,
    max_steps=20,
    planning_enabled=True,
    planning_model="openai:gpt-4o-mini",
    budget_awareness="limit:70",
    hitl_tools=["search_documents"],
    injected_tool_args={"ui_request_id": "Opaque UI correlation id"},
)
```

---

## TaskConfig

```python
class TaskConfig(BaseModel)
```

Configuration for a task. Immutable (frozen Pydantic model).

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | *(required)* | Unique identifier for the task. |
| `description` | `str` | `""` | Human-readable description of what the task does. |

### Example

```python
from orbiter.config import TaskConfig

config = TaskConfig(
    name="summarize",
    description="Summarize a document into key points.",
)
```

---

## RunConfig

```python
class RunConfig(BaseModel)
```

Configuration for a single run invocation. Immutable (frozen Pydantic model).

### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `max_steps` | `int` | `10` | Maximum LLM-tool round-trips for this run. Must be >= 1. |
| `timeout` | `float \| None` | `None` | Overall timeout in seconds for the run. |
| `stream` | `bool` | `False` | Whether to enable streaming output. |
| `verbose` | `bool` | `False` | Whether to enable verbose logging. |

### Example

```python
from orbiter.config import RunConfig

config = RunConfig(
    max_steps=20,
    timeout=120.0,
    stream=True,
    verbose=True,
)
```
