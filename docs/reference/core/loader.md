# exo.loader

YAML agent and swarm loader with variable substitution.

**Module:** `exo.loader`

```python
from exo.loader import (
    load_agents,
    load_swarm,
    load_yaml,
    register_agent_class,
    LoaderError,
)
```

---

## LoaderError

```python
class LoaderError(ExoError)
```

Raised for YAML loading or validation errors (file not found, missing sections, unknown agents, invalid format). Inherits from `ExoError`.

---

## load_agents()

```python
def load_agents(path: str | Path) -> dict[str, Any]
```

Load agents from a YAML file.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | `str \| Path` | *(required)* | Path to the YAML file. |

### Returns

Dict mapping agent name to `Agent` (or custom class) instance.

### Raises

- `LoaderError` -- if the file is not found.
- `LoaderError` -- if no `agents` section exists in the YAML.

### Example

```python
from exo.loader import load_agents

agents = load_agents("agents.yaml")
for name, agent in agents.items():
    print(f"{name}: {agent.describe()}")
```

---

## load_swarm()

```python
def load_swarm(path: str | Path) -> Swarm
```

Load a swarm (with agents) from a YAML file. If the YAML has no `swarm` section, creates a workflow-mode swarm with agents in declaration order.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | `str \| Path` | *(required)* | Path to the YAML file. |

### Returns

Configured `Swarm` instance.

### Raises

- `LoaderError` -- if the file is not found.
- `LoaderError` -- if no `agents` section exists in the YAML.
- `LoaderError` -- if swarm order references unknown agents.
- `LoaderError` -- if handoff edges reference unknown agents.

### Example

```python
from exo.loader import load_swarm
from exo import run

swarm = load_swarm("agents.yaml")
result = run.sync(swarm, "Analyze this dataset")
```

---

## load_yaml()

```python
def load_yaml(path: str | Path) -> dict[str, Any]
```

Load and substitute a YAML file, returning the raw dict. Performs variable substitution (`${ENV_VAR}` and `${vars.KEY}`) before returning.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | `str \| Path` | *(required)* | Path to the YAML file. |

### Returns

The substituted dict from the YAML file (with `vars` section removed).

### Raises

- `LoaderError` -- if the file is not found.
- `LoaderError` -- if the file does not contain a YAML dict.

---

## register_agent_class()

```python
def register_agent_class(name: str, cls: type) -> None
```

Register a custom agent class for YAML `type:` dispatch. When a YAML agent spec has `type: <name>`, the registered class is used instead of the builtin `Agent`.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | *(required)* | The type name used in YAML (e.g. `"custom"`). |
| `cls` | `type` | *(required)* | The class to instantiate. Must accept `name=` and other YAML spec keys as kwargs. |

### Example

```python
from exo.loader import register_agent_class, load_agents

class MyCustomAgent:
    def __init__(self, *, name: str, **kwargs):
        self.name = name
        self.config = kwargs

register_agent_class("custom", MyCustomAgent)

# In agents.yaml:
# agents:
#   my_bot:
#     type: custom
#     model: openai:gpt-4o
#     custom_field: value

agents = load_agents("agents.yaml")
```

---

## YAML File Format

### Basic Agent Definition

```yaml
agents:
  researcher:
    model: openai:gpt-4o
    instructions: "You are a research assistant."
    temperature: 0.7
    max_steps: 20

  writer:
    model: anthropic:claude-3-opus
    system_prompt: "You are a professional writer."
    max_tokens: 4096
```

### Variable Substitution

The loader supports two kinds of variable substitution:

**Environment variables** -- `${ENV_VAR}` is replaced with the value of the environment variable:

```yaml
agents:
  bot:
    model: ${MODEL_NAME}
    instructions: "Use API key ${API_KEY}"
```

**Internal variables** -- Define a `vars` section and reference with `${vars.KEY}`:

```yaml
vars:
  default_model: openai:gpt-4o
  base_prompt: "You are a helpful assistant."

agents:
  bot:
    model: ${vars.default_model}
    instructions: ${vars.base_prompt}
```

If a variable cannot be resolved, it remains as the literal string `${...}`.

### Swarm Configuration

```yaml
agents:
  researcher:
    model: openai:gpt-4o
    instructions: "Research the topic."
  writer:
    model: openai:gpt-4o
    instructions: "Write an article."

swarm:
  type: workflow          # workflow, handoff, or team
  flow: "researcher >> writer"
  order:                  # explicit agent order (optional)
    - researcher
    - writer
  max_handoffs: 10        # for handoff mode
  edges:                  # handoff edges (for handoff mode)
    - [researcher, writer]
```

### Agent Spec Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | `str` | Agent type: `"builtin"` (default) or a registered custom type. |
| `model` | `str` | Model string in `"provider:model_name"` format. |
| `instructions` | `str` | System prompt for the agent. |
| `system_prompt` | `str` | Alternative to `instructions` (same effect). |
| `temperature` | `float` | LLM sampling temperature. |
| `max_tokens` | `int` | Maximum output tokens per LLM call. |
| `max_steps` | `int` | Maximum LLM-tool round-trips. |
