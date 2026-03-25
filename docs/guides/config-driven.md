# Config-Driven Agents

Exo supports loading agents and swarms from YAML configuration files. This enables defining agent configurations declaratively, separating configuration from code, and using environment variables for secrets.

## Basic Usage

Create a YAML file defining your agents:

```yaml
# agents.yaml
vars:
  default_model: "openai:gpt-4o"

agents:
  researcher:
    model: "${vars.default_model}"
    instructions: "You research topics deeply and thoroughly."
    temperature: 0.7
    max_steps: 5

  writer:
    model: "${vars.default_model}"
    instructions: "You write clear, concise articles based on research."
    temperature: 0.9
```

Load and use them in Python:

```python
from exo.loader import load_agents, load_swarm

# Load individual agents
agents = load_agents("agents.yaml")
researcher = agents["researcher"]
writer = agents["writer"]

# Or load as a swarm
swarm = load_swarm("agents.yaml")
```

## YAML Format

### Agent Definition

Each agent is defined under the `agents` key with its name as the key:

```yaml
agents:
  agent_name:
    model: "provider:model_name"     # default: "openai:gpt-4o"
    instructions: "System prompt"     # or use system_prompt (alias)
    system_prompt: "System prompt"    # alias for instructions
    temperature: 0.7                  # default: 1.0
    max_tokens: 4096                  # default: None
    max_steps: 10                     # default: 10
    type: "builtin"                   # default: "builtin"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"openai:gpt-4o"` | Model string in `"provider:model_name"` format |
| `instructions` | `str` | `""` | System prompt for the agent |
| `system_prompt` | `str` | `""` | Alias for `instructions` |
| `temperature` | `float` | `1.0` | LLM sampling temperature |
| `max_tokens` | `int` | `None` | Maximum output tokens per LLM call |
| `max_steps` | `int` | `10` | Maximum LLM-tool round-trips |
| `type` | `str` | `"builtin"` | Agent class type (see [Custom Agent Classes](#custom-agent-classes)) |

### Variable Substitution

Variables are defined in the `vars` section and referenced with `${vars.KEY}`:

```yaml
vars:
  model: "openai:gpt-4o"
  tone: "professional and concise"

agents:
  assistant:
    model: "${vars.model}"
    instructions: "Be ${vars.tone} in all responses."
```

### Environment Variable Substitution

Environment variables are referenced with `${ENV_VAR}`:

```yaml
agents:
  assistant:
    model: "openai:${OPENAI_MODEL}"
    instructions: "API key is ${API_KEY}"
```

If an environment variable is not set, the reference is left as-is (e.g., `${UNDEFINED}` stays as the literal string `${UNDEFINED}`).

### Full-String vs Partial Substitution

- **Full-string match:** When the entire value is a single `${...}` reference, the original type is preserved (e.g., a number stays a number).
- **Partial match:** When `${...}` appears within a larger string, it is interpolated as a string.

```yaml
vars:
  temp: 0.7
  name: "helper"

agents:
  bot:
    temperature: "${vars.temp}"           # preserved as float 0.7
    instructions: "You are ${vars.name}"  # interpolated as string
```

## Swarm Definition

Add a `swarm` section to define orchestration:

```yaml
agents:
  researcher:
    instructions: "Research topics."
  writer:
    instructions: "Write articles."
  editor:
    instructions: "Edit for clarity."

swarm:
  type: "workflow"                    # "workflow", "handoff", or "team"
  flow: "researcher >> writer >> editor"
  order: ["researcher", "writer", "editor"]  # optional explicit order
  max_handoffs: 10                    # handoff mode only
  edges:                              # handoff edges
    - ["researcher", "writer"]
    - ["writer", "editor"]
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `str` | `"workflow"` | Swarm mode |
| `flow` | `str \| null` | `null` | Flow DSL string |
| `order` | `list[str] \| null` | `null` | Explicit agent execution order |
| `max_handoffs` | `int` | `10` | Maximum handoff transitions |
| `edges` | `list[[str, str]]` | `[]` | Handoff edges as `[source, target]` pairs |

When `order` is provided, agents are arranged in that order. Otherwise, agents run in declaration order. The `edges` field wires handoff targets between agents (for handoff mode).

## API Reference

### load_agents()

```python
def load_agents(path: str | Path) -> dict[str, Any]: ...
```

Load agents from a YAML file. Returns a dict mapping agent name to `Agent` instance.

```python
from exo.loader import load_agents

agents = load_agents("agents.yaml")
for name, agent in agents.items():
    print(f"{name}: {agent.model}")
```

Raises `LoaderError` if the file does not exist or has no `agents` section.

### load_swarm()

```python
def load_swarm(path: str | Path) -> Swarm: ...
```

Load a swarm (with agents) from a YAML file. If the YAML has no `swarm` section, creates a workflow-mode swarm with agents in declaration order.

```python
from exo.loader import load_swarm
from exo.runner import run

swarm = load_swarm("agents.yaml")
result = await run(swarm, "Write an article about AI")
```

### load_yaml()

```python
def load_yaml(path: str | Path) -> dict[str, Any]: ...
```

Low-level function that loads a YAML file, processes the `vars` section, performs all variable substitution, and returns the raw dict.

```python
from exo.loader import load_yaml

data = load_yaml("config.yaml")
print(data["agents"])
```

### register_agent_class()

```python
def register_agent_class(name: str, cls: type) -> None: ...
```

Register a custom agent class for YAML `type:` dispatch.

## Custom Agent Classes

You can register custom agent classes so the YAML loader creates instances of your class instead of the builtin `Agent`:

```python
from exo.agent import Agent
from exo.loader import register_agent_class, load_agents

class RAGAgent(Agent):
    """An agent with built-in retrieval-augmented generation."""

    def __init__(self, *, name: str, knowledge_base: str = "", **kwargs):
        super().__init__(name=name, **kwargs)
        self.knowledge_base = knowledge_base

# Register the class
register_agent_class("rag", RAGAgent)
```

Then in YAML:

```yaml
agents:
  research_bot:
    type: "rag"
    model: "openai:gpt-4o"
    knowledge_base: "/data/papers"
    instructions: "Answer questions using the knowledge base."
```

The loader passes all YAML fields (except `type`) as keyword arguments to your class constructor.

## Complete Example

```yaml
# production.yaml
vars:
  model: "anthropic:claude-sonnet-4-20250514"
  max_steps: 15

agents:
  triage:
    model: "${vars.model}"
    instructions: "Route customer queries to the appropriate specialist."
    max_steps: 3

  billing:
    model: "${vars.model}"
    instructions: "Handle billing and payment questions."
    max_steps: "${vars.max_steps}"

  technical:
    model: "${vars.model}"
    instructions: "Handle technical support questions."
    max_steps: "${vars.max_steps}"

swarm:
  type: "handoff"
  order: ["triage", "billing", "technical"]
  edges:
    - ["triage", "billing"]
    - ["triage", "technical"]
  max_handoffs: 5
```

```python
from exo.loader import load_swarm
from exo.runner import run

swarm = load_swarm("production.yaml")
result = await run(swarm, "I was charged twice for my subscription")
```

## Error Handling

```python
from exo.loader import LoaderError

try:
    agents = load_agents("missing.yaml")
except LoaderError as e:
    print(e)  # "YAML file not found: missing.yaml"

try:
    agents = load_agents("empty.yaml")
except LoaderError as e:
    print(e)  # "No 'agents' section in YAML"
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `load_agents()` | `exo.loader` | Load agents from YAML |
| `load_swarm()` | `exo.loader` | Load a swarm from YAML |
| `load_yaml()` | `exo.loader` | Load and substitute a YAML file |
| `register_agent_class()` | `exo.loader` | Register a custom agent class for YAML dispatch |
| `LoaderError` | `exo.loader` | YAML loading/validation error |
