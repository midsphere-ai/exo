# exo_cli.loader

Agent discovery and loading for the Exo CLI. Scans directories for agent definitions in three formats: Python files, YAML files, and Markdown files.

```python
from exo_cli.loader import (
    AgentLoadError,
    discover_agent_files,
    load_markdown_agent,
    load_python_agent,
    load_yaml_agents,
    scan_directory,
    validate_agent,
)
```

---

## AgentLoadError

```python
class AgentLoadError(Exception)
```

Raised when agent loading or validation fails.

---

## load_python_agent

```python
def load_python_agent(path: Path) -> dict[str, Any]
```

Load agents from a Python file. The module must define a `create_agent()` callable that returns an `Agent` (or dict of name to Agent for multi-agent files).

| Name | Type | Default | Description |
|---|---|---|---|
| `path` | `Path` | *(required)* | Path to the Python file |

**Returns:** `dict[str, Any]` -- Mapping of agent name to agent instance.

**Raises:** `AgentLoadError` -- If the module cannot be loaded, has no `create_agent()` function, or the factory raises an error.

### Example

```python
from pathlib import Path
from exo_cli import load_python_agent

# my_agent.py must have: def create_agent() -> Agent
agents = load_python_agent(Path("my_agent.py"))
print(list(agents.keys()))  # ["my_agent"]
```

---

## load_yaml_agents

```python
def load_yaml_agents(path: Path) -> dict[str, Any]
```

Load agents from a YAML config file. Delegates to `exo.loader.load_agents`.

| Name | Type | Default | Description |
|---|---|---|---|
| `path` | `Path` | *(required)* | Path to the YAML file |

**Returns:** `dict[str, Any]` -- Mapping of agent name to agent instance.

**Raises:** `AgentLoadError` -- If YAML loading fails.

### Example

```python
from pathlib import Path
from exo_cli import load_yaml_agents

agents = load_yaml_agents(Path("agents.yaml"))
```

---

## load_markdown_agent

```python
def load_markdown_agent(path: Path) -> dict[str, Any]
```

Load an agent from a Markdown file with YAML front-matter. The markdown body (after front-matter) is used as `instructions` unless an explicit `instructions` field is provided in the front-matter.

| Name | Type | Default | Description |
|---|---|---|---|
| `path` | `Path` | *(required)* | Path to the Markdown file |

**Returns:** `dict[str, Any]` -- Mapping of agent name to agent instance (single entry).

**Raises:** `AgentLoadError` -- If front-matter values are invalid (e.g., non-numeric temperature).

### Front-matter fields

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | file stem | Agent name |
| `model` | `str` | *(none)* | Model string (e.g. `openai:gpt-4o`) |
| `instructions` | `str` | markdown body | Explicit instructions (overrides body) |
| `temperature` | `float` | *(none)* | Temperature value |
| `max_tokens` | `int` | *(none)* | Maximum tokens |
| `max_steps` | `int` | *(none)* | Maximum steps |

### Example

Given a file `helper.md`:

```markdown
---
name: helper
model: openai:gpt-4o
temperature: 0.7
---

You are a helpful assistant that answers questions concisely.
```

```python
from pathlib import Path
from exo_cli import load_markdown_agent

agents = load_markdown_agent(Path("helper.md"))
agent = agents["helper"]
print(agent.name)          # "helper"
print(agent.instructions)  # "You are a helpful assistant..."
```

---

## discover_agent_files

```python
def discover_agent_files(directory: str | Path) -> list[Path]
```

Scan *directory* for agent definition files. Returns files with extensions `.py`, `.yaml`, and `.md` sorted by name for deterministic ordering. Only immediate children are scanned (no recursive walk).

| Name | Type | Default | Description |
|---|---|---|---|
| `directory` | `str \| Path` | *(required)* | Directory to scan |

**Returns:** `list[Path]` -- Sorted list of matching file paths.

**Raises:** `AgentLoadError` -- If *directory* is not a valid directory.

---

## validate_agent

```python
def validate_agent(name: str, agent: Any) -> None
```

Validate that *agent* looks like a usable agent instance. Checks for required attributes: `name` and `run`.

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | Agent name (used in error messages) |
| `agent` | `Any` | *(required)* | Agent instance to validate |

**Raises:** `AgentLoadError` -- If `name` or `run` attributes are missing.

---

## scan_directory

```python
def scan_directory(
    directory: str | Path,
    *,
    validate: bool = True,
) -> dict[str, Any]
```

Discover and load all agents from *directory*. Scans for `.py`, `.yaml`, and `.md` files, loads each via the appropriate loader, validates agent instances, and returns a merged dict of name to agent.

| Name | Type | Default | Description |
|---|---|---|---|
| `directory` | `str \| Path` | *(required)* | Path to scan for agent files |
| `validate` | `bool` | `True` | Whether to validate loaded agents |

**Returns:** `dict[str, Any]` -- Mapping of agent name to agent instance.

**Raises:** `AgentLoadError` -- On discovery, loading, validation errors, or duplicate agent names.

### Example

```python
from exo_cli import scan_directory

# Load all agents from a directory
agents = scan_directory("/path/to/agents")
for name, agent in agents.items():
    print(f"Loaded agent: {name}")

# Load without validation
agents = scan_directory("/path/to/agents", validate=False)
```
