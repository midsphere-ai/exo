# exo_cli.main

CLI entry point for the `exo` command. Supports agent/swarm execution from YAML config files with environment variable override, model selection, verbosity control, and streaming output.

```python
from exo_cli.main import CLIError, app, find_config, load_config, resolve_config
```

Config file search order (first found wins):

1. `--config` / `-c` flag (explicit path)
2. `.exo.yaml` in current directory
3. `exo.config.yaml` in current directory

---

## CLIError

```python
class CLIError(Exception)
```

Raised for CLI-level errors (config not found, parse failures).

---

## find_config

```python
def find_config(directory: str | Path | None = None) -> Path | None
```

Search *directory* (default: cwd) for a config file.

Returns the first matching path or `None` if no config exists. Searches for `.exo.yaml` and `exo.config.yaml` in the given directory.

| Name | Type | Default | Description |
|---|---|---|---|
| `directory` | `str \| Path \| None` | `None` | Directory to search. Defaults to current working directory |

**Returns:** `Path` to the found config file, or `None`.

### Example

```python
from exo_cli import find_config

# Auto-detect in cwd
config_path = find_config()

# Search specific directory
config_path = find_config("/path/to/project")
if config_path:
    print(f"Found config at: {config_path}")
```

---

## load_config

```python
def load_config(path: str | Path) -> dict[str, Any]
```

Load and validate a YAML config file. Delegates to `exo.loader.load_yaml` for variable substitution, then validates the top-level structure.

| Name | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| Path` | *(required)* | Path to the YAML config file |

**Returns:** `dict[str, Any]` -- Parsed configuration dictionary.

**Raises:** `CLIError` -- If the file does not exist or is not valid YAML dict.

### Example

```python
from exo_cli import load_config

config = load_config("agents.yaml")
print(config.keys())
```

---

## resolve_config

```python
def resolve_config(config_path: str | None) -> dict[str, Any] | None
```

Resolve config from explicit path or auto-discovery. If `config_path` is provided, loads it directly. Otherwise, searches the current directory for a config file.

| Name | Type | Default | Description |
|---|---|---|---|
| `config_path` | `str \| None` | *(required)* | Explicit path to config file, or `None` for auto-discovery |

**Returns:** Parsed config dict, or `None` if no config is available.

### Example

```python
from exo_cli import resolve_config

# Explicit path
config = resolve_config("my_agents.yaml")

# Auto-discover
config = resolve_config(None)
if config is None:
    print("No config found")
```

---

## app

```python
app = typer.Typer(name="exo", help="Exo — multi-agent framework CLI.", no_args_is_help=True)
```

The Typer CLI application instance. Register additional commands or groups on this object.

### Commands

#### run

```
exo run [OPTIONS] INPUT_TEXT
```

Run an agent or swarm with the given input.

| Option | Short | Type | Default | Description |
|---|---|---|---|---|
| `--config` | `-c` | `str` | `None` | Path to YAML config file |
| `--model` | `-m` | `str` | `None` | Model string (e.g. `openai:gpt-4o`) |
| `--stream` | `-s` | `bool` | `False` | Enable streaming output |

| Argument | Type | Description |
|---|---|---|
| `INPUT_TEXT` | `str` | Input text to send to the agent |

### Global options

| Option | Short | Type | Default | Description |
|---|---|---|---|---|
| `--verbose` | `-v` | `bool` | `False` | Enable verbose output |

### Example

```bash
# Run with explicit config
exo run --config agents.yaml "What is 2+2?"

# Run with model override
exo run -m openai:gpt-4o "Hello"

# Verbose streaming
exo --verbose run --stream "Explain Python decorators"
```
