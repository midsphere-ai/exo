# CLI

The `exo-cli` package provides a command-line interface for running agents, interactive REPL sessions, batch processing, and plugin-based extensibility. It supports agent discovery from Python, YAML, and Markdown files, with automatic config file detection and Rich-formatted output.

## Basic Usage

```bash
# Run an agent with input
exo run "What is Python?"

# Run with explicit config
exo run --config agents.yaml "Explain decorators"

# Run with model override
exo run -m openai:gpt-4o "Hello"

# Enable streaming
exo run --stream "Tell me a story"

# Verbose output
exo --verbose run "Debug this code"
```

## Config File Discovery

The CLI searches for configuration in this order:

1. Explicit `--config` / `-c` flag
2. `.exo.yaml` in the current directory
3. `exo.config.yaml` in the current directory

```python
from exo_cli import find_config, load_config, resolve_config

# Auto-discover config
path = find_config()  # searches cwd for .exo.yaml or exo.config.yaml

# Load explicit config
config = load_config("agents.yaml")

# Resolve (explicit or auto-discovered)
config = resolve_config(config_path=None)  # auto-discovers
config = resolve_config(config_path="my-config.yaml")  # explicit
```

## Agent Discovery and Loading

The CLI discovers agents from three file formats in a directory:

### Python Files (.py)

Must export a `create_agent()` function:

```python
# agents/research.py
from exo.agent import Agent

def create_agent():
    return Agent(
        name="researcher",
        model="openai:gpt-4o",
        instructions="You are a research assistant.",
    )
```

For multi-agent files, return a dict:

```python
def create_agent():
    return {
        "researcher": Agent(name="researcher", ...),
        "writer": Agent(name="writer", ...),
    }
```

### YAML Files (.yaml)

Agent definitions in YAML format (loaded via `exo.loader.load_agents`):

```yaml
# agents/team.yaml
agents:
  assistant:
    model: openai:gpt-4o
    instructions: You are a helpful assistant.
  coder:
    model: openai:gpt-4o
    instructions: You write Python code.
```

### Markdown Files (.md)

Agent definitions with YAML front-matter:

```markdown
---
name: writer
model: openai:gpt-4o
temperature: 0.7
max_tokens: 2000
---

You are a creative writer. You write engaging stories,
articles, and blog posts. Use vivid language and
compelling narratives.
```

Front-matter fields: `name`, `model`, `instructions`, `temperature`, `max_tokens`, `max_steps`.

The markdown body (after front-matter) is used as `instructions` unless an explicit `instructions` field is provided in the front-matter.

### Directory Scanning

```python
from exo_cli import discover_agent_files, scan_directory, validate_agent

# Find agent files in a directory
files = discover_agent_files("/path/to/agents")
# Returns .py, .yaml, .md files sorted by name

# Scan and load all agents
agents = scan_directory("/path/to/agents", validate=True)
# Returns dict: name -> agent instance

# Validate an agent instance
validate_agent("my_agent", agent)  # checks for 'name' and 'run' attributes
```

## Interactive Console

The `InteractiveConsole` provides a REPL for chatting with agents:

```python
from exo_cli import InteractiveConsole

console = InteractiveConsole(
    agents={"assistant": my_agent, "coder": coding_agent},
    run_fn=my_run_function,
    stream_fn=my_stream_function,  # optional
    streaming=True,
)

await console.start()
```

### Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help message |
| `/exit`, `/quit` | Exit the console |
| `/agents` | List available agents |
| `/switch <name>` | Switch to a different agent |
| `/clear` | Clear the screen |
| `/info` | Show current agent info |

### Command Parsing

```python
from exo_cli import parse_command

cmd, arg = parse_command("/switch researcher")
# cmd = "switch", arg = "researcher"

cmd, arg = parse_command("Hello, how are you?")
# cmd = "", arg = "Hello, how are you?"
```

### Agent Table Display

```python
from exo_cli import format_agents_table

table = format_agents_table(agents)
# Rich Table with Name, Model, Description columns
```

## Local Executor

The `LocalExecutor` wraps `exo.runner.run` with timeout, retry, and Rich output:

```python
from exo_cli import LocalExecutor

executor = LocalExecutor(
    agent=my_agent,
    provider=None,       # auto-resolved
    timeout=60.0,        # seconds (0 = no timeout)
    max_retries=3,
    verbose=True,
)

# Non-streaming execution
result = await executor.execute("What is Python?")
print(result.output)     # agent output text
print(result.steps)      # number of LLM steps
print(result.elapsed)    # wall-clock seconds
print(result.usage)      # {"prompt_tokens": ..., "output_tokens": ..., "total_tokens": ...}
print(result.summary())  # "3 step(s), 2.1s, 1500 tokens"

# Streaming execution
async for chunk in executor.stream("Tell me a story"):
    print(chunk, end="")

# Display helpers
executor.print_result(result)  # Rich panel with output
executor.print_error(error)    # Red error message
```

## Batch Processing

Process multiple inputs from JSON, CSV, or JSONL files:

```python
from exo_cli import load_batch_items, BatchItem, InputFormat
from exo_cli.batch import batch_execute, results_to_jsonl, results_to_csv

# Load inputs from file
items = load_batch_items(
    "inputs.jsonl",
    input_key="input",    # column/field with agent input
    id_key="id",          # column/field with item ID
)

# Execute batch with concurrency control
result = await batch_execute(
    agent=my_agent,
    items=items,
    concurrency=4,     # max parallel executions
    timeout=30.0,      # per-item timeout
)

print(result.summary())  # "100 items: 95 succeeded, 5 failed"

# Export results
jsonl = results_to_jsonl(result)
csv = results_to_csv(result)
```

### Input Formats

| Format | Extension | Structure |
|--------|-----------|-----------|
| JSON | `.json` | Array of objects |
| JSONL | `.jsonl` | One JSON object per line |
| CSV | `.csv` | Header row + data rows |

Each row/object must have an `input` field (configurable via `input_key`).

### Batch Items

```python
from exo_cli import BatchItem

item = BatchItem(
    id="item-1",
    input="What is Python?",
    metadata={"category": "programming"},
)
```

## Plugin System

Extend the CLI with plugins that hook into lifecycle events:

### Creating a Plugin

```python
from exo_cli import PluginSpec, PluginHook

async def on_startup(**kwargs):
    print("CLI starting up!")

async def on_pre_run(**kwargs):
    print("About to run agent...")

async def on_post_run(**kwargs):
    print("Agent finished!")

async def on_shutdown(**kwargs):
    print("CLI shutting down!")

plugin = PluginSpec(
    name="my-plugin",
    version="1.0.0",
    description="Example CLI plugin",
    hooks={
        PluginHook.STARTUP: on_startup,
        PluginHook.PRE_RUN: on_pre_run,
        PluginHook.POST_RUN: on_post_run,
        PluginHook.SHUTDOWN: on_shutdown,
    },
)
```

### Plugin Hooks

| Hook | When It Fires |
|------|--------------|
| `STARTUP` | CLI is initializing |
| `PRE_RUN` | Before agent execution |
| `POST_RUN` | After agent execution |
| `SHUTDOWN` | CLI is shutting down |

### Plugin Manager

```python
from exo_cli import PluginManager

manager = PluginManager()

# Load from Python entry points
loaded = manager.load_entrypoints()
print(f"Loaded {loaded} plugins from entry points")

# Load from a directory
loaded = manager.load_directory("/path/to/plugins")

# Register manually
manager.register(my_plugin_spec)

# Run lifecycle hooks
await manager.startup()
await manager.run_hook(PluginHook.PRE_RUN, agent=agent, input=text)
await manager.shutdown()
```

### Entry Point Discovery

Packages can register plugins via entry points:

```toml
# pyproject.toml
[project.entry-points."exo.plugins"]
my_plugin = "my_package:plugin"
```

### Directory Discovery

Place `.py` files in a plugins directory, each exporting a `plugin` attribute:

```python
# plugins/my_plugin.py
from exo_cli import PluginSpec, PluginHook

plugin = PluginSpec(
    name="my-plugin",
    hooks={PluginHook.STARTUP: on_startup},
)
```

## Advanced Patterns

### Custom Config Loading

```python
from exo_cli import find_config, load_config

config = load_config("custom-agents.yaml")
agents = config.get("agents", {})
```

### Programmatic Batch Processing

```python
from exo_cli import load_batch_items, InputFormat

# Force a specific format
items = load_batch_items("data.txt", fmt=InputFormat.JSONL)

# Custom input/id keys
items = load_batch_items("data.csv", input_key="prompt", id_key="case_id")
```

### Plugin-Augmented Execution

```python
manager = PluginManager()
manager.load_entrypoints()
manager.load_directory("./plugins")

await manager.startup()

try:
    await manager.run_hook(PluginHook.PRE_RUN, agent=agent, input=text)
    result = await executor.execute(text)
    await manager.run_hook(PluginHook.POST_RUN, result=result)
finally:
    await manager.shutdown()
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `app` | `exo_cli` | Typer CLI application |
| `find_config` | `exo_cli` | Auto-discover config file in cwd |
| `load_config` | `exo_cli` | Load and validate a YAML config file |
| `resolve_config` | `exo_cli` | Resolve config from explicit path or auto-discovery |
| `InteractiveConsole` | `exo_cli` | REPL console with slash commands |
| `parse_command` | `exo_cli` | Parse slash commands from input text |
| `format_agents_table` | `exo_cli` | Rich table of available agents |
| `LocalExecutor` | `exo_cli` | Agent executor with timeout and Rich output |
| `discover_agent_files` | `exo_cli` | Find agent files in a directory |
| `load_python_agent` | `exo_cli` | Load agents from a `.py` file |
| `load_yaml_agents` | `exo_cli` | Load agents from a `.yaml` file |
| `load_markdown_agent` | `exo_cli` | Load an agent from a `.md` file |
| `scan_directory` | `exo_cli` | Discover and load all agents from a directory |
| `validate_agent` | `exo_cli` | Validate an agent instance |
| `load_batch_items` | `exo_cli` | Load batch items from JSON/CSV/JSONL |
| `BatchItem` | `exo_cli` | Batch input item: `id`, `input`, `metadata` |
| `InputFormat` | `exo_cli` | Enum: `JSON`, `CSV`, `JSONL` |
| `results_to_jsonl` | `exo_cli` | Serialize batch results to JSONL |
| `results_to_csv` | `exo_cli` | Serialize batch results to CSV |
| `PluginManager` | `exo_cli` | Plugin discovery and lifecycle management |
| `PluginSpec` | `exo_cli` | Plugin definition with hooks |
| `PluginHook` | `exo_cli` | Enum: `STARTUP`, `SHUTDOWN`, `PRE_RUN`, `POST_RUN` |
