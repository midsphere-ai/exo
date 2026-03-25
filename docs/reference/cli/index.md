# exo_cli

Command-line agent runner for the Exo framework.

```python
from exo_cli import (
    # main — CLI entry point and config
    CLIError,
    app,
    find_config,
    load_config,
    resolve_config,
    # loader — agent discovery
    AgentLoadError,
    discover_agent_files,
    load_markdown_agent,
    load_python_agent,
    load_yaml_agents,
    scan_directory,
    validate_agent,
    # executor — local execution
    ExecutionResult,
    ExecutorError,
    LocalExecutor,
    # batch — batch processing
    BatchError,
    BatchItem,
    BatchResult,
    InputFormat,
    ItemResult,
    load_batch_items,
    results_to_csv,
    results_to_jsonl,
    # console — interactive REPL
    InteractiveConsole,
    format_agents_table,
    parse_command,
    # plugins — plugin system
    PluginError,
    PluginHook,
    PluginManager,
    PluginSpec,
)
```

**Install:** `pip install exo-cli`

---

## Submodules

| Module | Description |
|---|---|
| [`exo_cli.main`](main.md) | CLI entry point, config discovery, Typer app |
| [`exo_cli.loader`](loader.md) | Agent discovery and loading (Python, YAML, Markdown) |
| [`exo_cli.executor`](executor.md) | Local agent execution with Rich output |
| [`exo_cli.batch`](batch.md) | Batch input loading and concurrent execution |
| [`exo_cli.console`](console.md) | Interactive REPL console with slash commands |
| [`exo_cli.plugins`](plugins.md) | Plugin system with lifecycle hooks |

---

## Public API summary

| Export | Kind | Source |
|---|---|---|
| `CLIError` | Exception | `main` |
| `app` | `typer.Typer` | `main` |
| `find_config` | Function | `main` |
| `load_config` | Function | `main` |
| `resolve_config` | Function | `main` |
| `AgentLoadError` | Exception | `loader` |
| `discover_agent_files` | Function | `loader` |
| `load_markdown_agent` | Function | `loader` |
| `load_python_agent` | Function | `loader` |
| `load_yaml_agents` | Function | `loader` |
| `scan_directory` | Function | `loader` |
| `validate_agent` | Function | `loader` |
| `ExecutionResult` | Class | `executor` |
| `ExecutorError` | Exception | `executor` |
| `LocalExecutor` | Class | `executor` |
| `BatchError` | Exception | `batch` |
| `BatchItem` | Dataclass | `batch` |
| `BatchResult` | Dataclass | `batch` |
| `InputFormat` | StrEnum | `batch` |
| `ItemResult` | Dataclass | `batch` |
| `load_batch_items` | Function | `batch` |
| `results_to_csv` | Function | `batch` |
| `results_to_jsonl` | Function | `batch` |
| `InteractiveConsole` | Class | `console` |
| `format_agents_table` | Function | `console` |
| `parse_command` | Function | `console` |
| `PluginError` | Exception | `plugins` |
| `PluginHook` | StrEnum | `plugins` |
| `PluginManager` | Class | `plugins` |
| `PluginSpec` | Class | `plugins` |
