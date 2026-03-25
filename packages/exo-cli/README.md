# exo-cli

Command-line interface for the [Exo](../../README.md) multi-agent framework.

## Installation

```bash
pip install exo-cli
```

Requires Python 3.11+, `exo-core`, `exo-models`, `typer>=0.12`, and `rich>=13.0`.

## Usage

After installation, the `exo` command is available:

```bash
# Run an agent from a YAML config file
exo run agents.yaml

# Interactive console mode
exo console

# Run in batch mode
exo batch input.jsonl --output results.jsonl

# Discover agents in the current directory
exo list
```

## What's Included

- **Agent runner** -- load and run agents from YAML configuration or Python modules.
- **Interactive console** -- rich terminal UI for conversing with agents.
- **Batch processing** -- process input files with agents at scale.
- **Agent discovery** -- scan directories for agent definitions.
- **Plugin system** -- extend the CLI with custom commands.

## Documentation

- [CLI Guide](../../docs/guides/cli.md)
- [API Reference](../../docs/reference/cli/)
