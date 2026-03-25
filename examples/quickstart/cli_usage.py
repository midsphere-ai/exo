"""CLI usage reference — Exo command-line agent runner.

This file documents the ``exo`` CLI commands and options.
Run any command with ``--help`` for full details.

Install the CLI (included in the workspace):
    uv sync

Basic commands::

    # Run an agent with an explicit config file
    exo run --config agents.yaml "Summarize quantum computing"

    # Auto-discover config (.exo.yaml or exo.config.yaml in cwd)
    exo run "What is 2 + 2?"

    # Override the model for a single run
    exo run -m openai:gpt-4o "Explain decorators in Python"

    # Enable streaming output
    exo run --stream --config agents.yaml "Write a haiku"

    # Verbose mode — shows config keys, model, and streaming status
    exo --verbose run --config agents.yaml "Hello"

You can also invoke the CLI as a Python module::

    uv run python -m exo_cli.main run --config agents.yaml "Hello"

Programmatic equivalent (using the loader directly):

>>> from exo import run
>>> from exo.loader import load_swarm
>>> swarm = load_swarm("agents.yaml")
>>> result = run.sync(swarm, "Hello")
>>> print(result.output)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

YAML_PATH = Path(__file__).parent / "agents.yaml"


def show_help() -> None:
    """Print the CLI help text."""
    subprocess.run([sys.executable, "-m", "exo_cli.main", "--help"], check=False)


if __name__ == "__main__":
    show_help()
