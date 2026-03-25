"""Config-driven agents — load agents and swarms from YAML.

Demonstrates ``load_agents()`` and ``load_swarm()`` which let you
define agents in a YAML file with variable substitution, then run
them programmatically in Python.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python examples/quickstart/config_driven.py
"""

from pathlib import Path

from exo import run
from exo.loader import load_agents, load_swarm

YAML_PATH = Path(__file__).parent / "agents.yaml"

# --- Load individual agents from YAML ------------------------------------

agents = load_agents(YAML_PATH)
print(f"Loaded agents: {list(agents.keys())}")

# --- Load a full swarm (workflow pipeline) --------------------------------

swarm = load_swarm(YAML_PATH)
print(f"Swarm mode: {swarm.mode}, agents: {list(swarm.agents.keys())}")

if __name__ == "__main__":
    result = run.sync(swarm, "Explain why the ocean is salty.")
    print(result.output)
