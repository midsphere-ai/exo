"""DeepAgent — Multi-round deep research agent built on Orbiter.

A faithful port of SkyworkAI's DeepResearchAgent to the Orbiter framework,
using Swarm(mode="team") for orchestration.
"""

from .agents import build_deep_agent
from .config import DeepAgentConfig

__all__ = ["build_deep_agent", "DeepAgentConfig"]
