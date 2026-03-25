"""Human-in-the-loop — agent asks for human confirmation.

Demonstrates ``HumanInputTool`` with the default ``ConsoleHandler``
so the agent can pause and request input from the user via stdin.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python examples/quickstart/hitl.py
"""

from exo import Agent, run
from exo.human import HumanInputTool

agent = Agent(
    name="assistant",
    model="openai:gpt-4o-mini",
    instructions=(
        "You help with travel planning. Before booking, always use the "
        "human_input tool to confirm the destination with the user."
    ),
    tools=[HumanInputTool()],
)

if __name__ == "__main__":
    result = run.sync(agent, "Plan a weekend trip for me.")
    print(result.output)
