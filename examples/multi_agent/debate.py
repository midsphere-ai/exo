"""Collaborative debate — two agents argue opposing sides.

Demonstrates a multi-agent debate pattern where two agents present
opposing viewpoints and a moderator synthesises the conclusion.
Uses ``Swarm(mode="workflow")`` with a ``ParallelGroup`` for the
debaters followed by a serial moderator stage.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python examples/multi_agent/debate.py
"""

from exo import Agent, ParallelGroup, Swarm, run

# --- Debaters run in parallel, each given the same topic ----------------

affirmative = Agent(
    name="affirmative",
    model="openai:gpt-4o-mini",
    instructions=(
        "You argue IN FAVOUR of the proposition. "
        "Present 2-3 concise supporting arguments with evidence."
    ),
)

negative = Agent(
    name="negative",
    model="openai:gpt-4o-mini",
    instructions=(
        "You argue AGAINST the proposition. Present 2-3 concise counter-arguments with evidence."
    ),
)

debaters = ParallelGroup(name="debaters", agents=[affirmative, negative])

# --- Moderator synthesises both sides -----------------------------------

moderator = Agent(
    name="moderator",
    model="openai:gpt-4o-mini",
    instructions=(
        "You are a neutral debate moderator. "
        "Summarise both sides fairly, identify the strongest "
        "argument from each, and give a balanced conclusion."
    ),
)

debate = Swarm(
    agents=[debaters, moderator],
    flow="debaters >> moderator",
    mode="workflow",
)

if __name__ == "__main__":
    result = run.sync(debate, "Remote work is better than office work.")
    print(result.output)
