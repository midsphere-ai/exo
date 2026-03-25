"""Multi-root agent — team mode with a lead and workers.

Demonstrates ``Swarm(mode="team")`` where a lead agent delegates
to multiple worker agents via auto-generated ``delegate_to_*``
tools. The lead synthesises the final answer.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python examples/quickstart/multi_root_agent.py
"""

from exo import Agent, Swarm, run

lead = Agent(
    name="lead",
    model="openai:gpt-4o-mini",
    instructions=(
        "You coordinate research. Delegate history questions to the "
        "'historian' worker and science questions to the 'scientist' "
        "worker, then combine their answers."
    ),
)

historian = Agent(
    name="historian",
    model="openai:gpt-4o-mini",
    instructions="Answer with 2 historical facts about the topic.",
)

scientist = Agent(
    name="scientist",
    model="openai:gpt-4o-mini",
    instructions="Answer with 2 scientific facts about the topic.",
)

team = Swarm(
    agents=[lead, historian, scientist],
    mode="team",
)

if __name__ == "__main__":
    result = run.sync(team, "Tell me about volcanoes.")
    print(result.output)
