"""Master-worker — a lead agent delegates to specialist workers.

Demonstrates ``Swarm(mode="team")`` where a master agent delegates
sub-tasks to workers via auto-generated ``delegate_to_*`` tools,
then combines the results.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python examples/multi_agent/master_worker.py
"""

from exo import Agent, Swarm, run

worker_summary = Agent(
    name="summariser",
    model="openai:gpt-4o-mini",
    instructions="Summarise the given text in 2-3 sentences.",
)

worker_translate = Agent(
    name="translator",
    model="openai:gpt-4o-mini",
    instructions="Translate the given text into French.",
)

master = Agent(
    name="master",
    model="openai:gpt-4o-mini",
    instructions=(
        "You coordinate workers. First delegate to 'summariser' to get "
        "a summary, then delegate to 'translator' to translate it. "
        "Return the final translated summary."
    ),
)

team = Swarm(
    agents=[master, worker_summary, worker_translate],
    mode="team",
)

if __name__ == "__main__":
    result = run.sync(
        team,
        "The Exo framework provides a modern approach to building "
        "multi-agent systems with tools, handoffs, and swarm orchestration.",
    )
    print(result.output)
