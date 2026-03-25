"""Parallel task execution — run multiple agents concurrently.

Demonstrates ``ParallelGroup`` to run agents in parallel and
collect their outputs.  All agents receive the same input and
their results are joined with a separator.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python examples/quickstart/parallel_task.py
"""

from exo import Agent, ParallelGroup, Swarm, run

# Two researchers working in parallel
history = Agent(
    name="history",
    model="openai:gpt-4o-mini",
    instructions="Provide 2 historical facts about the topic.",
)

science = Agent(
    name="science",
    model="openai:gpt-4o-mini",
    instructions="Provide 2 scientific facts about the topic.",
)

research = ParallelGroup(name="research", agents=[history, science])

# Summariser takes the combined output
summariser = Agent(
    name="summariser",
    model="openai:gpt-4o-mini",
    instructions="Combine the research into a concise summary.",
)

pipeline = Swarm(
    agents=[research, summariser],
    flow="research >> summariser",
    mode="workflow",
)

if __name__ == "__main__":
    result = run.sync(pipeline, "The Moon")
    print(result.output)
