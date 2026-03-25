"""Workflow swarm — sequential agent pipeline.

Demonstrates ``Swarm(mode="workflow")`` where agents execute in
a defined order using the flow DSL (``"a >> b >> c"``).  Each
agent's output becomes the next agent's input.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python examples/quickstart/workflow.py
"""

from exo import Agent, Swarm, run

# Three-stage pipeline: research -> draft -> review
researcher = Agent(
    name="researcher",
    model="openai:gpt-4o-mini",
    instructions="You research a topic and return 3 key bullet points.",
)

drafter = Agent(
    name="drafter",
    model="openai:gpt-4o-mini",
    instructions="You take bullet-point research and write a short paragraph.",
)

reviewer = Agent(
    name="reviewer",
    model="openai:gpt-4o-mini",
    instructions="You review text for clarity and return the polished version.",
)

pipeline = Swarm(
    agents=[researcher, drafter, reviewer],
    flow="researcher >> drafter >> reviewer",
    mode="workflow",
)

if __name__ == "__main__":
    result = run.sync(pipeline, "Explain why the sky is blue.")
    print(result.output)
