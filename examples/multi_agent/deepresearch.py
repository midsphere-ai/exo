"""Deep research — multi-stage research with synthesis.

Demonstrates a research pattern where multiple specialist
researchers explore a topic in parallel, then a synthesiser
merges their findings into a comprehensive answer.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python examples/multi_agent/deepresearch.py
"""

from exo import Agent, ParallelGroup, Swarm, run

web_researcher = Agent(
    name="web_researcher",
    model="openai:gpt-4o-mini",
    instructions="Research the topic using web sources. Return 3 key findings.",
)

academic_researcher = Agent(
    name="academic_researcher",
    model="openai:gpt-4o-mini",
    instructions="Research the topic from an academic perspective. Return 3 findings with citations.",
)

researchers = ParallelGroup(name="researchers", agents=[web_researcher, academic_researcher])

synthesiser = Agent(
    name="synthesiser",
    model="openai:gpt-4o-mini",
    instructions=(
        "Combine the research findings into a concise, well-structured "
        "summary. Highlight agreements and contradictions between sources."
    ),
)

pipeline = Swarm(
    agents=[researchers, synthesiser],
    flow="researchers >> synthesiser",
    mode="workflow",
)

if __name__ == "__main__":
    result = run.sync(pipeline, "What are the health effects of intermittent fasting?")
    print(result.output)
