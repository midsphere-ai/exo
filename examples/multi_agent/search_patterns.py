"""Workflow search patterns — fan-out query, aggregate, rank.

Demonstrates a common search pattern where multiple specialist
search agents query different sources in parallel, an aggregator
merges the results, and a ranker produces a final ranked answer.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python examples/multi_agent/search_patterns.py
"""

from exo import Agent, ParallelGroup, Swarm, run

# --- Fan-out: parallel search agents query different sources ---------------

web_search = Agent(
    name="web_search",
    model="openai:gpt-4o-mini",
    instructions=(
        "Search the web for the given query. "
        "Return 3 results as a numbered list with title and one-line summary."
    ),
)

docs_search = Agent(
    name="docs_search",
    model="openai:gpt-4o-mini",
    instructions=(
        "Search internal documentation for the given query. "
        "Return 3 relevant doc excerpts as a numbered list."
    ),
)

code_search = Agent(
    name="code_search",
    model="openai:gpt-4o-mini",
    instructions=(
        "Search codebases for the given query. "
        "Return 3 relevant code references with file paths and descriptions."
    ),
)

search_group = ParallelGroup(
    name="searchers",
    agents=[web_search, docs_search, code_search],
)

# --- Aggregate: merge results from all sources ----------------------------

aggregator = Agent(
    name="aggregator",
    model="openai:gpt-4o-mini",
    instructions=(
        "Merge search results from multiple sources. "
        "De-duplicate overlapping findings and produce a unified list."
    ),
)

# --- Rank: score and order the aggregated results -------------------------

ranker = Agent(
    name="ranker",
    model="openai:gpt-4o-mini",
    instructions=(
        "Rank the aggregated search results by relevance to the "
        "original query. Return the top 5 results with relevance scores."
    ),
)

pipeline = Swarm(
    agents=[search_group, aggregator, ranker],
    flow="searchers >> aggregator >> ranker",
    mode="workflow",
)

if __name__ == "__main__":
    result = run.sync(pipeline, "How to implement retry logic in Python async code")
    print(result.output)
