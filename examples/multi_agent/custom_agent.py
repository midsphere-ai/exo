"""Custom agent — extend Agent with domain-specific defaults.

Shows how to create a reusable custom agent class that bundles
tools, instructions, and hooks so callers only provide a name.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python examples/multi_agent/custom_agent.py
"""

from exo import Agent, run, tool


@tool
async def search_docs(query: str) -> str:
    """Search internal documentation for relevant results."""
    return f"Found 3 results for '{query}': [doc1, doc2, doc3]"


@tool
async def create_ticket(title: str, body: str) -> str:
    """Create a support ticket in the tracking system."""
    return f"Created ticket: {title}"


def support_agent(name: str = "support") -> Agent:
    """Factory that builds a pre-configured support agent."""
    return Agent(
        name=name,
        model="openai:gpt-4o-mini",
        instructions=(
            "You are a support agent. Search docs first, then "
            "create a ticket if the issue cannot be resolved."
        ),
        tools=[search_docs, create_ticket],
        max_steps=5,
    )


if __name__ == "__main__":
    agent = support_agent()
    result = run.sync(agent, "My dashboard is showing stale data.")
    print(result.output)
