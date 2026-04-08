"""Human-in-the-loop — agent asks for human confirmation.

Demonstrates two HITL patterns:

1. ``HumanInputTool`` — LLM-directed: the agent decides when to ask.
2. ``ToolContext.require_approval()`` — tool-directed: the tool gates
   itself based on argument sensitivity.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python examples/quickstart/hitl.py
"""

from exo import Agent, ConsoleHandler, ToolContext, run, tool
from exo.human import HumanInputTool

# --- Pattern 1: LLM-directed HITL ---

agent_llm_directed = Agent(
    name="travel_planner",
    model="openai:gpt-4o-mini",
    instructions=(
        "You help with travel planning. Before booking, always use the "
        "human_input tool to confirm the destination with the user."
    ),
    tools=[HumanInputTool()],
)

# --- Pattern 2: Tool-directed HITL ---


@tool
async def book_flight(destination: str, price: float, ctx: ToolContext) -> str:
    """Book a flight to a destination.

    Args:
        destination: The destination city.
        price: The ticket price in USD.
    """
    if price > 500:
        await ctx.require_approval(
            f"Flight to {destination} costs ${price:.0f}. Approve booking?"
        )
    return f"Booked flight to {destination} for ${price:.0f}"


agent_tool_directed = Agent(
    name="booking_agent",
    model="openai:gpt-4o-mini",
    instructions="You help book flights. Use the book_flight tool.",
    tools=[book_flight],
    human_input_handler=ConsoleHandler(),
)

if __name__ == "__main__":
    print("=== Pattern 1: LLM-directed HITL ===")
    result = run.sync(agent_llm_directed, "Plan a weekend trip for me.")
    print(result.output)

    print("\n=== Pattern 2: Tool-directed HITL ===")
    result = run.sync(agent_tool_directed, "Book me a flight to Tokyo for $800.")
    print(result.output)
