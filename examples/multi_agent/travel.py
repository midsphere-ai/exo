"""Collaborative travel planner — specialised agents plan a trip.

Demonstrates handoff-based collaboration where a lead planner
delegates to specialised agents (flights, hotels, activities) and
synthesises a final itinerary.  Uses ``Swarm(mode="team")`` so
the lead can delegate via auto-generated ``delegate_to_*`` tools.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python examples/multi_agent/travel.py
"""

from exo import Agent, Swarm, run

# --- Specialist agents ---------------------------------------------------

flights = Agent(
    name="flights",
    model="openai:gpt-4o-mini",
    instructions=(
        "You are a flight specialist. Given a destination and dates, "
        "suggest 2 flight options with estimated prices."
    ),
)

hotels = Agent(
    name="hotels",
    model="openai:gpt-4o-mini",
    instructions=(
        "You are a hotel specialist. Given a destination and dates, "
        "suggest 2 hotel options with estimated nightly rates."
    ),
)

activities = Agent(
    name="activities",
    model="openai:gpt-4o-mini",
    instructions=(
        "You are an activities specialist. Given a destination, "
        "suggest 3 must-do activities with brief descriptions."
    ),
)

# --- Lead planner delegates and combines --------------------------------

planner = Agent(
    name="planner",
    model="openai:gpt-4o-mini",
    instructions=(
        "You are a travel planner. Delegate to 'flights', 'hotels', "
        "and 'activities' specialists, then combine their suggestions "
        "into a cohesive travel itinerary."
    ),
)

team = Swarm(
    agents=[planner, flights, hotels, activities],
    mode="team",
)

if __name__ == "__main__":
    result = run.sync(team, "Plan a 5-day trip to Tokyo in April.")
    print(result.output)
