"""Define agents with tools — minimal Exo example.

Shows how to create an Agent with a custom tool and run it
using the synchronous ``run.sync()`` convenience wrapper.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python examples/quickstart/define_agent.py
"""

from exo import Agent, run, tool


@tool
async def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"The weather in {city} is sunny, 22°C."


agent = Agent(
    name="weather-bot",
    model="openai:gpt-4o-mini",
    instructions="You are a helpful weather assistant. Use the get_weather tool.",
    tools=[get_weather],
)


if __name__ == "__main__":
    result = run.sync(agent, "What's the weather like in Tokyo?")
    print(result.output)
