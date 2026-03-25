# Your First Agent

This tutorial builds a multi-agent system step by step. By the end you will have:

- Multiple tools with the `@tool` decorator
- An agent that uses those tools
- A second agent with handoff
- A Swarm workflow
- Structured output with a Pydantic model

## Step 1: Define Tools

Start by creating tools the agent can use. Each tool is a Python function decorated with `@tool`. The decorator reads the function name, type hints, and Google-style docstring to generate a JSON schema automatically.

```python
from exo import tool


@tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city to check weather for.
    """
    # Simulated weather data
    weather_data = {
        "tokyo": "Sunny, 22 C",
        "london": "Cloudy, 15 C",
        "new york": "Rainy, 18 C",
    }
    return weather_data.get(city.lower(), f"No data for {city}")


@tool
async def get_forecast(city: str, days: int = 3) -> str:
    """Get a weather forecast for upcoming days.

    Args:
        city: Name of the city.
        days: Number of days to forecast (default: 3).
    """
    return f"Forecast for {city}: sunny for the next {days} days."


@tool
def get_time(timezone: str) -> str:
    """Get the current time in a timezone.

    Args:
        timezone: IANA timezone string (e.g., 'Asia/Tokyo').
    """
    # Sync functions work too -- they run via asyncio.to_thread()
    from datetime import datetime, timezone as tz
    return f"Current time: {datetime.now(tz.utc).isoformat()}"
```

You can verify the generated schema:

```python
print(get_weather.name)          # "get_weather"
print(get_weather.description)   # "Get the current weather for a city."
print(get_weather.parameters)
# {
#     "type": "object",
#     "properties": {
#         "city": {"type": "string", "description": "Name of the city to check weather for."}
#     },
#     "required": ["city"]
# }
```

## Step 2: Create an Agent

Create an agent that uses these tools:

```python
from exo import Agent

weather_agent = Agent(
    name="weather-assistant",
    model="openai:gpt-4o-mini",
    instructions=(
        "You are a weather assistant. Use the available tools to answer "
        "questions about weather, forecasts, and time zones. Be concise "
        "and helpful."
    ),
    tools=[get_weather, get_forecast, get_time],
    max_steps=5,
)
```

Inspect what the agent looks like:

```python
print(weather_agent.describe())
# {
#     'name': 'weather-assistant',
#     'model': 'openai:gpt-4o-mini',
#     'tools': ['get_weather', 'get_forecast', 'get_time'],
#     'handoffs': [],
#     'max_steps': 5,
#     'output_type': None,
# }
```

## Step 3: Run the Agent

Use `run.sync()` to execute the agent and inspect the result:

```python
from exo import run

result = run.sync(weather_agent, "What's the weather in Tokyo and what time is it there?")

print("Output:", result.output)
print(f"Steps: {result.steps}")
print(f"Tokens used: {result.usage.total_tokens}")
print(f"Messages in history: {len(result.messages)}")
```

The `RunResult` contains everything about the execution:

```python
# Inspect the full message history
for msg in result.messages:
    role = msg.role
    if role == "assistant" and hasattr(msg, "tool_calls") and msg.tool_calls:
        tools = [tc.name for tc in msg.tool_calls]
        print(f"  [{role}] Called tools: {tools}")
    elif role == "tool":
        print(f"  [{role}] {msg.tool_name}: {msg.content[:60]}...")
    else:
        content = getattr(msg, "content", "")
        print(f"  [{role}] {content[:80]}")
```

## Step 4: Add a Second Agent with Handoff

Create a travel planning agent that can hand off to the weather agent:

```python
@tool
async def book_flight(origin: str, destination: str, date: str) -> str:
    """Book a flight between two cities.

    Args:
        origin: Departure city.
        destination: Arrival city.
        date: Travel date in YYYY-MM-DD format.
    """
    return f"Flight booked: {origin} -> {destination} on {date}. Confirmation: FL-12345"


@tool
async def find_hotels(city: str, checkin: str, nights: int = 2) -> str:
    """Find available hotels in a city.

    Args:
        city: City to search for hotels.
        checkin: Check-in date in YYYY-MM-DD format.
        nights: Number of nights (default: 2).
    """
    return f"Found 3 hotels in {city} from {checkin} for {nights} nights."


travel_agent = Agent(
    name="travel-planner",
    model="openai:gpt-4o-mini",
    instructions=(
        "You are a travel planning assistant. Help users plan trips by "
        "booking flights and finding hotels. If the user asks about weather, "
        "hand off to the weather-assistant."
    ),
    tools=[book_flight, find_hotels],
    handoffs=[weather_agent],  # can delegate to weather_agent
)
```

Now run the travel agent directly. If it decides to delegate weather questions, it can hand off:

```python
result = run.sync(travel_agent, "Plan a trip to Tokyo next week. What's the weather like?")
print(result.output)
```

## Step 5: Wrap in a Swarm (Workflow Mode)

For more structured multi-agent execution, use a `Swarm`. In workflow mode, agents execute sequentially and each agent's output becomes the next agent's input:

```python
from exo import Swarm

# Research agent gathers information
researcher = Agent(
    name="researcher",
    model="openai:gpt-4o-mini",
    instructions=(
        "You are a research assistant. Given a travel destination, "
        "gather key facts: best time to visit, top attractions, "
        "and local customs. Be concise."
    ),
)

# Writer agent creates the final output
writer = Agent(
    name="writer",
    model="openai:gpt-4o-mini",
    instructions=(
        "You are a travel writer. Take the research notes you receive "
        "and write a polished, engaging 2-paragraph travel summary."
    ),
)

# Create a workflow pipeline: researcher -> writer
travel_swarm = Swarm(
    agents=[researcher, writer],
    flow="researcher >> writer",
    mode="workflow",
)

result = run.sync(travel_swarm, "Tell me about visiting Kyoto, Japan")
print(result.output)
```

### Team Mode

In team mode, the first agent is the lead and can delegate to workers via auto-generated tools:

```python
lead = Agent(
    name="lead",
    model="openai:gpt-4o-mini",
    instructions=(
        "You are a project lead. Delegate research to the researcher "
        "and writing to the writer. Synthesize their outputs into a "
        "final response."
    ),
)

team_swarm = Swarm(
    agents=[lead, researcher, writer],
    mode="team",
)

# The lead agent gets auto-generated tools:
#   delegate_to_researcher(task: str)
#   delegate_to_writer(task: str)
result = run.sync(team_swarm, "Create a travel guide for Kyoto, Japan")
print(result.output)
```

## Step 6: Structured Output

Use `output_type` to make the agent return validated, structured data:

```python
from pydantic import BaseModel


class WeatherReport(BaseModel):
    city: str
    temperature_celsius: float
    condition: str
    summary: str


structured_agent = Agent(
    name="structured-weather",
    model="openai:gpt-4o-mini",
    instructions="Return weather information in the requested structured format.",
    tools=[get_weather],
    output_type=WeatherReport,
)

result = run.sync(structured_agent, "What's the weather in Tokyo?")
print(result.output)
# The output will be a JSON string matching the WeatherReport schema
```

When `output_type` is set, the agent's instructions are augmented to guide the LLM toward producing output that matches the Pydantic model schema.

## Complete Example

Here is the full example in a single file:

```python
"""Multi-agent travel assistant with structured output."""

from pydantic import BaseModel

from exo import Agent, Swarm, run, tool


# -- Tools --

@tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city.
    """
    weather = {"tokyo": "Sunny, 22 C", "london": "Cloudy, 15 C"}
    return weather.get(city.lower(), f"No data for {city}")


@tool
async def book_flight(origin: str, destination: str, date: str) -> str:
    """Book a flight.

    Args:
        origin: Departure city.
        destination: Arrival city.
        date: Travel date (YYYY-MM-DD).
    """
    return f"Booked: {origin} -> {destination} on {date}"


@tool
async def find_hotels(city: str, nights: int = 2) -> str:
    """Find hotels in a city.

    Args:
        city: City to search.
        nights: Number of nights.
    """
    return f"Found 3 hotels in {city} for {nights} nights"


# -- Agents --

weather_agent = Agent(
    name="weather",
    model="openai:gpt-4o-mini",
    instructions="You are a weather assistant.",
    tools=[get_weather],
)

travel_agent = Agent(
    name="travel",
    model="openai:gpt-4o-mini",
    instructions="You are a travel planner.",
    tools=[book_flight, find_hotels],
    handoffs=[weather_agent],
)


# -- Swarm --

swarm = Swarm(
    agents=[travel_agent, weather_agent],
    flow="travel >> weather",
    mode="workflow",
)


# -- Structured Output --

class TripPlan(BaseModel):
    destination: str
    weather: str
    hotel: str
    flight: str


# -- Run --

if __name__ == "__main__":
    # Simple single-agent run
    result = run.sync(weather_agent, "Weather in Tokyo?")
    print(f"Weather: {result.output}\n")

    # Multi-agent workflow
    result = run.sync(swarm, "Plan a trip to Tokyo")
    print(f"Trip plan: {result.output}\n")

    # Inspect usage
    print(f"Total tokens: {result.usage.total_tokens}")
    print(f"Steps: {result.steps}")
```

## What You Learned

In this tutorial you:

1. **Defined tools** with the `@tool` decorator, using type hints and docstrings for automatic schema generation
2. **Created an Agent** with instructions, tools, and configuration
3. **Ran the agent** with `run.sync()` and inspected the `RunResult`
4. **Added handoffs** so one agent can delegate to another
5. **Built a Swarm** in workflow and team modes for structured multi-agent execution
6. **Used structured output** with `output_type` for validated responses

## Next Steps

- **[Core Concepts](concepts.md)** -- Deeper reference on all building blocks
- **[Guides](../guides/context/index.md)** -- Advanced topics like context engine, memory, and tracing
- **[API Reference](../reference/index.md)** -- Complete API documentation
