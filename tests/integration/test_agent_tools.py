"""Integration tests for agent multi-tool selection.

Tests that:
- An agent with multiple tools calls the correct tool when given a
  constrained prompt.
- An agent can chain two tools sequentially, calling them in the correct
  order when each step's output is required as input for the next.
"""

from __future__ import annotations

import json

import pytest


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_agent_calls_correct_tool_of_three(vertex_model: str) -> None:
    """Agent selects get_weather when explicitly instructed to use only that tool.

    The agent has three tools (get_weather, get_time, get_currency) but the
    prompt constrains it to call ONLY get_weather with city=Dublin.  We
    assert exactly one ToolCall with name=='get_weather' and 'Dublin'
    somewhere in its arguments string.
    """
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]
    from exo.tool import tool  # pyright: ignore[reportMissingImports]

    @tool
    def get_weather(city: str) -> str:
        """Return a brief weather description for a city.

        Args:
            city: The name of the city.
        """
        return f"Cloudy with a chance of rain in {city}."

    @tool
    def get_time(timezone: str) -> str:
        """Return the current time for a timezone.

        Args:
            timezone: IANA timezone identifier (e.g. 'Europe/Dublin').
        """
        return f"The current time in {timezone} is 12:00 PM."

    @tool
    def get_currency(country: str) -> str:
        """Return the currency used in a country.

        Args:
            country: The country name.
        """
        return f"The currency of {country} is EUR."

    provider = get_provider(vertex_model)
    agent = Agent(
        name="tool-selector",
        model=vertex_model,
        instructions=(
            "You are a helpful assistant. "
            "You have tools: get_weather, get_time, get_currency. "
            "When instructed to use only one tool, use ONLY that tool."
        ),
        tools=[get_weather, get_time, get_currency],
        max_steps=3,
    )

    result = await agent.run(
        "You MUST call ONLY the get_weather tool with city=Dublin. "
        "Do not call any other tool. Do not answer from memory.",
        provider=provider,
    )

    weather_calls = [tc for tc in result.tool_calls if tc.name == "get_weather"]
    assert len(weather_calls) == 1, (
        f"Expected exactly one get_weather call, got {len(weather_calls)}. "
        f"All tool calls: {[tc.name for tc in result.tool_calls]}"
    )
    args = weather_calls[0].arguments
    args_dict = json.loads(args) if args.strip() else {}
    city_value = str(args_dict.get("city", "")).lower()
    assert "dublin" in city_value or "dublin" in args.lower(), (
        f"Expected 'Dublin' in get_weather arguments, got: {args!r}"
    )


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_agent_chains_two_tools(vertex_model: str) -> None:
    """Agent chains get_capital then get_population in the correct order.

    The tools are chained: get_population requires the city returned by
    get_capital, so they must be called sequentially.  We assert that
    result.tool_calls contains exactly two ToolCall objects and that
    get_capital appears before get_population.
    """
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]
    from exo.tool import tool  # pyright: ignore[reportMissingImports]

    @tool
    def get_capital(country: str) -> str:
        """Return the capital city of a country.

        Args:
            country: The country name.
        """
        capitals = {
            "France": "Paris",
            "Germany": "Berlin",
            "Japan": "Tokyo",
            "Australia": "Canberra",
        }
        return capitals.get(country, f"Unknown capital for {country}")

    @tool
    def get_population(city: str) -> str:
        """Return the approximate population of a city.

        Args:
            city: The city name returned by get_capital.
        """
        populations = {
            "Paris": "2.1 million",
            "Berlin": "3.7 million",
            "Tokyo": "13.9 million",
            "Canberra": "450 thousand",
        }
        return populations.get(city, f"Population of {city} is unknown")

    provider = get_provider(vertex_model)
    agent = Agent(
        name="tool-chainer",
        model=vertex_model,
        instructions=(
            "You are a geography assistant. "
            "To answer population questions: "
            "FIRST call get_capital to get the capital city, "
            "THEN call get_population with that city name. "
            "Always use both tools in sequence."
        ),
        tools=[get_capital, get_population],
        max_steps=5,
    )

    result = await agent.run(
        "What is the population of the capital of France? "
        "You MUST call get_capital first, then get_population with the result. "
        "Use both tools in that exact order.",
        provider=provider,
    )

    tool_names = [tc.name for tc in result.tool_calls]
    assert len(result.tool_calls) == 2, (
        f"Expected exactly 2 tool calls, got {len(result.tool_calls)}: {tool_names}"
    )
    assert result.tool_calls[0].name == "get_capital", (
        f"Expected first tool call to be 'get_capital', got '{result.tool_calls[0].name}'"
    )
    assert result.tool_calls[1].name == "get_population", (
        f"Expected second tool call to be 'get_population', got '{result.tool_calls[1].name}'"
    )
