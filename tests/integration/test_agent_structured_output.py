"""Integration tests for structured output validation.

Tests that:
- output_type=PydanticModel produces a fully-populated, correctly-typed
  response for a simple factual prompt.
- output_type works correctly even when a tool call is required before
  the structured output is produced.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel


class CountryInfo(BaseModel):
    name: str
    capital: str
    population_millions: float
    continent: str


class WeatherReport(BaseModel):
    city: str
    temperature_celsius: float
    condition: str


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_structured_output_all_fields_populated(vertex_model: str) -> None:
    """Agent returns a fully-populated CountryInfo for France.

    The agent is instructed to reply with a JSON object matching CountryInfo.
    We assert that all four fields are present with correct types and values.
    """
    from exo._internal.output_parser import (  # pyright: ignore[reportMissingImports]
        parse_structured_output,
    )
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]

    provider = get_provider(vertex_model)
    agent = Agent(
        name="structured-output-agent",
        model=vertex_model,
        instructions=(
            "You are a knowledgeable assistant. "
            "When asked for country information, reply ONLY with a valid JSON object "
            "matching this schema: "
            '{"name": "<country name>", "capital": "<capital city>", '
            '"population_millions": <float>, "continent": "<continent name>"}. '
            "No other text, only the JSON object."
        ),
        max_steps=2,
    )

    result = await agent.run(
        "Return structured information about France. "
        "Reply with ONLY the JSON object, no other text.",
        provider=provider,
    )

    country_info = parse_structured_output(result.text, CountryInfo)
    assert isinstance(country_info, CountryInfo), (
        f"Expected CountryInfo instance, got {type(country_info)}"
    )
    assert country_info.capital.lower() == "paris", (
        f"Expected capital 'paris', got '{country_info.capital}'"
    )
    assert country_info.population_millions > 0, (
        f"Expected population_millions > 0, got {country_info.population_millions}"
    )
    assert country_info.continent, (
        "Expected non-empty continent string"
    )


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_structured_output_with_tool_call(vertex_model: str) -> None:
    """Agent produces a WeatherReport after calling get_raw_weather.

    The agent must call the get_raw_weather tool first, then use its
    result to populate all three fields of the WeatherReport JSON output.
    """
    from exo._internal.output_parser import (  # pyright: ignore[reportMissingImports]
        parse_structured_output,
    )
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]
    from exo.tool import tool  # pyright: ignore[reportMissingImports]

    @tool
    def get_raw_weather(city: str) -> str:
        """Return raw weather data for a city.

        Args:
            city: The name of the city to get weather for.
        """
        return f"Weather in {city}: 18.5 degrees Celsius, partly cloudy skies."

    provider = get_provider(vertex_model)
    agent = Agent(
        name="weather-report-agent",
        model=vertex_model,
        instructions=(
            "You are a weather assistant. "
            "You MUST call the get_raw_weather tool to get weather data, "
            "then reply ONLY with a valid JSON object matching this schema: "
            '{"city": "<city name>", "temperature_celsius": <float>, '
            '"condition": "<weather condition string>"}. '
            "No other text, only the JSON object."
        ),
        tools=[get_raw_weather],
        max_steps=3,
    )

    result = await agent.run(
        "You MUST call the get_raw_weather tool with city=London, "
        "then produce a JSON weather report. "
        "Reply with ONLY the JSON object.",
        provider=provider,
    )

    # Assert the tool was called
    weather_calls = [tc for tc in result.tool_calls if tc.name == "get_raw_weather"]
    assert len(weather_calls) >= 1, (
        f"Expected at least one get_raw_weather call, got {len(weather_calls)}. "
        f"All tool calls: {[tc.name for tc in result.tool_calls]}"
    )

    # Assert the structured output is correctly populated
    weather_report = parse_structured_output(result.text, WeatherReport)
    assert isinstance(weather_report, WeatherReport), (
        f"Expected WeatherReport instance, got {type(weather_report)}"
    )
    assert weather_report.city, "Expected non-empty city string"
    assert isinstance(weather_report.temperature_celsius, float), (
        f"Expected temperature_celsius to be float, got {type(weather_report.temperature_celsius)}"
    )
    assert weather_report.condition, "Expected non-empty condition string"
