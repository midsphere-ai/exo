"""Integration tests for structured output with multi-tool chain.

Tests that:
- output_type fields are correctly populated even after multiple tool calls
  are required to gather the necessary information.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel


class TravelReport(BaseModel):
    origin_city: str
    destination_city: str
    distance_km: float
    travel_tip: str


@pytest.mark.integration
@pytest.mark.timeout(30)
async def test_structured_output_after_two_tool_calls(vertex_model: str) -> None:
    """Agent produces TravelReport after calling get_distance then get_travel_tip.

    The tools are chained: the agent first calls get_distance(city1, city2) to
    obtain the distance, then calls get_travel_tip(destination) using the
    destination city.  After both tool calls the agent must output a valid
    TravelReport JSON object.  We assert all 4 fields are populated and that
    exactly 2 ToolCall objects are present.
    """
    from exo._internal.output_parser import (  # pyright: ignore[reportMissingImports]
        parse_structured_output,
    )
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]
    from exo.tool import tool  # pyright: ignore[reportMissingImports]

    @tool
    def get_distance(city1: str, city2: str) -> str:
        """Return the approximate driving distance in kilometres between two cities.

        Args:
            city1: The origin city name.
            city2: The destination city name.
        """
        distances: dict[tuple[str, str], int] = {
            ("London", "Paris"): 460,
            ("Paris", "London"): 460,
            ("Berlin", "Vienna"): 680,
            ("Vienna", "Berlin"): 680,
            ("Madrid", "Barcelona"): 620,
            ("Barcelona", "Madrid"): 620,
        }
        key = (city1.strip().title(), city2.strip().title())
        km = distances.get(key, 500)
        return f"The distance from {city1} to {city2} is approximately {km} km."

    @tool
    def get_travel_tip(destination: str) -> str:
        """Return a practical travel tip for a destination city.

        Args:
            destination: The destination city name.
        """
        tips: dict[str, str] = {
            "Paris": "Visit the Eiffel Tower early in the morning to avoid crowds.",
            "London": "Get an Oyster card for affordable public transport.",
            "Vienna": "Book the Vienna Philharmonic well in advance.",
            "Barcelona": "Watch out for pickpockets on Las Ramblas.",
            "Madrid": "Lunch is typically served from 2 PM onwards.",
        }
        tip = tips.get(
            destination.strip().title(),
            f"Enjoy your visit to {destination}!",
        )
        return tip

    provider = get_provider(vertex_model)
    agent = Agent(
        name="travel-report-agent",
        model=vertex_model,
        instructions=(
            "You are a travel assistant. "
            "When asked for a travel report: "
            "FIRST call get_distance with the origin and destination cities, "
            "THEN call get_travel_tip with the destination city. "
            "After BOTH tool calls, reply ONLY with a valid JSON object matching "
            "this schema: "
            '{"origin_city": "<origin>", "destination_city": "<destination>", '
            '"distance_km": <float>, "travel_tip": "<tip string>"}. '
            "No other text, only the JSON object."
        ),
        tools=[get_distance, get_travel_tip],
        max_steps=5,
    )

    result = await agent.run(
        "Generate a travel report for a trip from London to Paris. "
        "You MUST call get_distance first with city1=London and city2=Paris, "
        "then call get_travel_tip with destination=Paris. "
        "After both tool calls reply with ONLY the JSON travel report.",
        provider=provider,
    )

    # Assert exactly 2 tool calls
    assert len(result.tool_calls) == 2, (
        f"Expected exactly 2 tool calls, got {len(result.tool_calls)}: "
        f"{[tc.name for tc in result.tool_calls]}"
    )
    assert result.tool_calls[0].name == "get_distance", (
        f"Expected first tool call 'get_distance', got '{result.tool_calls[0].name}'"
    )
    assert result.tool_calls[1].name == "get_travel_tip", (
        f"Expected second tool call 'get_travel_tip', got '{result.tool_calls[1].name}'"
    )

    # Assert structured output is correctly populated
    travel_report = parse_structured_output(result.text, TravelReport)
    assert isinstance(travel_report, TravelReport), (
        f"Expected TravelReport instance, got {type(travel_report)}"
    )
    assert travel_report.origin_city, "Expected non-empty origin_city string"
    assert travel_report.destination_city, "Expected non-empty destination_city string"
    assert travel_report.distance_km > 0, (
        f"Expected distance_km > 0, got {travel_report.distance_km}"
    )
    assert travel_report.travel_tip, "Expected non-empty travel_tip string"
