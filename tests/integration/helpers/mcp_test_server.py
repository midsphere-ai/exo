"""Standalone MCP test server for integration tests.

Exposes tools:
- get_capital(country: str) -> str
- get_population(city: str) -> str
- get_large_dataset(topic: str) -> str
- long_running_task(steps: int) -> str  (emits progress notifications)

Run standalone: python mcp_test_server.py
"""

from __future__ import annotations

import asyncio

from mcp.server.fastmcp import Context as FastMCPContext  # pyright: ignore[reportMissingImports]

from exo.mcp import mcp_server  # pyright: ignore[reportMissingImports]

# Known capitals and populations for testing
_CAPITALS: dict[str, str] = {
    "japan": "Tokyo",
    "france": "Paris",
    "germany": "Berlin",
    "australia": "Canberra",
    "brazil": "Brasilia",
    "canada": "Ottawa",
    "india": "New Delhi",
    "china": "Beijing",
    "usa": "Washington D.C.",
    "united states": "Washington D.C.",
    "uk": "London",
    "united kingdom": "London",
    "italy": "Rome",
    "spain": "Madrid",
    "mexico": "Mexico City",
    "argentina": "Buenos Aires",
}

_POPULATIONS: dict[str, str] = {
    "tokyo": "approximately 14 million (city proper)",
    "paris": "approximately 2 million (city proper)",
    "berlin": "approximately 3.6 million",
    "canberra": "approximately 450,000",
    "brasilia": "approximately 3 million",
    "london": "approximately 9 million",
    "rome": "approximately 2.8 million",
    "madrid": "approximately 3.3 million",
    "dublin": "approximately 1.2 million",
    "sydney": "approximately 5.3 million",
    "new york": "approximately 8 million (city proper)",
    "osaka": "approximately 2.7 million",
}


@mcp_server(name="test-server")
class TestServer:
    """Test MCP server for integration tests."""

    def get_capital(self, country: str) -> str:
        """Return the capital city of the given country.

        Args:
            country: The name of the country.

        Returns:
            The capital city name or an informative message.
        """
        key = country.lower().strip()
        return _CAPITALS.get(key, f"Capital of {country} is not in the test database.")

    def get_population(self, city: str) -> str:
        """Return the approximate population of the given city.

        Args:
            city: The name of the city.

        Returns:
            The approximate population or an informative message.
        """
        key = city.lower().strip()
        return _POPULATIONS.get(key, f"Population of {city} is not in the test database.")


    def get_large_dataset(self, topic: str) -> str:
        """Return a large dataset about the given topic (>10 KB for workspace offload testing).

        Args:
            topic: The topic to generate data about.

        Returns:
            A large dataset string containing data about the topic.
            Always includes EXO_DATASET_KEYWORD_2024 for test verification.
        """
        keyword = "EXO_DATASET_KEYWORD_2024"
        header = (
            f"=== LARGE DATASET: {topic.upper()} ===\n"
            f"IDENTIFICATION_KEYWORD: {keyword}\n"
            f"This dataset contains comprehensive information about {topic}.\n"
            "It is intentionally large to test workspace offloading in the Exo framework.\n\n"
        )
        # Pad to exactly 15 KB (15360 bytes) with data entries
        target_bytes = 15 * 1024
        content = header
        entry_num = 0
        while len(content.encode("utf-8")) < target_bytes:
            content += (
                f"DATA_ENTRY[{entry_num}]: Detailed information about {topic} "
                f"— entry {entry_num} contains extensive measurements, "
                "observations, and related scientific data. " + "x" * 40 + "\n"
            )
            entry_num += 1
        # Trim to at most target_bytes, preserving UTF-8 boundaries
        encoded = content.encode("utf-8")
        if len(encoded) > target_bytes:
            content = encoded[:target_bytes].decode("utf-8", errors="ignore")
        return content


    async def long_running_task(self, steps: int, ctx: FastMCPContext) -> str:
        """Perform a long-running task that emits progress notifications.

        Args:
            steps: Number of progress steps to emit before completing.
            ctx: FastMCP context for progress reporting (injected by FastMCP).

        Returns:
            Completion message indicating how many steps were completed.
        """
        for i in range(steps):
            await ctx.report_progress(i + 1, steps, f"Step {i + 1} of {steps}")
            await asyncio.sleep(0.05)
        return f"Task completed successfully after {steps} steps."


if __name__ == "__main__":
    server = TestServer()
    server.run(transport="stdio")  # type: ignore[attr-defined]
