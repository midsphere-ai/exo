"""Search API tool — query web search engines.

Wraps a configurable search backend (defaults to a stub).  Set the
``SEARCH_API_KEY`` and ``SEARCH_ENGINE_ID`` environment variables to
enable the Google Custom Search JSON API, or replace the implementation
with any search provider.

Usage:
    from examples.common.tools.search import web_search
"""

from __future__ import annotations

import json
import os
from urllib.parse import quote_plus

from exo import tool


@tool
async def web_search(query: str, num_results: int = 5) -> str:
    """Search the web and return a list of results.

    Args:
        query: Search query string.
        num_results: Maximum number of results to return (1-10).
    """
    num_results = max(1, min(num_results, 10))

    api_key = os.environ.get("SEARCH_API_KEY", "")
    engine_id = os.environ.get("SEARCH_ENGINE_ID", "")

    if api_key and engine_id:
        return await _google_search(query, num_results, api_key, engine_id)
    return _stub_search(query, num_results)


async def _google_search(
    query: str,
    num: int,
    api_key: str,
    engine_id: str,
) -> str:
    """Call Google Custom Search JSON API."""
    import urllib.request

    url = (
        "https://www.googleapis.com/customsearch/v1"
        f"?key={api_key}&cx={engine_id}"
        f"&q={quote_plus(query)}&num={num}"
    )
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        return f"Search error: {exc}"

    items = data.get("items", [])
    if not items:
        return "No results found."

    results: list[str] = []
    for item in items:
        title = item.get("title", "")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        results.append(f"- **{title}**\n  {link}\n  {snippet}")
    return "\n\n".join(results)


def _stub_search(query: str, num: int) -> str:
    """Return placeholder results when no API key is configured."""
    lines = [
        f"[stub search] Results for '{query}' (set SEARCH_API_KEY for real results):",
    ]
    for i in range(1, num + 1):
        lines.append(f"  {i}. Example result {i} for '{query}'")
    return "\n".join(lines)
