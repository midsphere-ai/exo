"""Serper API search backend.

Uses the Serper API (serper.dev) for Google Search when ``SERPER_API_KEY``
is set.  Provides the same result format as the SearXNG backend so it can
be swapped in transparently.
"""

from __future__ import annotations

import json
import os
import urllib.request

_SERPER_BASE = "https://google.serper.dev"


def serper_search(
    query: str,
    categories: str = "general",
    engines: str = "",
    num_results: int = 10,
    timeout: int = 15,
) -> list[dict]:
    """Execute a search against the Serper API.

    Args:
        query: The search query string.
        categories: SearXNG-compatible category hint (``general``, ``science``,
            ``social media``).  Mapped to the appropriate Serper endpoint.
        engines: SearXNG-compatible engine hint.  When it contains ``reddit``,
            the query is scoped to ``site:reddit.com``.
        num_results: Maximum number of results to return.
        timeout: HTTP request timeout in seconds.
    """
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        return []

    # Determine endpoint and adjust query for social searches
    if categories == "science" or "scholar" in engines:
        endpoint = "/scholar"
    elif "reddit" in engines or categories == "social media":
        endpoint = "/search"
        if "site:reddit.com" not in query:
            query = f"{query} site:reddit.com"
    else:
        endpoint = "/search"

    url = f"{_SERPER_BASE}{endpoint}"
    payload = json.dumps({"q": query, "num": num_results}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return []

    # Scholar returns results under "organic"; news under "news"; search under "organic"
    items = data.get("organic", data.get("news", []))[:num_results]
    results: list[dict] = []
    for item in items:
        results.append(
            {
                "title": item.get("title", "Untitled"),
                "url": item.get("link", ""),
                "content": item.get("snippet", "") or item.get("title", ""),
            }
        )
    return results
