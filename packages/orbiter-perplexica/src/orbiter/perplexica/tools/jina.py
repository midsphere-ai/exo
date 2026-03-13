"""Jina Cloud API search and reader backend.

Uses the Jina Cloud API (``s.jina.ai`` for search, ``r.jina.ai`` for reading)
when ``JINA_API_KEY`` is set.  Provides the same result format as the SearXNG
and Serper backends so it can be swapped in transparently.
"""

from __future__ import annotations

import json
import os
import urllib.request


def jina_search(
    query: str,
    categories: str = "general",
    engines: str = "",
    num_results: int = 5,
    timeout: int = 15,
    api_key: str = "",
) -> list[dict]:
    """Execute a search against the Jina Cloud Search API.

    Args:
        query: The search query string.
        categories: SearXNG-compatible category hint (``general``, ``science``,
            ``social media``).  Mapped to query prefix modifiers.
        engines: SearXNG-compatible engine hint.  When it contains ``reddit``,
            the query is scoped to ``site:reddit.com``.
        num_results: Maximum number of results to return (Jina caps at 5).
        timeout: HTTP request timeout in seconds.
        api_key: Jina API key. Falls back to ``JINA_API_KEY`` env var.
    """
    api_key = api_key or os.environ.get("JINA_API_KEY", "")
    if not api_key:
        return []

    # Map categories to query modifiers
    if categories == "science" or "scholar" in engines:
        query = f"academic: {query}"
    elif ("reddit" in engines or categories == "social media") and "site:reddit.com" not in query:
        query = f"site:reddit.com {query}"

    num_results = min(num_results, 5)  # Jina caps at 5

    payload = json.dumps({"q": query, "num": num_results}).encode()
    req = urllib.request.Request(
        "https://s.jina.ai/",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return []

    items = data.get("data", [])[:num_results]
    results: list[dict] = []
    for item in items:
        content = item.get("content", "") or item.get("description", "") or item.get("title", "")
        results.append(
            {
                "title": item.get("title", "Untitled"),
                "url": item.get("url", ""),
                "content": content,
                "enriched": len(content) > 500,
            }
        )
    return results


def jina_reader_fetch(
    url: str,
    api_key: str,
    max_chars: int = 10_000,
    timeout: int = 15,
) -> str:
    """Fetch page content as markdown via the Jina Cloud Reader API.

    Args:
        url: The URL to read.
        api_key: Jina API key for authentication.
        max_chars: Maximum characters to return.
        timeout: HTTP request timeout in seconds.
    """
    req = urllib.request.Request(
        f"https://r.jina.ai/{url}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "X-Respond-With": "markdown",
            "X-With-Generated-Alt": "true",
            "X-Timeout": str(timeout),
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout + 5) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return ""

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n... [truncated]"
    return text
