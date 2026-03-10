"""SearXNG search tools matching Perplexica's tool interfaces.

Wraps a local SearXNG instance to provide web, academic, and social
search capabilities with multi-query parallel execution.  Set the
``SEARXNG_URL`` environment variable to point at your instance (defaults
to ``http://localhost:8888``).

Tools write their raw results to a module-level collector so the
researcher pipeline can retrieve them after the agent run completes.
"""

from __future__ import annotations

import asyncio
import json
import os
from urllib.parse import quote_plus

from orbiter import tool


# ---------------------------------------------------------------------------
# Shared result collector — tools append here, pipeline reads after run()
# ---------------------------------------------------------------------------

_collected_results: list[dict] = []


def get_collected_results() -> list[dict]:
    """Return all results collected during the current research run."""
    return list(_collected_results)


def clear_collected_results() -> None:
    """Clear the result collector before a new research run."""
    _collected_results.clear()


# ---------------------------------------------------------------------------
# SearXNG query helper
# ---------------------------------------------------------------------------


def _searxng_search(
    query: str,
    categories: str = "general",
    engines: str = "",
    num_results: int = 10,
    timeout: int = 15,
) -> list[dict]:
    """Execute a search against SearXNG, returning structured results."""
    import urllib.request

    base_url = os.environ.get("SEARXNG_URL", "http://localhost:8888")
    url = (
        f"{base_url}/search"
        f"?q={quote_plus(query)}"
        f"&format=json"
        f"&categories={quote_plus(categories)}"
    )
    if engines:
        url += f"&engines={quote_plus(engines)}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return [
            {
                "title": f"Example result for '{query}'",
                "url": f"https://example.com/{quote_plus(query)}",
                "content": f"Placeholder result for '{query}' (SearXNG unavailable)",
            }
        ]

    results = []
    for item in data.get("results", [])[:num_results]:
        results.append({
            "title": item.get("title", "Untitled"),
            "url": item.get("url", ""),
            "content": item.get("content", "") or item.get("title", ""),
        })
    return results


async def _multi_search(
    queries: list[str],
    categories: str = "general",
    engines: str = "",
    num_results: int = 10,
) -> str:
    """Run multiple queries in parallel, collect results, and return formatted text."""
    queries = queries[:3]

    tasks = [
        asyncio.to_thread(_searxng_search, q, categories, engines, num_results)
        for q in queries
    ]
    all_results = await asyncio.gather(*tasks)

    # Flatten and deduplicate by URL
    seen_urls: set[str] = set()
    unique_results: list[dict] = []
    for batch in all_results:
        for r in batch:
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                unique_results.append(r)

    # Side-effect: collect results for the pipeline
    _collected_results.extend(unique_results)

    if not unique_results:
        return "No results found."

    # Format as readable text for the LLM
    lines = []
    for i, r in enumerate(unique_results, 1):
        lines.append(f"[{i}] {r['title']} | {r['url']} | {r['content']}")
    return "\n".join(lines)


async def search_and_collect(
    queries: list[str], categories: str = "general", engines: str = ""
) -> list[dict]:
    """Search and return raw structured results (for pipeline use, not a tool)."""
    queries = queries[:3]
    tasks = [
        asyncio.to_thread(_searxng_search, q, categories, engines)
        for q in queries
    ]
    all_results = await asyncio.gather(*tasks)
    seen_urls: set[str] = set()
    unique: list[dict] = []
    for batch in all_results:
        for r in batch:
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                unique.append(r)
    return unique


@tool
async def web_search(queries: list[str]) -> str:
    """Perform web searches for up to 3 queries in parallel.

    Args:
        queries: An array of search queries to perform web searches for.
    """
    return await _multi_search(queries, categories="general")


@tool
async def academic_search(queries: list[str]) -> str:
    """Perform academic searches for scholarly articles and research. Up to 3 queries.

    Args:
        queries: List of academic search queries.
    """
    return await _multi_search(
        queries, categories="science", engines="arxiv,google scholar,pubmed"
    )


@tool
async def social_search(queries: list[str]) -> str:
    """Perform social media searches for discussions and trends. Up to 3 queries.

    Args:
        queries: List of social search queries.
    """
    return await _multi_search(
        queries, categories="social media", engines="reddit"
    )
