"""Search tools matching Perplexica's tool interfaces.

Uses Serper API (``SERPER_API_KEY``) by default for fast Google Search.
Falls back to a local SearXNG instance when ``SEARXNG_URL`` is set
without a Serper key.

Tools write their raw results to a module-level collector so the
researcher pipeline can retrieve them after the agent run completes.
"""

from __future__ import annotations

import asyncio
import json
import os
from urllib.parse import quote_plus

from orbiter import tool
from orbiter.observability.logging import get_logger  # pyright: ignore[reportMissingImports]

_log = get_logger(__name__)

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
# Search backend config — set from PerplexicaConfig before each pipeline run
# ---------------------------------------------------------------------------

_search_keys: dict[str, str] = {}


def configure_search_keys(
    serper_api_key: str = "",
    jina_api_key: str = "",
    searxng_url: str = "",
) -> None:
    """Set search backend API keys from config (falls back to env vars)."""
    _search_keys.clear()
    if serper_api_key:
        _search_keys["serper"] = serper_api_key
    if jina_api_key:
        _search_keys["jina"] = jina_api_key
    if searxng_url:
        _search_keys["searxng_url"] = searxng_url


# ---------------------------------------------------------------------------
# SearXNG query helper
# ---------------------------------------------------------------------------


_MAX_RETRIES = 3
_RETRY_DELAY = 2  # seconds between retries


def _search(
    query: str,
    categories: str = "general",
    engines: str = "",
    num_results: int = 10,
    timeout: int = 15,
) -> list[dict]:
    """Dispatch to Serper by default, fall back to SearXNG.

    Reads API keys from module-level ``_search_keys`` (set via
    ``configure_search_keys``), falling back to environment variables.
    """
    serper_key = _search_keys.get("serper") or os.environ.get("SERPER_API_KEY")
    jina_key = _search_keys.get("jina") or os.environ.get("JINA_API_KEY")
    searxng_url = _search_keys.get("searxng_url") or os.environ.get("SEARXNG_URL", "")

    backend = "serper" if serper_key else ("jina" if jina_key else "searxng")
    _log.debug("search backend=%s query=%r", backend, query)

    # Prefer Serper (faster, no retry/backoff needed)
    if serper_key:
        from .serper import serper_search

        return serper_search(
            query,
            categories,
            engines,
            num_results,
            timeout,
            api_key=serper_key,
        )

    # Jina Cloud Search
    if jina_key:
        from .jina import jina_search

        return jina_search(
            query,
            categories,
            engines,
            num_results,
            timeout,
            api_key=jina_key,
        )

    # Fall back to SearXNG
    return _searxng_search(query, categories, engines, num_results, timeout, searxng_url)


def _searxng_search(
    query: str,
    categories: str = "general",
    engines: str = "",
    num_results: int = 10,
    timeout: int = 15,
    searxng_url: str = "",
) -> list[dict]:
    """Execute a search against SearXNG with retry on empty results.

    Retries up to _MAX_RETRIES times when engines are suspended/rate-limited,
    giving them time to recover between attempts.
    """
    import time
    import urllib.request

    base_url = searxng_url or os.environ.get("SEARXNG_URL", "http://localhost:8888")
    url = f"{base_url}/search?q={quote_plus(query)}&format=json&categories={quote_plus(categories)}"
    if engines:
        url += f"&engines={quote_plus(engines)}"

    for attempt in range(_MAX_RETRIES):
        _log.debug("searxng attempt=%d/%d query=%r", attempt + 1, _MAX_RETRIES, query)
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            _log.warning("searxng attempt %d failed: %s", attempt + 1, exc)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY * (attempt + 1))
                continue
            _log.warning("searxng all retries exhausted for query=%r", query)
            return []

        results = []
        for item in data.get("results", [])[:num_results]:
            results.append(
                {
                    "title": item.get("title", "Untitled"),
                    "url": item.get("url", ""),
                    "content": item.get("content", "") or item.get("title", ""),
                }
            )

        if results or attempt == _MAX_RETRIES - 1:
            return results

        # Engines likely suspended — wait and retry
        time.sleep(_RETRY_DELAY * (attempt + 1))

    return []


async def _multi_search(
    queries: list[str],
    categories: str = "general",
    engines: str = "",
    num_results: int = 10,
) -> str:
    """Run multiple queries in parallel, collect results, and return formatted text."""
    queries = queries[:3]

    tasks = [asyncio.to_thread(_search, q, categories, engines, num_results) for q in queries]
    all_results = await asyncio.gather(*tasks)

    # Flatten and deduplicate by URL
    seen_urls: set[str] = set()
    unique_results: list[dict] = []
    for batch in all_results:
        for r in batch:
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                unique_results.append(r)

    _log.debug("multi_search queries=%s total_results=%d", queries, len(unique_results))

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
    queries = queries[:5]
    tasks = [asyncio.to_thread(_search, q, categories, engines) for q in queries]
    all_results = await asyncio.gather(*tasks)
    seen_urls: set[str] = set()
    unique: list[dict] = []
    for batch in all_results:
        for r in batch:
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                unique.append(r)
    _log.debug("search_and_collect queries=%d results=%d", len(queries), len(unique))
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
    return await _multi_search(queries, categories="science", engines="arxiv,google scholar,pubmed")


@tool
async def social_search(queries: list[str]) -> str:
    """Perform social media searches for discussions and trends. Up to 3 queries.

    Args:
        queries: List of social search queries.
    """
    return await _multi_search(queries, categories="social media", engines="reddit")
