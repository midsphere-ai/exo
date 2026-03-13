"""Web page content extraction tools.

Fetch and extract readable text content from web pages, stripping
navigation, scripts, and other non-content elements.

Usage:
    from examples.advanced.perplexica.tools.web_fetcher import fetch_page_content
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import re
from urllib.parse import quote, urlparse

from orbiter import tool

_MAX_CHARS = 10_000


def _fetch_via_jina(
    url: str,
    jina_url: str,
    max_chars: int = _MAX_CHARS,
    api_key: str = "",
) -> str:
    """Fetch page content as clean markdown via Jina Reader.

    When ``api_key`` is set, uses the Jina Cloud Reader API (``r.jina.ai``)
    for faster fetches with image captioning. Otherwise falls back to a
    self-hosted Jina Reader instance at ``jina_url``.
    """
    if api_key:
        from .jina import jina_reader_fetch

        return jina_reader_fetch(url, api_key, max_chars)

    import urllib.request

    reader_url = f"{jina_url}/{url}"
    req = urllib.request.Request(reader_url, headers={"X-Respond-With": "markdown"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n... [truncated]"
    return text


def _fetch_page_fallback(url: str) -> str:
    """Fetch a web page via direct HTTP and extract text with regex stripping."""
    import urllib.request

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Error: unsupported scheme '{parsed.scheme}'. Use http or https."

    safe_url = quote(url, safe=":/?#[]@!$&'()*+,;=-._~%")
    req = urllib.request.Request(safe_url, headers={"User-Agent": "OrbiterBot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return f"Error fetching {url}: {exc}"

    text = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.S)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S)
    text = re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=re.S)
    text = re.sub(r"<footer[^>]*>.*?</footer>", "", text, flags=re.S)
    text = re.sub(r"<header[^>]*>.*?</header>", "", text, flags=re.S)

    article_match = re.search(r"<article[^>]*>(.*?)</article>", text, flags=re.S)
    main_match = re.search(r"<main[^>]*>(.*?)</main>", text, flags=re.S)
    if article_match:
        text = article_match.group(1)
    elif main_match:
        text = main_match.group(1)

    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + "... [truncated]"
    return text


def _fetch_page(url: str) -> str:
    """Fetch a page, trying Jina Cloud / self-hosted Reader then falling back."""
    jina_api_key = os.environ.get("JINA_API_KEY", "")
    if jina_api_key:
        result = _fetch_via_jina(url, "", api_key=jina_api_key)
        if result:
            return result
    jina_url = os.environ.get("JINA_READER_URL", "")
    if jina_url:
        result = _fetch_via_jina(url, jina_url)
        if result:
            return result
    return _fetch_page_fallback(url)


@tool
async def fetch_page_content(url: str) -> str:
    """Fetch a web page and return its extracted text content.

    Strips scripts, styles, navigation, headers, and footers.  Prefers
    content from <article> or <main> tags when available.

    Args:
        url: Fully-qualified URL to fetch (e.g. https://example.com).
    """
    return await asyncio.to_thread(_fetch_page, url)


@tool
async def fetch_multiple_pages(urls_json: str) -> str:
    """Fetch multiple web pages concurrently and return their combined content.

    Args:
        urls_json: JSON array of URL strings to fetch (max 3).
    """
    try:
        urls = json.loads(urls_json)
    except (json.JSONDecodeError, TypeError):
        return "Error: urls_json must be a valid JSON array of URL strings."

    if not isinstance(urls, list):
        return "Error: urls_json must be a JSON array."

    # Limit to 3 concurrent fetches.
    urls = urls[:3]

    tasks = [asyncio.to_thread(_fetch_page, url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    sections: list[str] = []
    for url, result in zip(urls, results):
        sections.append(f"=== {url} ===")
        if isinstance(result, Exception):
            sections.append(f"Error fetching {url}: {result}")
        else:
            sections.append(result)
        sections.append("")

    return "\n".join(sections).strip()


@tool
async def scrape_url(urls: list[str]) -> str:
    """Scrape and extract content from up to 3 URLs.

    Args:
        urls: A list of URLs to scrape content from.
    """
    urls = urls[:3]
    tasks = [asyncio.to_thread(_fetch_page, url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    sections = []
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            sections.append(f"Error fetching {url}: {result}")
        else:
            sections.append(f"=== {url} ===\n{result}")
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Pipeline helper — enrich SearchResults with full page content
# ---------------------------------------------------------------------------


async def enrich_results(
    results: list,
    jina_url: str,
    max_results: int = 5,
    jina_api_key: str = "",
) -> list:
    """Scrape full page content for the top results via Jina Reader.

    Returns a new list with enriched content for the first ``max_results``
    items and original snippets for the rest. Skips results already marked
    as ``enriched`` (e.g., from Jina Search). Falls back gracefully when
    Jina Reader is unavailable or a page fails to fetch.
    """
    from ..types import SearchResult

    if (not jina_url and not jina_api_key) or not results:
        return results

    to_enrich = results[:max_results]

    async def _timed_fetch(url: str) -> str:
        return await asyncio.wait_for(
            asyncio.to_thread(_fetch_via_jina, url, jina_url, api_key=jina_api_key),
            timeout=8.0,
        )

    enriched: list[SearchResult] = []
    tasks_with_indices: list[tuple[int, asyncio.Task]] = []  # type: ignore[type-arg]
    for i, r in enumerate(to_enrich):
        if r.enriched:
            enriched.append(r)
        else:
            enriched.append(r)  # placeholder
            tasks_with_indices.append((i, asyncio.ensure_future(_timed_fetch(r.url))))

    if tasks_with_indices:
        task_results = await asyncio.gather(
            *[t for _, t in tasks_with_indices],
            return_exceptions=True,
        )
        for (idx, _), content in zip(tasks_with_indices, task_results, strict=True):
            r = to_enrich[idx]
            if isinstance(content, str) and content:
                enriched[idx] = SearchResult(
                    title=r.title,
                    url=r.url,
                    content=content,
                    enriched=True,
                )

    enriched.extend(results[max_results:])
    return enriched
