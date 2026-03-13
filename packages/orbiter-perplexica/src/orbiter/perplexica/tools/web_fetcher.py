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


def _fetch_via_jina(url: str, jina_url: str, max_chars: int = _MAX_CHARS) -> str:
    """Fetch page content as clean markdown via a Jina Reader instance."""
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
    """Fetch a page, trying Jina Reader first then falling back to direct fetch."""
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
) -> list:
    """Scrape full page content for the top results via Jina Reader.

    Returns a new list with enriched content for the first ``max_results``
    items and original snippets for the rest. Falls back gracefully when
    Jina Reader is unavailable or a page fails to fetch.
    """
    from ..types import SearchResult

    if not jina_url or not results:
        return results

    to_enrich = results[:max_results]

    async def _timed_fetch(url: str) -> str:
        return await asyncio.wait_for(
            asyncio.to_thread(_fetch_via_jina, url, jina_url), timeout=8.0,
        )

    tasks = [_timed_fetch(r.url) for r in to_enrich]
    contents = await asyncio.gather(*tasks, return_exceptions=True)

    enriched: list[SearchResult] = []
    for r, content in zip(to_enrich, contents, strict=True):
        if isinstance(content, Exception) or not content:
            enriched.append(r)
        else:
            enriched.append(SearchResult(title=r.title, url=r.url, content=content))

    enriched.extend(results[max_results:])
    return enriched
