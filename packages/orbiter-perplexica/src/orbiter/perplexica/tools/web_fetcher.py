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
import re
from urllib.parse import quote, urlparse

from orbiter import tool

_MAX_CHARS = 6000


def _fetch_page(url: str) -> str:
    """Fetch a web page and extract its readable text content.

    Args:
        url: Fully-qualified URL to fetch.
    """
    import urllib.request

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Error: unsupported scheme '{parsed.scheme}'. Use http or https."

    safe_url = quote(url, safe=":/?#[]@!$&'()*+,;=-._~%")
    req = urllib.request.Request(
        safe_url,
        headers={"User-Agent": "OrbiterBot/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return f"Error fetching {url}: {exc}"

    # Strip non-content elements.
    text = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.S)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S)
    text = re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=re.S)
    text = re.sub(r"<footer[^>]*>.*?</footer>", "", text, flags=re.S)
    text = re.sub(r"<header[^>]*>.*?</header>", "", text, flags=re.S)

    # Try to extract content from <article> or <main> tags first.
    article_match = re.search(
        r"<article[^>]*>(.*?)</article>", text, flags=re.S
    )
    main_match = re.search(r"<main[^>]*>(.*?)</main>", text, flags=re.S)

    if article_match:
        text = article_match.group(1)
    elif main_match:
        text = main_match.group(1)

    # Strip remaining HTML tags and clean up.
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + "... [truncated]"
    return text


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
