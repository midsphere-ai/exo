"""Browser tools — navigate web pages and capture screenshots.

These tools wrap HTTP fetching and (optionally) a headless browser for
richer page interaction.  They are intentionally simple so that example
agents can call them without heavyweight dependencies; swap the stub
implementations for real Playwright/Selenium calls in production.

Usage:
    from examples.common.tools.browser import browse_url, screenshot
"""

from __future__ import annotations

import html
import re
from urllib.parse import quote, urlparse

from exo import tool


@tool
async def browse_url(url: str) -> str:
    """Fetch a web page and return its text content.

    Args:
        url: Fully-qualified URL to fetch (e.g. https://example.com).
    """
    import urllib.request

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Error: unsupported scheme '{parsed.scheme}'. Use http or https."

    safe_url = quote(url, safe=":/?#[]@!$&'()*+,;=-._~%")
    req = urllib.request.Request(
        safe_url,
        headers={"User-Agent": "ExoBot/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return f"Error fetching {url}: {exc}"

    # Strip HTML tags and decode entities for a clean text view.
    text = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.S)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate to avoid blowing up context windows.
    max_chars = 8000
    if len(text) > max_chars:
        text = text[:max_chars] + "… [truncated]"
    return text


@tool
async def screenshot(url: str) -> str:
    """Describe what a screenshot of the page would contain.

    This is a stub that returns page metadata. In production, replace
    with a real headless-browser screenshot (Playwright / Selenium).

    Args:
        url: URL of the page to screenshot.
    """
    # In a real implementation this would launch a headless browser,
    # navigate to the URL, and return a base64-encoded image.
    return (
        f"[screenshot stub] Would capture a screenshot of {url}. "
        "Replace this tool with a Playwright-based implementation for "
        "real browser screenshots."
    )
