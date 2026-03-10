"""Content reader tool — fetches and extracts text from web pages.

Ported from SkyworkAI's WebFetcherTool. Supports httpx+BS4 and optional Jina Reader API.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from orbiter.tool import Tool

logger = logging.getLogger("deepagent")


class ContentReaderTool(Tool):
    """Fetch and extract readable text from a URL.

    Supports two backends:
    - httpx: Direct HTTP fetch + BeautifulSoup text extraction (default).
    - jina: Jina Reader API for cleaner extraction (requires API key).
    """

    def __init__(
        self,
        *,
        mode: str = "httpx",
        max_length: int = 4096,
        timeout: float = 20.0,
        jina_api_key: str | None = None,
    ) -> None:
        self.name = "read_webpage"
        self.description = (
            "Fetch and read the content of a web page. "
            "Returns the extracted text content from the given URL."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the web page to read.",
                },
            },
            "required": ["url"],
        }
        self._mode = mode
        self._max_length = max_length
        self._timeout = timeout
        self._jina_api_key = jina_api_key

    async def execute(self, **kwargs: Any) -> str:
        """Fetch and extract text from a URL.

        Args:
            url: The URL to fetch.

        Returns:
            Extracted text content from the page.
        """
        url: str = kwargs.get("url", "")
        if not url:
            return "Error: No URL provided."

        try:
            if self._mode == "jina" and self._jina_api_key:
                return await self._fetch_jina(url)
            return await self._fetch_httpx(url)
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return f"Error fetching URL: {e}"

    async def _fetch_httpx(self, url: str) -> str:
        """Fetch URL with httpx and extract text with BeautifulSoup."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return "Error: beautifulsoup4 is required. Install with: pip install beautifulsoup4"

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(self._timeout),
        ) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")

            if "text/html" in content_type or "text/plain" in content_type:
                html = response.text
                soup = BeautifulSoup(html, "html.parser")

                # Remove script and style elements
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()

                text = soup.get_text(separator="\n", strip=True)

                # Clean up excessive whitespace
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                text = "\n".join(lines)
            else:
                text = response.text

        if len(text) > self._max_length:
            text = text[: self._max_length] + "..."

        return text if text else "No readable content found."

    async def _fetch_jina(self, url: str) -> str:
        """Fetch URL content using Jina Reader API."""
        jina_url = f"https://r.jina.ai/{url}"
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
        ) as client:
            response = await client.get(
                jina_url,
                headers={
                    "Authorization": f"Bearer {self._jina_api_key}",
                    "Accept": "text/plain",
                },
            )
            response.raise_for_status()
            text = response.text.strip()

        if len(text) > self._max_length:
            text = text[: self._max_length] + "..."

        return text if text else "No readable content found."
