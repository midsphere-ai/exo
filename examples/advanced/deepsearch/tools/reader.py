"""Multi-provider content extraction from URLs.

Ported from node-DeepResearch/src/tools/read.ts.
Provides an abstract ContentReader with two concrete implementations:
  - JinaReader: uses Jina Reader API (r.jina.ai) for high-quality extraction
  - HttpReader: generic httpx + beautifulsoup4 + markdownify (no API key needed)
"""
from __future__ import annotations

import logging
import re

import httpx
from abc import ABC, abstractmethod

logger = logging.getLogger("deepsearch")


class ReadResult:
    """Container for the result of reading a URL."""

    __slots__ = ("url", "title", "content", "links", "images", "tokens", "success", "error")

    def __init__(
        self,
        url: str,
        title: str = "",
        content: str = "",
        links: list[tuple[str, str]] | None = None,
        images: dict[str, str] | None = None,
        tokens: int = 0,
        success: bool = True,
        error: str = "",
    ) -> None:
        self.url = url
        self.title = title
        self.content = content
        self.links = links or []
        self.images = images or {}
        self.tokens = tokens
        self.success = success
        self.error = error

    def __repr__(self) -> str:
        status = "ok" if self.success else f"error={self.error!r}"
        return f"<ReadResult url={self.url!r} title={self.title!r} {status} tokens={self.tokens}>"


class ContentReader(ABC):
    """Abstract base for content readers."""

    @abstractmethod
    async def read(
        self, url: str, with_links: bool = True, with_images: bool = False
    ) -> ReadResult:
        """Read and extract content from *url*, returning a ReadResult."""
        ...


class JinaReader(ContentReader):
    """High-quality content extraction via the Jina Reader API.

    Mirrors the original TypeScript implementation which POSTs to
    ``https://r.jina.ai/`` with the target URL in the JSON body and
    receives structured markdown content back.
    """

    ENDPOINT = "https://r.jina.ai/"

    def __init__(self, api_key: str, *, timeout: float = 60.0) -> None:
        if not api_key:
            raise ValueError("JinaReader requires a non-empty api_key")
        self.api_key = api_key
        self.timeout = timeout

    async def read(
        self, url: str, with_links: bool = True, with_images: bool = False
    ) -> ReadResult:
        url = url.strip()
        if not url:
            return ReadResult(url=url, success=False, error="URL cannot be empty")
        if not url.startswith(("http://", "https://")):
            return ReadResult(url=url, success=False, error="Only http/https URLs are supported")

        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Md-Link-Style": "discarded",
        }
        if with_links:
            headers["X-With-Links-Summary"] = "all"
        if with_images:
            headers["X-With-Images-Summary"] = "true"
        else:
            headers["X-Retain-Images"] = "none"

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=True
            ) as client:
                resp = await client.post(
                    self.ENDPOINT,
                    json={"url": url},
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

            item = data.get("data")
            if not item:
                return ReadResult(url=url, success=False, error="Invalid response data from Jina")

            logger.debug("Jina read: %s (%s)", item.get("title", ""), item.get("url", url))

            # Parse links — the API returns a mapping of {text: href} or a list of pairs
            raw_links = item.get("links", {})
            if isinstance(raw_links, dict):
                links = [(text, href) for text, href in raw_links.items()]
            elif isinstance(raw_links, list):
                links = [(entry[0], entry[1]) for entry in raw_links if len(entry) >= 2]
            else:
                links = []

            return ReadResult(
                url=item.get("url", url),
                title=item.get("title", ""),
                content=item.get("content", ""),
                links=links,
                images=item.get("images", {}),
                tokens=item.get("usage", {}).get("tokens", 0),
            )
        except httpx.HTTPStatusError as exc:
            msg = f"Jina API returned {exc.response.status_code}"
            logger.warning("Jina reader failed for %s: %s", url, msg)
            return ReadResult(url=url, success=False, error=msg)
        except httpx.TimeoutException:
            logger.warning("Jina reader timed out for %s", url)
            return ReadResult(url=url, success=False, error="Request timed out")
        except Exception as exc:
            logger.warning("Jina reader failed for %s: %s", url, exc)
            return ReadResult(url=url, success=False, error=str(exc))


class SelfHostedReader(ContentReader):
    """Self-hosted Jina Reader (intergalacticalvariable/reader).

    Uses the URL-prefix API format: ``{base_url}/{target_url}``
    with ``X-Respond-With: markdown`` header. No API key required.
    """

    def __init__(self, base_url: str, *, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def read(
        self, url: str, with_links: bool = True, with_images: bool = False
    ) -> ReadResult:
        url = url.strip()
        if not url:
            return ReadResult(url=url, success=False, error="URL cannot be empty")
        if not url.startswith(("http://", "https://")):
            return ReadResult(url=url, success=False, error="Only http/https URLs are supported")

        headers: dict[str, str] = {
            "X-Respond-With": "markdown",
        }
        if with_links:
            headers["X-With-Links-Summary"] = "all"
        if not with_images:
            headers["X-Retain-Images"] = "none"

        reader_url = f"{self.base_url}/{url}"

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=True
            ) as client:
                resp = await client.get(reader_url, headers=headers)
                resp.raise_for_status()
                content = resp.text

            # Extract title from first markdown heading if present
            title = ""
            for line in content.split("\n")[:5]:
                line = line.strip()
                if line.startswith("#"):
                    title = line.lstrip("#").strip()
                    break
                if line and not title:
                    title = line[:200]

            # Collapse excessive blank lines
            content = re.sub(r"\n{3,}", "\n\n", content).strip()

            logger.debug("Self-hosted reader: %s (%d chars)", title[:50], len(content))

            return ReadResult(
                url=url,
                title=title,
                content=content,
                tokens=len(content) // 4,
            )
        except httpx.HTTPStatusError as exc:
            msg = f"Reader returned {exc.response.status_code}"
            logger.warning("Self-hosted reader failed for %s: %s", url, msg)
            return ReadResult(url=url, success=False, error=msg)
        except httpx.TimeoutException:
            logger.warning("Self-hosted reader timed out for %s", url)
            return ReadResult(url=url, success=False, error="Request timed out")
        except Exception as exc:
            logger.warning("Self-hosted reader failed for %s: %s", url, exc)
            return ReadResult(url=url, success=False, error=str(exc))


class HttpReader(ContentReader):
    """Generic HTTP reader using beautifulsoup4 + markdownify.

    Falls back to naive HTML-tag stripping when the optional dependencies
    are not installed.
    """

    # Tags to remove before content extraction
    _NOISE_TAGS = ("script", "style", "nav", "footer", "header", "aside", "noscript")

    def __init__(self, *, timeout: float = 20.0, max_content_length: int = 50_000) -> None:
        self.timeout = timeout
        self.max_content_length = max_content_length

    async def read(
        self, url: str, with_links: bool = True, with_images: bool = False
    ) -> ReadResult:
        url = url.strip()
        if not url:
            return ReadResult(url=url, success=False, error="URL cannot be empty")
        if not url.startswith(("http://", "https://")):
            return ReadResult(url=url, success=False, error="Only http/https URLs are supported")

        try:
            html = await self._fetch(url)
        except httpx.HTTPStatusError as exc:
            msg = f"HTTP {exc.response.status_code}"
            logger.warning("HTTP reader failed for %s: %s", url, msg)
            return ReadResult(url=url, success=False, error=msg)
        except httpx.TimeoutException:
            logger.warning("HTTP reader timed out for %s", url)
            return ReadResult(url=url, success=False, error="Request timed out")
        except Exception as exc:
            logger.warning("HTTP reader failed for %s: %s", url, exc)
            return ReadResult(url=url, success=False, error=str(exc))

        return self._parse_html(url, html, with_links=with_links, with_images=with_images)

    async def _fetch(self, url: str) -> str:
        """Fetch raw HTML from *url*."""
        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DeepSearch/1.0)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text

    def _parse_html(
        self,
        url: str,
        html: str,
        *,
        with_links: bool = True,
        with_images: bool = False,
    ) -> ReadResult:
        """Parse HTML into a ReadResult, falling back to regex stripping if
        beautifulsoup4/markdownify are unavailable."""
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-untyped]
            import markdownify  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("bs4/markdownify not installed; falling back to tag stripping")
            return self._fallback_parse(url, html)

        soup = BeautifulSoup(html, "html.parser")

        # Extract title before we mutate the tree
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text().strip()

        # Remove noisy elements
        for tag in soup.find_all(list(self._NOISE_TAGS)):
            tag.decompose()

        # Prefer semantic content containers
        main = soup.find("main") or soup.find("article") or soup.find("body") or soup

        # Extract links before conversion (BeautifulSoup tag iteration)
        links: list[tuple[str, str]] = []
        if with_links:
            for anchor in main.find_all("a", href=True):
                text = anchor.get_text().strip()
                href = anchor["href"]
                if href.startswith("http") and text:
                    links.append((text, href))

        # Extract images
        images: dict[str, str] = {}
        if with_images:
            for img in main.find_all("img", src=True):
                alt = img.get("alt", "")
                src = img["src"]
                if src.startswith("http"):
                    images[alt or src] = src

        # Convert to markdown
        strip_tags = ["img"] if not with_images else []
        content: str = markdownify.markdownify(
            str(main), heading_style="ATX", strip=strip_tags
        )
        # Collapse excessive blank lines
        content = re.sub(r"\n{3,}", "\n\n", content).strip()

        # Truncate if too long
        if len(content) > self.max_content_length:
            content = content[: self.max_content_length]

        return ReadResult(
            url=url,
            title=title,
            content=content,
            links=links,
            images=images,
            tokens=len(content) // 4,
        )

    @staticmethod
    def _fallback_parse(url: str, html: str) -> ReadResult:
        """Minimal HTML-to-text conversion without external dependencies."""
        # Rough title extraction
        title = ""
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()

        # Strip tags
        content = re.sub(r"<[^>]+>", " ", html)
        content = re.sub(r"\s+", " ", content).strip()
        # Cap at a reasonable size
        if len(content) > 10_000:
            content = content[:10_000]

        return ReadResult(
            url=url,
            title=title,
            content=content,
            tokens=len(content) // 4,
        )


def get_reader(config: object) -> ContentReader:
    """Factory to create the appropriate content reader.

    Examines ``config.content_reader`` and ``config.jina_api_key`` to decide
    which implementation to return.  Falls back to :class:`HttpReader` when the
    Jina key is unavailable.
    """
    name = getattr(config, "content_reader", "http")
    reader_url = getattr(config, "reader_url", "")
    jina_key = getattr(config, "jina_api_key", "")
    if name == "selfhosted" and reader_url:
        logger.info("Using self-hosted reader at %s", reader_url)
        return SelfHostedReader(reader_url)
    if name == "jina" and jina_key:
        logger.info("Using Jina Reader for content extraction")
        return JinaReader(jina_key)
    logger.info("Using HTTP Reader for content extraction")
    return HttpReader()
