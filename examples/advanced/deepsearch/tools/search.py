"""Multi-provider web search.

Ported from node-DeepResearch's jina-search.ts, brave-search.ts, and serper-search.ts.
Provides a unified SearchProvider interface with concrete implementations for
Jina, Brave, Serper, and DuckDuckGo backends.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

logger = logging.getLogger("deepsearch")


class SearchProvider(ABC):
    """Abstract base class for web search providers."""

    @abstractmethod
    async def search(
        self,
        query: str,
        num_results: int = 10,
        tbs: str | None = None,
        location: str | None = None,
    ) -> list[dict]:
        """Execute a web search and return normalized results.

        Args:
            query: The search query string.
            num_results: Maximum number of results to return.
            tbs: Time-based search filter (e.g. "qdr:d" for past day).
            location: Geographic location hint for the search.

        Returns:
            List of dicts with keys: title, url, description, and optionally date.
        """
        ...


class JinaSearchProvider(SearchProvider):
    """Jina Search API (s.jina.ai).

    Uses Jina's search endpoint which returns structured results.
    Requires a Jina API key.
    """

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def search(
        self,
        query: str,
        num_results: int = 10,
        tbs: str | None = None,
        location: str | None = None,
    ) -> list[dict]:
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "X-Retain-Images": "none",
        }
        if location:
            headers["X-Location"] = location

        params: dict[str, Any] = {"q": query}
        if num_results:
            params["count"] = num_results

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    "https://s.jina.ai/",
                    params=params,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            logger.error("Jina search timed out for query: %s", query)
            return []
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Jina search HTTP error %d for query: %s",
                exc.response.status_code,
                query,
            )
            return []
        except Exception:
            logger.exception("Jina search unexpected error for query: %s", query)
            return []

        if not isinstance(data.get("data"), list):
            logger.warning("Jina search returned invalid response format for query: %s", query)
            return []

        results: list[dict] = []
        for item in data["data"]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": (item.get("description") or item.get("content") or "")[:500],
                "date": item.get("date"),
            })
        return results


class BraveSearchProvider(SearchProvider):
    """Brave Search API.

    Uses the Brave Web Search API which supports up to 20 results per request.
    Requires a Brave API subscription token.
    """

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def search(
        self,
        query: str,
        num_results: int = 10,
        tbs: str | None = None,
        location: str | None = None,
    ) -> list[dict]:
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
        }
        # Brave API caps at 20 results per request
        params: dict[str, Any] = {"q": query, "count": min(num_results, 20)}

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params=params,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            logger.error("Brave search timed out for query: %s", query)
            return []
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Brave search HTTP error %d for query: %s",
                exc.response.status_code,
                query,
            )
            return []
        except Exception:
            logger.exception("Brave search unexpected error for query: %s", query)
            return []

        results: list[dict] = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": (item.get("description") or "")[:500],
            })
        return results


class DuckDuckGoSearchProvider(SearchProvider):
    """DuckDuckGo search via the duckduckgo-search library.

    No API key required. Falls back gracefully if the library is not installed.
    """

    async def search(
        self,
        query: str,
        num_results: int = 10,
        tbs: str | None = None,
        location: str | None = None,
    ) -> list[dict]:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning(
                "duckduckgo-search not installed; returning empty results. "
                "Install with: pip install duckduckgo-search"
            )
            return []

        try:
            results: list[dict] = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=num_results, safesearch="moderate"):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", r.get("link", "")),
                        "description": (r.get("body") or r.get("snippet") or "")[:500],
                    })
            return results
        except Exception:
            logger.exception("DuckDuckGo search failed for query: %s", query)
            return []


class SearXNGSearchProvider(SearchProvider):
    """SearXNG meta-search engine (self-hosted).

    Queries a local SearXNG instance which aggregates results from multiple
    search engines (Google, Brave, DuckDuckGo, etc.) without API keys.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def search(
        self,
        query: str,
        num_results: int = 10,
        tbs: str | None = None,
        location: str | None = None,
    ) -> list[dict]:
        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "categories": "general",
        }
        # Map tbs time filters to SearXNG time_range
        if tbs:
            tbs_map = {
                "qdr:h": "day",
                "qdr:d": "day",
                "qdr:w": "week",
                "qdr:m": "month",
                "qdr:y": "year",
            }
            time_range = tbs_map.get(tbs)
            if time_range:
                params["time_range"] = time_range

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{self.base_url}/search",
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            logger.error("SearXNG search timed out for query: %s", query)
            return []
        except httpx.HTTPStatusError as exc:
            logger.error(
                "SearXNG search HTTP error %d for query: %s",
                exc.response.status_code,
                query,
            )
            return []
        except Exception:
            logger.exception("SearXNG search error for query: %s", query)
            return []

        results: list[dict] = []
        for item in data.get("results", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": (item.get("content") or "")[:500],
                "date": item.get("publishedDate"),
            })
        return results


class SerperSearchProvider(SearchProvider):
    """Serper.dev Google Search API.

    Provides access to Google search results including knowledge graph data
    and organic results. Supports time-based filtering and location targeting.
    Requires a Serper API key.
    """

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def search(
        self,
        query: str,
        num_results: int = 10,
        tbs: str | None = None,
        location: str | None = None,
    ) -> list[dict]:
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "q": query,
            "num": num_results,
            "autocorrect": False,
        }
        if location:
            payload["location"] = location
        if tbs:
            payload["tbs"] = tbs

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://google.serper.dev/search",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            logger.error("Serper search timed out for query: %s", query)
            return []
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Serper search HTTP error %d for query: %s",
                exc.response.status_code,
                query,
            )
            return []
        except Exception:
            logger.exception("Serper search unexpected error for query: %s", query)
            return []

        results: list[dict] = []

        # Knowledge graph entry (if present and has a description)
        kg = data.get("knowledgeGraph")
        if kg and kg.get("description"):
            results.append({
                "title": kg.get("title", ""),
                "url": kg.get("website") or kg.get("descriptionLink", ""),
                "description": (kg.get("description") or "")[:500],
            })

        # Organic results
        for item in data.get("organic", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "description": (item.get("snippet") or "")[:500],
                "date": item.get("date"),
            })

        return results


def get_search_provider(config: Any) -> SearchProvider:
    """Factory to create the appropriate search provider based on configuration.

    Provider selection logic:
    1. If config.search_provider names a specific provider and the matching
       API key is set, that provider is returned.
    2. Otherwise, falls back to DuckDuckGo (no key required).

    Args:
        config: A DeepSearchConfig instance (or any object with search_provider
                and *_api_key attributes).

    Returns:
        A concrete SearchProvider instance.
    """
    name = getattr(config, "search_provider", "duck")

    if name == "searxng" and getattr(config, "searxng_url", ""):
        logger.info("Using SearXNG search provider at %s", config.searxng_url)
        return SearXNGSearchProvider(config.searxng_url)
    elif name == "jina" and getattr(config, "jina_api_key", ""):
        logger.info("Using Jina search provider")
        return JinaSearchProvider(config.jina_api_key)
    elif name == "brave" and getattr(config, "brave_api_key", ""):
        logger.info("Using Brave search provider")
        return BraveSearchProvider(config.brave_api_key)
    elif name == "serper" and getattr(config, "serper_api_key", ""):
        logger.info("Using Serper search provider")
        return SerperSearchProvider(config.serper_api_key)
    else:
        if name not in ("duck", "auto"):
            logger.warning(
                "Requested search provider '%s' is not available (missing config); "
                "falling back to DuckDuckGo",
                name,
            )
        else:
            logger.info("Using DuckDuckGo search provider")
        return DuckDuckGoSearchProvider()
