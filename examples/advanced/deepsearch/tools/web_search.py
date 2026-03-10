"""Web search tool — multi-engine search with content fetching and summarization.

Faithful port of SkyworkAI's WebSearcherTool. Performs:
1. Parallel search across engines (DuckDuckGo, Brave, Serper, Jina)
2. Content fetching from result URLs
3. Per-page summarization via LLM
4. Merged summary report with citations
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from orbiter.tool import Tool
from orbiter.types import SystemMessage, UserMessage

from ..llm_utils import call_llm

logger = logging.getLogger("deepagent")


class WebSearchTool(Tool):
    """Search the web and return a comprehensive research report with citations.

    Mirrors SkyworkAI's WebSearcherTool: search -> fetch -> summarize -> merge.
    """

    def __init__(
        self,
        *,
        provider: str = "duckduckgo",
        num_results: int = 5,
        max_length: int = 4096,
        model: str = "openai:gpt-4o-mini",
        fetch_content: bool = True,
        summarize_pages: bool = True,
        merge_summaries: bool = True,
        brave_api_key: str | None = None,
        serper_api_key: str | None = None,
        jina_api_key: str | None = None,
    ) -> None:
        self.name = "web_search"
        self.description = (
            "Search the web for real-time information about any topic. "
            "Performs deep research by searching, fetching page content, "
            "summarizing each page, and merging into a comprehensive report with citations."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to submit.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of search results to return (default: 5).",
                },
                "filter_year": {
                    "type": "integer",
                    "description": "Optional year filter for search results.",
                },
            },
            "required": ["query"],
        }
        self._provider = provider
        self._num_results = num_results
        self._max_length = max_length
        self._model = model
        self._fetch_content = fetch_content
        self._summarize_pages = summarize_pages
        self._merge_summaries = merge_summaries
        self._brave_api_key = brave_api_key
        self._serper_api_key = serper_api_key
        self._jina_api_key = jina_api_key

    async def execute(self, **kwargs: Any) -> str:
        """Execute web search and return comprehensive report.

        Args:
            query: Search query.
            num_results: Number of results (optional).
            filter_year: Year filter (optional).

        Returns:
            Comprehensive research report with citations.
        """
        query: str = kwargs.get("query", "")
        num_results: int = kwargs.get("num_results", self._num_results)
        filter_year: int | None = kwargs.get("filter_year")

        if not query:
            return "Error: No search query provided."

        try:
            logger.info(f"Starting web search for: {query}")

            # Step 1: Search with retry
            results = await self._try_search(query, num_results, filter_year)
            if not results:
                return f"No search results found for: {query}"

            logger.info(f"Found {len(results)} search results")

            # Step 2: Fetch content from all pages
            if self._fetch_content:
                logger.info("Fetching content from web pages...")
                results = await self._fetch_content_for_results(results)
                fetched = len([r for r in results if r.get("raw_content")])
                logger.info(f"Fetched content from {fetched} pages")

            # Step 3: Summarize each page
            if self._summarize_pages:
                logger.info("Summarizing each page...")
                results = await self._summarize_results(results, query)

            # Step 4: Merge all summaries
            if self._merge_summaries and any(r.get("summary") for r in results):
                logger.info("Merging summaries into final report...")
                merged = await self._merge_summaries_report(results, query)
                return f"Web search results for query: {query}\n\n{merged}"

            # Fallback format
            lines = [f"Search results for '{query}':"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "").strip() or "No title"
                url = r.get("url", "")
                lines.append(f"\n{i}. {title}")
                lines.append(f"   URL: {url}")
                if r.get("description", "").strip():
                    lines.append(f"   Description: {r['description']}")
                if r.get("summary"):
                    lines.append(f"   Summary: {r['summary']}")
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return f"Error during web search: {e}"

    # ------------------------------------------------------------------
    # Search backends (ported from SkyworkAI search engines)
    # ------------------------------------------------------------------

    async def _try_search(
        self, query: str, num_results: int, filter_year: int | None
    ) -> list[dict[str, Any]]:
        """Try searching with configured provider, with retry."""
        max_retries = 3
        retry_delay = 10

        for attempt in range(max_retries + 1):
            try:
                results = await self._search(query, num_results, filter_year)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Search attempt {attempt + 1} failed: {e}")

            if attempt < max_retries:
                logger.warning(f"Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)

        return []

    async def _search(
        self, query: str, num_results: int, filter_year: int | None
    ) -> list[dict[str, Any]]:
        """Dispatch to the configured search provider."""
        if self._provider == "duckduckgo":
            return await self._search_duckduckgo(query, num_results, filter_year)
        elif self._provider == "brave":
            return await self._search_brave(query, num_results, filter_year)
        elif self._provider == "serper":
            return await self._search_serper(query, num_results, filter_year)
        elif self._provider == "jina":
            return await self._search_jina(query, num_results)
        else:
            raise ValueError(f"Unknown search provider: {self._provider}")

    async def _search_duckduckgo(
        self, query: str, num_results: int, filter_year: int | None
    ) -> list[dict[str, Any]]:
        """Search using DuckDuckGo via duckduckgo-search library."""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise ImportError(
                "duckduckgo-search is required. Install with: pip install duckduckgo-search"
            )

        def _do_search() -> list[dict[str, Any]]:
            ddgs = DDGS()
            kwargs: dict[str, Any] = {"keywords": query, "max_results": num_results}
            if filter_year:
                kwargs["timelimit"] = f"{filter_year}-01-01..{filter_year}-12-31"

            raw_results = ddgs.text(**kwargs)
            results = []
            for i, item in enumerate(raw_results):
                results.append({
                    "position": i + 1,
                    "url": item.get("href", item.get("link", "")),
                    "title": item.get("title", f"Result {i + 1}"),
                    "description": item.get("body", item.get("snippet", "")),
                    "source": "duckduckgo",
                    "raw_content": None,
                    "summary": None,
                })
            return results

        return await asyncio.to_thread(_do_search)

    async def _search_brave(
        self, query: str, num_results: int, filter_year: int | None
    ) -> list[dict[str, Any]]:
        """Search using Brave Search API."""
        if not self._brave_api_key:
            raise ValueError("Brave API key required. Set BRAVE_API_KEY.")

        params: dict[str, Any] = {"q": query, "count": num_results}
        if filter_year:
            params["freshness"] = f"{filter_year}-01-01to{filter_year}-12-31"

        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params=params,
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": self._brave_api_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for i, item in enumerate(data.get("web", {}).get("results", [])):
            results.append({
                "position": i + 1,
                "url": item.get("url", ""),
                "title": item.get("title", f"Result {i + 1}"),
                "description": item.get("description", ""),
                "source": "brave",
                "raw_content": None,
                "summary": None,
            })
        return results

    async def _search_serper(
        self, query: str, num_results: int, filter_year: int | None
    ) -> list[dict[str, Any]]:
        """Search using Serper.dev API."""
        if not self._serper_api_key:
            raise ValueError("Serper API key required. Set SERPER_API_KEY.")

        payload: dict[str, Any] = {"q": query, "num": num_results}
        if filter_year:
            payload["tbs"] = f"cdr:1,cd_min:1/1/{filter_year},cd_max:12/31/{filter_year}"

        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
            resp = await client.post(
                "https://google.serper.dev/search",
                json=payload,
                headers={
                    "X-API-KEY": self._serper_api_key,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for i, item in enumerate(data.get("organic", [])):
            results.append({
                "position": i + 1,
                "url": item.get("link", ""),
                "title": item.get("title", f"Result {i + 1}"),
                "description": item.get("snippet", ""),
                "source": "serper",
                "raw_content": None,
                "summary": None,
            })
        return results

    async def _search_jina(
        self, query: str, num_results: int
    ) -> list[dict[str, Any]]:
        """Search using Jina Search API."""
        if not self._jina_api_key:
            raise ValueError("Jina API key required. Set JINA_API_KEY.")

        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
            resp = await client.get(
                f"https://s.jina.ai/{query}",
                headers={
                    "Authorization": f"Bearer {self._jina_api_key}",
                    "Accept": "application/json",
                    "X-Max-Results": str(num_results),
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for i, item in enumerate(data.get("data", [])):
            results.append({
                "position": i + 1,
                "url": item.get("url", ""),
                "title": item.get("title", f"Result {i + 1}"),
                "description": item.get("description", ""),
                "source": "jina",
                "raw_content": None,
                "summary": None,
            })
        return results

    # ------------------------------------------------------------------
    # Content fetching (ported from SkyworkAI WebSearcherTool)
    # ------------------------------------------------------------------

    async def _fetch_content_for_results(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Fetch and add web content to search results in parallel."""
        if not results:
            return []

        fetched = await asyncio.gather(
            *[self._fetch_single(r) for r in results],
            return_exceptions=True,
        )

        final = []
        for i, result in enumerate(fetched):
            if isinstance(result, Exception):
                logger.warning(f"Exception fetching result {i + 1}: {result}")
                if i < len(results):
                    results[i]["raw_content"] = None
                    final.append(results[i])
            else:
                final.append(result)
        return final

    async def _fetch_single(self, result: dict[str, Any]) -> dict[str, Any]:
        """Fetch content for a single search result with timeout."""
        url = result.get("url")
        if not url:
            return result

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            result["raw_content"] = None
            return result

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=httpx.Timeout(20.0),
            ) as client:
                resp = await asyncio.wait_for(
                    client.get(
                        url,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/120.0.0.0 Safari/537.36"
                            ),
                        },
                    ),
                    timeout=20.0,
                )
                resp.raise_for_status()

                html = resp.text
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                text = "\n".join(lines)

                if len(text) > self._max_length:
                    text = text[: self._max_length] + "..."
                result["raw_content"] = text if text else None

        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching content from {url}")
            result["raw_content"] = None
        except Exception as e:
            logger.warning(f"Error fetching content from {url}: {e}")
            result["raw_content"] = None

        return result

    # ------------------------------------------------------------------
    # Summarization (ported from SkyworkAI WebSearcherTool)
    # ------------------------------------------------------------------

    async def _summarize_results(
        self, results: list[dict[str, Any]], query: str
    ) -> list[dict[str, Any]]:
        """Summarize each search result using LLM."""
        if not results:
            return results

        async def _summarize_one(result: dict[str, Any]) -> dict[str, Any]:
            raw_content = result.get("raw_content")
            if not raw_content:
                result["summary"] = "No content available to summarize."
                return result

            title = result.get("title", "Untitled")
            url = result.get("url", "")

            prompt = (
                f'Given this search query: "{query}"\n\n'
                f"And this webpage content:\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {raw_content[:self._max_length]}\n\n"
                f"Provide a concise summary (2-4 sentences) that:\n"
                f"1. Directly addresses the search query\n"
                f"2. Highlights the most relevant information from this page\n"
                f"3. Focuses on factual information and key insights\n"
                f"4. Is clear and actionable\n\n"
                f"Return only the summary, nothing else."
            )

            try:
                response = await call_llm(
                    model=self._model,
                    messages=[UserMessage(content=prompt)],
                )
                if response.success and response.message.strip():
                    result["summary"] = response.message.strip()
                else:
                    result["summary"] = "Failed to generate summary."
            except Exception as e:
                logger.warning(f"Failed to summarize {url}: {e}")
                result["summary"] = f"Summary unavailable: {e}"

            return result

        summarized = await asyncio.gather(
            *[_summarize_one(r) for r in results],
            return_exceptions=True,
        )

        final = []
        for i, result in enumerate(summarized):
            if isinstance(result, Exception):
                logger.error(f"Exception summarizing result {i}: {result}")
                if i < len(results):
                    results[i]["summary"] = f"Summary failed: {result}"
                    final.append(results[i])
            else:
                final.append(result)
        return final

    # ------------------------------------------------------------------
    # Merge summaries (ported from SkyworkAI WebSearcherTool)
    # ------------------------------------------------------------------

    async def _merge_summaries_report(
        self, results: list[dict[str, Any]], query: str
    ) -> str:
        """Merge all summaries into a comprehensive report with citations."""
        summarized = [r for r in results if r.get("summary")]
        if not summarized:
            return "No summaries available to merge."

        summaries_text = []
        for i, r in enumerate(summarized, 1):
            title = r.get("title", "").strip() or "Untitled"
            summary = r.get("summary", "")
            url = r.get("url", "")
            summaries_text.append(f"[{i}] {title} ({url})\nSummary: {summary}\n")

        prompt = (
            f"You are creating a comprehensive research report based on multiple web sources.\n\n"
            f'Search Query: "{query}"\n\n'
            f"Source Summaries:\n{''.join(summaries_text)}\n\n"
            f"Please create a well-structured, comprehensive report that:\n"
            f"1. Directly answers the search query\n"
            f"2. Synthesizes information from all sources\n"
            f"3. Organizes information logically\n"
            f"4. Includes inline citations using [1], [2], etc. format when referencing specific sources\n"
            f"5. Highlights key findings and insights\n"
            f"6. Is clear, professional, and easy to read\n\n"
            f"Format the report with:\n"
            f"- A clear introduction that addresses the query\n"
            f"- Main findings organized by topic or theme\n"
            f"- Inline citations [1], [2], etc. when referencing specific sources\n"
            f"- A conclusion that summarizes the key points\n\n"
            f"Return only the report content, nothing else."
        )

        try:
            response = await call_llm(
                model=self._model,
                messages=[
                    SystemMessage(
                        content="You are an expert at synthesizing information from multiple sources into comprehensive research reports."
                    ),
                    UserMessage(content=prompt),
                ],
            )

            if response.success and response.message.strip():
                report = response.message.strip()
                report += "\n\n## References:\n"
                for i, r in enumerate(summarized, 1):
                    title = r.get("title", "").strip() or "Untitled"
                    url = r.get("url", "")
                    report += f"[{i}] [{title}]({url})\n"
                return report
            return self._fallback_merge(summarized, query)
        except Exception as e:
            logger.error(f"Failed to merge summaries: {e}")
            return self._fallback_merge(summarized, query)

    def _fallback_merge(self, results: list[dict[str, Any]], query: str) -> str:
        """Fallback merge when LLM fails."""
        report = f"# Research Report: {query}\n\n## Summary\n\n"
        for i, r in enumerate(results, 1):
            title = r.get("title", "").strip() or "Untitled"
            summary = r.get("summary", "")
            report += f"### Source {i}: {title}\n{summary}\n\n"
        report += "## References\n\n"
        for i, r in enumerate(results, 1):
            title = r.get("title", "").strip() or "Untitled"
            url = r.get("url", "")
            report += f"{i}. [{title}]({url})\n"
        return report
