"""Perplexica search tools."""

from .searxng import (
    web_search,
    academic_search,
    social_search,
    search_and_collect,
    get_collected_results,
    clear_collected_results,
)
from .web_fetcher import fetch_page_content, fetch_multiple_pages, scrape_url
from .researcher_tools import done, reasoning_preamble

__all__ = [
    "web_search",
    "academic_search",
    "social_search",
    "search_and_collect",
    "fetch_page_content",
    "fetch_multiple_pages",
    "scrape_url",
    "done",
    "reasoning_preamble",
]
