"""Perplexica — AI-powered search engine matching Perplexica's architecture.

Usage:
    from orbiter.perplexica import search, search_with_details

    answer = await search("What is quantum computing?")
    result = await search_with_details("latest AI news", mode="balanced")
"""

from __future__ import annotations

from .config import PerplexicaConfig
from .conversation import ConversationManager
from .pipeline import run_search_pipeline
from .types import PerplexicaResponse, ResearchMode


async def search(
    query: str,
    mode: str = "balanced",
    chat_history: list[tuple[str, str]] | None = None,
    config: PerplexicaConfig | None = None,
) -> str:
    """Run a search query and return the answer with citations."""
    result = await run_search_pipeline(
        query=query,
        chat_history=chat_history,
        mode=mode,
        config=config,
    )
    return result.answer


async def search_with_details(
    query: str,
    mode: str = "balanced",
    chat_history: list[tuple[str, str]] | None = None,
    config: PerplexicaConfig | None = None,
) -> PerplexicaResponse:
    """Run a search query and return full response with sources and suggestions."""
    return await run_search_pipeline(
        query=query,
        chat_history=chat_history,
        mode=mode,
        config=config,
    )


__all__ = [
    "ConversationManager",
    "PerplexicaConfig",
    "PerplexicaResponse",
    "ResearchMode",
    "search",
    "search_with_details",
]
