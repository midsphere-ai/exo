"""Perplexica — AI-powered search engine matching Perplexica's architecture.

Usage:
    from orbiter.perplexica import search, search_with_details

    answer = await search("What is quantum computing?")
    result = await search_with_details("latest AI news", mode="balanced")
"""

from __future__ import annotations

from .config import PerplexicaConfig
from .conversation import ConversationManager
from .pipeline import run_search_pipeline, stream_search_pipeline
from .types import PerplexicaResponse, PipelineEvent, ResearchMode


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


def stream(
    query: str,
    mode: str = "balanced",
    chat_history: list[tuple[str, str]] | None = None,
    config: PerplexicaConfig | None = None,
):
    """Stream the search pipeline, yielding events as they happen.

    Yields PipelineEvent for stage transitions, orbiter StreamEvent for
    real-time text/tool events, and a final PerplexicaResponse.

    Usage::

        async for event in stream("quantum computing", mode="quality"):
            if isinstance(event, PipelineEvent):
                print(f"[{event.stage}] {event.status}")
            elif isinstance(event, TextEvent):
                print(event.text, end="", flush=True)
            elif isinstance(event, PerplexicaResponse):
                print(event.sources)
    """
    return stream_search_pipeline(
        query=query,
        chat_history=chat_history,
        mode=mode,
        config=config,
    )


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
    "PipelineEvent",
    "ResearchMode",
    "search",
    "search_with_details",
    "stream",
]
