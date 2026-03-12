"""Pipeline — programmatic orchestration matching Perplexica's SearchAgent.searchAsync()."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace

from orbiter.types import StreamEvent

from .agents.classifier import classify
from .agents.researcher import direct_search, parallel_research
from .agents.suggestion_generator import generate_suggestions
from .agents.writer import stream_write_answer, write_answer
from .config import PerplexicaConfig
from .types import PerplexicaResponse, PipelineEvent, SearchResult, Source


async def run_search_pipeline(
    query: str,
    chat_history: list[tuple[str, str]] | None = None,
    mode: str = "balanced",
    system_instructions: str = "",
    config: PerplexicaConfig | None = None,
) -> PerplexicaResponse:
    """Run the full Perplexica search pipeline.

    Matches Perplexica's SearchAgent.searchAsync():
    1. Classify query
    2. If skipSearch, go directly to writer with no context
    3. Otherwise, run researcher (iterative tool calls)
    4. Run writer with context from research
    5. Run suggestion generator
    """
    cfg = config or PerplexicaConfig()
    history = chat_history or []

    # Step 1 + 2: Classify and research
    search_results: list[SearchResult] = []
    if mode == "speed":
        # Fast path: classifier and direct SearXNG search in parallel
        classification, search_results = await asyncio.gather(
            classify(query, history, cfg),
            direct_search(query),
        )
    else:
        classification = await classify(query, history, cfg)
        if not classification.classification.skip_search:
            search_results = await parallel_research(
                query=classification.standalone_follow_up or query,
                classification=classification,
                chat_history=history,
                mode=mode,
                config=cfg,
            )
    effective_query = classification.standalone_follow_up or query

    # Cap sources for the writer — fewer sources = faster generation
    writer_cap = {"speed": 10, "balanced": 20}.get(mode, cfg.max_writer_sources)
    writer_results = search_results[:writer_cap]

    # Speed mode uses fast model for writing too
    writer_cfg = replace(cfg, model=cfg.fast_model) if mode == "speed" else cfg

    # Step 3 + 4: Write answer and generate suggestions in parallel
    write_task = write_answer(
        query=effective_query,
        search_results=writer_results,
        chat_history=history,
        system_instructions=system_instructions or cfg.system_instructions,
        mode=mode,
        config=writer_cfg,
    )
    suggest_task = generate_suggestions(history + [(query, "")], cfg)
    answer, suggestions = await asyncio.gather(write_task, suggest_task)

    # Sources list matches what the writer cited (same order, same indices)
    sources = [
        Source(title=r.title, url=r.url, content=r.content)
        for r in writer_results
    ]

    return PerplexicaResponse(
        answer=answer,
        sources=sources,
        suggestions=suggestions,
        query=effective_query,
        mode=mode,
    )


async def stream_search_pipeline(
    query: str,
    chat_history: list[tuple[str, str]] | None = None,
    mode: str = "balanced",
    system_instructions: str = "",
    config: PerplexicaConfig | None = None,
) -> AsyncIterator[PipelineEvent | StreamEvent | PerplexicaResponse]:
    """Stream the full search pipeline.

    Yields:
        PipelineEvent — stage transitions (classifier started/completed, etc.)
        StreamEvent — orbiter events from researcher (ToolCallEvent, TextEvent, etc.)
                      and writer (TextEvent for each token)
        PerplexicaResponse — final complete response as the last item
    """
    cfg = config or PerplexicaConfig()
    history = chat_history or []

    # Step 1 + 2: Classify and research
    search_results: list[SearchResult] = []
    if mode == "speed":
        yield PipelineEvent(stage="classifier", status="started")
        classification, search_results = await asyncio.gather(
            classify(query, history, cfg),
            direct_search(query),
        )
        yield PipelineEvent(stage="classifier", status="completed")
        yield PipelineEvent(
            stage="researcher", status="completed",
            message=f"{len(search_results)} results (direct)",
        )
    else:
        yield PipelineEvent(stage="classifier", status="started")
        classification = await classify(query, history, cfg)
        effective_query = classification.standalone_follow_up or query
        yield PipelineEvent(
            stage="classifier", status="completed",
            message=f"skip_search={classification.classification.skip_search}",
        )
        if not classification.classification.skip_search:
            yield PipelineEvent(stage="researcher", status="started")
            search_results = await parallel_research(
                query=effective_query,
                classification=classification,
                chat_history=history,
                mode=mode,
                config=cfg,
            )
            yield PipelineEvent(
                stage="researcher", status="completed",
                message=f"{len(search_results)} results",
            )
    effective_query = classification.standalone_follow_up or query

    # Cap sources for the writer — fewer sources = faster generation
    writer_cap = {"speed": 10, "balanced": 20}.get(mode, cfg.max_writer_sources)
    writer_results = search_results[:writer_cap]

    # Speed mode uses fast model for writing too
    writer_cfg = replace(cfg, model=cfg.fast_model) if mode == "speed" else cfg

    # Start suggestions concurrently with writer
    suggest_task = asyncio.create_task(
        generate_suggestions(history + [(query, "")], cfg)
    )

    # Step 3: Write answer (streaming text tokens)
    yield PipelineEvent(stage="writer", status="started")
    answer_parts: list[str] = []
    async for event in stream_write_answer(
        query=effective_query,
        search_results=writer_results,
        chat_history=history,
        system_instructions=system_instructions or cfg.system_instructions,
        mode=mode,
        config=writer_cfg,
    ):
        from orbiter.types import TextEvent
        if isinstance(event, TextEvent):
            answer_parts.append(event.text)
        yield event
    answer = "".join(answer_parts)
    yield PipelineEvent(stage="writer", status="completed")

    # Step 4: Suggestions (should be done by now)
    yield PipelineEvent(stage="suggestions", status="started")
    suggestions = await suggest_task
    yield PipelineEvent(stage="suggestions", status="completed")

    # Sources list matches what the writer cited (same order, same indices)
    sources = [
        Source(title=r.title, url=r.url, content=r.content)
        for r in writer_results
    ]
    yield PerplexicaResponse(
        answer=answer,
        sources=sources,
        suggestions=suggestions,
        query=effective_query,
        mode=mode,
    )
