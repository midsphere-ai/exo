"""Pipeline — programmatic orchestration matching Perplexica's SearchAgent.searchAsync()."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace

from orbiter.types import StreamEvent

from .agents.classifier import classify
from .agents.query_planner import hybrid_research
from .agents.researcher import direct_search
from .agents.suggestion_generator import generate_suggestions
from .agents.writer import stream_write_answer, write_answer
from .config import PerplexicaConfig
from .tools.embeddings import rerank_search_results
from .tools.web_fetcher import enrich_results
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
            search_results = await hybrid_research(
                query=classification.standalone_follow_up or query,
                classification=classification,
                chat_history=history,
                mode=mode,
                config=cfg,
            )
    effective_query = classification.standalone_follow_up or query

    # Rerank by relevance (skip for speed — not worth the latency)
    if mode != "speed" and search_results:
        search_results = await rerank_search_results(effective_query, search_results)

    # Cap sources — speed stays lean, balanced/quality get all results
    writer_cap = {"speed": 10}.get(mode, cfg.max_writer_sources)
    writer_results = search_results[:writer_cap]

    # Speed mode uses fast model for writing too
    writer_cfg = replace(cfg, model=cfg.fast_model) if mode == "speed" else cfg

    # Start suggestions early (independent of enrichment and writing)
    suggest_task = asyncio.create_task(
        generate_suggestions(history + [(query, "")], cfg)
    )

    # Enrich top results with full page content (skip for speed)
    enrich_cap = {"balanced": 5, "quality": 10}.get(mode, 0)
    if enrich_cap > 0 and writer_results:
        writer_results = await enrich_results(
            writer_results, cfg.jina_reader_url, max_results=enrich_cap,
        )

    # Step 3: Write answer
    answer = await write_answer(
        query=effective_query,
        search_results=writer_results,
        chat_history=history,
        system_instructions=system_instructions or cfg.system_instructions,
        mode=mode,
        config=writer_cfg,
    )

    # Step 4: Get suggestions (should be done by now)
    suggestions = await suggest_task

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
            search_results = await hybrid_research(
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

    # Rerank by relevance (skip for speed)
    if mode != "speed" and search_results:
        search_results = await rerank_search_results(effective_query, search_results)

    # Cap sources — speed stays lean, balanced/quality get all results
    writer_cap = {"speed": 10}.get(mode, cfg.max_writer_sources)
    writer_results = search_results[:writer_cap]

    # Speed mode uses fast model for writing too
    writer_cfg = replace(cfg, model=cfg.fast_model) if mode == "speed" else cfg

    # Start suggestions concurrently (independent of enrichment and writing)
    suggest_task = asyncio.create_task(
        generate_suggestions(history + [(query, "")], cfg)
    )

    # Enrich top results with full page content (skip for speed)
    enrich_cap = {"balanced": 5, "quality": 10}.get(mode, 0)
    if enrich_cap > 0 and writer_results:
        yield PipelineEvent(stage="enrichment", status="started")
        writer_results = await enrich_results(
            writer_results, cfg.jina_reader_url, max_results=enrich_cap,
        )
        yield PipelineEvent(
            stage="enrichment", status="completed",
            message=f"{min(enrich_cap, len(writer_results))} pages scraped",
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
