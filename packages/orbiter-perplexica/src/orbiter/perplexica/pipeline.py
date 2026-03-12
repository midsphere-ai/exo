"""Pipeline — programmatic orchestration matching Perplexica's SearchAgent.searchAsync()."""

from __future__ import annotations

from collections.abc import AsyncIterator

from orbiter.types import StreamEvent

from .agents.classifier import classify
from .agents.researcher import parallel_research, research, stream_research
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

    # Step 1: Classify
    classification = await classify(query, history, cfg)

    # Use the standalone follow-up as the effective query
    effective_query = classification.standalone_follow_up or query

    # Step 2: Research (if needed)
    search_results = []
    if not classification.classification.skip_search:
        _research = parallel_research if mode == "quality" else research
        search_results = await _research(
            query=effective_query,
            classification=classification,
            chat_history=history,
            mode=mode,
            config=cfg,
        )

    # Cap sources for the writer so citation indices stay accurate
    writer_results = search_results[:cfg.max_writer_sources]

    # Step 3: Write answer
    answer = await write_answer(
        query=effective_query,
        search_results=writer_results,
        chat_history=history,
        system_instructions=system_instructions or cfg.system_instructions,
        mode=mode,
        config=cfg,
    )

    # Step 4: Generate suggestions
    updated_history = history + [(query, answer)]
    suggestions = await generate_suggestions(updated_history, cfg)

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

    # Step 1: Classify (fast, non-streaming)
    yield PipelineEvent(stage="classifier", status="started")
    classification = await classify(query, history, cfg)
    effective_query = classification.standalone_follow_up or query
    yield PipelineEvent(
        stage="classifier", status="completed",
        message=f"skip_search={classification.classification.skip_search}",
    )

    # Step 2: Research (streaming tool calls)
    search_results: list[SearchResult] = []
    if not classification.classification.skip_search:
        yield PipelineEvent(stage="researcher", status="started")
        if mode == "quality":
            # Parallel sub-researchers — no per-event streaming, but ~5x faster
            search_results = await parallel_research(
                query=effective_query,
                classification=classification,
                chat_history=history,
                mode=mode,
                config=cfg,
            )
        else:
            async for event in stream_research(
                query=effective_query,
                classification=classification,
                chat_history=history,
                mode=mode,
                config=cfg,
            ):
                if isinstance(event, SearchResult):
                    search_results.append(event)
                else:
                    yield event
        yield PipelineEvent(
            stage="researcher", status="completed",
            message=f"{len(search_results)} results",
        )

    # Cap sources for the writer so citation indices stay accurate
    writer_results = search_results[:cfg.max_writer_sources]

    # Step 3: Write answer (streaming text tokens)
    yield PipelineEvent(stage="writer", status="started")
    answer_parts: list[str] = []
    async for event in stream_write_answer(
        query=effective_query,
        search_results=writer_results,
        chat_history=history,
        system_instructions=system_instructions or cfg.system_instructions,
        mode=mode,
        config=cfg,
    ):
        from orbiter.types import TextEvent
        if isinstance(event, TextEvent):
            answer_parts.append(event.text)
        yield event
    answer = "".join(answer_parts)
    yield PipelineEvent(stage="writer", status="completed")

    # Step 4: Suggestions (fast, non-streaming)
    yield PipelineEvent(stage="suggestions", status="started")
    updated_history = history + [(query, answer)]
    suggestions = await generate_suggestions(updated_history, cfg)
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
