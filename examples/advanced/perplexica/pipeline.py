"""Pipeline — programmatic orchestration matching Perplexica's SearchAgent.searchAsync()."""

from __future__ import annotations

from .agents.classifier import classify
from .agents.researcher import research
from .agents.writer import write_answer
from .agents.suggestion_generator import generate_suggestions
from .config import PerplexicaConfig
from .types import PerplexicaResponse, Source


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
        search_results = await research(
            query=effective_query,
            classification=classification,
            chat_history=history,
            mode=mode,
            config=cfg,
        )

    # Step 3: Write answer
    answer = await write_answer(
        query=effective_query,
        search_results=search_results,
        chat_history=history,
        system_instructions=system_instructions or cfg.system_instructions,
        mode=mode,
        config=cfg,
    )

    # Step 4: Generate suggestions
    updated_history = history + [(query, answer)]
    suggestions = await generate_suggestions(updated_history, cfg)

    # Build sources from search results
    sources = [
        Source(title=r.title, url=r.url, content=r.content)
        for r in search_results
    ]

    return PerplexicaResponse(
        answer=answer,
        sources=sources,
        suggestions=suggestions,
        query=effective_query,
        mode=mode,
    )
