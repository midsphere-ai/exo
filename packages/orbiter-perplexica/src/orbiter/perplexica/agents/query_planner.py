"""Query planner — adaptive structured-output research with hybrid orchestration."""

from __future__ import annotations

import asyncio
import datetime

from orbiter import Agent, run

from ..config import PerplexicaConfig
from ..tools.searxng import search_and_collect
from ..types import ClassifierOutput, QueryPlan, SearchResult


def _resolve_provider(model: str):
    try:
        from orbiter.models import get_provider
        return get_provider(model)
    except Exception:
        return None


_INITIAL_PROMPT = """\
You are a search query strategist. Given the user's question, generate exactly 3 \
targeted search queries that will find the most relevant, authoritative information.

Today's date: {today}

Requirements:
- Queries must be SEO-friendly keywords, NOT full sentences
- Cover different aspects/parts of the question
- Add year qualifiers for recent topics (e.g., "2025", "2026")
- Include specific entity names, technical terms, and domain keywords

Good: "nuclear fusion breakthroughs 2025 tokamak", "solar energy LCOE cost MWh 2025"
Bad: "What are the latest developments in fusion?" (sentence), "fusion" (too broad)

Set "sufficient" to false (you haven't searched yet).
"""

_FOLLOWUP_PROMPT = """\
You are a search query strategist refining a research plan. The user asked:
"{query}"

Here are the search results found so far:
{results_summary}

Identify what's MISSING or SHALLOW, then generate 3 targeted follow-up queries \
to fill those gaps. Focus on:
- Parts of the question not yet covered by any result
- Claims that need primary sources or specific data
- Recent developments that might have newer coverage

Today's date: {today}

If the existing results already comprehensively cover ALL parts of the question, \
set "sufficient" to true and return an empty queries list.
"""


async def _generate_query_plan(
    query: str,
    chat_history: list[tuple[str, str]],
    existing_results: list[SearchResult],
    config: PerplexicaConfig,
) -> QueryPlan:
    """Generate a query plan — initial queries or gap-filling follow-ups."""
    today = datetime.date.today().strftime("%B %d, %Y")

    if not existing_results:
        prompt = _INITIAL_PROMPT.format(today=today)
    else:
        lines = []
        for i, r in enumerate(existing_results[:20], 1):
            snippet = r.content[:150] if r.content else r.title
            lines.append(f"  {i}. {r.title}: {snippet}")
        prompt = _FOLLOWUP_PROMPT.format(
            query=query, results_summary="\n".join(lines), today=today,
        )

    parts = []
    for q, a in chat_history:
        parts.append(f"User: {q}")
        short_a = a[:500] + "..." if len(a) > 500 else a
        parts.append(f"Assistant: {short_a}")
    parts.append(f"User: {query}")
    formatted_input = "\n".join(parts)

    planner = Agent(
        name="query-planner",
        model=config.fast_model,
        instructions=prompt,
        output_type=QueryPlan,
        temperature=0.3,
        max_steps=1,
    )

    provider = _resolve_provider(config.fast_model)
    result = await run(planner, formatted_input, provider=provider)

    try:
        return QueryPlan.model_validate_json(result.output)
    except Exception:
        return QueryPlan(queries=[query], sufficient=False)


def _merge_results(
    primary: list[SearchResult],
    secondary: list[SearchResult],
) -> list[SearchResult]:
    """Merge two result lists, deduplicating by URL."""
    seen: set[str] = set()
    merged: list[SearchResult] = []
    for r in [*primary, *secondary]:
        if r.url and r.url not in seen:
            seen.add(r.url)
            merged.append(r)
    return merged


async def adaptive_research(
    query: str,
    chat_history: list[tuple[str, str]],
    mode: str,
    config: PerplexicaConfig,
) -> list[SearchResult]:
    """Adaptive multi-round research using structured output query generation.

    Each round the LLM sees existing results and either generates follow-up
    queries to fill gaps or signals that coverage is sufficient.
    """
    max_rounds = {"balanced": 2, "quality": 4}.get(mode, 2)
    all_results: list[SearchResult] = []

    for round_num in range(max_rounds):
        plan = await _generate_query_plan(query, chat_history, all_results, config)

        if plan.sufficient and round_num > 0:
            break
        if not plan.queries:
            break

        raw = await search_and_collect(plan.queries[:3])
        new_results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
            )
            for r in raw
        ]
        all_results = _merge_results(all_results, new_results)

    return all_results


async def hybrid_research(
    query: str,
    classification: ClassifierOutput,
    chat_history: list[tuple[str, str]],
    mode: str = "balanced",
    config: PerplexicaConfig | None = None,
) -> list[SearchResult]:
    """Hybrid research: adaptive structured queries + parallel researchers.

    Structured output path is guaranteed to produce results and adapts
    across rounds. Tool-calling researchers run concurrently as best-effort
    bonus depth, time-boxed to avoid stalling the pipeline.
    """
    from .researcher import parallel_research

    cfg = config or PerplexicaConfig()

    async def _timed_agents() -> list[SearchResult]:
        timeout = {"balanced": 15, "quality": 60}.get(mode, 15)
        try:
            return await asyncio.wait_for(
                parallel_research(query, classification, chat_history, mode, cfg),
                timeout=timeout,
            )
        except Exception:
            return []

    structured_task = adaptive_research(query, chat_history, mode, cfg)
    agent_task = _timed_agents()

    results = await asyncio.gather(structured_task, agent_task)
    return _merge_results(results[0], results[1])
