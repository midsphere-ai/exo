"""Researcher agent — iterative tool-calling research loop."""

from __future__ import annotations

from collections.abc import AsyncIterator

from orbiter import Agent, run
from orbiter.types import StreamEvent

from ..config import PerplexicaConfig
from ..prompts.instructions import (
    get_researcher_prompt,
    get_web_search_prompt,
    ACADEMIC_SEARCH_PROMPT,
    SOCIAL_SEARCH_PROMPT,
    SCRAPE_URL_PROMPT,
    DONE_PROMPT,
    REASONING_PREAMBLE_PROMPT,
)
from ..tools.searxng import (
    web_search,
    academic_search,
    social_search,
    clear_collected_results,
    get_collected_results,
)
from ..tools.web_fetcher import scrape_url
from ..tools.researcher_tools import done, reasoning_preamble
from ..types import ClassifierOutput, SearchResult


def _resolve_provider(model: str):
    try:
        from orbiter.models import get_provider
        return get_provider(model)
    except Exception:
        return None


_MODE_ITERATIONS = {"speed": 2, "balanced": 6, "quality": 25}

# Gemini thinking models that have built-in reasoning — reasoning_preamble is redundant.
_THINKING_MODEL_PATTERNS = ("gemini-2.5", "gemini-3",)


def _get_max_iterations(mode: str, override: int | None = None) -> int:
    if override is not None:
        return override
    return _MODE_ITERATIONS.get(mode, 25)


def _is_thinking_model(model: str) -> bool:
    """Check if the model has built-in thinking/reasoning capabilities."""
    model_lower = model.lower()
    return any(p in model_lower for p in _THINKING_MODEL_PATTERNS)


def _build_tools_and_action_desc(
    classification: ClassifierOutput,
    sources: list[str],
    mode: str,
    include_reasoning_preamble: bool = True,
) -> tuple[list, str]:
    """Build available tools and action descriptions based on classification."""
    tools = []
    action_lines = []

    # Web search (if web source enabled and search not skipped)
    if "web" in sources and not classification.classification.skip_search:
        tools.append(web_search)
        action_lines.append(f"- web_search: {get_web_search_prompt(mode)}")

    # Academic search
    if ("academic" in sources and classification.classification.academic_search
            and not classification.classification.skip_search):
        tools.append(academic_search)
        action_lines.append(f"- academic_search: {ACADEMIC_SEARCH_PROMPT}")

    # Social/discussion search
    if ("discussions" in sources and classification.classification.discussion_search
            and not classification.classification.skip_search):
        tools.append(social_search)
        action_lines.append(f"- social_search: {SOCIAL_SEARCH_PROMPT}")

    # Scrape URL (always available)
    tools.append(scrape_url)
    action_lines.append(f"- scrape_url: {SCRAPE_URL_PROMPT}")

    # Done (always available)
    tools.append(done)
    action_lines.append(f"- done: {DONE_PROMPT}")

    # Reasoning preamble (skip for speed mode and thinking models)
    if include_reasoning_preamble and mode != "speed":
        tools.append(reasoning_preamble)
        action_lines.append(f"- reasoning_preamble: {REASONING_PREAMBLE_PROMPT}")

    return tools, "\n".join(action_lines)


async def research(
    query: str,
    classification: ClassifierOutput,
    chat_history: list[tuple[str, str]],
    mode: str = "balanced",
    config: PerplexicaConfig | None = None,
) -> list[SearchResult]:
    """Run the researcher agent with iterative tool calls.

    Matches Perplexica's Researcher.research() flow.
    Uses a shared result collector in the search tools to gather results,
    since the orbiter framework doesn't expose tool call history in RunResult.
    """
    from ..config import PerplexicaConfig as Cfg
    cfg = config or Cfg()

    max_iterations = _get_max_iterations(mode, cfg.max_iterations)

    # Determine whether to include reasoning_preamble tool
    if cfg.use_reasoning_preamble is not None:
        include_reasoning = cfg.use_reasoning_preamble
    else:
        # Auto-detect: skip for thinking models that reason internally
        include_reasoning = not _is_thinking_model(cfg.model)

    tools, action_desc = _build_tools_and_action_desc(
        classification, cfg.sources, mode, include_reasoning
    )

    # Use no-reasoning-preamble prompt variant for thinking models
    effective_mode = mode if include_reasoning else f"{mode}_no_reasoning"
    instructions = get_researcher_prompt(
        action_desc=action_desc,
        mode=effective_mode,
        iteration=1,
        max_iteration=max_iterations,
    )

    # Build input with chat history context
    parts = []
    for q, a in chat_history:
        parts.append(f"User: {q}")
        short_a = a[:500] + "..." if len(a) > 500 else a
        parts.append(f"Assistant: {short_a}")
    parts.append(f"User: {query}")
    formatted_input = "\n".join(parts)

    researcher = Agent(
        name="researcher",
        model=cfg.model,
        instructions=instructions,
        tools=tools,
        temperature=0.1,
        max_steps=max_iterations * 3,
    )

    # Clear collector before run, collect results after
    clear_collected_results()

    provider = _resolve_provider(cfg.model)
    await run(researcher, formatted_input, provider=provider)

    # Get results from the shared collector (populated by tool side-effects)
    raw_results = get_collected_results()

    # Deduplicate and convert to SearchResult objects
    seen_urls: set[str] = set()
    results: list[SearchResult] = []
    for r in raw_results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            results.append(SearchResult(
                title=r.get("title", ""),
                url=url,
                content=r.get("content", ""),
            ))

    return results


async def stream_research(
    query: str,
    classification: ClassifierOutput,
    chat_history: list[tuple[str, str]],
    mode: str = "balanced",
    config: PerplexicaConfig | None = None,
) -> AsyncIterator[SearchResult | StreamEvent]:
    """Stream researcher events, then yield collected SearchResults at the end.

    Yields StreamEvent objects during execution (TextEvent, ToolCallEvent, etc.),
    then yields SearchResult objects once the agent finishes.
    """
    from ..config import PerplexicaConfig as Cfg
    cfg = config or Cfg()

    max_iterations = _get_max_iterations(mode, cfg.max_iterations)

    if cfg.use_reasoning_preamble is not None:
        include_reasoning = cfg.use_reasoning_preamble
    else:
        include_reasoning = not _is_thinking_model(cfg.model)

    tools, action_desc = _build_tools_and_action_desc(
        classification, cfg.sources, mode, include_reasoning
    )

    effective_mode = mode if include_reasoning else f"{mode}_no_reasoning"
    instructions = get_researcher_prompt(
        action_desc=action_desc,
        mode=effective_mode,
        iteration=1,
        max_iteration=max_iterations,
    )

    parts = []
    for q, a in chat_history:
        parts.append(f"User: {q}")
        short_a = a[:500] + "..." if len(a) > 500 else a
        parts.append(f"Assistant: {short_a}")
    parts.append(f"User: {query}")
    formatted_input = "\n".join(parts)

    researcher = Agent(
        name="researcher",
        model=cfg.model,
        instructions=instructions,
        tools=tools,
        temperature=0.1,
        max_steps=max_iterations * 3,
    )

    clear_collected_results()

    provider = _resolve_provider(cfg.model)
    async for event in run.stream(researcher, formatted_input, provider=provider, detailed=True):
        yield event

    # Yield collected search results
    raw_results = get_collected_results()
    seen_urls: set[str] = set()
    for r in raw_results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            yield SearchResult(
                title=r.get("title", ""),
                url=url,
                content=r.get("content", ""),
            )
