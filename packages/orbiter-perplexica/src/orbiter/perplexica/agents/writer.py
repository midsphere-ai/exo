"""Writer agent — generates final cited answer using Perplexica's 'Vane' prompt."""

from __future__ import annotations

from collections.abc import AsyncIterator

from orbiter import Agent, run
from orbiter.observability.logging import get_logger  # pyright: ignore[reportMissingImports]
from orbiter.types import StreamEvent

from ..config import PerplexicaConfig
from ..prompts.instructions import get_writer_prompt
from ..types import SearchResult

_log = get_logger(__name__)


def _resolve_provider(model: str):
    try:
        from orbiter.models import get_provider
        return get_provider(model)
    except Exception as exc:
        _log.warning("provider resolution failed for %s: %s", model, exc)
        return None


def format_results_as_context(search_results: list[SearchResult]) -> str:
    """Format search results as <result> context blocks for the writer prompt."""
    if not search_results:
        return ""
    parts = []
    for i, r in enumerate(search_results, 1):
        parts.append(
            f'<result index={i} title="{r.title}" url="{r.url}">\n{r.content}\n</result>'
        )
    return "\n".join(parts)


async def write_answer(
    query: str,
    search_results: list[SearchResult],
    chat_history: list[tuple[str, str]],
    system_instructions: str = "",
    mode: str = "balanced",
    config: PerplexicaConfig | None = None,
) -> str:
    """Generate final answer using Perplexica's writer prompt with context."""
    from ..config import PerplexicaConfig as Cfg
    cfg = config or Cfg()
    _log.debug("writer query=%r sources=%d model=%s", query, len(search_results), cfg.model)

    context = format_results_as_context(search_results)
    instructions = get_writer_prompt(context, system_instructions, mode, cfg.max_writer_words)

    # Build messages with chat history
    parts = []
    for q, a in chat_history:
        parts.append(f"User: {q}")
        short_a = a[:500] + "..." if len(a) > 500 else a
        parts.append(f"Assistant: {short_a}")
    parts.append(f"User: {query}")
    formatted_input = "\n".join(parts)

    writer = Agent(
        name="writer",
        model=cfg.model,
        instructions=instructions,
        temperature=0.2,
        max_steps=1,
    )

    provider = _resolve_provider(cfg.model)
    result = await run(writer, formatted_input, provider=provider)
    _log.info("writer done len=%d", len(result.output))
    return result.output


async def stream_write_answer(
    query: str,
    search_results: list[SearchResult],
    chat_history: list[tuple[str, str]],
    system_instructions: str = "",
    mode: str = "balanced",
    config: PerplexicaConfig | None = None,
) -> AsyncIterator[StreamEvent]:
    """Stream writer text tokens as they are generated."""
    from ..config import PerplexicaConfig as Cfg
    cfg = config or Cfg()
    _log.debug("stream_writer query=%r sources=%d model=%s", query, len(search_results), cfg.model)

    context = format_results_as_context(search_results)
    instructions = get_writer_prompt(context, system_instructions, mode, cfg.max_writer_words)

    parts = []
    for q, a in chat_history:
        parts.append(f"User: {q}")
        short_a = a[:500] + "..." if len(a) > 500 else a
        parts.append(f"Assistant: {short_a}")
    parts.append(f"User: {query}")
    formatted_input = "\n".join(parts)

    writer = Agent(
        name="writer",
        model=cfg.model,
        instructions=instructions,
        temperature=0.2,
        max_steps=1,
    )

    provider = _resolve_provider(cfg.model)
    async for event in run.stream(writer, formatted_input, provider=provider):
        yield event
