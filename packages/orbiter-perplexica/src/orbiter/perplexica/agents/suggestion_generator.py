"""Follow-up suggestion generator agent — structured output."""

from __future__ import annotations

import json

from orbiter import Agent, run
from orbiter.observability.logging import get_logger  # pyright: ignore[reportMissingImports]

from ..config import PerplexicaConfig
from ..prompts.instructions import get_suggestion_prompt
from ..types import SuggestionOutput

_log = get_logger(__name__)


def _resolve_provider(model: str):
    try:
        from orbiter.models import get_provider
        return get_provider(model)
    except Exception as exc:
        _log.warning("provider resolution failed for %s: %s", model, exc)
        return None


async def generate_suggestions(
    chat_history: list[tuple[str, str]],
    config: PerplexicaConfig | None = None,
) -> list[str]:
    """Generate follow-up suggestions based on conversation history."""
    _log.debug("generating suggestions history_len=%d", len(chat_history))
    from ..config import PerplexicaConfig as Cfg
    cfg = config or Cfg()

    # Format conversation for the suggestion generator
    parts = []
    for q, a in chat_history:
        parts.append(f"User: {q}")
        short_a = a[:500] + "..." if len(a) > 500 else a
        parts.append(f"Assistant: {short_a}")
    formatted_input = "\n".join(parts)

    suggestion_agent = Agent(
        name="suggestion_generator",
        model=cfg.fast_model,
        instructions=get_suggestion_prompt(),
        output_type=SuggestionOutput,
        temperature=0.8,
        max_steps=1,
    )

    provider = _resolve_provider(cfg.fast_model)
    result = await run(suggestion_agent, formatted_input, provider=provider)

    try:
        output = SuggestionOutput.model_validate_json(result.output)
        return output.suggestions
    except Exception:
        _log.warning("suggestion parse failed, trying raw JSON")
        # Try to parse as raw JSON
        try:
            data = json.loads(result.output)
            return data.get("suggestions", [])
        except Exception:
            _log.warning("suggestion parse failed completely")
            return []
