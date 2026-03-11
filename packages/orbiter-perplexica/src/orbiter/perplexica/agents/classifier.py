"""Classifier agent — determines search intent using Perplexica's classification schema."""

from __future__ import annotations

from orbiter import Agent, run

from ..config import PerplexicaConfig
from ..prompts.instructions import CLASSIFIER_PROMPT
from ..types import ClassifierOutput


def _resolve_provider(model: str):
    try:
        from orbiter.models import get_provider
        return get_provider(model)
    except Exception:
        return None


async def classify(query: str, chat_history: list[tuple[str, str]], config: PerplexicaConfig) -> ClassifierOutput:
    """Classify a query using Perplexica's classifier prompt."""
    # Format chat history for the classifier
    parts = []
    for q, a in chat_history:
        parts.append(f"User: {q}")
        short_a = a[:500] + "..." if len(a) > 500 else a
        parts.append(f"Assistant: {short_a}")
    parts.append(f"User: {query}")
    formatted_input = "\n".join(parts) if parts else query

    classifier = Agent(
        name="classifier",
        model=config.fast_model,
        instructions=CLASSIFIER_PROMPT,
        output_type=ClassifierOutput,
        temperature=0.2,
        max_steps=1,
    )

    provider = _resolve_provider(config.fast_model)
    result = await run(classifier, formatted_input, provider=provider)

    try:
        return ClassifierOutput.model_validate_json(result.output)
    except Exception:
        # Fallback: no skip, just web search
        from ..types import Classification
        return ClassifierOutput(
            classification=Classification(),
            standalone_follow_up=query,
        )
