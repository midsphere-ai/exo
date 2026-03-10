"""ReMe — Relevant Memory evolution strategy.

Extracts success/failure patterns from memory items with when-to-use
metadata, then deduplicates by content similarity.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol, runtime_checkable

from orbiter.memory.base import MemoryItem  # pyright: ignore[reportMissingImports]
from orbiter.memory.evolution import MemoryEvolutionStrategy  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@runtime_checkable
class PatternModel(Protocol):
    """Protocol for the LLM callable used by ``extract_patterns()``.

    Accepts a prompt string and returns a JSON string containing
    extracted patterns.
    """

    async def __call__(self, prompt: str) -> str: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _keyword_similarity(a: str, b: str) -> float:
    """Simple keyword overlap (Jaccard) similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    union = words_a | words_b
    return len(words_a & words_b) / len(union)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = (
    "Analyze the following memory items and extract success and failure patterns.\n"
    "For each pattern, provide:\n"
    "- content: a concise description of the pattern\n"
    "- pattern_type: either \"success\" or \"failure\"\n"
    "- when_to_use: a short description of when this pattern applies\n\n"
    "Memory items:\n{items}\n\n"
    "Respond with a JSON array of objects, each with keys: "
    "content, pattern_type, when_to_use.\n"
    'Example: [{{"content": "...", "pattern_type": "success", '
    '"when_to_use": "..."}}]'
)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class ReMeStrategy(MemoryEvolutionStrategy):
    """Relevant Memory — extracts success/failure patterns with when-to-use metadata.

    Uses an LLM to identify patterns from memory items, then deduplicates
    the resulting pattern set by content similarity.

    Args:
        similarity_threshold: Keyword similarity threshold for deduplication.
            Default ``0.85``.
    """

    name: str = "reme"

    def __init__(self, *, similarity_threshold: float = 0.85) -> None:
        self._similarity_threshold = similarity_threshold

    # -- public API ---------------------------------------------------------

    async def evolve(
        self,
        items: list[MemoryItem],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Extract success/failure patterns from items and deduplicate.

        If ``context`` contains a ``"model"`` key with a callable matching
        :class:`PatternModel`, it is used for LLM-based pattern extraction.
        Without a model, items are returned deduplicated as-is.

        Args:
            items: Input memory items to evolve.
            context: Optional context dict.  Pass ``{"model": model}`` to
                enable LLM-based pattern extraction.

        Returns:
            Deduplicated list of pattern memory items.
        """
        if not items:
            return []

        model: PatternModel | None = (context or {}).get("model")
        if model is not None:
            patterns = await self.extract_patterns(items, model)
        else:
            patterns = list(items)

        return self._deduplicate(patterns)

    async def extract_patterns(
        self,
        items: list[MemoryItem],
        model: PatternModel,
    ) -> list[MemoryItem]:
        """Extract success and failure patterns from memory items via LLM.

        Each extracted pattern is returned as a new :class:`MemoryItem` with
        ``when_to_use`` stored in ``metadata.extra``.

        Args:
            items: Source memory items to analyze.
            model: Async callable that accepts a prompt and returns JSON.

        Returns:
            List of new MemoryItem objects representing extracted patterns.
        """
        items_text = "\n".join(
            f"- {item.content}" for item in items
        )
        prompt = _EXTRACT_PROMPT.format(items=items_text)

        raw = await model(prompt)
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("ReMe: failed to parse model output as JSON")
            return list(items)

        if not isinstance(data, list):
            logger.warning("ReMe: model output is not a JSON array")
            return list(items)

        from orbiter.memory.base import MemoryMetadata  # pyright: ignore[reportMissingImports]

        patterns: list[MemoryItem] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            content = entry.get("content", "")
            if not content:
                continue
            when_to_use = entry.get("when_to_use", "")
            pattern_type = entry.get("pattern_type", "")
            patterns.append(
                MemoryItem(
                    content=content,
                    memory_type="pattern",
                    metadata=MemoryMetadata(
                        extra={
                            "when_to_use": when_to_use,
                            "pattern_type": pattern_type,
                        },
                    ),
                )
            )

        return patterns if patterns else list(items)

    # -- internals ----------------------------------------------------------

    def _deduplicate(self, items: list[MemoryItem]) -> list[MemoryItem]:
        """Remove near-duplicate items by keyword similarity.

        When two items exceed ``similarity_threshold``, the item with more
        content is kept and the shorter one is dropped.
        """
        if len(items) <= 1:
            return list(items)

        keep = [True] * len(items)

        for i in range(len(items)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(items)):
                if not keep[j]:
                    continue
                sim = _keyword_similarity(items[i].content, items[j].content)
                if sim >= self._similarity_threshold:
                    # Keep the longer / more detailed item
                    if len(items[j].content) > len(items[i].content):
                        keep[i] = False
                        break
                    else:
                        keep[j] = False

        return [items[i] for i in range(len(items)) if keep[i]]


__all__ = ["PatternModel", "ReMeStrategy"]
