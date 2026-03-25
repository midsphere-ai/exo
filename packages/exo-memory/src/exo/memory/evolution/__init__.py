"""Memory evolution strategies and composable pipelines.

Provides a base class for memory evolution algorithms that transform
lists of MemoryItem objects, plus pipeline operators for sequential
(>>) and parallel (|) composition.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Literal

from exo.memory.base import MemoryItem  # pyright: ignore[reportMissingImports]


class MemoryEvolutionStrategy(ABC):
    """Base class for memory evolution algorithms.

    Subclasses implement ``evolve()`` to transform a list of MemoryItem objects.
    Strategies compose via ``>>`` (sequential) and ``|`` (parallel) operators.

    Attributes:
        name: Human-readable identifier for this strategy.
    """

    name: str

    @abstractmethod
    async def evolve(
        self,
        items: list[MemoryItem],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Transform memory items according to this strategy.

        Args:
            items: Input memory items to evolve.
            context: Optional context dict (e.g., feedback, model config).

        Returns:
            Transformed list of memory items.
        """
        ...

    def __rshift__(self, other: MemoryEvolutionStrategy) -> MemoryEvolutionPipeline:
        """Sequential composition: ``self >> other``."""
        left = self._flatten("sequential")
        right = other._flatten("sequential")
        return MemoryEvolutionPipeline([*left, *right], mode="sequential")

    def __or__(self, other: MemoryEvolutionStrategy) -> MemoryEvolutionPipeline:
        """Parallel composition: ``self | other``."""
        left = self._flatten("parallel")
        right = other._flatten("parallel")
        return MemoryEvolutionPipeline([*left, *right], mode="parallel")

    def _flatten(self, mode: str) -> list[MemoryEvolutionStrategy]:
        """Return constituent strategies if this is a pipeline of the same mode."""
        return [self]


def _merge_results(results: list[list[MemoryItem]]) -> list[MemoryItem]:
    """Merge parallel evolution results (union by item ID, last-write-wins)."""
    seen: dict[str, MemoryItem] = {}
    for result_list in results:
        for item in result_list:
            seen[item.id] = item
    return list(seen.values())


class MemoryEvolutionPipeline(MemoryEvolutionStrategy):
    """Composes multiple strategies sequentially or in parallel.

    Sequential (``>>``): Output of each strategy feeds into the next.
    Parallel (``|``): All strategies run on the same input, results are
    merged (union by item ID, last-write-wins for duplicates).

    Attributes:
        name: Always ``"pipeline"``.
    """

    name: str = "pipeline"

    def __init__(
        self,
        strategies: list[MemoryEvolutionStrategy],
        mode: Literal["sequential", "parallel"] = "sequential",
    ) -> None:
        self._strategies = list(strategies)
        self._mode = mode

    @property
    def strategies(self) -> list[MemoryEvolutionStrategy]:
        """The constituent strategies in this pipeline."""
        return list(self._strategies)

    @property
    def mode(self) -> str:
        """The composition mode: ``'sequential'`` or ``'parallel'``."""
        return self._mode

    async def evolve(
        self,
        items: list[MemoryItem],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Execute the pipeline on the given items.

        Args:
            items: Input memory items.
            context: Optional context dict passed to each strategy.

        Returns:
            Transformed list of memory items.
        """
        if self._mode == "sequential":
            result = items
            for strategy in self._strategies:
                result = await strategy.evolve(result, context=context)
            return result

        # parallel
        results = await asyncio.gather(
            *(s.evolve(items, context=context) for s in self._strategies)
        )
        return _merge_results(list(results))

    def _flatten(self, mode: str) -> list[MemoryEvolutionStrategy]:
        """Flatten if composing with the same mode to avoid nesting."""
        if self._mode == mode:
            return list(self._strategies)
        return [self]


from exo.memory.evolution.ace import ACEStrategy  # pyright: ignore[reportMissingImports]
from exo.memory.evolution.reasoning_bank import (  # pyright: ignore[reportMissingImports]
    ReasoningBankStrategy,
    ReasoningEntry,
)
from exo.memory.evolution.reme import ReMeStrategy  # pyright: ignore[reportMissingImports]

__all__ = [
    "ACEStrategy",
    "MemoryEvolutionPipeline",
    "MemoryEvolutionStrategy",
    "ReMeStrategy",
    "ReasoningBankStrategy",
    "ReasoningEntry",
]
