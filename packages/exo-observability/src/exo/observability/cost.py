"""Cost estimation and tracking for LLM calls.

Records per-call costs based on model pricing tables and provides
breakdown and aggregation functions. Thread-safe for concurrent use.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger("exo.cost")


@dataclass(frozen=True)
class ModelPricing:
    """Pricing definition for a model pattern.

    *model_pattern* is a regex matched against model name strings.
    Costs are per 1,000 tokens.
    """

    model_pattern: str
    input_cost_per_1k: float
    output_cost_per_1k: float


@dataclass(frozen=True)
class CostEntry:
    """A single recorded LLM cost entry."""

    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Built-in pricing table
# ---------------------------------------------------------------------------

_DEFAULT_PRICING: list[ModelPricing] = [
    # OpenAI
    ModelPricing(
        model_pattern=r"gpt-4o-mini", input_cost_per_1k=0.00015, output_cost_per_1k=0.0006
    ),
    ModelPricing(model_pattern=r"gpt-4o", input_cost_per_1k=0.0025, output_cost_per_1k=0.01),
    # Anthropic
    ModelPricing(
        model_pattern=r"claude-sonnet-4-5-20250514",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
    ),
    ModelPricing(
        model_pattern=r"claude-haiku-3-5-20241022",
        input_cost_per_1k=0.0008,
        output_cost_per_1k=0.004,
    ),
    # Google
    ModelPricing(
        model_pattern=r"gemini-2\.0-flash", input_cost_per_1k=0.0001, output_cost_per_1k=0.0004
    ),
]


class CostTracker:
    """Tracks LLM call costs with thread-safe recording and aggregation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[CostEntry] = []
        self._pricing: list[ModelPricing] = list(_DEFAULT_PRICING)

    # -- Pricing management ---------------------------------------------------

    def register_pricing(self, pricing: ModelPricing) -> None:
        """Add or update a model pricing entry.

        New entries are prepended so they take priority over defaults.
        """
        with self._lock:
            self._pricing.insert(0, pricing)
        logger.debug(
            "registered pricing for pattern %r: in=$%.6f/1k out=$%.6f/1k",
            pricing.model_pattern,
            pricing.input_cost_per_1k,
            pricing.output_cost_per_1k,
        )

    def _find_pricing(self, model: str) -> ModelPricing | None:
        """Find the first pricing entry whose pattern matches *model*."""
        for p in self._pricing:
            if re.search(p.model_pattern, model):
                return p
        return None

    # -- Recording ------------------------------------------------------------

    def record(self, model: str, input_tokens: int, output_tokens: int) -> CostEntry:
        """Record a single LLM call and return the computed :class:`CostEntry`."""
        pricing = self._find_pricing(model)

        if pricing is None:
            logger.warning("No pricing found for model %r — cost recorded as 0", model)
            input_cost = 0.0
            output_cost = 0.0
        else:
            input_cost = (input_tokens / 1000) * pricing.input_cost_per_1k
            output_cost = (output_tokens / 1000) * pricing.output_cost_per_1k

        entry = CostEntry(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
        )

        with self._lock:
            self._entries.append(entry)

        logger.debug(
            "recorded cost: model=%s tokens=%d+%d cost=$%.6f",
            model,
            input_tokens,
            output_tokens,
            entry.total_cost,
        )
        return entry

    # -- Queries --------------------------------------------------------------

    def get_total(self) -> float:
        """Return total cost across all recorded entries."""
        with self._lock:
            return sum(e.total_cost for e in self._entries)

    def get_breakdown(self) -> dict[str, float]:
        """Return per-model cost breakdown."""
        with self._lock:
            breakdown: dict[str, float] = {}
            for e in self._entries:
                breakdown[e.model] = breakdown.get(e.model, 0.0) + e.total_cost
            return breakdown

    def get_entries(self, since: datetime | None = None) -> list[CostEntry]:
        """Return entries, optionally filtered to those after *since*."""
        with self._lock:
            if since is None:
                return list(self._entries)
            return [e for e in self._entries if e.timestamp >= since]

    # -- Management -----------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded entries (pricing table is preserved)."""
        with self._lock:
            self._entries.clear()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_global_tracker = CostTracker()


def get_tracker() -> CostTracker:
    """Return the global cost tracker."""
    return _global_tracker


def reset() -> None:
    """Reset global cost tracker — for testing only."""
    _global_tracker.reset()
