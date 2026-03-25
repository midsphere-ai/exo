"""Pluggable stop-condition detectors for the Ralph refinement loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from exo.eval.ralph.config import (  # pyright: ignore[reportMissingImports]
    LoopState,
    StopConditionConfig,
    StopType,
)

# ---------------------------------------------------------------------------
# StopDecision — result of a detector check
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StopDecision:
    """Outcome of a single detector evaluation."""

    should_stop: bool
    stop_type: StopType = StopType.NONE
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.should_stop


_CONTINUE = StopDecision(should_stop=False)


# ---------------------------------------------------------------------------
# StopDetector ABC
# ---------------------------------------------------------------------------


class StopDetector(ABC):
    """Base class for pluggable stop-condition detectors.

    Each detector examines the current :class:`LoopState` and the static
    :class:`StopConditionConfig`, returning a :class:`StopDecision`.
    """

    __slots__ = ()

    @abstractmethod
    async def check(
        self,
        state: LoopState,
        config: StopConditionConfig,
    ) -> StopDecision:
        """Return a stop decision for the current loop state."""


# ---------------------------------------------------------------------------
# Built-in detectors
# ---------------------------------------------------------------------------


class MaxIterationDetector(StopDetector):
    """Stops when the loop has reached ``config.max_iterations``."""

    __slots__ = ()

    async def check(
        self,
        state: LoopState,
        config: StopConditionConfig,
    ) -> StopDecision:
        if state.iteration >= config.max_iterations:
            return StopDecision(
                should_stop=True,
                stop_type=StopType.MAX_ITERATIONS,
                reason=f"Reached max iterations ({state.iteration}/{config.max_iterations})",
                metadata={"current": state.iteration, "max": config.max_iterations},
            )
        return _CONTINUE


class TimeoutDetector(StopDetector):
    """Stops when elapsed wall-clock time exceeds ``config.timeout``.

    A timeout of ``0.0`` (the default) disables this detector.
    """

    __slots__ = ()

    async def check(
        self,
        state: LoopState,
        config: StopConditionConfig,
    ) -> StopDecision:
        if config.timeout <= 0.0:
            return _CONTINUE
        elapsed = state.elapsed()
        if elapsed >= config.timeout:
            return StopDecision(
                should_stop=True,
                stop_type=StopType.TIMEOUT,
                reason=f"Timeout exceeded ({elapsed:.1f}s/{config.timeout:.1f}s)",
                metadata={"elapsed": elapsed, "timeout": config.timeout},
            )
        return _CONTINUE


class CostLimitDetector(StopDetector):
    """Stops when cumulative cost meets or exceeds ``config.max_cost``.

    A max_cost of ``0.0`` (the default) disables this detector.
    """

    __slots__ = ()

    async def check(
        self,
        state: LoopState,
        config: StopConditionConfig,
    ) -> StopDecision:
        if config.max_cost <= 0.0:
            return _CONTINUE
        if state.cumulative_cost >= config.max_cost:
            return StopDecision(
                should_stop=True,
                stop_type=StopType.MAX_COST,
                reason=f"Cost limit reached ({state.cumulative_cost:.3f}/{config.max_cost:.3f})",
                metadata={"current": state.cumulative_cost, "max": config.max_cost},
            )
        return _CONTINUE


class ConsecutiveFailureDetector(StopDetector):
    """Stops when consecutive failures reach ``config.max_consecutive_failures``."""

    __slots__ = ()

    async def check(
        self,
        state: LoopState,
        config: StopConditionConfig,
    ) -> StopDecision:
        if config.max_consecutive_failures <= 0:
            return _CONTINUE
        if state.consecutive_failures >= config.max_consecutive_failures:
            return StopDecision(
                should_stop=True,
                stop_type=StopType.MAX_CONSECUTIVE_FAILURES,
                reason=(
                    f"Too many consecutive failures "
                    f"({state.consecutive_failures}/{config.max_consecutive_failures})"
                ),
                metadata={
                    "current": state.consecutive_failures,
                    "max": config.max_consecutive_failures,
                },
            )
        return _CONTINUE


class ScoreThresholdDetector(StopDetector):
    """Stops when the latest score meets or exceeds ``config.score_threshold``.

    Uses the mean of all scorer values in the most recent score snapshot.
    A score_threshold of ``0.0`` (the default) disables this detector.
    """

    __slots__ = ()

    async def check(
        self,
        state: LoopState,
        config: StopConditionConfig,
    ) -> StopDecision:
        if config.score_threshold <= 0.0:
            return _CONTINUE
        latest = state.latest_score()
        if not latest:
            return _CONTINUE
        mean_score = sum(latest.values()) / len(latest)
        if mean_score >= config.score_threshold:
            return StopDecision(
                should_stop=True,
                stop_type=StopType.SCORE_THRESHOLD,
                reason=f"Score threshold met ({mean_score:.3f} >= {config.score_threshold:.3f})",
                metadata={"mean_score": mean_score, "threshold": config.score_threshold},
            )
        return _CONTINUE


# ---------------------------------------------------------------------------
# CompositeDetector — runs multiple detectors, first-match wins
# ---------------------------------------------------------------------------


class CompositeDetector(StopDetector):
    """Aggregates multiple detectors and returns the first triggered decision."""

    __slots__ = ("_detectors",)

    def __init__(self, detectors: list[StopDetector] | None = None) -> None:
        self._detectors: list[StopDetector] = list(detectors) if detectors else []

    def add(self, detector: StopDetector) -> CompositeDetector:
        """Append a detector (supports chaining)."""
        self._detectors.append(detector)
        return self

    async def check(
        self,
        state: LoopState,
        config: StopConditionConfig,
    ) -> StopDecision:
        for detector in self._detectors:
            decision = await detector.check(state, config)
            if decision.should_stop:
                return decision
        return _CONTINUE

    def __len__(self) -> int:
        return len(self._detectors)

    def __repr__(self) -> str:
        return f"CompositeDetector(detectors={len(self._detectors)})"
